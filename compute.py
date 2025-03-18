"""
Analysis functions for dispersion calculations.

These functions help analyze dispersion energies, compare different methods,
and calculate statistics.
"""
import numpy as np


def has_cpp_extensions():
    """
    Check if C++ extensions are available.
    
    Returns
    -------
    bool
        True if the C++ extensions are available, False otherwise
    """
    try:
        from . import dispersion
        return True
    except ImportError:
        return False


def calculate_dispersion_energy(positions, cartesians, c6s, method='BJ', params=None):
    """
    Calculate dispersion energy using the specified method.
    
    Parameters
    ----------
    positions : numpy.ndarray
        Array of atomic numbers
    cartesians : numpy.ndarray
        Array of atomic coordinates
    c6s : numpy.ndarray
        C6 coefficient matrix
    method : str
        Method to use: 'BJ' or 'TT'
    params : numpy.ndarray, optional
        Parameters for dispersion calculation. If None, default parameters are used.
        
    Returns
    -------
    float
        Dispersion energy in kcal/mol
    """
    if not has_cpp_extensions():
        raise ImportError("C++ extensions are not available. Cannot calculate dispersion energy.")
    
    from . import dispersion
    
    if params is None:
        if method.upper() == 'BJ':
            params = np.array([1.0, 1.0, 0.55, 2.0], dtype=np.float64)
        elif method.upper() == 'TT':
            params = np.array([1.0, 1.0, -0.33, 4.39], dtype=np.float64)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'BJ' or 'TT'.")
    
    if method.upper() == 'BJ':
        energy = dispersion.disp.disp_2B(positions, cartesians, c6s, params)
    elif method.upper() == 'TT':
        energy = dispersion.disp.disp_2B_TT(positions, cartesians, c6s, params)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'BJ' or 'TT'.")
    
    return hartree_to_kcalmol(energy)


def calculate_interaction_energy(positions, cartesians, c6s, indices_a, indices_b=None, 
                               method='BJ', params=None, atm=True, atm_params=None):
    """
    Calculate interaction energy between two fragments.
    
    Parameters
    ----------
    positions : numpy.ndarray
        Array of atomic numbers
    cartesians : numpy.ndarray
        Array of atomic coordinates
    c6s : numpy.ndarray
        C6 coefficient matrix
    indices_a : list or numpy.ndarray
        Indices of atoms in fragment A
    indices_b : list or numpy.ndarray, optional
        Indices of atoms in fragment B. If not provided, all atoms not in A are used.
    method : str
        Method to use: 'BJ' or 'TT'
    params : numpy.ndarray, optional
        Parameters for dispersion calculation. If None, default parameters are used.
    atm : bool
        Whether to include ATM (three-body) dispersion
    atm_params : numpy.ndarray, optional
        Parameters for ATM dispersion. If None, default parameters are used.
        
    Returns
    -------
    float
        Interaction energy in kcal/mol
    """
    if not has_cpp_extensions():
        raise ImportError("C++ extensions are not available. Cannot calculate interaction energy.")
    
    from . import dispersion
    
    indices_a = np.array(indices_a)
    
    if indices_b is None:
        # All indices not in A
        indices_b = np.array([i for i in range(len(positions)) if i not in indices_a])
    else:
        indices_b = np.array(indices_b)
    
    pA = positions[indices_a]
    cA = cartesians[indices_a]
    pB = positions[indices_b]
    cB = cartesians[indices_b]
    
    # Extract C6 coefficients for fragments
    n_atoms = len(positions)
    c6s_A = np.zeros((len(indices_a), len(indices_a)), dtype=np.float64)
    c6s_B = np.zeros((len(indices_b), len(indices_b)), dtype=np.float64)
    
    # Fill in C6 coefficients for fragments
    for i in range(len(indices_a)):
        for j in range(len(indices_a)):
            c6s_A[i, j] = c6s[indices_a[i], indices_a[j]]
    
    for i in range(len(indices_b)):
        for j in range(len(indices_b)):
            c6s_B[i, j] = c6s[indices_b[i], indices_b[j]]
    
    if params is None:
        if method.upper() == 'BJ':
            params = np.array([1.0, 1.0, 0.55, 2.0], dtype=np.float64)
        elif method.upper() == 'TT':
            params = np.array([1.0, 1.0, -0.33, 4.39], dtype=np.float64)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'BJ' or 'TT'.")
    
    if atm and atm_params is None:
        if method.upper() == 'BJ':
            atm_params = np.array([1.0, 1.0, 0.55, 2.0, 0.75], dtype=np.float64)
        elif method.upper() == 'TT':
            atm_params = np.array([1.0, 1.0, -0.31, 3.43, 0.75], dtype=np.float64)
    
    if method.upper() == 'BJ':
        if atm:
            c6s_ATM = c6s.copy()  # In real case, you might use different coefficients for ATM
            c6s_ATM_A = c6s_A.copy()
            c6s_ATM_B = c6s_B.copy()
            energy = dispersion.disp.disp_2B_BJ_ATM_CHG(
                positions, cartesians, c6s, c6s_ATM,
                pA, cA, c6s_A, c6s_ATM_A,
                pB, cB, c6s_B, c6s_ATM_B,
                params, atm_params
            )
        else:
            energy = dispersion.disp.disp_2B_dimer(
                positions, cartesians, c6s,
                pA, cA, c6s_A,
                pB, cB, c6s_B,
                params
            )
    elif method.upper() == 'TT':
        if atm:
            c6s_ATM = c6s.copy()  # In real case, you might use different coefficients for ATM
            c6s_ATM_A = c6s_A.copy()
            c6s_ATM_B = c6s_B.copy()
            energy = dispersion.disp.disp_2B_TT_ATM_TT(
                positions, cartesians, c6s, c6s_ATM,
                pA, cA, c6s_A, c6s_ATM_A,
                pB, cB, c6s_B, c6s_ATM_B,
                params, atm_params
            )
        else:
            energy = dispersion.disp.disp_2B_TT_dimer(
                positions, cartesians, c6s,
                pA, cA, c6s_A,
                pB, cB, c6s_B,
                params
            )
    
    return hartree_to_kcalmol(energy)
