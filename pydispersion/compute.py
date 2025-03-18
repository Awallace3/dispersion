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
    
    return energy
