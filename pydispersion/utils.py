"""
Utility functions for working with dispersion calculations.

These functions provide convenience methods for preparing input data,
handling file formats, and converting between different units.
"""
import numpy as np


def split_molecule(positions, cartesians, indices_a, indices_b=None):
    """
    Split molecule into two fragments.
    
    Parameters
    ----------
    positions : numpy.ndarray
        Array of atomic numbers
    cartesians : numpy.ndarray
        Array of atomic coordinates
    indices_a : list or numpy.ndarray
        Indices of atoms in fragment A
    indices_b : list or numpy.ndarray, optional
        Indices of atoms in fragment B. If not provided, all atoms not in A are used.
        
    Returns
    -------
    tuple
        (pA, cA, pB, cB) where:
        - pA: positions for fragment A
        - cA: cartesians for fragment A
        - pB: positions for fragment B
        - cB: cartesians for fragment B
    """
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
    
    return pA, cA, pB, cB


def prepare_default_params(method='BJ'):
    """
    Prepare default parameters for dispersion calculations.
    
    Parameters
    ----------
    method : str
        Method to use: 'BJ' (Becke-Johnson) or 'TT' (Tang-Toennies)
        
    Returns
    -------
    tuple
        (params_2B, params_ATM) where:
        - params_2B: parameters for two-body dispersion
        - params_ATM: parameters for ATM dispersion
    """
    if method.upper() == 'BJ':
        # Default parameters for BJ damping
        params_2B = np.array([1.0, 1.0, 0.55, 2.0], dtype=np.float64)
        params_ATM = np.array([1.0, 1.0, 0.55, 2.0, 0.75], dtype=np.float64)
    elif method.upper() == 'TT':
        # Default parameters for Tang-Toennies damping
        params_2B = np.array([1.0, 1.0, -0.33, 4.39], dtype=np.float64)
        params_ATM = np.array([1.0, 1.0, -0.31, 3.43, 0.75], dtype=np.float64)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'BJ' or 'TT'.")
    
    return params_2B, params_ATM
