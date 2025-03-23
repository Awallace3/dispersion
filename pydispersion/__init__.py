"""
Dispersion module for Python with C++ acceleration.

This package provides dispersion energy calculations using C++ implementations
for performance-critical parts combined with Python utilities for ease of use.
"""

__version__ = "0.1.0"

try:
    # Import the compiled C++ modules
    from . import dispersion
    from .dispersion import disp
except ImportError:
    import warnings
    warnings.warn("Could not import C++ extension modules. Some functionality will be unavailable.")

# Import Python utilities
from .utils import *
from .compute import *

# Cleanup function for uninstallation
import os
import atexit
import pkg_resources

def _cleanup_on_uninstall():
    # This function will be called when Python exits during uninstallation
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if conda_prefix:
        binary_path = os.path.join(conda_prefix, 'bin', 'dftd4')
        if os.path.exists(binary_path):
            try:
                os.remove(binary_path)
                print(f"Removed dftd4 binary from {binary_path}")
            except OSError as e:
                print(f"Error removing dftd4 binary: {e}")

# Register cleanup function
atexit.register(_cleanup_on_uninstall)
