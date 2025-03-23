"""
Dispersion module for Python with C++ acceleration.

This package provides dispersion energy calculations using C++ implementations
for performance-critical parts combined with Python utilities for ease of use.
"""

__version__ = "0.1.0"

try:
    # Import the compiled C++ modules
    import dispersion
except ImportError:
    import warnings
    warnings.warn("Could not import C++ extension modules. Some functionality will be unavailable.")

# Import Python utilities
from .utils import *
from .compute import *
