# -- src/fluid/__init__.py

"""
Subpackage for obtaining and processing floc data from PARTIES output files.

This package provides:
- floc_statistics: Functions to calculate fluid flow statistics (like mean, reynolds stresses and more)
"""

from . import floc_statistics

from .find_flocs import find_flocs

__all__ = [
    "floc_statistics",
    "find_flocs",
]
