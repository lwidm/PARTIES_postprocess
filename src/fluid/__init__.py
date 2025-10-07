# -- src/fluid/__init__.py

"""
Subpackage for obtaining and processing fluid data from PARTIES output files.

This package provides:
- flow_statistics: Functions to calculate fluid flow statistics (like mean, reynolds stresses and more)
"""

from . import flow_statistics

__all__ = [
    "flow_statistics",
]
