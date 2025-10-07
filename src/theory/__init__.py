# -- thoery/__init__.py
"""
Subpackage for physics from theory.

This package provides:
- law_of_the_wall: function to calculate law of the wall profiles and fitting of its parameters
"""

from . import law_of_the_wall

__all__ = [
    "law_of_the_wall",
]
