# -- myio/__init__.py
"""
MyIO subpackage for data input/output operations.

This package provides:
- myio: NumPy-based I/O operations
- myio_torch: PyTorch tensor I/O operations
- utexas: reading of UT Austin data files
"""

from . import myio
from . import myio_torch
from . import utexas

from .myio import (
    load_columns_from_txt_numpy,
    list_parties_data_files,
    read_domain_info,
    read_particle_data,
    combine_dicts,
    read_coh_range,
)
from .myio_torch import load_columns_from_txt_torch

__all__ = [
    "myio",
    "myio_torch",
    "utexas",
    "load_columns_from_txt_numpy",
    "load_columns_from_txt_torch",
    "read_domain_info",
    "read_coh_range",
    "list_parties_data_files",
    "read_particle_data",
    "combine_dicts",
]
