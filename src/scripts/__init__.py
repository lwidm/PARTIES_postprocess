# -- src/scripts/__init__.py

"""
Subpackage all scripts
"""

from . import (
    run_floc_analysis,
    run_fluid_wall_analysis,
    run_statist_steady_analysis,
    run_all_plots,
    run_phi_eulerian,
    run_double_check,
)

__all__ = [
    "run_floc_analysis",
    "run_fluid_wall_analysis",
    "run_statist_steady_analysis",
    "run_all_plots",
    "run_phi_eulerian",
    "run_double_check",
]
