# -- src/scripts/__init__.py

"""
Subpackage all scripts
"""

from . import (
    run_fluid_wall_analysis,
    run_statist_steady_analysis,
    run_all_plots,
    run_phi_eulerian,
    run_double_check,
    run_get_start_steady,
    run_plot_biggest_floc,
    run_plot_floc_slice,
)

__all__ = [
    "run_fluid_wall_analysis",
    "run_statist_steady_analysis",
    "run_all_plots",
    "run_phi_eulerian",
    "run_double_check",
    "run_get_start_steady",
    "run_plot_biggest_floc",
    "run_plot_floc_slice",
]
