# -- __init__.py

"""
Lukas Widmer's (lwidm / lukwidmer / x-lwidmer) python package for postprocessing PARTIES simulations
"""

from . import myio, theory, fluid, flocs, scripts, plotting, globals

__version__ = "1.0.0"
__author__ = "Lukas Widmer"
__all__ = ["myio", "theory", "fluid", "flocs", "scripts", "plotting", "globals"]

if __name__ == "__main__":
    print(f"MyIO package {__version__}")
