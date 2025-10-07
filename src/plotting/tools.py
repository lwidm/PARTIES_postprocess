# -- src/plotting/tools.py

from matplotlib import pyplot as plt
from matplotlib.axes import Axes


def update_rcParams():
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern"],
            "text.latex.preamble": r"\usepackage{amsmath}",
        }
    )


def format_plot_axes(axes: Axes) -> Axes:
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    axes.spines["left"].set_linewidth(1.2)
    axes.spines["bottom"].set_linewidth(1.0)
    axes.tick_params(axis="both", which="both", direction="out", labelsize=12)
    axes.legend(frameon=False, fontsize=12)
    plt.tight_layout()
    return axes
