from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union, Literal

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib import colors as mcolors
import colorsys

from src import globals

NumericArray = Union[np.ndarray, float, int]
PlotMethod = Literal[
    "plot", "semilogx", "semilogy", "loglog", "pcolormesh", "imshow", "scatter", "bar", "err_plot", "err_semilogx", "err_semilogy", "err_loglog"
]


@dataclass
class PlotSeries:
    data: Dict[str, Any]
    x_key: Optional[str] = "x"
    y_key: Optional[str] = "y"
    plot_method: Optional[PlotMethod] = "plot"
    kwargs: Dict[str, Any] = field(default_factory=dict)


SeriesLike = Union[PlotSeries, Sequence[PlotSeries]]


# ------------------------- rc / axis helpers -------------------------


def update_plot_params() -> None:
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
    # legend handling is left to callers; keep the frame off by default
    try:
        axes.legend(frameon=False, fontsize=12)
    except Exception:
        # no legend present yet
        pass
    plt.tight_layout()
    return axes


# ------------------------- generic plotting helpers -------------------------


def _extract_xy(
    series: PlotSeries,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Return numeric (x, y) arrays or (None, None) if not available."""
    x: Optional[np.ndarray] = None
    y: Optional[np.ndarray] = None
    try:
        if series.x_key is not None:
            x = np.asarray(series.data[series.x_key])
    except Exception:
        x = None
    try:
        if series.y_key is not None:
            y = np.asarray(series.data[series.y_key])
    except Exception:
        y = None
    return x, y


def _hex_to_rgb01(hexcolor):
    return mcolors.to_rgb(hexcolor)

def _adjust_color(hexcolor, lighter=0.0, sat_mul=1.0):
    """
    Return an RGB tuple (0..1) that is the input color shifted in lightness
    by `lighter` (positive => lighter, negative => darker) and saturation scaled by sat_mul.
    """
    r, g, b = _hex_to_rgb01(hexcolor)
    h, l, s = colorsys.rgb_to_hls(r, g, b)  # note: HLS (not HSL)
    l = min(max(0.0, l + lighter), 1.0)
    s = min(max(0.0, s * sat_mul), 1.0)
    r2, g2, b2 = colorsys.hls_to_rgb(h, l, s)
    return (r2, g2, b2)

def _plot_one(ax: Axes, series: PlotSeries) -> None:
    method: PlotMethod = series.plot_method or "plot"
    if method in ("plot", "semilogx", "semilogy", "loglog", "scatter", "err_plot", "err_semilogx", "err_semilogy", "err_loglog"):
        x, y = _extract_xy(series)
        ax_values_tuple: Tuple
        if x is None:
            if y is None:
                raise ValueError("Could not extract both x and y from PlotSeries")
            ax_values_tuple = (x,)
        elif y is None:
            ax_values_tuple = (y,)
        else:
            ax_values_tuple = (x, y)
        plot_kwargs = series.kwargs

        if method == "plot":
            ax.plot(*(ax_values_tuple), **plot_kwargs)
        elif method == "semilogx":
            ax.semilogx(*(ax_values_tuple), **plot_kwargs)
        elif method == "semilogy":
            ax.semilogy(*(ax_values_tuple), **plot_kwargs)
        elif method == "loglog":
            ax.loglog(x, y, **plot_kwargs)
        elif method in ["err_semilogx", "err_semilogy", "err_plot", "err_loglog"]:
            if "x_err" in series.data and "y_err" in series.data:
                ax.errorbar(*(ax_values_tuple), xerr=series.data["x_err"], yerr=series.data["y_err"], **plot_kwargs)
            elif "x_err" in series.data:
                ax.errorbar(*(ax_values_tuple), xerr=series.data["x_err"], **plot_kwargs)
            elif "y_err" in series.data:
                ax.errorbar(*(ax_values_tuple), yerr=series.data["y_err"], **plot_kwargs)
            else:
                raise ValueError('Either "x_err" or "y_err" must be defined in PlotSeries data dict')

            if method == "err_semilogx":
                ax.set_xscale('log')
            elif method == "err_semilogy":
                ax.set_yscale('log')
            elif method == "err_loglog":
                ax.set_xscale('log')
                ax.set_yscale('log')
        elif method == "scatter":
            if x is None or y is None:
                raise ValueError(
                    "Failed to extract either x or y when trying to create scatter plot"
                )
            ax.scatter(x, y, **plot_kwargs)

    elif method == "bar":
        plot_kwargs = series.kwargs
        counts: np.ndarray = series.data["counts"]
        edges: np.ndarray = series.data["edges"]
        widths = edges[1:] - edges[:-1]
        edge_lighter=-0.18
        edge_sat=1.3
        base_colour: Optional[str] = series.kwargs["color"]
        if base_colour is None:
            base_colour = "red"
        face_alpha: float = 0.38
        face_rgb: Tuple[float, float, float] = _hex_to_rgb01(base_colour)
        face_rgba: Tuple[float, float, float, float] = (face_rgb[0], face_rgb[1], face_rgb[2], face_alpha)
        edge_rgb: Tuple[float, float, float] =  _adjust_color(base_colour, lighter=edge_lighter, sat_mul=edge_sat)
        plot_kwargs.update({"align": "edge", "facecolor": face_rgba, "edgecolor": edge_rgb, "linewidth": 2.8, "zorder": 2})
        ax.bar(edges[:-1], counts, width=widths, **plot_kwargs)
    elif method == "pcolormesh":
        plot_kwargs = series.kwargs
        data = series.data
        X = data.get("X")
        Y = data.get("Y")
        C = np.asarray(data.get("C") or data.get("u"))
        # fallback: try to construct mesh from x and y
        if X is None or Y is None:
            x, y = _extract_xy(series)
            if x is None or y is None:
                raise ValueError(
                    "Failed to extract either X, Y, x or y when trying to create scatter plot"
                )
            X, Y = np.meshgrid(x, y)
        ax.pcolormesh(
            X, Y, C, shading=plot_kwargs.pop("shading", "auto"), **plot_kwargs
        )
    elif method == "imshow":
        data = series.data
        C = np.asarray(data.get("C") or data.get("u"))
        plot_kwargs = series.kwargs
        ax.imshow(C, **plot_kwargs)
    else:
        raise ValueError("Plot method specified not implemented yet")


def generic_plot(
    output_path: Union[str, Path],
    series_list: Sequence[PlotSeries],
    figsize: Tuple[float, float] = (6.5, 5.5),
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    legend_loc: Optional[str] = None,
    legend_bbox: Optional[Tuple[float, float]] = None,
    dpi: int = 300,
) -> None:
    update_plot_params()
    fig, ax = plt.subplots(figsize=figsize)

    for s in series_list:
        _plot_one(ax, s)

    if xlabel:
        ax.set_xlabel(xlabel, fontsize=14)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=14)
    if title:
        ax.set_title(title)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    legend_kwargs = {}
    if legend_loc is not None:
        legend_kwargs["loc"] = legend_loc
    if legend_bbox is not None:
        legend_kwargs["bbox_to_anchor"] = legend_bbox
    if legend_kwargs:
        ax.legend(**legend_kwargs)

    ax = format_plot_axes(ax)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=dpi)

    if not globals.on_anvil:
        plt.show()
    plt.close(fig)

def generic_hist_plot(
    output_path: Union[str, Path],
    series_list: Sequence[PlotSeries],
    figsize: Tuple[float, float] = (6.5, 5.5),
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    legend_loc: Optional[str] = None,
    legend_bbox: Optional[Tuple[float, float]] = None,
    dpi: int = 300,
) -> None:
    update_plot_params()
    fig, ax = plt.subplots(figsize=figsize)

    for s in series_list:
        _plot_one(ax, s)

    if xlabel:
        ax.set_xlabel(xlabel, fontsize=14)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=14)
    if title:
        ax.set_title(title)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    legend_kwargs = {}
    if legend_loc is not None:
        legend_kwargs["loc"] = legend_loc
    if legend_bbox is not None:
        legend_kwargs["bbox_to_anchor"] = legend_bbox
    if legend_kwargs:
        ax.legend(**legend_kwargs)

    ax = format_plot_axes(ax)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=dpi)

    if not globals.on_anvil:
        plt.show()
    plt.close(fig)
