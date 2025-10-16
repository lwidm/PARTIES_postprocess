from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union, Literal

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from src import globals

NumericArray = Union[np.ndarray, float, int]
PlotMethod = Literal[
    "plot", "semilogx", "semilogy", "loglog", "pcolormesh", "imshow", "scatter"
]


@dataclass
class PlotSeries:
    data: Dict[str, Any]
    x_key: Optional[str] = "x"
    y_key: Optional[str] = "y"
    label: Optional[str] = None
    color: Optional[str] = None
    linestyle: Optional[str] = None
    marker: Optional[str] = None
    linewidth: Optional[float] = None
    plot_method: Optional[PlotMethod] = "plot"
    kwargs: Dict[str, Any] = field(default_factory=dict)


SeriesLike = Union[PlotSeries, Sequence[PlotSeries]]


# ------------------------- rc / axis helpers -------------------------


def update_rcParams() -> None:
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


def _clean_plot_kwargs(series: PlotSeries) -> Dict[str, Any]:
    kw: Dict[str, Any] = {}
    if series.label is not None:
        kw["label"] = series.label
    if series.color is not None:
        kw["color"] = series.color
    if series.linestyle is not None:
        kw["linestyle"] = series.linestyle
    if series.marker is not None:
        kw["marker"] = series.marker
    if series.linewidth is not None:
        kw["linewidth"] = series.linewidth
    kw.update(series.kwargs or {})
    return kw


def _plot_one(ax: Axes, series: PlotSeries) -> None:
    method = (series.plot_method or "plot").lower()
    if method in ("plot", "semilogx", "semilogy", "loglog", "scatter"):
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
        plot_kwargs = _clean_plot_kwargs(series)
        if method == "plot":
            ax.plot(*(ax_values_tuple), **plot_kwargs)
        elif method == "semilogx":
            ax.semilogx(*(ax_values_tuple), **plot_kwargs)
        elif method == "semilogy":
            ax.semilogy(*(ax_values_tuple), **plot_kwargs)
        elif method == "loglog":
            ax.loglog(x, y, **plot_kwargs)
        elif method == "scatter":
            if x is None or y is None:
                raise ValueError(
                    "Failed to extract either x or y when trying to create scatter plot"
                )
            ax.scatter(x, y, **plot_kwargs)
    elif method == "pcolormesh":
        plot_kwargs = _clean_plot_kwargs(series)
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
        plot_kwargs = _clean_plot_kwargs(series)
        ax.imshow(C, **plot_kwargs)
    else:
        raise ValueError("Plot method specified not implemented yet")


def generic_line_plot(
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
    update_rcParams()
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
