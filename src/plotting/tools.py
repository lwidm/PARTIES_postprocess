from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union, Literal, Callable

import numpy as np
from numpy.fft import fft2, ifft2
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib import colors as mcolors
import colorsys

NumericArray = Union[np.ndarray, float, int]
PlotMethod = Literal[
    "plot",
    "semilogx",
    "semilogy",
    "loglog",
    "pcolormesh",
    "imshow",
    "scatter",
    "bar",
    "contour",
    "contourf",
    "err_plot",
    "err_semilogx",
    "err_semilogy",
    "err_loglog",
]


@dataclass
class PlotSeries:
    data: Dict[str, Any]
    x_key: Optional[str] = "x"
    y_key: Optional[str] = "y"
    plot_method: Optional[PlotMethod] = "plot"
    kwargs: Dict[str, Any] = field(default_factory=dict)


SeriesLike = Union[PlotSeries, Sequence[PlotSeries]]

default_kwargs: dict[str, dict] = {
    "kLineWidth": {"linewidth": 0.9},  # default linewidth for generic plots
    "kBarLineWidth": {"linewidth": 1.0},  # default linewidth for bar plots
    "kELineWidth": {"linewidth": 0.6},  # default elinwidth (linewidth for errorbars)
    "kECapSize": {"capsize": 2.0},  # default capsize (cap size for errorbars)
    "kECapThick": {"capthick": 0.8},  # default capthick (cap thinkness for errorbars)
    "kBarsAbove": {"barsabove": True},  # default barsabove
    "kMarkerSize": {"markersize": 5.0},  # default markersize
}


def _add_default_kwargs(defaults: list[str], plot_kwargs: dict) -> None:
    for default in defaults:
        keys = default_kwargs[default].keys()
        for key in keys:
            if key not in plot_kwargs:
                plot_kwargs[key] = default_kwargs[default][key]


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


def _gaussian_filter_2d(data: np.ndarray, sigma: float) -> np.ndarray:
    valid_mask = ~np.isnan(data)
    if not np.any(valid_mask):
        return np.full_like(data, np.nan)

    data_zeroed = np.nan_to_num(data, nan=0.0)

    kernel_size = int(4 * sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    ax = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel = kernel / np.sum(kernel)

    data_fft = fft2(data_zeroed)

    kernel_padded = np.zeros_like(data)
    kernel_padded[:kernel_size, :kernel_size] = kernel
    kernel_padded = np.roll(kernel_padded, -kernel_size // 2, axis=(0, 1))
    kernel_fft = fft2(kernel_padded)

    conv = np.real(ifft2(data_fft * kernel_fft))

    mask_fft = fft2(valid_mask.astype(float))
    norm = np.real(ifft2(mask_fft * kernel_fft))

    result = conv / np.where(norm == 0, 1, norm)

    return result


def _plot_one(ax: Axes, series: PlotSeries)  -> Any:
    method: PlotMethod = series.plot_method or "plot"

    if method in (
        "plot",
        "semilogx",
        "semilogy",
        "loglog",
        "scatter",
        "err_plot",
        "err_semilogx",
        "err_semilogy",
        "err_loglog",
    ):
        other: Any = None

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
        _add_default_kwargs(["kLineWidth", "kMarkerSize"], plot_kwargs)

        if method == "plot":
            other = ax.plot(*(ax_values_tuple), **plot_kwargs)
        elif method == "semilogx":
            other = ax.semilogx(*(ax_values_tuple), **plot_kwargs)
        elif method == "semilogy":
            other = ax.semilogy(*(ax_values_tuple), **plot_kwargs)
        elif method == "loglog":
            other = ax.loglog(x, y, **plot_kwargs)
        elif method in ["err_semilogx", "err_semilogy", "err_plot", "err_loglog"]:
            _add_default_kwargs(
                ["kELineWidth", "kECapSize", "kECapThick", "kBarsAbove"], plot_kwargs
            )
            if "x_err" in series.data and "y_err" in series.data:
                other = ax.errorbar(
                    *(ax_values_tuple),
                    xerr=series.data["x_err"],
                    yerr=series.data["y_err"],
                    **plot_kwargs
                )
            elif "x_err" in series.data:
                other = ax.errorbar(
                    *(ax_values_tuple), xerr=series.data["x_err"], **plot_kwargs
                )
            elif "y_err" in series.data:
                other = ax.errorbar(
                    *(ax_values_tuple), yerr=series.data["y_err"], **plot_kwargs
                )
            else:
                raise ValueError(
                    'Either "x_err" or "y_err" must be defined in PlotSeries data dict'
                )

            if method == "err_semilogx":
                ax.set_xscale("log")
            elif method == "err_semilogy":
                ax.set_yscale("log")
            elif method == "err_loglog":
                ax.set_xscale("log")
                ax.set_yscale("log")
        elif method == "scatter":
            if x is None or y is None:
                raise ValueError(
                    "Failed to extract either x or y when trying to create scatter plot"
                )
            other = ax.scatter(x, y, **plot_kwargs)

    elif method == "bar":
        plot_kwargs: dict = series.kwargs
        _add_default_kwargs(["kBarLineWidth"], plot_kwargs)
        counts: np.ndarray = series.data["counts"]
        edges: np.ndarray = series.data["edges"]
        widths = edges[1:] - edges[:-1]
        edge_lighter = -0.18
        edge_sat = 1.3
        base_colour: Optional[str] = series.kwargs["color"]
        if base_colour is None:
            base_colour = "red"
        face_alpha: float = 0.38
        face_rgb: Tuple[float, float, float] = _hex_to_rgb01(base_colour)
        face_rgba: Tuple[float, float, float, float] = (
            face_rgb[0],
            face_rgb[1],
            face_rgb[2],
            face_alpha,
        )
        edge_rgb: Tuple[float, float, float] = _adjust_color(
            base_colour, lighter=edge_lighter, sat_mul=edge_sat
        )
        plot_kwargs.update(
            {
                "align": "edge",
                "facecolor": face_rgba,
                "edgecolor": edge_rgb,
                "zorder": 2,
            }
        )
        other = ax.bar(edges[:-1], counts, width=widths, **plot_kwargs)
    elif method == "pcolormesh":
        plot_kwargs = series.kwargs
        data = series.data
        X = data.get("X")
        Y = data.get("Y")
        if "C" in data:
            C = data.get("C")
        elif "u" in data:
            C = data.get("u")
        else:
            raise ValueError(
                    "Failed to extract either C or u when trying to create pcolormesh plot"
                )
        # fallback: try to construct mesh from x and y
        if X is None or Y is None:
            x, y = _extract_xy(series)
            if x is None or y is None:
                raise ValueError(
                    "Failed to extract either X, Y, x or y when trying to create pcolormesh plot"
                )
            X, Y = np.meshgrid(x, y)
        other = ax.pcolormesh(
            X, Y, C, shading=plot_kwargs.pop("shading", "auto"), **plot_kwargs
        )
    elif method == "imshow":
        data = series.data
        C = np.asarray(data.get("C") or data.get("u"))
        plot_kwargs = series.kwargs
        other = ax.imshow(C, **plot_kwargs)
    elif method in ("contour", "contourf"):
        from scipy.interpolate import RectBivariateSpline

        plot_kwargs = series.kwargs
        data = series.data
        X = data.get("X")
        Y = data.get("Y")
        if "C" in data:
            C = data.get("C")
        elif "u" in data:
            C = data.get("u")
        else:
            raise ValueError(
                "Failed to extract either C or u when trying to create contour plot"
            )
        if X is None or Y is None:
            x, y = _extract_xy(series)
            if x is None or y is None:
                raise ValueError(
                    "Failed to extract either X, Y, x or y when trying to create contour plot"
                )
            X, Y = np.meshgrid(x, y)

        # Check if interpolation is requested
        interp_factor = plot_kwargs.pop("interp_factor", 5)

        if interp_factor is not None and interp_factor > 1:
            # Extract 1D arrays from meshgrid
            x_1d = X[0, :]
            y_1d = Y[:, 0]

            # Create interpolation function
            spline = RectBivariateSpline(y_1d, x_1d, C, kx=3, ky=3)

            # Create finer grid
            x_fine = np.linspace(x_1d.min(), x_1d.max(), len(x_1d) * interp_factor)
            y_fine = np.linspace(y_1d.min(), y_1d.max(), len(y_1d) * interp_factor)
            X_fine, Y_fine = np.meshgrid(x_fine, y_fine)

            # Interpolate
            C_fine = spline(y_fine, x_fine)

            # Use interpolated data
            X, Y, C = X_fine, Y_fine, C_fine

        levels = plot_kwargs.pop("levels", 10)
        if method == "contour":
            other = ax.contour(X, Y, C, levels=levels, **plot_kwargs)
        else:
            other = ax.contourf(X, Y, C, levels=levels, **plot_kwargs)
    else:
        raise ValueError("Plot method specified not implemented yet")

    return other


def generic_plot(
    series_list: Sequence[PlotSeries],
    legend: bool,
    figsize: Tuple[float, float] = (6.5, 5.5),
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    xlim: Optional[Tuple[Optional[float], Optional[float]]] = None,
    ylim: Optional[Tuple[Optional[float], Optional[float]]] = None,
    title: Optional[str] = None,
    legend_loc: Optional[str] = None,
    legend_bbox: Optional[Tuple[float, float]] = None,
    additional_objects: Optional[Sequence[Callable[[Axes], Any]]] = None,
) -> Tuple[Axes, Figure, list[Any]]:
    other: Any = [None for _ in range(len(series_list))]
    update_plot_params()
    fig, ax = plt.subplots(figsize=figsize)

    for i, s in enumerate(series_list):
        other[i] = _plot_one(ax, s)

    if additional_objects:
        for obj_func in additional_objects:
            obj_func(ax)

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

    if legend:
        legend_kwargs: Dict[str, Any] = {"frameon": False, "fontsize": 12}
        if legend_loc is not None:
            legend_kwargs["loc"] = legend_loc
        if legend_bbox is not None:
            legend_kwargs["bbox_to_anchor"] = legend_bbox
        ax.legend(**legend_kwargs)

    ax = format_plot_axes(ax)

    return ax, fig, other


def my_save_fig(output_path: Path, fig: Figure, dpi: float = 300):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path) + ".png", dpi=dpi)
    fig.savefig(str(output_path) + ".eps", dpi=dpi)
