from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence, Literal, Callable

import numpy as np
from numpy.fft import fft2, ifft2
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib import colors as mcolors
from matplotlib.patches import Circle
import colorsys

NumericArray = np.ndarray | float | int

# ---- global font size defaults ----
kFontScale: float = 1.5
kTickLabelSize: float = 12.0 * kFontScale
kAxisLabelSize: float = 14.0 * kFontScale
kTitleSize: float = 14.0 * kFontScale
kLegendSize: float = 14.0 * kFontScale
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
    "hline",
    "vline",
    "err_plot",
    "err_semilogx",
    "err_semilogy",
    "err_loglog",
]


@dataclass
class PlotSeries:
    data: dict[str, Any]
    x_key: str | None = "x"
    y_key: str | None = "y"
    plot_method: PlotMethod | None = "plot"
    kwargs: dict[str, Any] = field(default_factory=dict)


SeriesLike = PlotSeries | Sequence[PlotSeries]

# fmt: off
default_kwargs: dict[str, dict] = {
    "kLineWidth": {"linewidth": 0.9 * kFontScale},  # default linewidth for generic plots
    "kBarLineWidth": {"linewidth": 1.0 * kFontScale},  # default linewidth for bar plots
    "kELineWidth": {"linewidth": 0.6 * kFontScale},  # default elinwidth (linewidth for errorbars)
    "kECapSize": {"capsize": 2.0 * kFontScale},  # default capsize (cap size for errorbars)
    "kECapThick": {"capthick": 0.8 * kFontScale},  # default capthick (cap thinkness for errorbars)
    "kBarsAbove": {"barsabove": True},  # default barsabove
    "kMarkerSize": {"markersize": 5.0 * kFontScale},  # default markersize
    "kMarkerEdgeWidth": {"markeredgewidth": 0.7 * kFontScale},  # default marker edge width
}
# fmt: on


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


def set_integer_log_xticks(
    ax: Axes,
    xlim: tuple[float, float],
) -> None:
    lo, hi = xlim
    # 1-9 individually, then 10, 20, 30, ..., 100, 200, 300, ... etc.
    candidates: list[int] = []
    decade = 1
    while decade <= hi:
        for d in range(1, 10):
            val = d * decade
            if val > hi:
                break
            candidates.append(val)
        decade *= 10
    ticks = [t for t in candidates if lo <= t <= hi]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])


def format_plot_axes(axes: Axes) -> Axes:
    # axes.spines["top"].set_visible(True)
    # axes.spines["right"].set_visible(True)
    # axes.spines["left"].set_linewidth(1.0)
    # axes.spines["bottom"].set_linewidth(1.0)
    axes.tick_params(
        axis="both",
        which="both",
        direction="in",
        labelsize=kTickLabelSize,
        top=True,
        right=True,
        bottom=True,
        left=True,
    )
    plt.tight_layout()
    return axes


# ------------------------- generic plotting helpers -------------------------


def _extract_xy(
    series: PlotSeries,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return numeric (x, y) arrays or (None, None) if not available."""
    x: np.ndarray | None = None
    y: np.ndarray | None = None
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
    if sigma <= 0:
        return data.copy()
    valid_mask = ~np.isnan(data)
    if not np.any(valid_mask):
        return np.full_like(data, np.nan)

    data_zeroed = np.nan_to_num(data, nan=0.0)

    kernel_size = int(4 * sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Ensure kernel size doesn't exceed data dimensions
    min_data_dim = min(data.shape[0], data.shape[1])
    if kernel_size > min_data_dim:
        kernel_size = min_data_dim if min_data_dim % 2 == 1 else min_data_dim - 1

    ax = np.arange(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
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

    # Only restore NaN for boundary NaN cells (outside the data region).
    # A NaN cell is "boundary" if the entire upper-right quadrant
    # (all cells with larger row AND larger col) has no valid data.
    nan_mask = ~valid_mask
    quadrant_sum = np.cumsum(np.cumsum(valid_mask[::-1, ::-1], axis=0), axis=1)[
        ::-1, ::-1
    ]
    boundary_nan = (quadrant_sum == 0) & nan_mask
    result[boundary_nan] = np.nan

    return result


def _plot_one(ax: Axes, series: PlotSeries) -> Any:
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
        ax_values_tuple: tuple[np.ndarray, ...]
        if x is None:
            if y is None:
                raise ValueError("Could not extract both x and y from PlotSeries")
            ax_values_tuple = (y,)
        elif y is None:
            ax_values_tuple = (x,)
        else:
            ax_values_tuple = (x, y)

        plot_kwargs = series.kwargs
        _add_default_kwargs(
            ["kLineWidth", "kMarkerSize", "kMarkerEdgeWidth"], plot_kwargs
        )

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
        base_colour: str | None = series.kwargs["color"]
        if base_colour is None:
            base_colour = "red"
        face_alpha: float = 0.38
        face_rgb: tuple[float, float, float] = _hex_to_rgb01(base_colour)
        face_rgba: tuple[float, float, float, float] = (
            face_rgb[0],
            face_rgb[1],
            face_rgb[2],
            face_alpha,
        )
        edge_rgb: tuple[float, float, float] = _adjust_color(
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

        levels = plot_kwargs.pop("levels", 10)
        if method == "contour":
            other = ax.contour(X, Y, C, levels=levels, **plot_kwargs)
        else:
            other = ax.contourf(X, Y, C, levels=levels, **plot_kwargs)
    elif method == "hline":
        plot_kwargs = series.kwargs
        _add_default_kwargs(["kLineWidth"], plot_kwargs)
        y_value = series.data.get("y")
        if y_value is None:
            raise ValueError("hline requires 'y' value in data dict")
        other = ax.axhline(y=y_value, **plot_kwargs)
    elif method == "vline":
        plot_kwargs = series.kwargs
        _add_default_kwargs(["kLineWidth"], plot_kwargs)
        x_value = series.data.get("x")
        if x_value is None:
            raise ValueError("vline requires 'x' value in data dict")
        other = ax.axvline(x=x_value, **plot_kwargs)
    else:
        raise ValueError("Plot method specified not implemented yet")

    return other


def generic_plot(
    series_list: Sequence[PlotSeries],
    legend: bool,
    figsize: tuple[float, float] = (6.5, 5.5),
    xlabel: str | None = None,
    ylabel: str | None = None,
    xlim: tuple[float | None, float | None] | None = None,
    ylim: tuple[float | None, float | None] | None = None,
    title: str | None = None,
    legend_loc: str | None = None,
    legend_bbox: tuple[float, float] | None = None,
    additional_objects: Sequence[Callable[[Axes], Any]] | None = None,
    legend_handles: list[Any] | None = None,
    legend_labels: list[str] | None = None,
    legend_handler_map: dict | None = None,
) -> tuple[Axes, Figure, list[Any]]:
    other: Any = [None for _ in range(len(series_list))]
    update_plot_params()
    fig, ax = plt.subplots(figsize=figsize)

    for i, s in enumerate(series_list):
        other[i] = _plot_one(ax, s)

    if additional_objects:
        for obj_func in additional_objects:
            obj_func(ax)

    if xlabel:
        ax.set_xlabel(xlabel, fontsize=kAxisLabelSize)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=kAxisLabelSize)
    if title:
        ax.set_title(title, fontsize=kTitleSize)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    if legend:
        legend_kwargs: dict[str, Any] = {"frameon": False, "fontsize": kLegendSize}
        if legend_loc is not None:
            legend_kwargs["loc"] = legend_loc
        if legend_bbox is not None:
            legend_kwargs["bbox_to_anchor"] = legend_bbox
        if legend_handler_map is not None:
            legend_kwargs["handler_map"] = legend_handler_map
        if legend_handles is not None:
            ax.legend(legend_handles, legend_labels or [], **legend_kwargs)
        else:
            ax.legend(**legend_kwargs)

    ax = format_plot_axes(ax)
    fig.tight_layout()

    return ax, fig, other


def _draw_particle(
    ax: Axes,
    x: float,
    y: float,
    radius: float,
    color: tuple[float, float, float],
    lw: float,
    zorder: float,
) -> None:
    n: int = 128
    lin: np.ndarray = np.linspace(-1, 1, n)
    xx: np.ndarray
    yy: np.ndarray
    xx, yy = np.meshgrid(lin, lin)

    r_sq: np.ndarray = xx**2 + yy**2
    inside: np.ndarray = r_sq <= 1.0

    edge_factor: np.ndarray = np.sqrt(np.clip(1.0 - r_sq, 0, 1))
    hl_dist: np.ndarray = np.sqrt((xx + 0.35) ** 2 + (yy - 0.35) ** 2)
    highlight: np.ndarray = np.exp(-(hl_dist**2) * 1.5)

    brightness: np.ndarray = 0.35 * edge_factor + 0.45 * highlight * edge_factor
    brightness: np.ndarray = np.clip(brightness, 0.12, 0.92)

    rgba: np.ndarray = np.zeros((n, n, 4))
    rgba[:, :, 0] = np.clip(color[0] * brightness * 2.0, 0, 1)
    rgba[:, :, 1] = np.clip(color[1] * brightness * 2.0, 0, 1)
    rgba[:, :, 2] = np.clip(color[2] * brightness * 2.0, 0, 1)
    rgba[:, :, 3] = inside.astype(float)

    xlim_cur: tuple[float, float]
    ylim_cur: tuple[float, float]
    xlim_cur, ylim_cur = ax.get_xlim(), ax.get_ylim()
    ax.imshow(
        rgba,
        extent=(x - radius, x + radius, y - radius, y + radius),
        origin="lower",
        interpolation="bilinear",
        aspect="equal",
        zorder=zorder,
    )
    ax.set_xlim(xlim_cur)
    ax.set_ylim(ylim_cur)

    edge = Circle(
        (x, y),
        radius,
        fill=False,
        edgecolor="#404040",
        lw=lw,
        zorder=zorder + 1,
    )
    ax.add_patch(edge)


def particle_event_plot(
    h_vals: np.ndarray,
    v_vals: np.ndarray,
    radii: np.ndarray,
    floc_ids: np.ndarray,
    floc_color_map: dict[int, tuple[float, float, float]],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    title: str | None,
    figsize: tuple[float, float],
) -> tuple[Axes, Figure]:
    update_plot_params()
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")

    unique_fids: np.ndarray
    counts: np.ndarray
    unique_fids, counts = np.unique(floc_ids, return_counts=True)
    floc_size: dict[int, int] = dict(zip(unique_fids.tolist(), counts.tolist()))
    sort_idx: np.ndarray = np.argsort([-floc_size[int(f)] for f in floc_ids])

    idx: int
    i: int
    for idx, i in enumerate(sort_idx):
        _draw_particle(
            ax,
            float(h_vals[i]),
            float(v_vals[i]),
            float(radii[i]),
            color=floc_color_map[int(floc_ids[i])],
            lw=0.3 * kFontScale,
            zorder=2 + idx,
        )

    ax.set_xticks([])
    ax.set_yticks([])
    if title is not None:
        ax.set_title(title, fontsize=kTitleSize)
    fig.tight_layout()

    return ax, fig


def my_save_fig(output_path: Path, fig: Figure, dpi: float = 300):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path) + ".png", dpi=dpi, bbox_inches="tight")
    # fig.savefig(str(output_path) + ".eps", dpi=dpi, bbox_inches="tight")
    fig.savefig(str(output_path) + ".pdf", dpi=dpi, bbox_inches="tight")
