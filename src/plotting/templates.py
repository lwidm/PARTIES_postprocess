# -- src/plotting/templates.py
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from matplotlib import pyplot as plt

from src.plotting.tools import (
    PlotSeries,
    generic_plot,
    _plot_one,
    update_plot_params,
    format_plot_axes,
)
from src import globals


def velocity_profile_wall(
    output_dir: Union[str, Path],
    series_list: Sequence[PlotSeries],
    figsize: Tuple[float, float] = (6.5, 5.5),
    xlabel: str = r"$y^+$",
    ylabel: str = r"$u^+$",
    legend_loc: str = "lower right",
    legend_bbox: Tuple[float, float] = (1.0, 0.20),
    dpi: int = 300,
) -> None:
    if not series_list:
        raise ValueError("series_list must contain at least one PlotSeries")

    update_plot_params()
    fig, ax = plt.subplots(figsize=figsize)

    for s in series_list:
        _plot_one(ax, s)

    x_candidates: List[float] = []
    y_candidates: List[float] = []
    for s in series_list:
        try:
            if s.x_key and s.data.get(s.x_key) is not None:
                x_candidates.append(np.max(np.asarray(s.data[s.x_key])))
        except Exception:
            pass
        try:
            if s.y_key and s.data.get(s.y_key) is not None:
                y_candidates.append(np.max(np.asarray(s.data[s.y_key])))
        except Exception:
            pass

    if x_candidates:
        x_max = min(max(x_candidates), 1e2)
        ax.set_xlim(1.0, x_max)
    if y_candidates:
        ax.set_ylim(0.0, 1.1 * max(y_candidates))

    viscous_sublayer_boundary = 5.0
    buffer_layer_boundary = 30.0
    for boundary_position in (viscous_sublayer_boundary, buffer_layer_boundary):
        ax.axvline(
            x=boundary_position,
            color="0.25",
            linewidth=0.8,
            linestyle=":",
            alpha=0.7,
            zorder=0,
        )

    x_max = ax.get_xlim()[1]
    y_max = ax.get_ylim()[1]
    label_y_position = 0.99 * y_max

    viscous_center: float = np.sqrt(1.0 * viscous_sublayer_boundary)
    buffer_center: float = np.sqrt(viscous_sublayer_boundary * buffer_layer_boundary)
    log_center: float = np.sqrt(buffer_layer_boundary * x_max)

    label_style = {
        "ha": "center",
        "va": "top",
        "fontsize": 12,
        "bbox": {"facecolor": "white", "edgecolor": "none", "alpha": 0.0},
    }

    ax.text(
        viscous_center, label_y_position, "Viscous sublayer\n$y^+<5$", **label_style
    )
    ax.text(buffer_center, label_y_position, "Buffer layer\n$5<y^+<30$", **label_style)
    ax.text(log_center, label_y_position, "Log-law region\n$30<y^+$", **label_style)

    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.legend(loc=legend_loc, bbox_to_anchor=legend_bbox)
    ax = format_plot_axes(ax)

    Re_val: Optional[float] = None
    Re_tau_val: Optional[float] = None
    for s in series_list:
        if Re_val is None and "Re" in s.data:
            try:
                Re_val = float(s.data["Re"])
            except Exception:
                pass
        if Re_tau_val is None and "Re_tau" in s.data:
            try:
                Re_tau_val = float(s.data["Re_tau"])
            except Exception:
                pass

    out_path = Path(output_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if Re_val is not None and Re_tau_val is not None:
        plot_filename = out_path / f"Re={Re_val:.0f}_Re_tau={Re_tau_val:.0f}-y+_u+.png"
    else:
        plot_filename = out_path / "y+_u+.png"

    fig.savefig(str(plot_filename), dpi=dpi)

    if not globals.on_anvil:
        plt.show()
    plt.close(fig)


def normal_stress_wall(
    output_dir: Union[str, Path],
    series_list: Sequence[PlotSeries],
) -> None:
    xmin: float = 0.0
    xmax: Optional[float] = None
    ymin: float = 0.0
    ymax: Optional[float] = None
    for series in series_list:
        try:
            if ymax is None:
                ymax = float(np.max(series.data["y"]))
            if xmax is None:
                xmax = float(np.max(series.data["x"]))
            ymax = max(ymax, float(np.max(series.data["x"])))
            xmax = max(xmax, float(np.max(series.data["y"])))
        except:
            pass

    xlim: Optional[Tuple[float, float]] = None
    ylim: Optional[Tuple[float, float]] = None
    if ymax is not None:
        ylim = (ymin, min(1.1 * ymax, 8.0))
    if xmax is not None:
        xlim = (xmin, min(xmax, 80.0))

    generic_plot(
        Path(output_dir) / "wall_normal_stress.png",
        list(series_list),
        xlabel=r"$y^+$",
        ylabel="$\\left\\{\\langle u^\\prime u^\\prime \\rangle,\\; \\langle v^\\prime v^\\prime \\rangle,\\; \\langle w^\\prime w^\\prime \\rangle,\\; \\langle k \\rangle\\right\\}/u_\\tau^2$",
        figsize=(6.5, 5.5),
        legend_loc="lower right",
        legend_bbox=(1.0, 0.70),
        xlim=xlim,
        ylim=ylim,
    )
    return


def floc_count_evolution(
    output_dir: Path, series_list: Sequence[PlotSeries], normalised: bool
) -> None:
    out_path = Path(output_dir) / "floc_count_evolution.png"
    ylabel: str = r"\# Flocs"
    if normalised:
        ylabel = r"(\# Flocs) / (\# Particles)"
    generic_plot(
        out_path,
        list(series_list),
        xlabel="time [-]",
        ylabel=ylabel,
        figsize=(6.5, 5.5),
        legend_loc="lower right",
        legend_bbox=(1.0, 0.80),
        dpi=150,
    )


def fluid_Ekin_evolution(output_dir: Union[str, Path], series_list) -> None:
    out_path = Path(output_dir) / "E_kin_evolution.png"
    generic_plot(
        out_path,
        list(series_list),
        xlabel="time [-]",
        ylabel=r"E_kin [-]",
        figsize=(6.5, 5.5),
        legend_loc="lower right",
        legend_bbox=(1.0, 0.80),
        dpi=150,
    )


def _pdf(
    output_dir: Union[str, Path],
    series_list,
    name: str,
    xlabel: str,
    ylabel: str,
    xmin: float,
    xmax: float,
    ymin: float,
) -> None:
    out_path = Path(output_dir) / f"{name}.png"
    generic_plot(
        out_path,
        list(series_list),
        xlabel=xlabel,
        ylabel=ylabel,
        xlim=(xmin, xmax),
        ylim=(ymin, 1.1e0),
        figsize=(6.5, 5.5),
        legend_loc="lower right",
        legend_bbox=(1.0, 0.80),
        dpi=150,
    )


def n_p_pdf(output_dir: Union[str, Path], series_list):
    _pdf(output_dir, series_list, r"PDF_n_p", r"$n_p$", r"$PDF(n_p)$", 0.9, 4.5, 1e-3)


def D_f_pdf(output_dir: Union[str, Path], series_list):
    _pdf(
        output_dir,
        series_list,
        r"PDF_D_f",
        r"$D_f / D_p$",
        r"$PDF(D_f)$",
        0.0,
        20,
        1e-6,
    )


def D_g_pdf(output_dir: Union[str, Path], series_list):
    _pdf(
        output_dir,
        series_list,
        r"PDF_D_g",
        r"$D_g / D_p$",
        r"$PDF(D_g)$",
        0.0,
        20,
        1e-6,
    )


def _avg_floc_dir(
    output_dir: Union[str, Path], series_list, name: str, ylabel: str, inner_units: bool
) -> None:
    xlabel: str = r"$y$"
    if inner_units:
        xlabel: str = r"$y^+$"
    out_path = Path(output_dir) / f"{name}.png"
    generic_plot(
        out_path,
        list(series_list),
        xlabel=xlabel,
        ylabel=ylabel,
        figsize=(6.5, 5.5),
        legend_loc="lower right",
        legend_bbox=(1.0, 0.80),
        dpi=150,
    )


def avg_D_f(
    output_dir: Union[str, Path], series_list: List[PlotSeries], inner_units: bool
):
    _avg_floc_dir(
        output_dir, series_list, r"avg_D_f", r"$\langle D_f \rangle$", inner_units
    )


def avg_D_g(
    output_dir: Union[str, Path], series_list: List[PlotSeries], inner_units: bool
):
    _avg_floc_dir(
        output_dir, series_list, r"avg_D_g", r"$\langle D_g \rangle$", inner_units
    )


def mass_avg_D_f(
    output_dir: Union[str, Path], series_list: List[PlotSeries], inner_units: bool
):
    _avg_floc_dir(
        output_dir,
        series_list,
        r"mass_avg_D_f",
        r"$\langle D_f \rangle_\text{mass}$",
        inner_units,
    )


def mass_avg_D_g(
    output_dir: Union[str, Path], series_list: List[PlotSeries], inner_units: bool
):
    _avg_floc_dir(
        output_dir,
        series_list,
        r"mass_avg_D_g",
        r"$\langle D_g \rangle_\text{mass}$",
        inner_units,
    )
