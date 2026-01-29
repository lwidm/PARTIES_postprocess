# -- src/plotting/templates.py
from pathlib import Path
from typing import Sequence, Literal, Callable, Any

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import ScalarFormatter

from src.plotting.tools import (
    PlotSeries,
    generic_plot,
    _plot_one,
    update_plot_params,
    format_plot_axes,
    my_save_fig,
)


def velocity_profile_wall(
    output_dir: Path,
    series_list: Sequence[PlotSeries],
    figsize: tuple[float, float] = (6.5, 5.5),
    xlabel: str = r"$y^+$",
    ylabel: str = r"$u^+$",
    legend_loc: str = "lower right",
    legend_bbox: tuple[float, float] = (1.0, 0.20),
    dpi: int = 300,
) -> None:
    if not series_list:
        raise ValueError("series_list must contain at least one PlotSeries")

    update_plot_params()
    fig, ax = plt.subplots(figsize=figsize)

    for s in series_list:
        _plot_one(ax, s)

    x_candidates: list[float] = []
    y_candidates: list[float] = []
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

    Re_val: float | None = None
    Re_tau_val: float | None = None
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

    output_dir.parent.mkdir(parents=True, exist_ok=True)

    if Re_val is not None and Re_tau_val is not None:
        plot_filename = output_dir / f"Re={Re_val:.0f}_Re_tau={Re_tau_val:.0f}-y+_u+"
    else:
        plot_filename = output_dir / "y+_u+"

    fig.savefig(str(plot_filename), dpi=dpi)


def normal_stress_wall(
    output_dir: Path,
    series_list: Sequence[PlotSeries],
    use_marker: bool,
) -> None:
    xmin: float = 0.0
    xmax: float | None = None
    ymin: float = 0.0
    ymax: float | None = None
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

    xlim: tuple[float, float] | None = None
    ylim: tuple[float, float] | None = None
    if ymax is not None:
        ylim = (ymin, min(1.1 * ymax, 8.0))
    if xmax is not None:
        xlim = (xmin, min(xmax, 80.0))

    name: str = "wall_normal_stress"
    if use_marker:
        name: str = "wall_normal_stress_marker"
    ax, fig, _ = generic_plot(
        list(series_list),
        legend=True,
        xlabel=r"$y^+$",
        ylabel=r"$\left\{\langle u^\prime u^\prime \rangle, \langle v^\prime v^\prime \rangle, \langle w^\prime w^\prime \rangle\right\}/u_\tau^2$",
        figsize=(6.5, 5.5),
        legend_loc="lower right",
        legend_bbox=(1.0, 0.50),
        xlim=xlim,
        ylim=ylim,
    )
    my_save_fig(output_dir / name, fig)

    return


def floc_count_evolution(
    output_dir: Path, series_list: Sequence[PlotSeries], normalised: bool
) -> None:
    out_path = output_dir / "floc_count_evolution"
    ylabel: str = r"\#Flocs"
    if normalised:
        ylabel = r"(\#Flocs) / (\#Particles)"
    ax, fig, _ = generic_plot(
        list(series_list),
        legend=True,
        xlabel=r"Dimensionless time, $\tau = L/U$ [-]",
        ylabel=ylabel,
        figsize=(6.5, 5.5),
        legend_loc="lower right",
        legend_bbox=(1.0, 0.80),
    )
    my_save_fig(out_path, fig, dpi=150)


def fluid_Ekin_evolution(output_dir: Path, series_list) -> None:
    out_path = output_dir / "E_kin_evolution"
    ax, fig, _ = generic_plot(
        list(series_list),
        legend=True,
        xlabel=r"Dimensionless time, $\tau = L/U$ [-]",
        ylabel=r"Dimensionless energy, $E_{kin}$ [-]",
        figsize=(6.5, 5.5),
        legend_loc="lower right",
        legend_bbox=(1.0, 0.80),
    )
    my_save_fig(out_path, fig, dpi=150)


def _pdf(
    output_dir: Path,
    series_list: list[PlotSeries],
    name: str,
    xlabel: str,
    ylabel: str,
    xmin: float | None,
    xmax: float | None,
    ymin: float | None,
    ymax: float | None,
    additional_objects: Sequence[Callable[[Axes], Any]] | None = None,
) -> None:
    out_path = output_dir / f"{name}"
    ax, fig, _ = generic_plot(
        list(series_list),
        legend=True,
        xlabel=xlabel,
        ylabel=ylabel,
        xlim=(xmin, xmax),
        ylim=(ymin, ymax),
        figsize=(6.5, 5.5),
        legend_loc="best",
        # legend_bbox=(1.0, 0.80),
        additional_objects=additional_objects,
    )
    my_save_fig(out_path, fig, dpi=150)


def n_p_pdf(output_dir: Path, series_list: list[PlotSeries]) -> None:
    _pdf(
        output_dir,
        series_list,
        r"PDF_n_p",
        r"\#Particles in floc, $n_p$",
        r"$PDF(n_p)$",
        0.9,
        20,
        1e-3,
        1.1e0,
    )


def D_f_pdf(output_dir: Path, series_list: list[PlotSeries]) -> None:
    _pdf(
        output_dir,
        series_list,
        r"PDF_D_f",
        r"$D_f / D_p$",
        r"$PDF(D_f)$",
        0.0,
        20,
        1e-6,
        1.1e0,
    )


def D_g_pdf(output_dir: Path, series_list: list[PlotSeries]) -> None:
    _pdf(
        output_dir,
        series_list,
        r"PDF_D_g",
        r"$D_g / D_p$",
        r"$PDF(D_g)$",
        0.0,
        20,
        1e-6,
        1.1e0,
    )


def n_p_mass_pdf(output_dir: Path, series_list: list[PlotSeries]) -> None:
    _pdf(
        output_dir,
        series_list,
        r"PDF_n_p_mass",
        r"\#Particles in floc, $n_p$",
        r"Mass-weighted $PDF(n_p)$",
        0.9,
        20,
        1e-3,
        1.1e0,
    )


def D_f_mass_pdf(output_dir: Path, series_list: list[PlotSeries]) -> None:
    _pdf(
        output_dir,
        series_list,
        r"PDF_D_f_mass",
        r"$D_f / D_p$",
        r"Mass-weighted $PDF(D_f)$",
        0.0,
        20,
        1e-6,
        1.1e0,
    )


def D_g_mass_pdf(output_dir: Path, series_list: list[PlotSeries]) -> None:
    _pdf(
        output_dir,
        series_list,
        r"PDF_D_g_mass",
        r"$D_g / D_p$",
        r"Mass-weighted $PDF(D_g)$",
        0.0,
        20,
        1e-6,
        1.1e0,
    )


def _avg_floc_dir(
    output_dir: Path, series_list, name: str, ylabel: str, inner_units: bool
) -> None:
    xlabel: str = r"$y = \tilde y/L$ [-]"
    if inner_units:
        xlabel: str = r"$y^+$"
    out_path = output_dir / f"{name}"
    ax, fig, _ = generic_plot(
        list(series_list),
        legend=True,
        xlabel=xlabel,
        ylabel=ylabel,
        figsize=(6.5, 5.5),
        legend_loc="best",
        # legend_bbox=(1.0, 0.80),
    )
    my_save_fig(out_path, fig, dpi=150)


def avg_D_f(output_dir: Path, series_list: list[PlotSeries], inner_units: bool):
    _avg_floc_dir(
        output_dir,
        series_list,
        r"avg_D_f",
        r"$\langle D_f / d_{p} \rangle$",
        inner_units,
    )


def avg_D_g(output_dir: Path, series_list: list[PlotSeries], inner_units: bool):
    _avg_floc_dir(
        output_dir,
        series_list,
        r"avg_D_g",
        r"$\langle D_g  / d_{p} \rangle$",
        inner_units,
    )


def mass_avg_D_f(output_dir: Path, series_list: list[PlotSeries], inner_units: bool):
    _avg_floc_dir(
        output_dir,
        series_list,
        r"mass_avg_D_f",
        r"$\langle D_f / d_{p} \rangle_\text{mass}$",
        inner_units,
    )


def mass_avg_D_g(output_dir: Path, series_list: list[PlotSeries], inner_units: bool):
    _avg_floc_dir(
        output_dir,
        series_list,
        r"mass_avg_D_g",
        r"$\langle D_g  / d_{p}\rangle_\text{mass}$",
        inner_units,
    )


# -------------------- Fluid volume fraction --------------------


def phi_eulerian(
    output_dir: Path, series_list: list[PlotSeries], normalised: bool
) -> None:
    out_path: Path = output_dir / f"phi_eulerian{"_norm" if normalised else ""}"
    xlabel: str = r"$y = \tilde y/L$ [-]"
    ylabel: str = r"$\langle \phi \rangle$ [%]"
    if normalised:
        ylabel: str = r"$\langle \phi / \phi_0 \rangle$ [-]"

    ax, fig, _ = generic_plot(
        list(series_list),
        legend=True,
        xlabel=xlabel,
        ylabel=ylabel,
        figsize=(6.5, 5.5),
        legend_loc="lower center",
    )
    my_save_fig(out_path, fig, dpi=150)


# -------------------- Lagrangian data pdf --------------------


def lagrangian_acceleration_pdf(
    output_dir: Path, series_list: list[PlotSeries], label: str | None
) -> None:
    name: str
    if label is not None:
        name = f"lagrangian_acceleration_pdf_{label}"
    else:
        name = f"lagrangian_acceleration_pdf"
    _pdf(
        output_dir=output_dir,
        series_list=series_list,
        name=name,
        xlabel=r"$a_{p,i} / \sigma_{a_{p,i}}$",
        ylabel=r"PDF",
        xmin=-15.0,
        xmax=15.0,
        ymin=1e-5,
        ymax=1.1e0,
    )


def lagrangian_up_pdf(
    output_dir: Path, series_list: list[PlotSeries], label: str | None
) -> None:
    name: str
    if label is not None:
        name = f"lagrangian_up_pdf_{label}"
    else:
        name = f"lagrangian_up_pdf"
    _pdf(
        output_dir=output_dir,
        series_list=series_list,
        name=name,
        xlabel=r"$u^+_{p}$",
        ylabel=r"PDF",
        xmin=None,
        xmax=None,
        ymin=1e-5,
        ymax=None,
    )


# -------------------- familiy tree --------------------


def family_tree_breakup_formation_pdf(
    output_dir: Path,
    series_list: list[PlotSeries],
    label: str | None,
    filtered_unfiltered: Literal[
        1, 2, 3
    ],  # 1 just filtered, 2 superimpose filterd unfiltered, 3 just unfiltred
) -> None:
    name = f"family_tree_breakup_formation_pdf"
    if filtered_unfiltered == 2:
        name += "_superimposed"
    if filtered_unfiltered == 3:
        name += "_unfiltered"
    if label is not None:
        name += f"_{label}"
    _pdf(
        output_dir=output_dir,
        series_list=series_list,
        name=name,
        xlabel=r"$y / L $",
        ylabel=r"PDF",
        xmin=None,
        xmax=1,
        ymin=0,
        ymax=None,
    )


def floc_timescale(
    output_dir: Path,
    series_list: list[PlotSeries],
    label: str,
    additional_objects: Sequence[Callable[[Axes], Any]] | None = None,
) -> None:
    name = f"floc_timescale_{label}"
    _pdf(
        output_dir=output_dir,
        series_list=series_list,
        name=name,
        xlabel=r"$y / L $",
        ylabel=r"PDF",
        xmin=None,
        xmax=1,
        ymin=0,
        ymax=None,
        additional_objects=additional_objects,
    )


def noncohesive_floc_lifetime(
    output_dir: Path,
    series_list: list[PlotSeries],
    label: str,
    additional_objects: Sequence[Callable[[Axes], Any]] | None = None,
) -> None:
    name = f"noncohesive_floc_lifetime_{label}"
    _pdf(
        output_dir=output_dir,
        series_list=series_list,
        name=name,
        xlabel=r"$y / L $",
        ylabel=r"$t_{floc} \cdot L / U$",
        xmin=0.0,
        xmax=1.0,
        ymin=None,
        ymax=None,
        additional_objects=additional_objects,
    )


def coagulation_kernel(
    output_dir: Path,
    series_pcolormesh: PlotSeries,
    series_contour: PlotSeries | None,
    name: str,
    x_axis_value: Literal["np", "D", "DD"],
) -> None:
    out_path: Path = output_dir / f"coagulation_kernel_{name}"

    K: np.ndarray = series_pcolormesh.data["C"]
    X: np.ndarray = series_pcolormesh.data["X"]
    Y: np.ndarray = series_pcolormesh.data["Y"]
    xlim: tuple[float, float] = series_pcolormesh.data["xlim"]
    ylim: tuple[float, float] = series_pcolormesh.data["ylim"]

    mask = (
        (X[0, :] >= xlim[0])
        & (X[0, :] <= xlim[1])
        & (Y[:, 0] >= ylim[0])
        & (Y[:, 0] <= ylim[1])
    )

    cmax: float = np.nanmax(K[mask])

    s_list = (
        [series_pcolormesh, series_contour]
        if series_contour is not None
        else [series_pcolormesh]
    )

    ylabel: str
    xlabel: str
    if x_axis_value == "np":
        xlabel = "$x, \\quad n_p$"
        ylabel = "$y, \\quad n_p$"
    elif x_axis_value == "D":
        xlabel = "$x, \\quad D/d_p \\sim \\sqrt[3]{n_p}$"
        ylabel = "$y, \\quad D/d_p \\sim \\sqrt[3]{n_p}$"
    else:
        xlabel = "$x^2, \\quad D^2/d_p^2 \\sim (\\sqrt[3]{n_p})^2$"
        ylabel = "$y^2, \\quad D^2/d_p^2 \\sim (\\sqrt[3]{n_p})^2$"

    ax, fig, mesh = generic_plot(
        s_list,
        legend=True,
        xlabel=xlabel,
        ylabel=ylabel,
        xlim=xlim,
        ylim=ylim,
        figsize=(6.5, 5.5),
        legend_loc="best",
    )

    # mesh[0].set_clim(0, cmax)
    cbar = plt.colorbar(mesh[0], ax=ax, orientation="horizontal")
    ax.set_aspect("equal", adjustable="box")

    my_save_fig(out_path, fig, dpi=150)


def fragment_size_distribution(
    output_dir: Path,
    series_pcolormesh: PlotSeries,
    series_contour: PlotSeries | None,
    name: str,
) -> None:
    out_path: Path = output_dir / f"fragment_size_distribution_{name}"
    xlim: tuple[float, float] = series_pcolormesh.data["xlim"]
    ylim: tuple[float, float] = series_pcolormesh.data["ylim"]

    s_list = (
        [series_pcolormesh, series_contour]
        if series_contour is not None
        else [series_pcolormesh]
    )

    ax, fig, mesh = generic_plot(
        s_list,
        legend=True,
        xlabel="$x \\quad (n_p)$",
        ylabel="$y \\quad (n_p)$",
        xlim=xlim,
        ylim=ylim,
        figsize=(6.5, 5.5),
        legend_loc="best",
    )

    cbar = plt.colorbar(mesh[0], ax=ax, orientation="horizontal")
    ax.set_aspect("equal", adjustable="box")
    my_save_fig(out_path, fig, dpi=150)


def breakage_rate(
    output_dir: Path,
    series_list: Sequence[PlotSeries],
    n_p_max: float,
    D_dp_max: float,
    x_axis_value: Literal["np", "D"],
) -> None:
    out_path = output_dir / "breakage_rate"

    F_list: list[np.ndarray] = []
    x_list: list[np.ndarray] = []
    for s in series_list:
        F_list.append(s.data["y"])
        x_list.append(s.data["x"])

    ymax: float = max([np.nanmax(F[x < n_p_max]) for x, F in zip(x_list, F_list)])

    xlabel: str
    xlim: tuple[float | None, float | None]
    if x_axis_value == "np":
        xlim=(2, n_p_max)
        xlabel = r"floc size: $x \quad (n_p)$"
    else:
        xlim=(0.9, D_dp_max)
        xlim=(None, None)
        xlabel = r"floc size: $x \quad (D/d_p \sim \sqrt[3]{n_p})$"

    ax, fig, _ = generic_plot(
        list(series_list),
        legend=True,
        xlabel=xlabel,
        ylabel="breakage rate: $F(x)$",
        xlim=xlim,
        ylim=(0, ymax),
        figsize=(6.5, 5.5),
        legend_loc="best",
    )
    current_ticks = list(ax.get_xticks())
    if x_axis_value == "np":
        if 1 not in current_ticks:
            current_ticks.append(2)
            current_ticks.sort()
            ax.set_xticks(current_ticks)
    else:
        ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9])
        ax.set_xticklabels(['1', '2', '3', '4', '5', '6', '7', '8', '9'])
    my_save_fig(out_path, fig, dpi=150)


def coalescence_kernel_colletti(
    output_dir: Path,
    series_list: Sequence[PlotSeries],
    n_p_max: float,
    D_dp_max: float,
    x_axis_value: Literal["np", "D", "DD"],
) -> None:
    out_path = output_dir / "breakage_rate"

    y_list: list[np.ndarray] = []
    x_list: list[np.ndarray] = []
    for s in series_list:
        y_list.append(s.data["y"])
        x_list.append(s.data["x"])

    ymax: float = max([np.nanmax(F[x < n_p_max]) for x, F in zip(x_list, y_list)])

    xlabel: str
    xlim: tuple[float | None, float | None]
    if x_axis_value == "np":
        xlim=(2, n_p_max)
        xlabel = r"floc size: $x_1 + x_2, \quad n_{p,1} + n_{p,2})$"
    elif x_axis_value == "D":
        xlim=(0.9, D_dp_max)
        xlim=(None, None)
        xlabel = r"floc size: $x_1 + x_2, \quad (D_1 + D_2)/d_p \quad D/d_p \sim \sqrt[3]{n_p})$"
    else:
        xlim=(0.9, D_dp_max)
        xlim=(None, None)
        xlabel = r"floc size: $x_1^2 + x_2^2, \quad (D_1^2 + D_2^2)/d_p^2 \quad D/d_p \sim \sqrt[3]{n_p})$"

    ax, fig, _ = generic_plot(
        list(series_list),
        legend=True,
        xlabel=xlabel,
        ylabel="Coalescence kernel: $K(x_1, x_2)$",
        xlim=xlim,
        ylim=(0, ymax),
        figsize=(6.5, 5.5),
        legend_loc="best",
    )
    current_ticks = list(ax.get_xticks())
    if x_axis_value == "np":
        if 1 not in current_ticks:
            current_ticks.append(2)
            current_ticks.sort()
            ax.set_xticks(current_ticks)
    else:
        ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9])
        ax.set_xticklabels(['1', '2', '3', '4', '5', '6', '7', '8', '9'])
    my_save_fig(out_path, fig, dpi=150)

def number_density_evo_sink_source(
    output_dir: Path,
    series_list: Sequence[PlotSeries],
    name: str | None,
    xmax: float | None,
    mass_weighted: bool,
) -> None:
    if name is not None:
        name = f"number_density_evo_{name}"
    else:
        name = "number_density_evo"
    out_path = output_dir / name

    xlabel: str = r"floc size: $n_p$"
    ylabel: str
    if mass_weighted:
        ylabel = r"$\frac{\partial n(n_p)}{\partial t} \cdot n_p$"
    else:
        ylabel = r"$\frac{\partial n(n_p)}{\partial t}$"
    ax, fig, _ = generic_plot(
        list(series_list),
        legend=True,
        xlabel=xlabel,
        ylabel=ylabel,
        xlim=(1, xmax),
        ylim=(None, None),
        figsize=(6.5, 5.5),
        legend_loc="best",
    )
    current_ticks = list(ax.get_xticks())
    if 1 not in current_ticks:
        current_ticks.append(1)
        current_ticks.sort()
        ax.set_xticks(current_ticks)
    my_save_fig(out_path, fig, dpi=150)


def cumulative_floculation_balance(
    output_dir: Path,
    series_list: Sequence[PlotSeries],
    name: str | None,
    xmax: float | None,
    mass_weighted: bool,
) -> None:
    if name is not None:
        name = f"cumulative_floculation_balance_{name}"
    else:
        name = "cumulative_floculation_balance"
    out_path = output_dir / name

    xlabel: str = r"floc size: $n_p$"
    ylabel: str
    if mass_weighted:
        ylabel = r"$\sum_{i=1}^{n_p} \left( \frac{\partial n_p}{\partial t} \cdot n_p \right)$"
    else:
        ylabel = r"$\sum_{i=1}^{n_p} \frac{\partial n_p}{\partial t}$"
    ax, fig, _ = generic_plot(
        list(series_list),
        legend=True,
        xlabel=xlabel,
        ylabel=ylabel,
        xlim=(1, xmax),
        ylim=(None, None),
        figsize=(6.5, 5.5),
        legend_loc="best",
    )
    current_ticks = list(ax.get_xticks())
    if 1 not in current_ticks:
        current_ticks.append(1)
        current_ticks.sort()
        ax.set_xticks(current_ticks)
    my_save_fig(out_path, fig, dpi=150)


def total_frequency_plot(
    output_dir: Path,
    series_list_floc: Sequence[PlotSeries],
    series_list_break: Sequence[PlotSeries],
) -> None:
    out_path = output_dir / "total_frequency"

    text_scale_factor = 1.5

    combined_series: list[PlotSeries] = list(series_list_floc) + list(series_list_break)

    ax, fig, _ = generic_plot(
        combined_series,
        legend=True,
        xlabel=r"Particle volume fraction $\phi$ [\%]",
        ylabel=r"Aggregation/Breakup frequency, $\tfrac{f \cdot L^4}{U}$ [-]",
        xlim=(None, None),
        ylim=(None, None),
        figsize=(6.5, 5.5),
        legend_loc="best",
    )

    ax.xaxis.label.set_fontsize(14 * text_scale_factor)
    ax.yaxis.label.set_fontsize(14 * text_scale_factor)
    ax.tick_params(axis="both", which="major", labelsize=12 * text_scale_factor)
    ax.legend(frameon=False, fontsize=12 * text_scale_factor, loc="best")

    plt.tight_layout()

    my_save_fig(out_path, fig, dpi=150)
