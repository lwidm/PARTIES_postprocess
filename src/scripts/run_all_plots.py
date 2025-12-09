from os import sep
from typing import Literal, Callable
from pathlib import Path
import seaborn as sns
import re

from src.plotting.tools import PlotSeries
from src.plotting import series as plt_series
from src.plotting import templates as plt_templ
from src import globals

from matplotlib import pyplot as plt
from matplotlib.colors import Colormap
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, Colormap

# parent_dir: Path = Path("/media/usb/UCSB/")
parent_dir: Path = Path("./")


def fuild_velocity_profile(
    utexas_dir: Path,
    plot_dir: Path,
    data_dir: Path,
    data_names: list[str],
    labels: list[str],
    colours: list[str | tuple[float, float, float, float]],
    use_markers: bool,
):
    data_dirs: list[Path] = [data_dir / data_name for data_name in data_names]
    Num_data: int = len(data_names)
    linestyles: tuple[str, str, str] = ("-", ":", ":")

    separate_labels: bool = False

    use_label_texas: bool = False
    use_label_log_fit_texas: bool = False
    use_label_visc_fit_texas: bool = False
    use_label: bool = False
    use_label_log_fit: bool = False
    use_label_visc_fit: bool = False
    if separate_labels:
        use_label_texas = True
        use_label_log_fit_texas = True
        use_label_visc_fit_texas = False
        use_label = True
        use_label_log_fit = True
        use_label_visc_fit = False

    utexas_wall_series: list[PlotSeries] = plt_series.u_plus_mean_utexas(
        utexas_dir,
        "utexas",
        "k",
        True,
        True,
        use_label_texas,
        use_label_log_fit_texas,
        use_label_visc_fit_texas,
        linestyles,
    )
    parties_wall_series: list[list[PlotSeries]] = []
    for i in range(Num_data):
        parties_wall_series.append(
            plt_series.u_plus_mean_parties(
                data_dirs[i],
                label=labels[i],
                colour=colours[i],
                log_fit=True,
                visc_fit=False,
                use_label=use_label,
                use_label_log_fit=use_label_log_fit,
                use_label_visc_fit=use_label_visc_fit,
                linestyles=linestyles,
            )
        )

    all_wall_series: list[PlotSeries] = utexas_wall_series
    for series in parties_wall_series:
        all_wall_series += series

    if not separate_labels:
        colours_proxy: list[str | tuple[float, float, float, float]] = colours
        colours_proxy[Num_data] = "k"
        proxy_series = plt_series.u_plus_proxies(
            linestyles=list(linestyles),
            labels=labels + ["utexas"],
            colours=colours_proxy,
        )
        all_wall_series += proxy_series

    plt_templ.velocity_profile_wall(plot_dir, all_wall_series)


def fluid_wall_normal(
    utexas_dir: Path,
    plot_dir: Path,
    data_dir: Path,
    data_names: list[str],
    labels: list[str],
    colours: list[str | tuple[float, float, float, float]],
    use_markers: bool,
):

    data_dirs: list[Path] = [data_dir / data_name for data_name in data_names]
    Num_data: int = len(data_names)

    linestyles_utexas: list[str] = ["-", "-.", "--", ":"]
    utexas_colour: str = "k"
    if use_markers:
        linestyles: list[str] = [":" for _ in range(4)]
    else:
        linestyles: list[str] = linestyles_utexas
    if use_markers:
        markers: list[str] = ["o", "d", "^", "s"]
    else:
        markers: list[str] = ["None" for _ in range(len(linestyles_utexas))]
    # colours: list[str] = ["k", "k", "k", "k", "k"]

    # ==============================
    # Automation
    # ==============================

    utexas_stress_series: list[PlotSeries] = plt_series.normal_stress_wall_utexas(
        utexas_dir, linestyles_utexas, utexas_colour
    )
    parties_stress_series: list[list[PlotSeries]] = []
    for i in range(Num_data):
        parties_stress_series.append(
            plt_series.normal_stress_wall_parties(
                data_dirs[i],
                linestyles=linestyles,
                markers=markers,
                label=labels[i],
                colour=colours[i],
            )
        )
    colours_proxy: list[str | tuple[float, float, float, float]] = colours
    colours_proxy[Num_data] = "k"
    marker_cases: list[str]
    linestyle_cases: list[str]
    if use_markers:
        marker_cases = ["s" for _ in range(Num_data)] + ["None"]
        linestyle_cases = ["None" for _ in range(Num_data)] + ["-"]
    else:
        marker_cases = ["s" for _ in range(Num_data + 1)]
        linestyle_cases = ["None" for _ in range(Num_data + 1)]
    proxy_series = plt_series.normal_stress_wall_label_proxies(
        linestyles=linestyles_utexas,
        markers=markers,
        labels=labels + ["utexas"],
        colours=colours_proxy,
        marker_cases=marker_cases,
        linestyle_cases=linestyle_cases,
    )
    all_stress_series: list[PlotSeries] = utexas_stress_series
    for series in parties_stress_series:
        all_stress_series += series
    all_stress_series += proxy_series
    plt_templ.normal_stress_wall(plot_dir, all_stress_series, use_markers)


def floc(
    plot_dir: Path,
    data_dir: Path,
    data_names: list[str],
    labels: list[str],
    colours: list[str | tuple[float, float, float, float]],
    markers: list[str],
    linestyles: list[str],
    plot_evo_fit: bool,
) -> None:

    data_dirs: list[Path] = [data_dir / data_name for data_name in data_names]

    def get_series_floc_evolution(
        csv_dir: Path,
        colour: str | tuple[float, float, float, float],
        label: str,
    ) -> PlotSeries:
        s: PlotSeries = plt_series.floc_count_evolution(
            csv_dir,
            colour,
            label,
            normalised=True,
            reset_time=True,
        )
        return s

    def get_series_floc_evolution_fit(
        csv_dir: Path,
        colour: str | tuple[float, float, float, float],
        label: str,
    ) -> PlotSeries:
        s: PlotSeries = plt_series.floc_count_evolution_fit(
            csv_dir,
            colour,
            label,
            normalised=True,
            reset_time=True,
        )
        return s

    def get_series_pdf(
        data_dir: Path,
        colour: str | tuple[float, float, float, float],
        label: str,
        marker: str,
    ) -> tuple[
        PlotSeries,
        PlotSeries,
        PlotSeries,
        PlotSeries,
        PlotSeries,
        PlotSeries,
        PlotSeries,
        PlotSeries,
        PlotSeries,
        PlotSeries,
        PlotSeries,
        PlotSeries,
    ]:
        (
            s_n_p_PDF,
            s_D_f_d_particle_PDF,
            s_D_g_d_particle_PDF,
            s_n_p_PDF_err,
            s_D_f_d_particle_PDF_err,
            s_D_g_d_particle_PDF_err,
            s_mass_n_p_PDF,
            s_mass_D_f_d_particle_PDF,
            s_mass_D_g_d_particle_PDF,
            s_mass_n_p_PDF_err,
            s_mass_D_f_d_particle_PDF_err,
            s_mass_D_g_d_particle_PDF_err,
        ) = plt_series.floc_pdf(
            floc_dir=data_dir,
            labels=[label for _ in range(6)],
            colours=[colour for _ in range(6)],
            markers=[marker for _ in range(6)],
        )

        return (
            s_n_p_PDF,
            s_D_f_d_particle_PDF,
            s_D_g_d_particle_PDF,
            s_n_p_PDF_err,
            s_D_f_d_particle_PDF_err,
            s_D_g_d_particle_PDF_err,
            s_mass_n_p_PDF,
            s_mass_D_f_d_particle_PDF,
            s_mass_D_g_d_particle_PDF,
            s_mass_n_p_PDF_err,
            s_mass_D_f_d_particle_PDF_err,
            s_mass_D_g_d_particle_PDF_err,
        )

    def get_series_avg(
        data_dir: Path,
        label: str,
        colour: str | tuple[float, float, float, float],
        marker: str,
    ) -> tuple[
        PlotSeries,
        PlotSeries,
        PlotSeries,
        PlotSeries,
        PlotSeries,
        PlotSeries,
        PlotSeries,
        PlotSeries,
    ]:
        (
            s_D_f_d_particle_avg,
            s_D_g_d_particle_avg,
            s_D_f_d_particle_mass_avg,
            s_D_g_d_particle_mass_avg,
            s_D_f_d_particle_err,
            s_D_g_d_particle_err,
            s_D_f_d_particle_mass_err,
            s_D_g_d_particle_mass_err,
        ) = plt_series.floc_avg_dir(
            floc_dir=data_dir,
            labels=[label for _ in range(4)],
            colours=[colour for _ in range(4)],
            markers=[marker for _ in range(4)],
        )
        return (
            s_D_f_d_particle_avg,
            s_D_g_d_particle_avg,
            s_D_f_d_particle_mass_avg,
            s_D_g_d_particle_mass_avg,
            s_D_f_d_particle_err,
            s_D_g_d_particle_err,
            s_D_f_d_particle_mass_err,
            s_D_g_d_particle_mass_err,
        )

    plot_dir.mkdir(parents=True, exist_ok=True)
    s_evo_list: list[PlotSeries] = []
    s_evo_fit_list: list[PlotSeries] = []
    s_pdf_np_list: list[PlotSeries] = []
    s_pdf_Df_list: list[PlotSeries] = []
    s_pdf_Dg_list: list[PlotSeries] = []
    s_pdf_np_err_list: list[PlotSeries] = []
    s_pdf_Df_err_list: list[PlotSeries] = []
    s_pdf_Dg_err_list: list[PlotSeries] = []
    s_pdf_np_mass_list: list[PlotSeries] = []
    s_pdf_Df_mass_list: list[PlotSeries] = []
    s_pdf_Dg_mass_list: list[PlotSeries] = []
    s_pdf_np_mass_err_list: list[PlotSeries] = []
    s_pdf_Df_mass_err_list: list[PlotSeries] = []
    s_pdf_Dg_mass_err_list: list[PlotSeries] = []
    s_avg_Df_list: list[PlotSeries] = []
    s_avg_Dg_list: list[PlotSeries] = []
    s_mass_avg_Df_list: list[PlotSeries] = []
    s_mass_avg_Dg_list: list[PlotSeries] = []
    s_avg_Df_err_list: list[PlotSeries] = []
    s_avg_Dg_err_list: list[PlotSeries] = []
    s_mass_avg_Df_err_list: list[PlotSeries] = []
    s_mass_avg_Dg_err_list: list[PlotSeries] = []
    for i in range(len(data_dirs)):
        s_evo = get_series_floc_evolution(
            data_dirs[i],
            colours[i],
            labels[i],
        )
        s_evo_list.append(s_evo)

        s_evo_fit = get_series_floc_evolution_fit(
            data_dirs[i],
            colours[i],
            labels[i],
        )
        s_evo_fit_list.append(s_evo_fit)
        (
            s_np,
            s_Df,
            s_Dg,
            s_np_err,
            s_Df_err,
            s_Dg_err,
            s_np_mass,
            s_Df_mass,
            s_Dg_mass,
            s_np_mass_err,
            s_Df_mass_err,
            s_Dg_mass_err,
        ) = get_series_pdf(
            data_dirs[i],
            colours[i],
            labels[i],
            markers[i],
        )
        s_pdf_np_list.append(s_np)
        s_pdf_Df_list.append(s_Df)
        s_pdf_Dg_list.append(s_Dg)
        s_pdf_np_err_list.append(s_np_err)
        s_pdf_Df_err_list.append(s_Df_err)
        s_pdf_Dg_err_list.append(s_Dg_err)
        s_pdf_np_mass_list.append(s_np_mass)
        s_pdf_Df_mass_list.append(s_Df_mass)
        s_pdf_Dg_mass_list.append(s_Dg_mass)
        s_pdf_np_mass_err_list.append(s_np_mass_err)
        s_pdf_Df_mass_err_list.append(s_Df_mass_err)
        s_pdf_Dg_mass_err_list.append(s_Dg_mass_err)

        (
            s_avg_Df,
            s_avg_Dg,
            s_mass_avg_Df,
            s_mass_avg_Dg,
            s_err_Df,
            s_err_Dg,
            s_mass_err_Df,
            s_mass_err_Dg,
        ) = get_series_avg(data_dirs[i], labels[i], colours[i], markers[i])
        s_avg_Df_list.append(s_avg_Df)
        s_avg_Dg_list.append(s_avg_Dg)
        s_mass_avg_Df_list.append(s_mass_avg_Df)
        s_mass_avg_Dg_list.append(s_mass_avg_Dg)
        s_avg_Df_err_list.append(s_err_Df)
        s_avg_Dg_err_list.append(s_err_Dg)
        s_mass_avg_Df_err_list.append(s_mass_err_Df)
        s_mass_avg_Dg_err_list.append(s_mass_err_Dg)

    if plot_evo_fit:
        s_evo_list = s_evo_fit_list + s_evo_list
    plt_templ.floc_count_evolution(plot_dir, s_evo_list, normalised=True)
    plt_templ.n_p_pdf(plot_dir, s_pdf_np_err_list + s_pdf_np_list)
    plt_templ.D_f_pdf(plot_dir, s_pdf_Df_err_list + s_pdf_Df_list)
    plt_templ.D_g_pdf(plot_dir, s_pdf_Dg_err_list + s_pdf_Dg_list)
    plt_templ.n_p_mass_pdf(plot_dir, s_pdf_np_mass_err_list + s_pdf_np_mass_list)
    plt_templ.D_f_mass_pdf(plot_dir, s_pdf_Df_mass_err_list + s_pdf_Df_mass_list)
    plt_templ.D_g_mass_pdf(plot_dir, s_pdf_Dg_mass_err_list + s_pdf_Dg_mass_list)

    if True:
        s_avg_Df_list = s_avg_Df_err_list + s_avg_Df_list
        s_avg_Dg_list = s_avg_Dg_err_list + s_avg_Dg_list
        s_avg_Dg_list = s_avg_Dg_err_list + s_avg_Dg_list
        s_mass_avg_Df_list = s_mass_avg_Df_err_list + s_mass_avg_Df_list
        s_mass_avg_Dg_list = s_mass_avg_Dg_err_list + s_mass_avg_Dg_list
    plt_templ.avg_D_f(
        plot_dir,
        s_avg_Df_list,
        inner_units=False,
    )
    plt_templ.avg_D_g(
        plot_dir,
        s_avg_Dg_list,
        inner_units=False,
    )
    plt_templ.mass_avg_D_f(
        plot_dir,
        s_mass_avg_Df_list,
        inner_units=False,
    )
    plt_templ.mass_avg_D_g(
        plot_dir,
        s_mass_avg_Dg_list,
        inner_units=False,
    )


def phi_eulerian(
    plot_dir: Path,
    data_dir: Path,
    data_names: list[str],
    labels: list[str],
    colours: list[str | tuple[float, float, float, float]],
    show_errs: bool,
) -> None:
    csv_dirs: list[Path] = [data_dir / data_name for data_name in data_names]

    s_list: list[PlotSeries] = []
    s_err_list: list[PlotSeries | None] = []
    i = 0
    for fluid_dir in csv_dirs:
        s_vfu, s_err_vfu = plt_series.phi_eulerian_vfu(
            csv_dir=fluid_dir,
            colour=colours[i],
            linestyle="-",
            # label=labels[i] + " (vfu)",
            label=labels[i],
            normalised=True,
            show_err=show_errs,
        )
        s_list.append(s_vfu)
        s_err_list.append(s_err_vfu)

        # s_vfu, s_err_vfu = plt_series.phi_eulerian_ana(
        #     csv_dir=fluid_dir,
        #     colour=colours[i + 1],
        #     linestyle="--",
        #     label=labels[i] + " (ana)",
        #     phi_tot=None,
        # )
        # s_list.append(s_vfu)
        # s_err_list.append(s_err_vfu)

        # i += 2
        i += 1

    s_plot: list[PlotSeries] = []
    if show_errs:
        if any(x is None for x in s_err_list):
            raise ValueError("s_err_lsit contains None entries")
        s_plot += s_err_list  # type: ignore
    s_plot += s_list

    plt_templ.phi_eulerian(output_dir=plot_dir, series_list=s_plot, normalised=True)


def lagrangian_data(
    plot_dir: Path,
    data_dir: Path,
    data_names: list[str],
    labels: list[str],
    colours: list[str | tuple[float, float, float, float]],
    markers: list[str],
    show_errs: bool,
    separate_plots: bool,
) -> None:
    csv_dirs: list[Path] = [data_dir / data_name for data_name in data_names]

    s_a_list: list[list[PlotSeries]] = []
    s_a_fit_list: list[PlotSeries] = []
    s_a_err_list: list[list[PlotSeries]] = []

    s_up_list: list[list[PlotSeries]] = []
    s_up_err_list: list[list[PlotSeries]] = []

    s_up_all_list: list[PlotSeries] = []
    s_up_err_all_list: list[PlotSeries] = []

    yp_list: list[float] = [5.0, 30.0, 180.0]

    for i, csv_dir in enumerate(csv_dirs):

        colours_local: list[str | tuple[float, float, float, float]]
        labels_local: list[str | None]
        if not separate_plots:
            colours_local = [colours[i] for _ in range(max(3, len(csv_dirs)))]
            labels_local = [labels[i] for _ in range(max(3, len(csv_dirs)))]
            markers_local = markers
        else:
            colours_local = colours
            labels_local = [None for _ in range(max(3, len(csv_dirs)))]
            markers_local = markers

        s_ax, s_ay, s_az, s_ax_err, s_ay_err, s_az_err, s_a_fit = (
            plt_series.lagrangian_acceleration_pdf(
                csv_dir=csv_dir,
                labels=labels_local,
                colours=colours_local,
                markers=markers_local,
                show_legend=separate_plots,
            )
        )
        s_a_list.append([s_ax, s_ay, s_az])
        s_a_fit_list.append(s_a_fit)
        s_a_err_list.append([s_ax_err, s_ay_err, s_az_err])

        s_up_list.append([])
        s_up_err_list.append([])
        for j, yp in enumerate(yp_list):
            s_up, s_up_err = plt_series.lagrangian_u_p_pdf(
                csv_dir=csv_dir,
                yp=yp,
                label=labels_local[j],
                colour=colours_local[j],
                marker=markers_local[j],
                show_legend=separate_plots,
            )
            s_up_list[i].append(s_up)
            s_up_err_list[i].append(s_up_err)

        if separate_plots:
            labels_local = [labels[i] for _ in range(max(3, len(csv_dirs)))]
        s_up_all, s_up_err_all = plt_series.lagrangian_u_p_pdf(
            csv_dir=csv_dir,
            yp=None,
            label=labels_local[i],
            colour=colours_local[i],
            marker=markers[i],
            show_legend=True,
        )
        s_up_all_list.append(s_up_all)
        s_up_err_all_list.append(s_up_err_all)

    s_a_proxies: list[PlotSeries] = plt_series.lagrangian_acceleration_pdf_proxies(
        markers=markers,
        labels=labels,
        colours=colours,
        marker_cases=["s" for _ in range(len(labels))],
    )
    s_up_proxies: list[PlotSeries] = plt_series.lagrangian_up_pdf_proxies(
        markers=markers,
        labels=labels,
        yp_list=yp_list,
        colours=colours,
        marker_cases=["s" for _ in range(len(labels))],
    )

    if show_errs:
        for i, csv_dir in enumerate(csv_dirs):
            s_a_list[i] = s_a_err_list[i] + s_a_list[i]
            s_up_list[i] = s_up_err_list[i] + s_up_list[i]
        s_up_all_list = s_up_err_all_list + s_up_all_list

    if separate_plots:
        for i, csv_dir in enumerate(csv_dirs):
            plt_templ.lagrangian_acceleration_pdf(
                plot_dir, [s_a_fit_list[i]] + s_a_list[i], labels[i]
            )
            plt_templ.lagrangian_up_pdf(plot_dir, s_up_list[i], labels[i])
    else:
        s_a_plot: list[PlotSeries] = [s_a_fit_list[0]]
        s_up_plot: list[PlotSeries] = []
        for i, csv_dir in enumerate(csv_dirs):
            s_a_plot += s_a_list[i]
            s_up_plot += s_up_list[i]
        plt_templ.lagrangian_acceleration_pdf(plot_dir, s_a_plot + s_a_proxies, None)
        plt_templ.lagrangian_up_pdf(plot_dir, s_up_plot + s_up_proxies, None)
    plt_templ.lagrangian_up_pdf(plot_dir, s_up_all_list, "all")


def fam_tree(
    plot_dir: Path,
    data_dir: Path,
    data_names: list[str],
    labels: list[str],
    colours: list[str | tuple[float, float, float, float]],
    markers: list[str],
    linestyles: list[str],
    separate_plots: bool,
    unfiltered: Literal[
        1, 2, 3
    ],  # 1 dont show them, 2 superimpose them, 3 separately plot them
) -> None:
    csv_dirs: list[Path] = [data_dir / data_name for data_name in data_names]

    s_list_filtered: list[list[PlotSeries]] = []
    s_list_unfiltered: list[list[PlotSeries]] = []

    use_markers: bool = False
    if use_markers:
        linestyles = ["None" for _ in range(2)]
    else:
        markers = ["None" for _ in range(2)]

    for i, csv_dir in enumerate(csv_dirs):

        labels_local: list[str | None]
        if not separate_plots:
            labels_local = [labels[i] for _ in range(2)]
            markers_local = markers
            linestyles_local = linestyles
            colours_local = [colours[i] for _ in range(2)]
        else:
            labels_local = [None for _ in range(2)]
            markers_local = markers
            colours_local = colours
            linestyles_local = linestyles

        s_formation_filtered, _ = plt_series.family_tree_breakup_formation_pdf(
            csv_dir=csv_dir,
            label=labels_local[0],
            colour=colours_local[0],
            marker=markers_local[0],
            linestyle=linestyles_local[0],
            type="formation",
            filtered_t_min=True,
            name=None,
            show_label=True,
        )
        s_breakup_filtered, _ = plt_series.family_tree_breakup_formation_pdf(
            csv_dir=csv_dir,
            label=labels_local[1],
            colour=colours_local[1],
            marker=markers_local[1],
            linestyle=linestyles_local[1],
            type="breakup",
            filtered_t_min=True,
            name=None,
            show_label=True,
        )
        s_formation_unfiltered = plt_series.family_tree_breakup_formation_pdf(
            csv_dir=csv_dir,
            label=labels_local[0],
            colour=colours_local[0],
            marker=markers_local[0],
            linestyle=linestyles_local[0],
            type="formation",
            filtered_t_min=False,
            name=None,
            show_label=True,
        )
        s_breakup_unfiltered = plt_series.family_tree_breakup_formation_pdf(
            csv_dir=csv_dir,
            label=labels_local[1],
            colour=colours_local[1],
            marker=markers_local[1],
            linestyle=linestyles_local[1],
            type="breakup",
            filtered_t_min=False,
            name=None,
            show_label=True,
        )
        s_list_filtered.append([s_breakup_filtered, s_formation_filtered])
        if unfiltered == 2:
            s_list_unfiltered.append(
                [s_breakup_unfiltered[1], s_formation_unfiltered[1]]
            )
        elif unfiltered == 3:
            s_list_unfiltered.append(
                [s_breakup_unfiltered[0], s_formation_unfiltered[0]]
            )

    if separate_plots:
        for i, csv_dir in enumerate(csv_dirs):
            if unfiltered == 1:
                plt_templ.family_tree_breakup_formation_pdf(
                    plot_dir, s_list_filtered[i], labels[i], 1
                )
            elif unfiltered == 2:
                plt_templ.family_tree_breakup_formation_pdf(
                    plot_dir, s_list_filtered[i] + s_list_unfiltered[i], labels[i], 2
                )
            elif unfiltered == 3:
                plt_templ.family_tree_breakup_formation_pdf(
                    plot_dir, s_list_filtered[i], labels[i], 1
                )
                plt_templ.family_tree_breakup_formation_pdf(
                    plot_dir, s_list_unfiltered[i], labels[i], 3
                )
    else:
        s_list_all: list[PlotSeries] = []
        s_list_all_unfiltered: list[PlotSeries] = []
        for i, csv_dir in enumerate(csv_dirs):
            if unfiltered == 1:
                s_list_all += [s_list_filtered[i][0]]
            elif unfiltered == 2:
                s_list_all += [s_list_filtered[i][0], s_list_unfiltered[i][0]]
            elif unfiltered == 3:
                s_list_all += [s_list_filtered[i][0]]
                s_list_all_unfiltered += [s_list_unfiltered[i][0]]
        for i, csv_dir in enumerate(csv_dirs):
            if unfiltered == 1:
                s_list_all += [s_list_filtered[i][1]]
            elif unfiltered == 2:
                s_list_all += [s_list_filtered[i][1], s_list_unfiltered[i][1]]
            elif unfiltered == 3:
                s_list_all += [s_list_filtered[i][1]]
                s_list_all_unfiltered += [s_list_unfiltered[i][1]]
        if unfiltered == 1:
            plt_templ.family_tree_breakup_formation_pdf(plot_dir, s_list_all, None, 1)
        elif unfiltered == 2:
            plt_templ.family_tree_breakup_formation_pdf(plot_dir, s_list_all, None, 2)
        elif unfiltered == 3:
            plt_templ.family_tree_breakup_formation_pdf(plot_dir, s_list_all, None, 1)
            plt_templ.family_tree_breakup_formation_pdf(
                plot_dir, s_list_all_unfiltered, None, 3
            )


def floc_timescale(
    plot_dir: Path,
    data_dir: Path,
    data_names: list[str],
    labels: list[str],
    cmap_breakup: Colormap,
    cmap_formation: Colormap,
) -> None:
    csv_dirs: list[Path] = [data_dir / data_name for data_name in data_names]

    s_list: list[list[PlotSeries]] = [[] for _ in range(len(csv_dirs))]
    colorbar_funcs_list: list[list[Callable[[Axes], None]]] = []
    min_color_val = 0.3

    for i, csv_dir in enumerate(csv_dirs):

        files = list((csv_dir / "floc_timescale").glob("*breakup_t_min=*.csv"))
        timescales: list[float] = []
        for file in files:
            match = re.search(r"t_min=(\d+\.\d+)", file.name)
            if match:
                timescales.append(float(match.group(1)))

        timescales = sorted(set(timescales))

        max_t = max(timescales)

        colours_formation: list[str | tuple[float, float, float, float]] = [
            cmap_formation(t_min / max_t * (1 - min_color_val) + min_color_val)
            for t_min in timescales
        ]
        colours_breakup: list[str | tuple[float, float, float, float]] = [
            cmap_breakup(t_min / max_t * (1 - min_color_val) + min_color_val)
            for t_min in timescales
        ]

        colorbar_funcs = create_colorbar_functions(
            timescales, max_t, cmap_formation, cmap_breakup, min_val=min_color_val
        )
        colorbar_funcs_list.append(colorbar_funcs)

        labels_local: list[str | None]

        for j, t_min in enumerate(timescales):
            _, s_formation = plt_series.family_tree_breakup_formation_pdf(
                csv_dir=csv_dir / "floc_timescale",
                # label=f"$t_{{floc,min}} = {t_min}$",
                label=None,
                colour=colours_formation[j],
                marker="None",
                linestyle="-",
                type="formation",
                filtered_t_min=True,
                name=f"formation_t_min={t_min}.csv",
                show_label=False,
            )
            _, s_breakup = plt_series.family_tree_breakup_formation_pdf(
                csv_dir=csv_dir / "floc_timescale",
                label=None,
                colour=colours_breakup[j],
                marker="None",
                linestyle="-",
                type="breakup",
                filtered_t_min=True,
                name=f"breakup_t_min={t_min}.csv",
                show_label=False,
            )
            s_list[i].append(s_breakup)
            s_list[i].append(s_formation)

        _, s_formation_estimate = plt_series.family_tree_breakup_formation_pdf(
            csv_dir=csv_dir / "floc_timescale",
            label=f"$t_{{floc,min}} = $ poisseulle estimate",
            colour="k",
            marker="None",
            linestyle="-",
            type="formation",
            filtered_t_min=True,
            name=f"formation_t_min=poisseulle.csv",
            show_label=True,
        )
        _, s_breakup_estimate = plt_series.family_tree_breakup_formation_pdf(
            csv_dir=csv_dir / "floc_timescale",
            label=f"$t_{{floc,min}} = $ poisseulle estimate",
            colour="k",
            marker="None",
            linestyle="--",
            type="breakup",
            filtered_t_min=True,
            name=f"breakup_t_min=poisseulle.csv",
            show_label=True,
        )

        s_list[i].append(s_breakup_estimate)
        s_list[i].append(s_formation_estimate)

        _, s_formation_advanced_estimate = plt_series.family_tree_breakup_formation_pdf(
            csv_dir=csv_dir,
            label=f"advanced filter",
            colour="k",
            marker="None",
            linestyle=":",
            type="formation",
            filtered_t_min=True,
            name=f"floc_formation_advanced_filtered_pdf.csv",
            show_label=True,
        )
        _, s_breakup_advanced_estimate = plt_series.family_tree_breakup_formation_pdf(
            csv_dir=csv_dir,
            label=f"advanced filter",
            colour="k",
            marker="None",
            linestyle="-.",
            type="breakup",
            filtered_t_min=True,
            name=f"floc_breakup_advanced_filtered_pdf.csv",
            show_label=True,
        )

        s_list[i].append(s_breakup_advanced_estimate)
        s_list[i].append(s_formation_advanced_estimate)

    for i, csv_dir in enumerate(csv_dirs):
        plt_templ.floc_timescale(
            plot_dir, s_list[i], labels[i], additional_objects=colorbar_funcs_list[i]
        )


def create_colorbar_functions(
    timescales: list[float],
    max_t: float,
    cmap_formation: Colormap,
    cmap_breakup: Colormap,
    min_val: float,
) -> list[Callable[[Axes], None]]:

    fraction: float = 1.2

    def add_formation_colorbar(ax: Axes) -> None:
        fig = ax.get_figure()
        width = 0.3 * fraction  # 0.3 is base width at top
        left = 0.85 - width  # Center in right half
        cax = fig.add_axes([left, 0.7, width, 0.03])  # type: ignore
        effective_max = max_t / (1 - min_val)
        norm = Normalize(vmin=-min_val * effective_max, vmax=effective_max)
        sm = ScalarMappable(norm=norm, cmap=cmap_formation)
        sm.set_array([])
        sm.set_clim(0, max_t)
        cbar = plt.colorbar(sm, cax=cax, ax=ax, orientation="horizontal")
        cbar.set_label("Formation", fontsize=10)
        cbar.ax.xaxis.set_ticks([])

    def add_breakup_colorbar(ax: Axes) -> None:
        fig = ax.get_figure()
        width = 0.3 * fraction
        left = 0.85 - width
        cax = fig.add_axes([left, 0.62, width, 0.03])  # Below formation # type: ignore
        effective_max = max_t / (1 - min_val)
        norm = Normalize(vmin=-min_val * effective_max, vmax=effective_max)
        sm = ScalarMappable(norm=norm, cmap=cmap_breakup)
        sm.set_array([])
        sm.set_clim(0, max_t)
        cbar = plt.colorbar(sm, cax=cax, ax=ax, orientation="horizontal")
        cbar.set_label("Breakup", fontsize=10)
        tick_positions = [0, max_t]
        tick_labels = ["0", f"{max_t:.2f}"]
        cbar.set_ticks(tick_positions)
        cbar.set_ticklabels(tick_labels, fontsize=8)

    return [add_formation_colorbar, add_breakup_colorbar]


def noncohesive_floc_lifetime(plot_dir: Path):
    data_name: str = "phi5p0_noCo"
    label: str = r"$\phi_{5\%}$ no cohesion"
    data_dir: Path = parent_dir / "data/"
    csv_dir: Path = data_dir / data_name

    s_list: list[PlotSeries] = list(plt_series.noncohesive_floc_lifetime(csv_dir))
    plt_templ.noncohesive_floc_lifetime(
        output_dir=plot_dir,
        series_list=s_list,
        label=label,
    )


def coagulation_kernel(
    plot_dir: Path,
    data_dir: Path,
    data_names: list[str],
    labels: list[str],
    contour_sigmas: list[float],
):

    cmap: Colormap = plt.get_cmap("Blues")
    s_pcolormesh_list: list[PlotSeries] = []
    s_contour_list: list[PlotSeries] = []
    for i, data_name in enumerate(data_names):
        s_pcolormesh, s_contour = plt_series.coagulation_kernel(
            data_dir / data_name,
            labels[i],
            cmap,
            xlim=(1, 50),
            pcolormesh_log_scale=True,
            contour_log_scale=True,
            contour_interp_factor=1,
            contour_sigma=contour_sigmas[i],
            contour_levels=10,
        )
        s_pcolormesh_list.append(s_pcolormesh)
        s_contour_list.append(s_contour)
    for i in range(len(data_names)):
        # plt_templ.coagulation_kernel(
        #     plot_dir, s_pcolormesh_list[i], s_contour_list[i], data_names[i]
        # )
        plt_templ.coagulation_kernel(
            plot_dir, s_pcolormesh_list[i], None, data_names[i]
        )


def fragment_size_distribution(
    plot_dir: Path,
    data_dir: Path,
    data_names: list[str],
    labels: list[str],
):

    cmap: Colormap = plt.get_cmap("Blues")
    s_pcolormesh_list: list[PlotSeries] = []
    s_contour_list: list[PlotSeries] = []
    for i, data_name in enumerate(data_names):
        s_pcolormesh, s_contour = plt_series.fragment_size_distribution(
            data_dir / data_name, labels[i], cmap, xlim=(1, 50)
        )
        s_pcolormesh_list.append(s_pcolormesh)
        s_contour_list.append(s_contour)
    for i in range(len(data_names)):
        plt_templ.fragment_size_distribution(
            plot_dir, s_pcolormesh_list[i], s_contour_list[i], data_names[i]
        )


def breakage_rate(
    plot_dir: Path,
    data_dir: Path,
    data_names: list[str],
    labels: list[str],
    colours: list[str | tuple[float, float, float, float]],
) -> None:
    s_list: list[PlotSeries] = []
    for i, data_name in enumerate(data_names):
        s_list.append(
            plt_series.breakage_rate(
                pickle_dir=data_dir / data_name,
                colour=colours[i],
                linestyle="-",
                label=labels[i],
            )
        )
    plt_templ.breakage_rate(plot_dir, s_list, n_p_max=20)


def number_density_evo_sink_source(
    plot_dir: Path,
    data_dir: Path,
    data_names: list[str],
    labels: list[str],
    linestyles: list[str],
    markers: list[str],
    colours: list[str | tuple[float, float, float, float]],
    mass_weighted: bool,
    separate_plots: bool,
):
    markers = ["" for _ in range(len(markers))]
    s_list_gain_coag: list[PlotSeries]
    s_list_loss_coag: list[PlotSeries]
    s_list_gain_frag: list[PlotSeries]
    s_list_loss_frag: list[PlotSeries]
    s_list_dn_dt: list[PlotSeries]
    s_cases: list[PlotSeries]
    s_quantities: list[PlotSeries]
    s_hline_zero: PlotSeries
    (
        s_list_gain_coag,
        s_list_loss_coag,
        s_list_gain_frag,
        s_list_loss_frag,
        s_list_dn_dt,
        s_cases,
        s_quantities,
        s_hline_zero,
    ) = plt_series.number_density_evo_sink_source(
        data_dir=data_dir,
        data_names=data_names,
        labels=labels,
        colours=colours,
        linestyles=linestyles,
        markers=markers,
        mass_weighted=mass_weighted,
        separate_plots=separate_plots,
    )
    if separate_plots:
        for i in range(len(data_names)):
            series_list: list[PlotSeries] = [
                s_hline_zero,
                s_list_gain_coag[i],
                s_list_loss_coag[i],
                s_list_gain_frag[i],
                s_list_loss_frag[i],
                s_list_dn_dt[i],
            ]
            plt_templ.number_density_evo_sink_source(
                output_dir=plot_dir, series_list=series_list, name=data_names[i], xmax=20, mass_weighted=mass_weighted
            )
    else:
        series_list: list[PlotSeries] = (
            s_quantities
            +s_cases
            + [s_hline_zero]
            + s_list_gain_coag
            + s_list_loss_coag
            + s_list_gain_frag
            + s_list_loss_frag
            + s_list_dn_dt
        )
        plt_templ.number_density_evo_sink_source(
            output_dir=plot_dir, series_list=series_list, name=None, xmax=20, mass_weighted=mass_weighted
        )


def main() -> None:

    plot_dir: Path = Path("./output/plots")
    data_names: list[str] = [
        "phi5p0_noCo",
        "phi1p5",
        "phi3p0",
        "phi5p0",
        # "test"
    ]
    labels: list[str] = [
        r"$\phi_{5\%}$ no cohesion",
        r"$\phi_{1.5\%}$",
        r"$\phi_{3\%}$",
        r"$\phi_{5\%}$",
        # "test"
    ]
    coagulation_kernel_sigmas: list[float] = [
        2,
        5,
        5,
        5,
    ]
    data_dir: Path = parent_dir / "data/"
    cmap = plt.get_cmap("tab10")
    cb_palette = sns.color_palette("colorblind", n_colors=5)
    blue_cmap = plt.get_cmap("Blues")
    red_cmap = plt.get_cmap("Reds")

    # colours: list[str | tuple[float, float, float, float]] = [
    #     cmap(x) for x in np.linspace(0.3, 0.9, 5)
    # ]
    colours: list[str | tuple[float, float, float, float]] = list(cb_palette)
    colours_fam_tree: list[str | tuple[float, float, float, float]] = [
        blue_cmap(0.55),
        red_cmap(0.55),
    ]
    markers: list[str] = ["o", "s", "^", "v", "P"]
    linestyles: list[str] = ["-", "--", "-.", ":", ":"]
    # floc(
    #     plot_dir,
    #     data_dir,
    #     data_names,
    #     labels,
    #     colours,
    #     markers,
    #     linestyles,
    #     plot_evo_fit=False,
    # )
    # fluid_wall_normal(
    #     data_dir, plot_dir, data_dir, data_names, labels, colours, use_markers=True
    # )
    # fluid_wall_normal(
    #     data_dir, plot_dir, data_dir, data_names, labels, colours, use_markers=False
    # )
    # fuild_velocity_profile(
    #     data_dir, plot_dir, data_dir, data_names, labels, colours, use_markers=True
    # )
    # phi_eulerian(plot_dir, data_dir, data_names, labels, colours, False)
    # lagrangian_data(
    #     plot_dir,
    #     data_dir,
    #     data_names,
    #     labels,
    #     colours,
    #     markers,
    #     show_errs=False,
    #     separate_plots=False,
    # )
    # fam_tree(
    #     plot_dir,
    #     data_dir,
    #     data_names,
    #     labels,
    #     colours_fam_tree,
    #     markers,
    #     linestyles,
    #     separate_plots=True,
    #     unfiltered=3,
    # )
    # floc_timescale(
    #     plot_dir=plot_dir,
    #     data_dir=data_dir,
    #     data_names=data_names,
    #     labels=labels,
    #     cmap_breakup=red_cmap,
    #     cmap_formation=blue_cmap,
    # )
    # noncohesive_floc_lifetime(plot_dir)
    coagulation_kernel(
        plot_dir, data_dir, data_names, labels, contour_sigmas=coagulation_kernel_sigmas
    )
    fragment_size_distribution(
        plot_dir,
        data_dir,
        data_names,
        labels,
    )
    breakage_rate(
        plot_dir,
        data_dir,
        data_names,
        labels,
        colours,
    )
    number_density_evo_sink_source(
        plot_dir=plot_dir,
        data_dir=data_dir,
        data_names=data_names,
        labels=labels,
        linestyles=linestyles,
        markers=markers,
        colours=colours,
        mass_weighted=True,
        separate_plots=True,
    )

    if not globals.on_anvil:
        plt.show()
