from typing import Optional, Tuple, List
from pathlib import Path

from src import scripts
from src.plotting.tools import PlotSeries
from src.plotting import series as plt_series
from src.plotting import templates as plt_templ
from src import globals

from matplotlib import pyplot as plt

# parent_dir: Path = Path("/media/usb/UCSB/")
parent_dir: Path = Path("./")


def fluid(
    utexas_dir: Path,
    plot_dir: Path,
    data_dir: Path,
    data_names: List[str],
    labels: List[str],
):

    data_dirs: List[Path] = [data_dir / data_name for data_name in data_names]

    colours: List[str] = ["C0", "C1", "C2", "C3", "C4"]
    # colours: List[str] = ["k", "k", "k", "k", "k"]

    # ==============================
    # Automation
    # ==============================

    Num_data: int = len(data_names)
    utexas_wall_series: List[PlotSeries] = plt_series.u_plus_mean_utexas(
        utexas_dir, "utexas", "k", True, True, None
    )
    parties_wall_series: List[List[PlotSeries]] = []
    for i in range(Num_data):
        parties_wall_series.append(
            plt_series.u_plus_mean_parties(
                data_dirs[i],
                label=labels[i],
                colour=colours[i],
                log_fit=True,
                visc_fit=False,
                linestyles=("-", "--", "--"),
            )
        )

    all_wall_series: List[PlotSeries] = utexas_wall_series
    for series in parties_wall_series:
        all_wall_series += series
    plt_templ.velocity_profile_wall(plot_dir, all_wall_series)

    utexas_stress_series: List[PlotSeries] = plt_series.normal_stress_wall_utexas(
        utexas_dir
    )
    parties_stress_series: List[List[PlotSeries]] = []
    for i in range(Num_data):
        parties_stress_series.append(
            plt_series.normal_stress_wall_parties(
                data_dirs[i],
                label=labels[i],
                colour=colours[i],
            )
        )
    all_stress_series: List[PlotSeries] = utexas_stress_series
    for series in parties_stress_series:
        all_stress_series += series
    plt_templ.normal_stress_wall(plot_dir, all_stress_series)


def floc(
    plot_dir: Path,
    data_dir: Path,
    data_names: List[str],
    labels: List[str],
    colours: List[str],
    markers: List[str],
    linestyles: List[str],
) -> None:

    data_dirs: List[Path] = [data_dir / data_name for data_name in data_names]

    def get_series_floc_evolution(
        csv_dir: Path,
        colour: str,
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
        colour: str,
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
        colour: str,
        label: str,
        marker: str,
    ) -> Tuple[
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

    def get_series_avg(data_dir: Path, label: str, colour: str, marker: str) -> Tuple[
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
    s_evo_list: List[PlotSeries] = []
    s_evo_fit_list: List[PlotSeries] = []
    s_pdf_np_list: List[PlotSeries] = []
    s_pdf_Df_list: List[PlotSeries] = []
    s_pdf_Dg_list: List[PlotSeries] = []
    s_pdf_np_err_list: List[PlotSeries] = []
    s_pdf_Df_err_list: List[PlotSeries] = []
    s_pdf_Dg_err_list: List[PlotSeries] = []
    s_pdf_np_mass_list: List[PlotSeries] = []
    s_pdf_Df_mass_list: List[PlotSeries] = []
    s_pdf_Dg_mass_list: List[PlotSeries] = []
    s_pdf_np_mass_err_list: List[PlotSeries] = []
    s_pdf_Df_mass_err_list: List[PlotSeries] = []
    s_pdf_Dg_mass_err_list: List[PlotSeries] = []
    s_avg_Df_list: List[PlotSeries] = []
    s_avg_Dg_list: List[PlotSeries] = []
    s_mass_avg_Df_list: List[PlotSeries] = []
    s_mass_avg_Dg_list: List[PlotSeries] = []
    s_avg_Df_err_list: List[PlotSeries] = []
    s_avg_Dg_err_list: List[PlotSeries] = []
    s_mass_avg_Df_err_list: List[PlotSeries] = []
    s_mass_avg_Dg_err_list: List[PlotSeries] = []
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

    if True:
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
    data_names: List[str],
    labels: List[str],
    output_dir: Path,
    colours: List[str],
    show_errs: bool,
) -> None:
    fluid_dirs: List[Path] = [
        output_dir / data_name / "fluid" for data_name in data_names
    ]

    s_list: List[PlotSeries] = []
    s_err_list: List[Optional[PlotSeries]] = []
    for i, fluid_dir in enumerate(fluid_dirs):
        s, s_err = plt_series.phi_eulerian(
            fluid_dir=fluid_dir,
            colour=colours[i],
            linestyle="-",
            label=labels[i],
            normalised=True,
            show_err=show_errs,
        )
        s_list.append(s)
        s_err_list.append(s_err)

    s_plot: List[PlotSeries] = []
    if show_errs:
        if any(x is None for x in s_err_list):
            raise ValueError("s_err_lsit contains None entries")
        s_plot += s_err_list  # type: ignore
    s_plot += s_list

    plt_templ.phi_eulerian(output_dir=plot_dir, series_list=s_plot, normalised=True)


def main() -> None:

    plot_dir: Path = Path("./output/plots")
    data_names: List[str] = [
        # "phi1p5",
        # "phi3p0",
        # "phi5p0",
        "test"
    ]
    labels: List[str] = [
        # r"$\phi_{1.5\%}$",
        # r"$\phi_{3\%}$",
        # r"$\phi_{5\%}$",
        "test"
    ]
    data_dir: Path = parent_dir / "data/"
    colours: List[str] = ["C0", "C1", "C2", "C3", "C4"]
    markers: List[str] = ["o", "s", "^", "v", "P"]
    linestyles: List[str] = ["-", "--", "-.", ":"]
    floc(
        plot_dir,
        data_dir,
        data_names,
        labels,
        colours,
        markers,
        linestyles,
    )
    fluid(data_dir, plot_dir, data_dir, data_names, labels)
    # phi_eulerian(plot_dir, [data_names[1]], [labels[1]], output_dir, colours, False)
    if not globals.on_anvil:
        plt.show()
