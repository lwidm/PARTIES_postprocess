from typing import Optional, Tuple, List
from pathlib import Path

from src import scripts
from src.myio import myio
from src.myio.myio import MyPath
from src.plotting.tools import PlotSeries
from src.plotting import series as plt_series
from src.plotting import templates as plt_templ
from src.scripts.run_floc_analysis import process_flocs


def fluid(utexas_dir: MyPath, plot_dir: MyPath):

    # ==============================
    # Inputs
    # ==============================

    data_names: List[str] = [
        # "phi1p5",
        "phi5p0",
        # "phi5p0_new",
    ]
    labels: List[str] = [
        # r"$\phi_{1.5\%}$",
        r"$\phi_{5\%}$",
        # r"$\phi_{5\%}$ new",
    ]
    parties_data_dir: str = "/media/usb/UCSB/data/"
    output_dir: str = "/media/usb/UCSB/output/"
    # colours: List[str] = ["C0", "C1", "C2", "C3", "C4"]
    colours: List[str] = ["k"]

    # ==============================
    # Automation
    # ==============================

    Num_data: int = len(data_names)
    utexas_h5 = Path(utexas_dir) / "utexas.h5"
    parties_processed_filename = "parties_reynolds.h5"
    utexas_wall_series: List[PlotSeries] = plt_series.u_plus_mean_wall_utexas(utexas_h5)
    parties_wall_series: List[List[PlotSeries]] = []
    for i in range(Num_data):
        parties_wall_series.append(
            plt_series.u_plus_mean_wall_parties(
                Path(output_dir + data_names[i] + "/fluid")
                / parties_processed_filename,
                label=labels[i],
                colour=colours[i],
                linestyles=("-", "--"),
            )
        )

    all_wall_series: List[PlotSeries] = utexas_wall_series
    for series in parties_wall_series:
        all_wall_series += series
    plt_templ.velocity_profile_wall(plot_dir, all_wall_series)

    utexas_stress_series: List[PlotSeries] = plt_series.normal_stress_wall_utexas(
        utexas_h5
    )
    parties_stress_series: List[List[PlotSeries]] = []
    for i in range(Num_data):
        parties_stress_series.append(
            plt_series.normal_stress_wall_parties(
                Path(output_dir + data_names[i] + "/fluid")
                / parties_processed_filename,
                label=labels[i],
                colour=colours[i],
            )
        )
    all_stress_series: List[PlotSeries] = utexas_stress_series
    for series in parties_stress_series:
        all_stress_series += series
    plt_templ.normal_stress_wall(plot_dir, all_stress_series)


def floc(plot_dir: MyPath):

    # ==============================
    # Inputs
    # ==============================

    compute: bool = True
    compute_flocs: List[bool] = [
        False,
        False,
        # False,
    ]
    data_names: List[str] = [
        "phi1p5",
        "phi5p0",
        # "phi5p0_new",
    ]
    labels: List[str] = [
        r"$\phi_{1.5\%}$",
        r"$\phi_{5\%}$",
        # r"$\phi_{5\%}$ new",
    ]
    trn: List[bool] = [False, True, True]
    Re_tau: List[float] = [
        189.54087993838434,
        180,
        # 180,
    ]
    parties_data_dir: str = "/media/usb/UCSB/data/"
    output_dir: str = "/media/usb/UCSB/output/"
    min_file_indices: List[Optional[int]] = [
        None,
        None,
        # None,
    ]
    max_file_indices: List[Optional[int]] = [
        None,
        None,
        # None,
    ]
    min_steady_times: List[Optional[float]] = [
        200,
        None,
        # None,
    ]
    max_steady_indices: List[Optional[int]] = [
        None,
        None,
        # None,
    ]
    colours: List[str] = ["C0", "C1", "C2", "C3", "C4"]
    markers: List[str] = ["o", "s", "^", "v", "P"]
    linestyles: List[str] = ["-", "--", "-.", ":"]

    # ==============================
    # Automation
    # ==============================

    Num_data: int = len(data_names)

    parties_data_dirs: List[str] = [
        parties_data_dir + data_name for data_name in data_names
    ]
    output_dirs: List[str] = [output_dir + data_name for data_name in data_names]

    time_idx_info: List[dict] = [{} for _ in range(Num_data)]
    for i in range(Num_data):
        min_steady_time = min_steady_times[i]
        if min_steady_time is None:
            time_idx_info[i] = {"file_idx": None}
        else:
            time_idx_info[i] = myio.find_idx_from_time(
                file_prefix="Particle",
                data_dir=parties_data_dirs[i],
                target_time=min_steady_time,
            )
    min_steady_indices: List[Optional[int]] = [
        info["file_idx"] for info in time_idx_info
    ]

    plot_dir = Path(plot_dir)
    if compute:
        for i in range(len(parties_data_dirs)):
            scripts.run_floc_analysis.main(
                parties_data_dir=parties_data_dirs[i],
                output_dir=output_dirs[i],
                trn=trn[i],
                Re_tau=Re_tau[i],
                process_flocs=compute_flocs[i],
                min_file_index=min_file_indices[i],
                max_file_index=max_file_indices[i],
                min_steady_index=min_steady_indices[i],
                max_steady_index=max_steady_indices[i],
                num_workers=6,
                use_threading=False,
            )

    def get_series_floc_evolution(
        output_dir: MyPath,
        colour: str,
        label: str,
        min_file_index: Optional[int],
        max_file_index: Optional[int],
    ) -> PlotSeries:
        s: PlotSeries = plt_series.floc_count_evolution(
            Path(output_dir) / "flocs",
            colour,
            label,
            min_file_index,
            max_file_index,
            normalised=True,
            reset_time=False,
        )
        return s

    def get_series_pdf(
        output_dir: MyPath,
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
    ]:
        s_n_p_PDF, s_D_f_PDF, s_D_g_PDF, s_n_p_PDF_err, s_D_f_PDF_err, s_D_g_PDF_err = (
            plt_series.floc_pdf(
                floc_dir=Path(output_dir) / "flocs",
                labels=[label for _ in range(3)],
                colours=[colour for _ in range(3)],
                markers=[marker for _ in range(3)],
            )
        )

        return (
            s_n_p_PDF,
            s_D_f_PDF,
            s_D_g_PDF,
            s_n_p_PDF_err,
            s_D_f_PDF_err,
            s_D_g_PDF_err,
        )

    def get_series_avg(
        output_dir: MyPath, label: str, colour: str, marker: str
    ) -> Tuple[
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
            s_D_f_avg,
            s_D_g_avg,
            s_D_f_mass_avg,
            s_D_g_mass_avg,
            s_D_f_err,
            s_D_g_err,
            s_D_f_mass_err,
            s_D_g_mass_err,
        ) = plt_series.floc_avg_dir(
            floc_dir=Path(output_dir) / "flocs",
            labels=[label for _ in range(4)],
            colours=[colour for _ in range(4)],
            markers=[marker for _ in range(4)],
            inner_units=False,
        )
        return (
            s_D_f_avg,
            s_D_g_avg,
            s_D_f_mass_avg,
            s_D_g_mass_avg,
            s_D_f_err,
            s_D_g_err,
            s_D_f_mass_err,
            s_D_g_mass_err,
        )

    plot_dir.mkdir(parents=True, exist_ok=True)
    s_evo_list: List[PlotSeries] = []
    s_pdf_np_list: List[PlotSeries] = []
    s_pdf_Df_list: List[PlotSeries] = []
    s_pdf_Dg_list: List[PlotSeries] = []
    s_pdf_np_err_list: List[PlotSeries] = []
    s_pdf_Df_err_list: List[PlotSeries] = []
    s_pdf_Dg_err_list: List[PlotSeries] = []
    s_avg_Df_list: List[PlotSeries] = []
    s_avg_Dg_list: List[PlotSeries] = []
    s_mass_avg_Df_list: List[PlotSeries] = []
    s_mass_avg_Dg_list: List[PlotSeries] = []
    s_avg_Df_err_list: List[PlotSeries] = []
    s_avg_Dg_err_list: List[PlotSeries] = []
    s_mass_avg_Df_err_list: List[PlotSeries] = []
    s_mass_avg_Dg_err_list: List[PlotSeries] = []
    for i in range(len(output_dirs)):
        s_evo = get_series_floc_evolution(
            output_dirs[i],
            colours[i],
            labels[i],
            min_file_indices[i],
            max_file_indices[i],
        )
        s_evo_list.append(s_evo)
        s_np, s_Df, s_Dg, s_np_err, s_Df_err, s_Dg_err = get_series_pdf(
            output_dirs[i],
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

        (
            s_avg_Df,
            s_avg_Dg,
            s_mass_avg_Df,
            s_mass_avg_Dg,
            s_err_Df,
            s_err_Dg,
            s_mass_err_Df,
            s_mass_err_Dg,
        ) = get_series_avg(output_dirs[i], labels[i], colours[i], markers[i])
        s_avg_Df_list.append(s_avg_Df)
        s_avg_Dg_list.append(s_avg_Dg)
        s_mass_avg_Df_list.append(s_mass_avg_Df)
        s_mass_avg_Dg_list.append(s_mass_avg_Dg)
        s_avg_Df_err_list.append(s_err_Df)
        s_avg_Dg_err_list.append(s_err_Dg)
        s_mass_avg_Df_err_list.append(s_mass_err_Df)
        s_mass_avg_Dg_err_list.append(s_mass_err_Dg)

    plt_templ.floc_count_evolution(plot_dir, s_evo_list, normalised=True)
    plt_templ.n_p_pdf(plot_dir, s_pdf_np_err_list + s_pdf_np_list)
    plt_templ.D_f_pdf(plot_dir, s_pdf_Df_err_list + s_pdf_Df_list)
    plt_templ.D_g_pdf(plot_dir, s_pdf_Dg_err_list + s_pdf_Dg_list)

    if False:
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


def main() -> None:
    plot_dir: Path = Path("./output/plots")
    floc(plot_dir)
    fluid("/media/usb/UCSB/output", plot_dir)
