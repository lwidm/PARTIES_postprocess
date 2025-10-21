from typing import Optional, Tuple, List
from pathlib import Path

from numpy import inner

from src import scripts
from src.myio import myio
from src.myio.myio import MyPath
from src.plotting.tools import PlotSeries
from src.plotting import series as plt_series
from src.plotting import templates as plt_templ


def fluid(utexas_dir: MyPath, plot_dir: MyPath):

    utexas_h5 = Path(utexas_dir) / "utexas.h5"
    parties_processed_filename = "parties_reynolds.h5"
    utexas_wall_series: List[PlotSeries] = plt_series.u_plus_mean_wall_utexas(utexas_h5)
    parties_wall_series_phi1p5: List[PlotSeries] = plt_series.u_plus_mean_wall_parties(
        Path("/media/usb/UCSB/output/phi1p5") / parties_processed_filename,
        label=r"$\phi_{1.5\%}$",
        colour="C0",
    )
    parties_wall_series_phi5p0: List[PlotSeries] = plt_series.u_plus_mean_wall_parties(
        Path("/media/usb/UCSB/output/phi5p0") / parties_processed_filename,
        label=r"$\phi_{5\%}$ no cohesion",
        colour="C1",
    )
    parties_wall_series_phi5p0_co: List[PlotSeries] = (
        plt_series.u_plus_mean_wall_parties(
            Path("/media/usb/UCSB/output/phi5p0_co") / parties_processed_filename,
            label=r"\phi_{5\%}$",
            colour="C2",
        )
    )

    all_wall_series: List[PlotSeries] = (
        utexas_wall_series
        + parties_wall_series_phi5p0_co
        + parties_wall_series_phi5p0
        + parties_wall_series_phi1p5
    )
    plt_templ.velocity_profile_wall(plot_dir, all_wall_series)

    utexas_stress_series: List[PlotSeries] = plt_series.normal_stress_wall_utexas(
        utexas_h5
    )
    parties_stress_series_phi1p5: List[PlotSeries] = (
        plt_series.normal_stress_wall_parties(
            Path("/media/usb/UCSB/output/phi1p5") / parties_processed_filename,
            label=r"$\phi_{1.5\%}$",
            colour="C0",
        )
    )
    parties_stress_series_phi5p0: List[PlotSeries] = (
        plt_series.normal_stress_wall_parties(
            Path("/media/usb/UCSB/output/phi5p0") / parties_processed_filename,
            label=r"$\phi_{5\%}$ no cohesion",
            colour="C1",
        )
    )
    parties_stress_series_phi5p0_co: List[PlotSeries] = (
        plt_series.normal_stress_wall_parties(
            Path("/media/usb/UCSB/output/phi5p0_co") / parties_processed_filename,
            label=r"$\phi_{5\%}$",
            colour="C2",
        )
    )
    all_stress_series: List[PlotSeries] = (
        utexas_stress_series
        + parties_stress_series_phi5p0_co
        + parties_stress_series_phi5p0
        + parties_stress_series_phi1p5
    )
    plt_templ.normal_stress_wall(plot_dir, all_stress_series)


def floc(plot_dir: MyPath):
    plot_dir = Path(plot_dir)
    time_idx_info: List[dict] = [{}, {}, {}]
    time_idx_info[0] = myio.find_idx_from_time(
        file_prefix="Particle",
        data_dir="/media/usb/UCSB/data/phi1p5",
        target_time=200.0,
    )
    time_idx_info[1] = myio.find_idx_from_time(
        file_prefix="Particle",
        data_dir="/media/usb/UCSB/data/phi5p0",
        target_time=30.0,
    )
    time_idx_info[2] = myio.find_idx_from_time(
        file_prefix="Particle",
        data_dir="/media/usb/UCSB/data/phi5p0_co",
        target_time=100.0,
    )
    print(time_idx_info[0])
    min_file_indices: List[Optional[int]] = [None, 101, None, None]
    max_file_indices: List[Optional[int]] = [None, None, None, None]
    min_steady_indices: List[Optional[int]] = [inf["file_idx"] for inf in time_idx_info]
    max_steady_indices: List[Optional[int]] = [None, None, None, None]
    parties_data_dirs: List[str] = [
        "/media/usb/UCSB/data/phi1p5",
        "/media/usb/UCSB/data/phi5p0",
        "/media/usb/UCSB/data/phi5p0_co",
    ]
    output_dirs: List[str] = [
        "/media/usb/UCSB/output/phi1p5",
        "/media/usb/UCSB/output/phi5p0",
        "/media/usb/UCSB/output/phi5p0_co",
    ]
    compute: bool = True
    if compute:
        trn: List[bool] = [False, False, True]
        Re_tau: List[float] = [80.6542800, 189.54087993838434, 80.6542800]
        for i in range(len(parties_data_dirs)):
            scripts.run_floc_analysis.main(
                parties_data_dir=parties_data_dirs[i],
                output_dir=output_dirs[i],
                trn=trn[i],
                Re_tau=Re_tau[i],
                min_file_index=min_file_indices[i],
                max_file_index=max_file_indices[i],
                min_steady_index=min_steady_indices[i],
                max_steady_index=max_steady_indices[i],
                num_workers=6,
            )

    def get_series_floc_evolution(
        output_dir: MyPath,
        colour: str,
        label: str,
        min_file_index: Optional[int],
        max_file_index: Optional[int],
    ) -> PlotSeries:
        s: PlotSeries = plt_series.floc_count_evolution(
            output_dir,
            colour,
            label,
            min_file_index,
            max_file_index,
            normalised=True,
            reset_time=True,
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
    ]:
        s_n_p_PDF, s_D_f_PDF, s_D_g_PDF = plt_series.floc_pdf(
            floc_dir=output_dir,
            labels=[label for _ in range(3)],
            colours=[colour for _ in range(3)],
            markers=[marker for _ in range(3)],
        )

        return (
            s_n_p_PDF,
            s_D_f_PDF,
            s_D_g_PDF,
        )

    def get_series_avg(
        output_dir: MyPath,
        label: str,
        linestyle: str,
    ) -> Tuple[
        PlotSeries,
        PlotSeries,
        PlotSeries,
        PlotSeries,
    ]:
        s_D_f_avg, s_D_g_avg, s_D_f_mass_avg, s_D_g_mass_avg = plt_series.floc_avg_dir(
            floc_dir=output_dir,
            labels=[label for _ in range(4)],
            colours=["k" for _ in range(4)],
            linestyles=[linestyle for _ in range(4)],
            inner_units=False,
        )
        return (
            s_D_f_avg,
            s_D_g_avg,
            s_D_f_mass_avg,
            s_D_g_mass_avg,
        )

    plot_dir.mkdir(parents=True, exist_ok=True)
    labels: List[str] = [r"$\phi_{1.5\%}$" r"$\phi_{5\%}$ no cohesion", r"$\phi_{5\%}$"]
    colours: List[str] = ["C0", "C1", "C2"]
    markers: List[str] = ["o", "s", "^"]
    linestyles: List[str] = ["-", "--", "-."]
    s_evo_list: List[PlotSeries] = []
    s_pdf_np_list: List[PlotSeries] = []
    s_pdf_Df_list: List[PlotSeries] = []
    s_pdf_Dg_list: List[PlotSeries] = []
    s_avg_Df_list: List[PlotSeries] = []
    s_avg_Dg_list: List[PlotSeries] = []
    s_mass_avg_Df_list: List[PlotSeries] = []
    s_mass_avg_Dg_list: List[PlotSeries] = []
    for i in range(3):
        s_evo_list[i] = get_series_floc_evolution(
            output_dirs[i],
            colours[i],
            labels[i],
            min_file_indices[i],
            max_file_indices[i],
        )
        s_np, s_Df, s_Dg= get_series_pdf(
            output_dirs[i],
            colours[i],
            labels[i],
            markers[i],
        )
        s_pdf_np_list[i] = s_np
        s_pdf_Df_list[i] = s_Df
        s_pdf_Dg_list[i] = s_Dg

        s_avg_Df, s_avg_Dg, s_mass_avg_Df, s_mass_avg_Dg = get_series_avg(
                output_dirs[i],
                labels[i],
                linestyles[i])
        s_avg_Df_list[i] = s_avg_Df
        s_avg_Dg_list[i] = s_avg_Dg
        s_mass_avg_Df_list[i] = s_mass_avg_Df
        s_mass_avg_Dg_list[i] = s_mass_avg_Dg

    plt_templ.floc_count_evolution(
        plot_dir, s_evo_list, normalised=True
    )
    plt_templ.n_p_pdf(plot_dir, s_pdf_np_list)
    plt_templ.D_f_pdf(plot_dir, s_pdf_Df_list)
    plt_templ.D_g_pdf(plot_dir, s_pdf_Dg_list)

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
