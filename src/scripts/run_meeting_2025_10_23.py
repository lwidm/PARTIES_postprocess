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
    compute: bool = True
    if compute:
        scripts.run_floc_analysis.main(
            parties_data_dir="/media/usb/UCSB/data/phi1p5",
            output_dir="/media/usb/UCSB/output/phi1p5",
            trn=False,
            u_tau=0.0288051,
            min_file_index=min_file_indices[0],
            max_file_index=max_file_indices[0],
            min_steady_index=min_steady_indices[0],
            max_steady_index=max_steady_indices[0],
            num_workers=6,
        )

        scripts.run_floc_analysis.main(
            parties_data_dir="/media/usb/UCSB/data/phi5p0",
            output_dir="/media/usb/UCSB/output/phi5p0",
            trn=False,
            u_tau=0.0288051,
            min_file_index=min_file_indices[1],
            max_file_index=max_file_indices[1],
            min_steady_index=min_steady_indices[1],
            max_steady_index=max_steady_indices[1],
            num_workers=6,
        )
        scripts.run_floc_analysis.main(
            parties_data_dir="/media/usb/UCSB/data/phi5p0_co",
            output_dir="/media/usb/UCSB/output/phi5p0_co",
            trn=True,
            u_tau=0.0288051,
            min_file_index=min_file_indices[2],
            max_file_index=max_file_indices[2],
            min_steady_index=min_steady_indices[2],
            max_steady_index=max_steady_indices[2],
            num_workers=6,
        )

    def get_plot_series_1(
        output_dir: MyPath,
        colour: str,
        label: str,
        marker: str,
        min_file_index: Optional[int],
        max_file_index: Optional[int],
    ) -> Tuple[
        PlotSeries,
        PlotSeries,
        PlotSeries,
        PlotSeries,
    ]:
        s: PlotSeries = plt_series.floc_count_evolution(
            output_dir,
            colour,
            label,
            min_file_index,
            max_file_index,
            normalised=True,
            reset_time=True,
        )
        s_n_p_PDF, s_D_f_PDF, s_D_g_PDF = plt_series.floc_pdf(
            floc_dir=output_dir,
            labels=[label, label, label],
            colours=[colour, colour, colour],
            markers=[marker, marker, marker],
            linestyles=["None", "None", "None"],
        )

        return (
            s,
            s_n_p_PDF,
            s_D_f_PDF,
            s_D_g_PDF,
        )

    def get_plot_series_2(
        output_dir: MyPath,
        colour: str,
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
            labels=[label, label, label, label],
            colours=[colour, colour, colour, colour],
            markers=["None", "None", "None", "None"],
            linestyles=[linestyle, linestyle, linestyle, linestyle],
            inner_units=False,
        )
        return (
            s_D_f_avg,
            s_D_g_avg,
            s_D_f_mass_avg,
            s_D_g_mass_avg,
        )

    plot_dir.mkdir(parents=True, exist_ok=True)
    tuple_s_1p5 = get_plot_series_1(
        "/media/usb/UCSB/output/phi1p5/flocs",
        "C0",
        r"$\phi_{1.5\%}$",
        "o",
        min_file_indices[0],
        max_file_indices[0],
    )
    tuple_s_5p0 = get_plot_series_1(
        "/media/usb/UCSB/output/phi5p0/flocs",
        "C1",
        r"$\phi_{5\%}$ no cohesion",
        "s",
        min_file_indices[1],
        max_file_indices[1],
    )
    tuple_s_5p0_co = get_plot_series_1(
        "/media/usb/UCSB/output/phi5p0_co/flocs",
        "C2",
        r"$\phi_{5\%}$",
        "^",
        min_file_indices[2],
        max_file_indices[2],
    )

    plt_templ.floc_count_evolution(
        plot_dir, [tuple_s_1p5[0], tuple_s_5p0[0], tuple_s_5p0_co[0]], normalised=True
    )
    plt_templ.n_p_pdf(plot_dir, [tuple_s_1p5[1], tuple_s_5p0[1], tuple_s_5p0_co[1]])
    plt_templ.D_f_pdf(plot_dir, [tuple_s_1p5[2], tuple_s_5p0[2], tuple_s_5p0_co[2]])
    plt_templ.D_g_pdf(plot_dir, [tuple_s_1p5[3], tuple_s_5p0[3], tuple_s_5p0_co[3]])

    tuple_s_1p5_avg = get_plot_series_2(
        "/media/usb/UCSB/output/phi1p5/flocs", "k", r"$\phi_{1.5\%}$", "-."
    )
    tuple_s_5p0_avg = get_plot_series_2(
        "/media/usb/UCSB/output/phi5p0/flocs", "k", r"$\phi_{5\%}$ no cohesion", "-"
    )
    tuple_s_5p0_co_avg = get_plot_series_2(
        "/media/usb/UCSB/output/phi5p0_co/flocs", "k", r"$\phi_{5\%}$", "--"
    )

    plt_templ.avg_D_f(plot_dir, [tuple_s_1p5_avg[0], tuple_s_5p0_avg[0], tuple_s_5p0_co_avg[0]], inner_units=False)
    plt_templ.avg_D_g(plot_dir, [tuple_s_1p5_avg[1], tuple_s_5p0_avg[1], tuple_s_5p0_co_avg[1]], inner_units=False)
    plt_templ.mass_avg_D_f(
        plot_dir, [tuple_s_1p5_avg[2], tuple_s_5p0_avg[2], tuple_s_5p0_co_avg[2]], inner_units=False
    )
    plt_templ.mass_avg_D_g(
        plot_dir, [tuple_s_1p5_avg[3], tuple_s_5p0_avg[3], tuple_s_5p0_co_avg[3]], inner_units=False
    )


def main() -> None:
    plot_dir: Path = Path("./output/plots")
    floc(plot_dir)
