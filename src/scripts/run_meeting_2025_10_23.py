from typing import Optional, Tuple, List
from pathlib import Path

from src import scripts
from src.myio import myio
from src.myio.myio import MyPath
from src.plotting.tools import PlotSeries
from src.plotting import series as plt_series
from src.plotting import templates as plt_templ


def main() -> None:

    time_idx_info: List[dict] = [{}, {}, {}]
    time_idx_info[0] = myio.find_idx_from_time(
        file_prefix="Particle",
        data_dir="/media/usb/UCSB/phi1p5",
        target_time=200.0,
    )
    time_idx_info[1] = myio.find_idx_from_time(
        file_prefix="Particle",
        data_dir="/media/usb/UCSB/phi5p0",
        target_time=30.0,
    )
    time_idx_info[2] = myio.find_idx_from_time(
        file_prefix="Particle",
        data_dir="/media/usb/UCSB/phi5p0_co",
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
            parties_data_dir="/media/usb/UCSB/phi1p5",
            output_dir="./output/phi1p5",
            trn=False,
            min_file_index=min_file_indices[0],
            max_file_index=max_file_indices[0],
            min_steady_index=min_steady_indices[0],
            max_steady_index=max_steady_indices[0],
            num_workers=6,
        )

        scripts.run_floc_analysis.main(
            parties_data_dir="/media/usb/UCSB/phi5p0",
            output_dir="./output/phi5p0",
            trn=False,
            min_file_index=min_file_indices[1],
            max_file_index=max_file_indices[1],
            min_steady_index=min_steady_indices[1],
            max_steady_index=max_steady_indices[1],
            num_workers=6,
        )
        scripts.run_floc_analysis.main(
            parties_data_dir="/media/usb/UCSB/phi5p0_co",
            output_dir="./output/phi5p0_co",
            trn=True,
            min_file_index=min_file_indices[2],
            max_file_index=max_file_indices[2],
            min_steady_index=min_steady_indices[2],
            max_steady_index=max_steady_indices[2],
            num_workers=6,
        )

    def get_plot_series(
        output_dir: MyPath,
        colour: str,
        label: str,
        marker: str,
        min_file_index: Optional[int],
        max_file_index: Optional[int],
    ) -> Tuple[PlotSeries, PlotSeries, PlotSeries, PlotSeries]:
        s: PlotSeries = plt_series.floc_count_evolution(
            output_dir,
            colour,
            label,
            min_file_index,
            max_file_index,
            normalised=True,
            reset_time=True,
        )
        s_n_p, s_D_f, s_D_g = plt_series.floc_pdf(
            floc_dir=output_dir,
            labels=[label, label, label],
            colours=[colour, colour, colour],
            markers=[marker, marker, marker],
            linestyles=["None", "None", "None"],
        )
        return s, s_n_p, s_D_f, s_D_g

    plot_dir: Path = Path("./output/plots")
    plot_dir.mkdir(parents=True, exist_ok=True)
    tuple_s_1p5 = get_plot_series(
        "./output/phi1p5/flocs",
        "C0",
        r"$\phi_{1.5\%}$",
        "o",
        min_file_indices[0],
        max_file_indices[0],
    )
    tuple_s_5p0 = get_plot_series(
        "./output/phi5p0/flocs",
        "C1",
        r"$\phi_{5\%}$ no cohesion",
        "s",
        min_file_indices[1],
        max_file_indices[1],
    )
    tuple_s_5p0_co = get_plot_series(
        "./output/phi5p0_co/flocs",
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
