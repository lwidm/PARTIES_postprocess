from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from matplotlib import pyplot as plt

from src.myio import myio
from src.myio.myio import MyPath
from src.plotting.tools import PlotSeries
from src.plotting import series as plt_series
from src.plotting import templates as plt_templ
from src import globals

def main():
    plot_dir: Path = Path("./output/plots_tmp")
    # parties_data_dir: Path = Path("/media/usb/UCSB/data/")
    parties_data_dir: Path = Path("./data/")
    # output_dir: MyPath = Path("/media/usb/UCSB/output/")
    output_dir: MyPath = Path("./output")

    data_names: List[str] = [
        "phi1p5",
        # "phi5p0",
        "phi5p0_new",
    ]
    labels: List[str] = [
        r"$\phi_{1.5\%}$",
        # r"$\phi_{5\%}$",
        r"$\phi_{5\%}$ new",
    ]
    min_file_indices: List[Optional[int]] = [
        None,
        # None,
        None,
    ]
    max_file_indices: List[Optional[int]] = [
        None,
        # None,
        None,
    ]
    trn: List[bool] = [
        False,
        # True,
        True,
    ]
    colours: List[str] = ["C0", "C1", "C2", "C3", "C4"]

    plot_dir = Path(plot_dir)
    output_dir = Path(output_dir)

    parties_data_dirs: List[Path] = [
        parties_data_dir / data_name for data_name in data_names
    ]
    output_dirs: List[Path] = [output_dir / data_name for data_name in data_names]

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


    plot_dir.mkdir(parents=True, exist_ok=True)
    s_evo_list: List[PlotSeries] = []
    for i in range(len(output_dirs)):
        s_evo = get_series_floc_evolution(
            output_dirs[i],
            colours[i],
            labels[i],
            min_file_indices[i],
            max_file_indices[i],
        )
        s_evo_list.append(s_evo)

    for i, s_evo in enumerate(s_evo_list):
        plt_templ.floc_count_evolution(plot_dir, s_evo_list, normalised=True)
        xy: List[Tuple] = plt.ginput(n=1, show_clicks=True)
        time: float = xy[0][0]
        info: Dict[str, Any] = myio.find_idx_from_time("Particle", parties_data_dirs[i], time)
        print(f"{data_names[i]}: time: {time}, idx: {info["file_idx"]}")
        if trn[i]:
            info2: Dict[str, Any] = myio.find_idx_from_time("Particle", parties_data_dirs[i] / "trn", time)
            print(f"TRN: {data_names[i]}: time: {time}, idx: {info2["file_idx"]}")

    if not globals.on_anvil:
        plt.show()
