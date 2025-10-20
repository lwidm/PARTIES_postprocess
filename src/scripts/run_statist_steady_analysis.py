# -- src/scripts/run_fuild_wall_analysis.py

from typing import Optional, List, Dict, Union, Literal
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
import h5py # type: ignore
import tqdm # type: ignore

from src import globals
from src.myio import myio
from src.fluid import flow_statistics as fstat
from src import plotting


def compute_fluid_Ekin_evolution(
    parties_data_dir: Union[str, Path],
    output_dir: Union[str, Path],
    Re: float,
    min_file_index: Optional[int],
    max_file_index: Optional[int],
) -> Dict[str, np.ndarray]:

    print("Starting fluid kinetic energy computation...")
    print(
        f'Looking for datafile in directory: "{parties_data_dir}" with min_file_index: {min_file_index} and max_file_index: {max_file_index}'
    )
    data_files: List[Path] = myio.list_data_files(
        parties_data_dir, "Data", min_file_index, max_file_index
    )

    time: np.ndarray = myio.get_time_array(
        "Data", parties_data_dir, min_file_index, max_file_index
    )
    E_kin: np.ndarray = np.zeros(len(data_files))

    for i, fluid_file in enumerate(tqdm.tqdm(data_files, desc="Processing total fluid kinetic energy")):
        E_kin[i] = fstat.calc_tot_fluid_Ekin(fluid_file, Re)

    results: Dict[str, np.ndarray] = {"E_kin": E_kin, "time": time}

    myio.save_to_h5(Path(output_dir) / "E_kin.h5", results)

    return results



def main(
    parties_data_dir: Union[str, Path],
    output_dir: Union[str, Path],
    min_file_index: Optional[int] = None,
    max_file_index: Optional[int] = None,
):

    # =============================================================================
    # CONFIGURATION AND CONSTANTS
    # =============================================================================


    Re: float = 2800.0
    output_dir = Path(output_dir) / "fluid"
    plot_dir = output_dir / "plots"

    num_workers_single_component: Optional[int] = 5
    num_workers_cross_component: Optional[int] = 2

    if globals.on_anvil:
        num_workers_single_component = 8
        num_workers_cross_component = 4

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # processing_method: Literal["load", "compute"] = "load"
    processing_method: Literal["load", "compute"] = "compute"

    # =============================================================================
    # Computation and plotting
    # =============================================================================

    Ekin_results: Dict[str, np.ndarray] = {}
    if processing_method == "load":
        with h5py.File(output_dir / "E_kin.h5", "r") as f:
            Ekin_results["E_kin"] = f["E_kin"][:]  # type: ignore
            Ekin_results["time"] = f["time"][:]  # type: ignore
    elif processing_method == "compute":
        Ekin_results = compute_fluid_Ekin_evolution(
            parties_data_dir, output_dir, Re, min_file_index, max_file_index
        )

    s = plotting.series.Ekin_evolution(output_dir / "E_kin.h5", 'k', '-', 'None', 'None')

    plotting.templates.fluid_Ekin_evolution(plot_dir, [s])
