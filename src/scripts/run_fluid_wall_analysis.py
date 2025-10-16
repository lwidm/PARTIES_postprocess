# -- src/scripts/run_fuild_wall_analysis.py

from typing import Optional, List, Dict, Union, Literal
from pathlib import Path
import numpy as np

from src import globals
from src.myio import myio
from src.fluid import flow_statistics as fstat
from src import theory
from src import plotting
from src.plotting.tools import PlotSeries


def compute_all_reynolds_stresses(
    parties_data_dir: Union[str, Path],
    output_dir: Union[str, Path],
    Re: float,
    min_file_index: Optional[int] = None,
    max_file_index: Optional[int] = None,
    num_workers_single_component: Optional[int] = None,
    num_workers_cross_component: Optional[int] = None,
    use_threads: bool = False,
    save_intermediates: bool = True,
) -> Dict[str, Union[np.ndarray, float]]:
    """
    Compute all Reynolds stresses using ADLeonelli's code

    This function processes components one at a time, saves intermediate results,
    and computes Reynolds stresses while minimizing memory usage.

    Args:
        min_file_index: Minimum file index to process
        max_file_index: Maximum file index to process
        num_workers: Number of parallel workers
        use_threads: Use threads instead of processes
        save_intermediates: Whether to save intermediate results

    Returns:
        Dictionary containing all final results including wall units
    """
    print("Starting Reynolds stress computation...")
    print(
        f'Looking for datafile in directory: "{parties_data_dir}" with min_file_index: {min_file_index} and max_file_index: {max_file_index}'
    )
    data_files: List[Path] = myio.list_parties_data_files(
        parties_data_dir, "Data", min_file_index, max_file_index
    )
    print(f"Processing {len(data_files)} for fuild analysis ...")
    if not data_files:
        return {}

    grid: Dict[str, np.ndarray] = fstat.get_grid(data_files[0])

    results: Dict[str, np.ndarray] = fstat.process_mean_flow(data_files, grid)
    results = results | fstat.process_fluctuations(data_files, results, grid)

    tau_w: float
    u_tau: float
    tau_w, u_tau = fstat.calc_friction_velocity(results, grid, Re)

    results_wall: Dict[str, Union[np.ndarray, float]] = fstat.get_wall_units(
        results, grid, Re, tau_w, u_tau
    )

    myio.save_to_h5(
        f"{output_dir}/reynolds_stresses.h5",
        results | results_wall | {"Re": Re},
        {
            "min_index": min_file_index,
            "max_index": max_file_index,
            "num_files_processed": len(data_files),
        },
    )

    return results | results_wall | {"Re": Re}


def compute_and_save_utexas(
    utexas_data_dir: Union[str, Path],
    out_h5: Union[str, Path],
) -> Dict[str, Union[np.ndarray, float]]:
    """Load UTEXAS text files, compute law-of-the-wall fits & profiles, save to HDF5 and return dict."""
    utexas_mean_data_file = f"{utexas_data_dir}/LM_Channel_0180_mean_prof.dat"
    utexas_fluc_data_file = f"{utexas_data_dir}/LM_Channel_0180_vel_fluc_prof.dat"

    (
        _,  # utexas_y_delta
        utexas_y_plus,
        utexas_U_plus,
        _,  # utexas_velocity_gradient
        _,  # utexas_w
        _,  # utexas_p
    ) = myio.load_columns_from_txt_numpy(utexas_mean_data_file)

    (
        _,
        _,
        utexas_upup_plus,
        utexas_vpvp_plus,
        utexas_wpwp_plus,
        _,  # utexas_upvp_plus
        _,  # utexas_upwp_plus
        _,  # utexas_vpwp_plus
        utexas_k_plus,
    ) = myio.load_columns_from_txt_numpy(utexas_fluc_data_file)

    # law-of-the-wall fit + profile
    utexas_kappa, utexas_constant = theory.law_of_the_wall.fit_parameters(
        utexas_y_plus, utexas_U_plus
    )
    (
        utexas_viscous_y_plus,
        utexas_viscous_U_plus,
        utexas_log_y_plus,
        utexas_log_U_plus,
    ) = theory.law_of_the_wall.generate_profile(
        utexas_y_plus, utexas_kappa, utexas_constant
    )

    utexas_stats: Dict[str, Union[np.ndarray, float]] = {
        "utexas_y_plus": utexas_y_plus,
        "utexas_U_plus": utexas_U_plus,
        "utexas_upup_plus": utexas_upup_plus,
        "utexas_vpvp_plus": utexas_vpvp_plus,
        "utexas_wpwp_plus": utexas_wpwp_plus,
        "utexas_k_plus": utexas_k_plus,
        "utexas_kappa": utexas_kappa,
        "utexas_constant": utexas_constant,
        "utexas_viscous_y_plus": utexas_viscous_y_plus,
        "utexas_viscous_U_plus": utexas_viscous_U_plus,
        "utexas_log_y_plus": utexas_log_y_plus,
        "utexas_log_U_plus": utexas_log_U_plus,
    }

    # save
    myio.save_to_h5(out_h5, utexas_stats, {"source": "utexas_text_files"})
    return utexas_stats


def compute_and_save_parties(
    parties_data_dir: Union[str, Path],
    output_dir: Union[str, Path],
    output_h5_filename: str,
    Re: float,
    min_file_index: Optional[int] = None,
    max_file_index: Optional[int] = None,
    num_workers_single_component: Optional[int] = None,
    num_workers_cross_component: Optional[int] = None,
    use_threads: bool = False,
) -> Dict[str, Union[np.ndarray, float]]:
    parties_results = compute_all_reynolds_stresses(
        parties_data_dir,
        output_dir,
        Re,
        min_file_index,
        max_file_index,
        num_workers_single_component,
        num_workers_cross_component,
        use_threads,
    )

    # compute law-of-the-wall fits & profiles for PARTIES
    parties_yc_plus: np.ndarray = parties_results["yc_plus"]  # type: ignore
    parties_U_plus: np.ndarray = parties_results["U_plus"]  # type: ignore
    parties_kappa, parties_constant = theory.law_of_the_wall.fit_parameters(
        parties_yc_plus, parties_U_plus
    )
    (
        parties_viscous_yc_plus,
        parties_viscous_U_plus,
        parties_log_yc_plus,
        parties_log_U_plus,
    ) = theory.law_of_the_wall.generate_profile(
        parties_yc_plus, parties_kappa, parties_constant
    )

    parties_results.update(
        {
            "parties_kappa": parties_kappa,
            "parties_constant": parties_constant,
            "parties_viscous_yc_plus": parties_viscous_yc_plus,
            "parties_viscous_U_plus": parties_viscous_U_plus,
            "parties_log_yc_plus": parties_log_yc_plus,
            "parties_log_U_plus": parties_log_U_plus,
        }
    )

    # Save final parties dict (metadata: indices if available)
    meta = {
        "min_index": min_file_index,
        "max_index": max_file_index,
    }
    myio.save_to_h5(Path(output_dir) / output_h5_filename, parties_results, meta)
    return parties_results


def main(
    parties_data_dir: Union[str, Path],
    utexas_data_dir: Union[str, Path],
    output_dir: Union[str, Path],
    min_file_index: Optional[int] = None,
    max_file_index: Optional[int] = None,
):

    # =============================================================================
    # CONFIGURATION AND CONSTANTS
    # =============================================================================

    Re: float = 2800.0

    output_dir = Path(output_dir) / "fluid"
    output_dir.mkdir(parents=True, exist_ok=True)

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

    plot_dir = output_dir / "plots"
    utexas_h5 = output_dir / "utexas.h5"
    results_utexas = compute_and_save_utexas(utexas_data_dir, utexas_h5)
    parties_processed_filename = "parties_reynolds.h5"
    results_parties = compute_and_save_parties(
        parties_data_dir,
        output_dir,
        parties_processed_filename,
        Re,
        min_file_index,
        max_file_index,
    )

    plot_dir: Path = Path(output_dir) / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    utexas_wall_series: List[PlotSeries] = plotting.series.u_plus_mean_wall_utexas(
        utexas_h5
    )
    parties_wall_series: List[PlotSeries] = plotting.series.u_plus_mean_wall_parties(
        output_dir / parties_processed_filename, label="PARTIES", colour="C1"
    )
    all_wall_series: List[PlotSeries] = utexas_wall_series + parties_wall_series
    plotting.templates.velocity_profile_wall(plot_dir, all_wall_series)

    utexas_stress_series: List[PlotSeries] = plotting.series.normal_stress_wall_utexas(
        utexas_h5
    )
    parties_stress_series: List[PlotSeries] = (
        plotting.series.normal_stress_wall_parties(
            output_dir / parties_processed_filename, label="PARTIES", colour="C1"
        )
    )
    all_stress_series: List[PlotSeries] = utexas_stress_series + parties_stress_series
    plotting.templates.normal_stress_wall(plot_dir, all_stress_series)

    data_files: List[Path] = myio.list_parties_data_files(parties_data_dir, "Data")
    data_file: Path = data_files[-1]
    phi: float = fstat.calc_tot_vol_frac(data_file)
    print(f"Total volume fraction is {phi*100} %")
