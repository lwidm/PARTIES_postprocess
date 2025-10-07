# -- src/scripts/run_fuild_wall_analysis.py

from typing import Optional, List, Dict, Union, Literal
from pathlib import Path
import numpy as np

from src import globals
from src.myio import myio
from src.fluid import flow_statistics as fstat
from src import theory
from src import plotting
from src.plotting.templates import (
    create_velocity_profile_wall_plot,
    create_normal_stress_wall_plot,
    create_particle_slice_plot,
)


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

    data_files: List[Path] = myio.list_parties_data_files(
        parties_data_dir, "Data", min_file_index, max_file_index
    )
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


def collect_flow_statistics(
    processing_method: Literal["load", "compute"],
    Re: float,
    output_dir: Union[str, Path],
    utexas_data_dir: Union[str, Path],
    parties_data_dir: Optional[Union[str, Path]] = None,
    min_file_index: Optional[int] = None,
    max_file_index: Optional[int] = None,
    num_workers_single_component: Optional[int] = None,
    num_workers_cross_component: Optional[int] = None,
    use_threads: bool = False,
) -> Dict[str, Union[np.ndarray, float]]:
    """
    Main data retrieval function that coordinates data processing and theory.

    Args:
        processing_method: One of "load" or "compute"

    Returns:
        The parties_results dict updated with utexas data, derived profiles,
        fit parameters and scalar metadata.
    """
    utexas_mean_data_file: str = f"{utexas_data_dir}/LM_Channel_0180_mean_prof.dat"
    utexas_fluc_data_file: str = f"{utexas_data_dir}/LM_Channel_0180_vel_fluc_prof.dat"
    (
        _,  # utexas_y_delta,
        utexas_y_plus,
        utexas_U_plus,
        _,  # utexas_velocity_gradient,
        _,  # utexas_w,
        _,  # utexas_p,
    ) = myio.load_columns_from_txt_numpy(utexas_mean_data_file)
    (
        _,
        _,
        utexas_upup_plus,
        utexas_vpvp_plus,
        utexas_wpwp_plus,
        _,  # utexas_upvp_plus,
        _,  # utexas_upwp_plus,
        _,  # utexas_vpwp_plus,
        utexas_k_plus,
    ) = myio.load_columns_from_txt_numpy(utexas_fluc_data_file)

    parties_results: Dict[str, Union[np.ndarray, float]]
    if processing_method == "compute":
        if parties_data_dir is None:
            raise ValueError(
                "Specify parties_data_dir in collect_flow_statistics() if reynolds_stresses are to be computed"
            )
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
    elif processing_method == "load":
        parties_results, _ = myio.load_from_h5(f"{output_dir}/reynolds_stresses.h5")
    else:
        raise ValueError(
            f'processing_method must be one of ["load", "compute"]. Got: {processing_method}'
        )

    parties_yc_plus: np.ndarray = parties_results["yc_plus"]  # type: ignore
    parties_U_plus: np.ndarray = parties_results["U_plus"]  # type: ignore
    u_tau: float = parties_results["u_tau"]  # type: ignore
    tau_w: float = parties_results["tau_w"]  # type: ignore

    print(f"u_tau: {u_tau}, tau_w: {tau_w}")

    # Fit law of the wall parameters to both datasets
    utexas_kappa, utexas_constant = theory.law_of_the_wall.fit_parameters(
        utexas_y_plus, utexas_U_plus
    )
    parties_kappa, parties_constant = theory.law_of_the_wall.fit_parameters(
        parties_yc_plus, parties_U_plus
    )

    (
        utexas_viscous_y_plus,
        utexas_viscous_U_plus,
        utexas_log_y_plus,
        utexas_log_U_plus,
    ) = theory.law_of_the_wall.generate_profile(
        utexas_y_plus, utexas_kappa, utexas_constant
    )

    (
        parties_viscous_yc_plus,
        parties_viscous_U_plus,
        parties_log_yc_plus,
        parties_log_U_plus,
    ) = theory.law_of_the_wall.generate_profile(
        parties_yc_plus, parties_kappa, parties_constant
    )

    print(
        f"Law of the wall parameters (utexas):  κ={utexas_kappa:.3f}, C+={utexas_constant:.3f}\n"
        f"Law of the wall parameters (PARTIES): κ={parties_kappa:.3f}, C+={parties_constant:.3f}"
    )

    new_entries: Dict[str, Union[np.ndarray, float]] = {
        # utexas reference data
        "utexas_y_plus": utexas_y_plus,
        "utexas_U_plus": utexas_U_plus,
        "utexas_viscous_y_plus": utexas_viscous_y_plus,
        "utexas_viscous_U_plus": utexas_viscous_U_plus,
        "utexas_log_y_plus": utexas_log_y_plus,
        "utexas_log_U_plus": utexas_log_U_plus,
        "utexas_upup_plus": utexas_upup_plus,
        "utexas_vpvp_plus": utexas_vpvp_plus,
        "utexas_wpwp_plus": utexas_wpwp_plus,
        "utexas_k_plus": utexas_k_plus,
        # parties (computed) data (also include profiles to keep everything together)
        "parties_viscous_yc_plus": parties_viscous_yc_plus,
        "parties_viscous_U_plus": parties_viscous_U_plus,
        "parties_log_yc_plus": parties_log_yc_plus,
        "parties_log_U_plus": parties_log_U_plus,
        # law-of-the-wall fits
        "utexas_kappa": utexas_kappa,
        "utexas_constant": utexas_constant,
        "parties_kappa": parties_kappa,
        "parties_constant": parties_constant,
    }

    parties_results.update(new_entries)

    return parties_results


def main():

    # =============================================================================
    # CONFIGURATION AND CONSTANTS
    # =============================================================================

    plotting.tools.update_rcParams()

    Re: float = 2800.0

    output_dir: Path = Path("./output/fluid")
    parties_data_dir: Path = Path("./data")
    utexas_data_dir: Path = Path("./data")

    num_workers_single_component: Optional[int] = 5
    num_workers_cross_component: Optional[int] = 2
    min_file_index: Optional[int] = None
    max_file_index: Optional[int] = None

    if globals.on_anvil:
        output_dir = Path("/home/x-lwidmer/Documents/PARTIES_postprocess/output/fluid")
        utexas_data_dir = Path("/home/x-lwidmer/Documents/PARTIES_postprocess/data")
        parties_data_dir = Path("/anvil/scratch/x-lwidmer/RUN5")
        num_workers_single_component = 8
        num_workers_cross_component = 4
        min_file_index = 280

    output_dir.mkdir(exist_ok=True)

    # processing_method: Literal["load", "compute"] = "load"
    processing_method: Literal["load", "compute"] = "compute"

    # =============================================================================
    # Computation and plotting
    # =============================================================================

    results: Dict[str, Union[np.ndarray, float]] = collect_flow_statistics(
        processing_method,
        Re,
        output_dir,
        utexas_data_dir,
        parties_data_dir,
        min_file_index,
        max_file_index,
        num_workers_single_component,
        num_workers_cross_component,
        use_threads=False,
    )

    create_velocity_profile_wall_plot(output_dir, results)

    create_normal_stress_wall_plot(output_dir, results)

    create_particle_slice_plot(
        parties_data_dir,
        output_dir,
        Re,
        float(results["Re_tau"]),
        min_file_index,
        max_file_index,
    )

    data_files: List[Path] = myio.list_parties_data_files(parties_data_dir, "Data")
    data_file: Path = data_files[-1]
    phi: float = fstat.calc_tot_vol_frac(data_file)
    print(f"Total volume fraction is {phi*100} %")
