# -- src/scripts/run_floc_analysis.py

import numpy as np
from typing import Dict, Union, Optional, Tuple, List
from pathlib import Path
import tqdm  # type: ignore
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

from src.myio import myio
from src.flocs.find_flocs import find_flocs
from src.flocs import floc_statistics as floc_stat
from src import plotting


def analyze_floc(
    particle_data: Dict[str, Union[np.ndarray, float]],
    domain: Dict[str, Union[int, float]],
) -> Tuple[Dict[str, Union[np.ndarray, float]], Dict[str, Union[int, float]]]:
    N_particles: int = len(np.asarray(particle_data["r"]))
    particle_diameter: float = 2 * np.asarray(particle_data["r"])[0]

    X_p: np.ndarray = np.column_stack(
        (particle_data["x"], particle_data["y"], particle_data["z"])
    )
    U_p: np.ndarray = np.column_stack(
        (particle_data["u"], particle_data["v"], particle_data["w"])
    )

    shifts, X_com = floc_stat.calc_CoM(X_p, np.asarray(particle_data["r"]), domain)
    U_com: np.ndarray = floc_stat.calc_velocity(U_p)

    feret_diam: float = floc_stat.calc_feret_diam(particle_diameter, X_p, X_com, shifts)
    gyration_diam: float = floc_stat.calc_gyration_diam(
        particle_diameter, X_p, X_com, shifts
    )
    fractal_dim: float = floc_stat.calc_fractal_dim(
        particle_diameter, feret_diam, N_particles
    )
    orientation: np.ndarray = floc_stat.calc_orientation(X_p, X_com, shifts)
    theta: np.ndarray = floc_stat.calc_theta(orientation, 3)
    pitch: float = floc_stat.calc_pitch(orientation, N_particles)
    # rotational_vel: np.ndarray = floc_stat.calc_rotational_velocity(x, u, shifts, floc_com, n_par)

    floc_results = {
        "floc_id": np.asarray(particle_data["floc_id"])[0],
        "n_p": N_particles,
        "x": X_com[0],
        "y": X_com[1],
        "z": X_com[2],
        "u": U_com[0],
        "v": U_com[1],
        "w": U_com[2],
        "D_f": feret_diam,
        "D_g": gyration_diam,
        "n_f": fractal_dim,
        "l_x": orientation[0],
        "l_y": orientation[1],
        "l_z": orientation[2],
        # "omega_x": rotational_vel[0],
        # "omega_y": rotational_vel[1],
        # "omega_z": rotational_vel[2],
        "theta_x": theta[0],
        "theta_y": theta[1],
        "theta_z": theta[2],
        "pitch": pitch,
        "time": float(particle_data["time"]),
    }
    particle_data.update(
        {"x_shift": shifts[:, 0], "y_shift": shifts[:, 1], "z_shift": shifts[:, 2]}
    )

    return particle_data, floc_results


def compute_floc_stats(
    particle_data_all: Dict[str, Union[np.ndarray, float]],
    domain: Dict[str, Union[int, float]],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], float]:
    """Compute statistics for all flocs."""
    particle_results: List[Dict[str, Union[np.ndarray, float]]] = []
    floc_results: List[Dict[str, Union[int, float]]] = []

    floc_id: int
    for floc_id in np.unique(particle_data_all["floc_id"]):
        single_floc: Dict[str, Union[np.ndarray, float]] = extract_floc(
            particle_data_all, floc_id
        )
        particle_data: Dict[str, Union[np.ndarray, float]]
        floc_stats: Dict[str, Union[int, float]]
        particle_data, floc_stats = analyze_floc(single_floc, domain)
        particle_results.append(particle_data)
        floc_results.append(floc_stats)
    combined_particle_results: Dict[str, np.ndarray] = myio.combine_dicts(
        particle_results
    )
    combined_floc_results: Dict[str, np.ndarray] = myio.combine_dicts(floc_results)
    time: float = float(particle_data_all["time"])

    # par_results = {key: np.array(value) for key, value in par_results.items()}
    # sorted_indices = np.argsort(par_results['id'])
    # par_results = {key: value[sorted_indices] for key, value in par_results.items()}

    return combined_particle_results, combined_floc_results, time


def extract_floc(
    particle_data: Dict[str, Union[np.ndarray, float]], floc_id: int
) -> Dict[str, Union[np.ndarray, float]]:
    mask: np.ndarray = np.asarray(particle_data["floc_id"]) == floc_id
    result = {}

    for key, value in particle_data.items():
        if key == "time":
            # Keep time as scalar
            result[key] = float(value)
        else:
            # Assume everything else is an array that needs masking
            result[key] = np.asarray(value)[mask]

    return result


def process_flocs(
    in_file: Union[str, Path],
    out_file: Union[str, Path],
    domain: Dict[str, Union[int, float]],
    coh_range: float,
) -> Dict[str, np.ndarray]:

    particle_data: Dict[str, Union[np.ndarray, float]] = myio.read_particle_data(
        in_file
    )
    flocs: Dict[str, Union[np.ndarray, float]] = find_flocs(
        particle_data, coh_range, domain
    )

    particle_results: Dict[str, np.ndarray]
    floc_results: Dict[str, np.ndarray]
    time: float
    particle_results, floc_results, time = compute_floc_stats(flocs, domain)

    # compute_floc_stats_obj(flocs, domain)

    # TODO :
    # if prev_particle_results is not None:
    #     reindex_flocs(par_results, prev_particle_results, floc_results)
    myio.save_to_h5(
        out_file, {"particles": particle_results, "flocs": floc_results, "time": time}
    )

    return particle_results


def process_all_flocs(
    parties_data_dir: Union[str, Path],
    output_dir: Union[str, Path],
    min_file_index: Optional[int],
    max_file_index: Optional[int],
    num_workers: Optional[int],
    use_threading: bool,
):
    output_dir = Path(output_dir)

    print(f"Looking for particle files in {parties_data_dir} ...")
    particle_files: List[Path] = myio.list_data_files(
        parties_data_dir, "Particle", min_file_index, max_file_index
    )

    def floc_filename(fp: Path) -> str:
        return "Flocs_" + fp.stem.split("_")[-1] + ".h5"

    out_files: List[Path] = [output_dir / floc_filename(fp) for fp in particle_files]

    init_file_path: Path = particle_files[0]
    domain: Dict[str, Union[int, float]] = myio.read_domain_info(init_file_path)
    coh_range: float = myio.read_coh_range(parties_data_dir, init_file_path)

    if num_workers is not None:
        if use_threading:
            executor = ThreadPoolExecutor
        else:
            executor = ProcessPoolExecutor
        with executor(max_workers=num_workers) as ex:
            futures = {
                ex.submit(process_flocs, in_f, out_f, domain, coh_range): (
                    in_f,
                    out_f,
                )
                for in_f, out_f in zip(particle_files, out_files)
            }

            for _ in tqdm.tqdm(
                as_completed(futures), total=len(futures), desc="Processing flocs"
            ):
                pass
    else:
        prev_results = None
        for in_file, out_file in tqdm.tqdm(
            zip(particle_files, out_files), total=len(particle_files)
        ):
            prev_results = process_flocs(in_file, out_file, domain, coh_range)


def main(
    parties_data_dir: Union[str, Path],
    output_dir: Union[str, Path],
    trn: bool,
    u_tau: float,
    min_file_index: Optional[int] = None,
    max_file_index: Optional[int] = None,
    min_steady_index: Optional[int] = None,
    max_steady_index: Optional[int] = None,
    num_workers: Optional[int] = None,
    use_threading: bool = False,
):
    # =============================================================================
    # CONFIGURATION AND CONSTANTS
    # =============================================================================

    output_dir = Path(output_dir) / "flocs"

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    plotting.tools.update_plot_params()

    # =============================================================================
    # Computation and plotting
    # =============================================================================

    Re: float = myio.read_Re(parties_data_dir)
    L: float = myio.read_channel_half_height(parties_data_dir)

    if trn:
        parties_data_dir = Path(parties_data_dir) / "trn"

    process_all_flocs(
        parties_data_dir,
        output_dir,
        min_file_index,
        max_file_index,
        num_workers,
        use_threading,
    )

    floc_stat.calc_PDF(
        output_dir=output_dir,
        bin_widths=(1, 1, 1),
        floc_dir=output_dir,
        min_file_index=min_steady_index,
        max_file_index=max_steady_index,
        num_workers=num_workers,
        use_threading=False,
    )

    floc_stat.CalcAvgDiam(
        output_dir=output_dir,
        floc_dir=output_dir,
        channel_half_width=L,
        Re=Re,
        u_tau=u_tau,
        n_bins=100,
        n_bins_inner=100,
        min_file_index=min_steady_index,
        max_file_index=max_steady_index,
        num_workers=num_workers,
        use_threading=False,
    )

    # TODO :
    # family_tree = fam_tree.FamilyTree(floc_dir)
    # family_tree.build()
    #
    # tree_file = analysis_dir / "family_tree.pkl"
    # myio.save_to_pickle(tree_file, family_tree.family_tree)
