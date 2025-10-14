# -- src/scripts/run_floc_analysis.py

import h5py  # type: ignore
import numpy as np
from typing import Dict, Union, Optional, Tuple, List, Literal
from pathlib import Path
import tqdm  # type: ignore
from matplotlib import pyplot as plt

from src.myio import myio
from src.flocs.find_flocs import find_flocs
from src.flocs import floc_statistics as floc_stat
from src import globals
from src import plotting


def analyze_floc(
    particle_data: Dict[str, np.ndarray], domain: Dict[str, Union[int, float]]
) -> Tuple[Dict[str, np.ndarray], Dict[str, Union[int, float]]]:
    """Analyze a single floc and compute its statistics.
    Args:
        particle_data: Dict containing particle data for a single floc.
        domain: Dict containing domain size and periodicity information.
    Returns:
        Tuple containing updated particle data and a dict with floc statistics.
    """
    N_particles: int = len(particle_data["r"])
    particle_diameter: float = 2 * particle_data["r"][0]

    X_p: np.ndarray = np.column_stack(
        (particle_data["x"], particle_data["y"], particle_data["z"])
    )
    U_p: np.ndarray = np.column_stack(
        (particle_data["u"], particle_data["v"], particle_data["w"])
    )

    shifts, X_com = floc_stat.calc_CoM(X_p, particle_data["r"], domain)
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
        "floc_id": particle_data["floc_id"][0],
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
    }
    particle_data.update(
        {"x_shift": shifts[:, 0], "y_shift": shifts[:, 1], "z_shift": shifts[:, 2]}
    )

    return particle_data, floc_results


def compute_floc_stats(
    particle_data_all: Dict[str, np.ndarray], domain: Dict[str, Union[int, float]]
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Compute statistics for all flocs."""
    particle_results: List[Dict[str, np.ndarray]] = []
    floc_results: List[Dict[str, Union[int, float]]] = []

    floc_id: int
    for floc_id in np.unique(particle_data_all["floc_id"]):
        single_floc: Dict[str, np.ndarray] = extract_floc(particle_data_all, floc_id)
        particle_data: Dict[str, np.ndarray]
        floc_stats: Dict[str, Union[int, float]]
        particle_data, floc_stats = analyze_floc(single_floc, domain)
        particle_results.append(particle_data)
        floc_results.append(floc_stats)
    combined_particle_results: Dict[str, np.ndarray] = myio.combine_dicts(
        particle_results
    )
    combined_floc_results: Dict[str, np.ndarray] = myio.combine_dicts(
        floc_results, scalar=True
    )

    # par_results = {key: np.array(value) for key, value in par_results.items()}
    # sorted_indices = np.argsort(par_results['id'])
    # par_results = {key: value[sorted_indices] for key, value in par_results.items()}

    return combined_particle_results, combined_floc_results


def extract_floc(
    particle_data: Dict[str, np.ndarray], floc_id: int
) -> Dict[str, np.ndarray]:
    """Extract particles belonging to a specific floc."""
    mask = particle_data["floc_id"] == floc_id
    return {key: value[mask] for key, value in particle_data.items()}


def process_flocs(
    in_file: Union[str, Path],
    out_file: Union[str, Path],
    domain: Dict[str, Union[int, float]],
    coh_range: float,
    prev_particle_results: Optional[Dict[str, np.ndarray]],
) -> Dict[str, np.ndarray]:
    """Perform floc analysis on for a single Particle_xx.h5 file."""

    particle_data: Dict[str, np.ndarray] = myio.read_particle_data(in_file)
    flocs: Dict[str, np.ndarray] = find_flocs(particle_data, coh_range, domain)

    particle_results: Dict[str, np.ndarray]
    floc_results: Dict[str, np.ndarray]
    particle_results, floc_results = compute_floc_stats(flocs, domain)

    # compute_floc_stats_obj(flocs, domain)

    # if prev_particle_results is not None:
    #     reindex_flocs(par_results, prev_particle_results, floc_results)
    myio.save_to_h5(out_file, {"particles": particle_results, "flocs": floc_results})

    return particle_results


def plot_floc_count_evolution(floc_dir: Path) -> None:
    """Minimal floc count plot."""
    floc_files: List[Path] = myio.list_parties_data_files(floc_dir, "Flocs")
    # timesteps: List[int] = []
    counts: List[int] = []

    for floc_file in floc_files:
        with h5py.File(floc_file, "r") as f:
            if "flocs" in f and "floc_id" in f["flocs"]:  # type: ignore
                # timestep = int(floc_file.stem.split("_")[-1])
                floc_count = len(np.unique(f["flocs"]["floc_id"][:]))  # type: ignore
                # timesteps.append(timestep)
                counts.append(floc_count)

    time: np.ndarray = np.loadtxt(f"{floc_dir}/time.csv", delimiter=",")

    figure, axes = plt.subplots(figsize=(6.5, 5.5))
    # axes.plot(timesteps, counts, "bx-")
    axes.plot(time, counts, "k-")
    axes.set_xlabel("time [-]")
    axes.set_ylabel(r"\# Flocs")
    axes.grid(True)

    axes.legend(loc="lower right", bbox_to_anchor=(1.0, 0.80))
    axes = plotting.tools.format_plot_axes(axes)

    figure.savefig(floc_dir / "floc_count_evolution.png", dpi=150, bbox_inches="tight")
    if not globals.on_anvil:
        plt.show()
    plt.close()


def main(parties_data_dir: Union[str, Path], output_dir: Union[str, Path]):
    # =============================================================================
    # CONFIGURATION AND CONSTANTS
    # =============================================================================

    output_dir = Path(output_dir) / "flocs"

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    plotting.tools.update_rcParams()

    Re: float = 2800.0

    num_workers_single_component: Optional[int] = 5
    num_workers_cross_component: Optional[int] = 2

    if globals.on_anvil:
        num_workers_single_component = 8
        num_workers_cross_component = 4
        min_file_index = 250

    # processing_method: Literal["load", "compute"] = "load"
    processing_method: Literal["load", "compute"] = "compute"

    # =============================================================================
    # Computation and plotting
    # =============================================================================

    if processing_method == "compute":
        particle_files: List[Path] = myio.list_parties_data_files(
            parties_data_dir, "Particle"
        )

        def floc_filename(fp: Path) -> str:
            return "Flocs_" + fp.stem.split("_")[-1] + ".h5"

        out_files: List[Path] = [
            output_dir / floc_filename(fp) for fp in particle_files
        ]

        init_file_path: Path = particle_files[0]
        domain: Dict[str, Union[int, float]] = myio.read_domain_info(init_file_path)
        coh_range: float = myio.read_coh_range(parties_data_dir, init_file_path)

        prev_results = None
        for in_file, out_file in tqdm.tqdm(
            zip(particle_files, out_files), total=len(particle_files)
        ):
            prev_results = process_flocs(
                in_file, out_file, domain, coh_range, prev_results
            )


        time: np.ndarray = myio.get_time_array("Particle", parties_data_dir, None, None)
        np.savetxt(f"{output_dir}/time.csv", time, delimiter=",")

        # TODO :
        # family_tree = fam_tree.FamilyTree(floc_dir)
        # family_tree.build()
        #
        # tree_file = analysis_dir / "family_tree.pkl"
        # myio.save_to_pickle(tree_file, family_tree.family_tree)
    elif processing_method == "load":
        pass

    plot_floc_count_evolution(output_dir)

