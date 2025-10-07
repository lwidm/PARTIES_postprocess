# -- scripts/run_floc_analysis.py

import h5py  # type: ignore
import numpy as np
from typing import Dict, Union, Optional, Tuple, List
from pathlib import Path
import sys
import tqdm  # type: ignore
from matplotlib import pyplot as plt

from myio import myio
from flocs.find_flocs import find_flocs
from flocs import floc_statistics


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

    shifts, X_com = floc_statistics.calc_CoM(X_p, particle_data["r"], domain)
    # U_com = floc_statistics.calc_velocity(U_p)

    # feret_diam = floc_statistics.calc_feret_diam(particle_diameter, X_p, X_com, shifts)
    # gyration_diam = floc_statistics.calc_gyration_diam(particle_diameter, X_p, X_com, shifts)
    # fractal_dim = floc_statistics.calc_fractal_dim(particle_diameter, feret_diam, N_particles)
    # orientation = floc_statistics.calc_orientation(X_p, X_com, shifts)
    # theta = floc_statistics.calc_theta(orientation, 3)
    # pitch = floc_statistics.calc_pitch(orientation, N_particles)
    # rotational_vel = floc_statistics.calc_rotational_velocity(x, u, shifts, floc_com, n_par)

    floc_results = {
        "floc_id": particle_data["floc_id"][0],
        "n_p": N_particles,
        "x": X_com[0],
        "y": X_com[1],
        "z": X_com[2],
        # "u": U_com[0],
        # "v": U_com[1],
        # "w": U_com[2],
        # "D_f": feret_diam,
        # "D_g": gyration_diam,
        # "n_f": fractal_dim,
        # "l_x": orientation[0],
        # "l_y": orientation[1],
        # "l_z": orientation[2],
        # "omega_x": rotational_vel[0],
        # "omega_y": rotational_vel[1],
        # "omega_z": rotational_vel[2],
        # "theta_x": theta[0],
        # "theta_y": theta[1],
        # "theta_z": theta[2],
        # "pitch": pitch,
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


def simple_floc_plot(floc_dir: Path) -> None:
    """Minimal floc count plot."""
    floc_files: List[Path] = myio.list_parties_data_files(floc_dir, "Flocs")
    timesteps: List[int] = []
    counts: List[int] = []

    for floc_file in floc_files:
        with h5py.File(floc_file, "r") as f:
            if "flocs" in f and "floc_id" in f["flocs"]:  # type: ignore
                timestep = int(floc_file.stem.split("_")[-1])
                floc_count = len(np.unique(f["flocs"]["floc_id"][:]))  # type: ignore
                timesteps.append(timestep)
                counts.append(floc_count)

    plt.figure(figsize=(8, 5))
    plt.plot(timesteps, counts, "bo-")
    plt.xlabel("Timestep")
    plt.ylabel("Number of Flocs")
    plt.title("Floc Count Over Time")
    plt.grid(True)
    plt.savefig(floc_dir / "simple_floc_count.png", dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


def main():
    data_dir: Path = Path(sys.argv[1])

    analysis_dir: Path = data_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)
    floc_dir: Path = analysis_dir / "flocs"
    floc_dir.mkdir(exist_ok=True)

    particle_files: List[Path] = myio.list_parties_data_files(data_dir, "Particle")

    def floc_filename(fp: Path) -> str:
        return "Flocs_" + fp.stem.split("_")[-1] + ".h5"

    out_files: List[Path] = [floc_dir / floc_filename(fp) for fp in particle_files]

    init_file_path: Path = particle_files[0]
    domain: Dict[str, Union[int, float]] = myio.read_domain_info(init_file_path)
    coh_range: float = myio.read_coh_range(data_dir, init_file_path)

    prev_results = None
    for in_file, out_file in tqdm.tqdm(
        zip(particle_files, out_files), total=len(particle_files)
    ):
        prev_results = process_flocs(in_file, out_file, domain, coh_range, prev_results)

    # TODO :
    # family_tree = fam_tree.FamilyTree(floc_dir)
    # family_tree.build()
    #
    # tree_file = analysis_dir / "family_tree.pkl"
    # myio.save_to_pickle(tree_file, family_tree.family_tree)

    simple_floc_plot(floc_dir)


if __name__ == "__main__":
    main()
