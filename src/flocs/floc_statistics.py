# -- src/flocs/floc_statistics.py

from pathlib import Path
import h5py
import numpy as np
from typing import Dict, Union, Tuple, Optional, List
from tqdm import tqdm


from src.myio import myio


def calc_CoM(
    X_p: np.ndarray, r: np.ndarray, domain: Dict[str, Union[int, float]]
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate the center of mass of a system of particles.

    Args:
        X_p: Particle coordinates.
        r: Particle radii.
        domain: Information on domain size and periodicity in each direction.

    Returns:
        Particle data with six aditional keys, `x_CoM`, `y_CoM`, `z_CoM`, denoting
        the center of mass location in each direction for that aggregat and `x_shift`,
        `y_shift`, `z_shift`, indicating if the particle should be moved to the other
        side of the domain in that direction in order to form a "complete" floc across
        the periodic boundaries.
    """

    N_particles: int = len(r)
    X_CoM: np.ndarray = np.zeros(3)
    shifts: np.ndarray = np.zeros((N_particles, 3))
    i: int
    dir: str
    for i, dir in enumerate(["x", "y", "z"]):
        CoM: np.floating
        shift: np.ndarray
        if domain[f"{dir}_periodic"]:
            CoM, shift = _calc_CoM_1d_periodic(X_p[:, i], r, domain[f"L{dir}"])
        else:
            CoM = _calc_CoM_1d(X_p[:, i], r)
            shift = np.zeros_like(X_p[:, i])
        X_CoM[i] = CoM
        shifts[:, i] = shift
    return shifts, X_CoM


def _calc_CoM_1d_periodic(
    x_p: np.ndarray, r: np.ndarray, L: float
) -> Tuple[np.floating, np.ndarray]:
    """Calculate the center of mass for a periodic, one dimensional system of particles.

    Args:
        x_p: Particle positions along a single dimension.
        r: Particle radii.
        L: Domain length in the relevant direction.

    Returns:
        The center of mass of the 1D system and the "shift" for every particle in the
        aggregate. The shift indicates if the particle should be moved to the other
        side of the domain in order to form a "complete" floc across the periodic
        boundaries.
    """

    shift: np.ndarray = np.zeros_like(x_p)
    L_half: float = L / 2

    in_left: np.ndarray = x_p < L_half
    in_right: np.ndarray = x_p > L_half

    x_left: np.ndarray = x_p[in_left]
    r_left: np.ndarray = r[in_right]
    x_right: np.ndarray = x_p[in_right]
    r_right: np.ndarray = r[in_right]

    if len(x_left) == 0:
        CoM: np.floating = _calc_CoM_1d(x_right, r_right)
        return CoM, shift
    if len(x_right) == 0:
        CoM: np.floating = _calc_CoM_1d(x_left, r_left)
        return CoM, shift

    m_left: float = calc_mass(r_left)
    m_right: float = calc_mass(r_right)
    com_left: np.floating = _calc_CoM_1d(x_left, r_left)
    com_right: np.floating = _calc_CoM_1d(x_right, r_right)

    if com_right - com_left > L_half:
        if m_left * com_left > m_right * (L - com_right):
            com_right -= L
            shift[in_right] = -L
        else:
            com_left += L
            shift[in_left] = L
    com = (com_left * m_left + com_right * m_right) / (m_left + m_right)
    return com, shift


def _calc_CoM_1d(x_p: np.ndarray, r: np.ndarray) -> np.floating:
    """Calculate the center of mass for a 1D system of particles.

    Note:
        Need to add an additional vector, rho, for dealing with particles of
        multiple densities.
    """
    # return np.average(x, weights=r**3)
    return np.mean(x_p)


def calc_mass(r: np.ndarray) -> float:
    """Calculate the total mass of a system of particles with radii `r`.

    Note:
        Need to add an additional vector, rho, for dealing with particles of
        multiple densities.
    """
    return 4 / 3 * np.pi * np.sum(r**3)


def calc_velocity(U: np.ndarray) -> np.ndarray:
    """Calculate the floc velocity."""
    return U.mean(axis=0)


def calc_feret_diam(
    particle_diam: float,
    X_p: np.ndarray,
    X_CoM: np.ndarray,
    shift: np.ndarray,
) -> float:
    """Calculate the floc Feret diameter."""
    dist: float = np.linalg.norm(X_p - X_CoM + shift, axis=1)
    return 2 * np.max(dist) + particle_diam


def calc_gyration_diam(
    particle_diam: float,
    X_p: np.ndarray,
    X_CoM: np.ndarray,
    shift: np.ndarray,
) -> float:
    """Calculates the floc diameter of gyration."""
    N_particles: int = len(X_p)
    if N_particles == 1:
        return particle_diam
    if N_particles == 2:
        return np.sqrt(1.6) * particle_diam
    dist: np.ndarray = X_p - X_CoM + shift
    return 2 * np.sqrt((dist * dist).sum() / N_particles)


# BUG: Does not handle polydispersity
def calc_fractal_dim(
    particle_diam: float, feret_diam: float, N_particles: int
) -> float:
    """Calculate the floc fractal dimension."""
    if N_particles < 2:
        return 1
    return np.log(N_particles) / np.log(feret_diam / particle_diam)


def calc_orientation(
    X_p: np.ndarray, X_CoM: np.ndarray, shift: np.ndarray
) -> np.ndarray:
    """Calculates the floc orientation

    The orientation is the vector from the center of mass to the furthest particle.
    """
    dist: np.ndarray = X_p - X_CoM + shift
    dist2: np.ndarray = (dist * dist).sum(axis=1)
    return dist[np.argmax(dist2)]


def calc_pitch(orientation: np.ndarray, N_partices: int) -> float:
    """Calculate the floc pitch.

    The pitch is the angle the floc makes with the x-axis in the y-direction.
    """
    if N_partices < 2:
        return 0
    return np.arctan(orientation[1] / orientation[0])


def calc_theta(orientation: np.ndarray, N_particles: int) -> np.ndarray:
    """Calculate the floc angle.

    The angle is the angle the floc makes with the x-axis in the z-direction.
    """
    if N_particles < 2:
        return np.zeros(3)
    return np.arccos(orientation / np.linalg.norm(orientation))


def _process_single_pdf(
    edges_list: List[np.ndarray],
    n_p_arr: np.ndarray,
    D_f_arr: np.ndarray,
    D_g_arr: np.ndarray,
    N_flocs: int,
    tot_mass_local: float,
) -> Tuple[
    List[np.ndarray],
    List[np.ndarray],
    List[np.ndarray],
    List[np.ndarray],
    List[np.ndarray],
    List[np.ndarray],
    List[np.ndarray],
]:

    means_list: List[np.ndarray] = []
    medians_list: List[np.ndarray] = []
    counts_list: List[np.ndarray] = []
    probabs_list: List[np.ndarray] = []

    mass_means_list: List[np.ndarray] = []
    masses_list: List[np.ndarray] = []
    mass_probabs_list: List[np.ndarray] = []

    for i, vals in enumerate([n_p_arr, D_f_arr, D_g_arr]):
        bin_indices: np.ndarray = np.digitize(vals, edges_list[i])

        means: np.ndarray = np.full_like(edges_list[i][:-1], np.nan, dtype=float)
        medians: np.ndarray = np.full_like(edges_list[i][:-1], np.nan, dtype=float)
        counts: np.ndarray = np.zeros_like(edges_list[i][:-1], dtype=float)
        probabs: np.ndarray = np.zeros_like(edges_list[i][:-1], dtype=float)

        mass_means: np.ndarray = np.full_like(edges_list[i][:-1], np.nan, dtype=float)
        masses: np.ndarray = np.zeros_like(edges_list[i][:-1], dtype=float)
        mass_probabs: np.ndarray = np.zeros_like(edges_list[i][:-1], dtype=float)

        for j in np.arange(1, len(edges_list[i])):
            array_index = j - 1

            mask: np.ndarray = bin_indices == j
            binned_vals: np.ndarray = vals[mask]
            binned_n_p: np.ndarray = n_p_arr[mask]

            if binned_vals.size > 0:
                means[array_index] = np.mean(binned_vals)
                medians[array_index] = np.median(binned_vals)
                counts[array_index] = binned_vals.size
                probabs[array_index] = (
                    counts[array_index] / float(N_flocs) if N_flocs > 0 else 0.0
                )
            else:
                means[array_index] = np.nan
                medians[array_index] = np.nan
                counts[array_index] = 0
                probabs[array_index] = 0.0

            if binned_vals.size > 0 and np.sum(binned_n_p) > 0:
                mass_means[array_index] = np.average(binned_vals, weights=binned_n_p)
            else:
                mass_means[array_index] = np.nan

            masses[array_index] = np.sum(binned_n_p)
            mass_probabs[array_index] = (
                masses[array_index] / tot_mass_local if tot_mass_local > 0 else 0.0
            )

        means_list.append(means)
        medians_list.append(medians)
        counts_list.append(counts)
        probabs_list.append(probabs)

        mass_means_list.append(mass_means)
        masses_list.append(masses)
        mass_probabs_list.append(mass_probabs)

    return (
        means_list,
        medians_list,
        counts_list,
        probabs_list,
        mass_means_list,
        masses_list,
        mass_probabs_list,
    )


def calc_PDF(
    output_dir: Path,
    floc_dir: Path,
    bin_widths: tuple[float, float, float],
    min_file_index: Optional[int],
    max_file_index: Optional[int],
    num_workers: Optional[int],
    use_threading: bool,
):

    floc_files: List[Path] = myio.list_data_files(
        floc_dir,
        "Flocs",
        min_file_index,
        max_file_index,
    )

    d: float
    with h5py.File(str(floc_files[0]), "r") as f:
        d = f["particles/r"][0] * 2  # type: ignore

    N_flocs_total: int = 0
    N_flocs_list: List[int] = []
    mass_total: float = 0
    mass_list: List[float] = []
    n_p_list: List[np.ndarray] = []
    D_f_list: List[np.ndarray] = []
    D_g_list: List[np.ndarray] = []

    for h5_file in tqdm(
        floc_files,
        total=len(floc_files),
        desc="Generating edges lists for pdfs",
    ):
        with h5py.File(str(h5_file), "r") as f:
            floc_ids: np.ndarray = np.asarray(f["flocs/floc_id"][:])  # type: ignore
            n_p: np.ndarray = np.asarray(f["flocs/n_p"][:])  # type: ignore
            D_f: np.ndarray = np.asarray(f["flocs/D_f"][:]) / d  # type: ignore
            D_g: np.ndarray = np.asarray(f["flocs/D_g"][:]) / d  # type: ignore

        unique_vals, first_indices = np.unique(floc_ids, return_index=True)
        N_flocs_list.append(len(unique_vals))
        N_flocs_total += N_flocs_list[-1]
        n_p_list.append(n_p[first_indices])
        D_f_list.append(D_f[first_indices])
        D_g_list.append(D_g[first_indices])
        mass_list.append(np.sum(n_p_list[-1]))
        mass_total += mass_list[-1]

    edges_list: List[np.ndarray] = []
    centers_list: List[np.ndarray] = []

    for i, vals in enumerate([n_p_list, D_f_list, D_g_list]):
        min_val = (
            np.min([np.min(local_data) for local_data in vals]) - bin_widths[i] / 2
        )
        max_val = (
            np.max([np.max(local_data) for local_data in vals]) - bin_widths[i] / 2
        )
        num_bins = int(np.ceil((max_val - min_val) / bin_widths[i]))
        edges: np.ndarray = np.linspace(
            min_val, min_val + num_bins * bin_widths[i], num_bins + 1
        )
        edges_list.append(edges)
        centers_list.append(edges[:-1] - edges[1:])

    all_stats: Dict[str, List[List[np.ndarray]]] = {
        key: []
        for key in [
            "means",
            "medians",
            "counts",
            "probabs",
            "mass_means",
            "masses",
            "mass_probabs",
        ]
    }
    mean_stats: Optional[Dict[str, List[np.ndarray]]] = None

    for i in tqdm(
        range(len(floc_files)),
        total=len(floc_files),
        desc="averaging pdf between snapshots",
    ):
        processed_results: Tuple[
            List[np.ndarray],
            List[np.ndarray],
            List[np.ndarray],
            List[np.ndarray],
            List[np.ndarray],
            List[np.ndarray],
            List[np.ndarray],
        ] = _process_single_pdf(
            edges_list,
            n_p_list[i],
            D_f_list[i],
            D_g_list[i],
            N_flocs_list[i],
            mass_list[i],
        )
        current: Dict[str, List[np.ndarray]] = {}
        for idx, key in enumerate(list(all_stats.keys())):
            current[key] = processed_results[idx]
            all_stats[key].append(current[key])

        if mean_stats is None:
            mean_stats = {}
            for key in current.keys():
                mean_stats[key] = current[key].copy()
        else:
            new_mean_stats: Dict[str, List[np.ndarray]] = {}
            for key in mean_stats.keys():
                new_arrays: List[np.ndarray] = []
                for idx in range(len(current[key])):
                    if key in ["means", "medians", "mass_means"]:
                        # Replace NaN with 0 for summation, but keep track of valid counts
                        current_val = np.nan_to_num(current[key][idx], nan=0.0)
                        mean_val = np.nan_to_num(mean_stats[key][idx], nan=0.0)
                        new_array: np.ndarray = mean_val + current_val
                    else:
                        # For counts, probabs, masses, mass_probabs, use regular addition
                        new_array: np.ndarray = mean_stats[key][idx] + current[key][idx]
                    new_arrays.append(new_array)
                new_mean_stats[key] = new_arrays
            mean_stats = new_mean_stats

    if mean_stats is None:
        raise ValueError("no mean stats computed")

    valid_counts: Dict[str, List[np.ndarray]] = {}
    for key in all_stats:
        if key in ["means", "medians", "mass_means"]:
            valid_counts[key] = []
            for j in range(len(all_stats[key][0])):
                count_arr = np.zeros_like(all_stats[key][0][j])
                for i in range(len(floc_files)):
                    valid_mask = ~np.isnan(all_stats[key][i][j])
                    count_arr[valid_mask] += 1
                valid_counts[key].append(count_arr)

    for key in mean_stats.keys():
        new_arrays: List[np.ndarray] = []
        for idx, arr in enumerate(mean_stats[key]):
            if key in ["means", "medians", "mass_means"]:
                valid_count = valid_counts[key][idx]
                with np.errstate(divide="ignore", invalid="ignore"):
                    new_array: np.ndarray = np.where(
                        valid_count > 0, arr / valid_count, np.nan
                    )
            else:
                new_array: np.ndarray = arr / len(floc_files)
            new_arrays.append(new_array)
        mean_stats[key] = new_arrays

    diff_stats: Dict[str, List[List[np.ndarray]]] = {}
    for key in all_stats:
        diff_stats[key] = []
        for i in range(len(floc_files)):
            snapshot_diffs: List[np.ndarray] = []
            for j in range(len(all_stats[key][i])):
                diff_array: np.ndarray = all_stats[key][i][j] - mean_stats[key][j]
                snapshot_diffs.append(diff_array)
            diff_stats[key].append(snapshot_diffs)

    std_stats: Dict[str, List[np.ndarray]] = {}
    for key in mean_stats:
        std_stats[key] = []
        for j in range(len(mean_stats[key])):
            squared_diffs: List[np.ndarray] = []
            for i in range(len(floc_files)):
                if key in ["means", "medians", "mass_means"]:
                    if not np.isnan(all_stats[key][i][j]).all():
                        diff_clean = np.nan_to_num(diff_stats[key][i][j], nan=0.0)
                        squared_array: np.ndarray = diff_clean**2
                        squared_diffs.append(squared_array)
                else:
                    squared_array: np.ndarray = diff_stats[key][i][j] ** 2
                    squared_diffs.append(squared_array)

            if squared_diffs:
                sum_squared: np.ndarray = np.zeros_like(squared_diffs[0])
                for sq_arr in squared_diffs:
                    sum_squared += sq_arr
                if key in ["means", "medians", "mass_means"]:
                    valid_count = valid_counts[key][j]
                    with np.errstate(divide="ignore", invalid="ignore"):
                        mean_squared = np.where(
                            valid_count > 0, sum_squared / valid_count, np.nan
                        )
                else:
                    mean_squared = sum_squared / len(floc_files)
                std_arr: np.ndarray = np.sqrt(mean_squared)
                std_stats[key].append(std_arr)
            else:
                std_arr: np.ndarray = np.full_like(mean_stats[key][j], np.nan)
                std_stats[key].append(std_arr)

    results: Dict[str, Union[float, np.ndarray]] = {
        "d": d,
        "N_flocs": N_flocs_total,
        "bin_width_n_p": bin_widths[0],
        "bin_width_D_f": bin_widths[1],
        "bin_width_D_g": bin_widths[2],
        "edges_n_p": edges_list[0],
        "edges_D_f": edges_list[1],
        "edges_D_g": edges_list[2],
        "centers_n_p": centers_list[0],
        "centers_D_f": centers_list[1],
        "centers_D_g": centers_list[2],
        # Mean statistics
        "means_n_p": mean_stats["means"][0],
        "means_D_f": mean_stats["means"][1],
        "means_D_g": mean_stats["means"][2],
        "medians_n_p": mean_stats["medians"][0],
        "medians_D_f": mean_stats["medians"][1],
        "medians_D_g": mean_stats["medians"][2],
        "counts_n_p": mean_stats["counts"][0],
        "counts_D_f": mean_stats["counts"][1],
        "counts_D_g": mean_stats["counts"][2],
        "probab_n_p": mean_stats["probabs"][0],
        "probab_D_f": mean_stats["probabs"][1],
        "probab_D_g": mean_stats["probabs"][2],
        # Mass-weighted mean statistics
        "mass_means_n_p": mean_stats["mass_means"][0],
        "mass_means_D_f": mean_stats["mass_means"][1],
        "mass_means_D_g": mean_stats["mass_means"][2],
        "mass_counts_n_p": mean_stats["masses"][0],
        "mass_counts_D_f": mean_stats["masses"][1],
        "mass_counts_D_g": mean_stats["masses"][2],
        "mass_probab_n_p": mean_stats["mass_probabs"][0],
        "mass_probab_D_f": mean_stats["mass_probabs"][1],
        "mass_probab_D_g": mean_stats["mass_probabs"][2],
        # Standard deviations
        "std_means_n_p": std_stats["means"][0],
        "std_means_D_f": std_stats["means"][1],
        "std_means_D_g": std_stats["means"][2],
        "std_medians_n_p": std_stats["medians"][0],
        "std_medians_D_f": std_stats["medians"][1],
        "std_medians_D_g": std_stats["medians"][2],
        "std_counts_n_p": std_stats["counts"][0],
        "std_counts_D_f": std_stats["counts"][1],
        "std_counts_D_g": std_stats["counts"][2],
        "std_probab_n_p": std_stats["probabs"][0],
        "std_probab_D_f": std_stats["probabs"][1],
        "std_probab_D_g": std_stats["probabs"][2],
        # Mass-weighted standard deviations
        "std_mass_means_n_p": std_stats["mass_means"][0],
        "std_mass_means_D_f": std_stats["mass_means"][1],
        "std_mass_means_D_g": std_stats["mass_means"][2],
        "std_mass_counts_n_p": std_stats["masses"][0],
        "std_mass_counts_D_f": std_stats["masses"][1],
        "std_mass_counts_D_g": std_stats["masses"][2],
        "std_mass_probab_n_p": std_stats["mass_probabs"][0],
        "std_mass_probab_D_f": std_stats["mass_probabs"][1],
        "std_mass_probab_D_g": std_stats["mass_probabs"][2],
    }

    myio.save_to_h5(Path(output_dir) / "floc_PDF.h5", results)

    return results


def CalcAvgDiam(
    output_dir: Path,
    floc_dir: Path,
    channel_half_width: float,
    Re: float,
    u_tau: float,
    n_bins: int,
    n_bins_inner: int,
    min_file_index: Optional[int],
    max_file_index: Optional[int],
    num_workers: Optional[int],
    use_threading: bool,
):

    floc_files: List[Path] = myio.list_data_files(
        floc_dir,
        "Flocs",
        min_file_index,
        max_file_index,
    )

    r_p: float
    with h5py.File(str(floc_files[0]), "r") as f:
        r_p = f["particles/r"][0]  # type: ignore

    def to_wall_units(y: np.ndarray) -> np.ndarray:
        y = np.asarray(y)
        return np.asarray(Re * y * u_tau)

    y: np.ndarray = np.linspace(0.0, channel_half_width, n_bins, endpoint=True)
    yp: np.ndarray = np.linspace(0.0, to_wall_units(y[-1]), n_bins_inner, endpoint=True)
    y_left_arr: np.ndarray = y[:-1]
    y_right_arr: np.ndarray = y[1:]
    yp_left_arr: np.ndarray = yp[:-1]
    yp_right_arr: np.ndarray = yp[1:]

    y_center: np.ndarray = (y_left_arr + y_right_arr) / 2
    yp_center: np.ndarray = (yp_left_arr + yp_right_arr) / 2

    all_D_f_avg: List[np.ndarray] = []
    all_D_g_avg: List[np.ndarray] = []
    all_D_f_mass_avg: List[np.ndarray] = []
    all_D_g_mass_avg: List[np.ndarray] = []
    all_inner_D_f_avg: List[np.ndarray] = []
    all_inner_D_g_avg: List[np.ndarray] = []
    all_inner_D_f_mass_avg: List[np.ndarray] = []
    all_inner_D_g_mass_avg: List[np.ndarray] = []
    all_y_mean: List[np.ndarray] = []
    all_yp_mean: List[np.ndarray] = []

    for floc_file in tqdm(
        floc_files, desc="Processing average diameter", total=len(floc_files)
    ):
        y_floc_arr: np.ndarray
        yp_floc_arr: np.ndarray
        n_p_arr: np.ndarray
        D_f_arr: np.ndarray
        D_g_arr: np.ndarray
        with h5py.File(str(floc_file), "r") as f:
            floc_ids: np.ndarray = f["flocs/floc_id"][:]  # type: ignore
            _, first_indices = np.unique(floc_ids, return_index=True)
            y_floc_arr = f["flocs/y"][first_indices]  # type: ignore
            yp_floc_arr = to_wall_units(y_floc_arr)
            n_p_arr = f["flocs/n_p"][first_indices]  # type: ignore
            D_f_arr = f["flocs/D_f"][first_indices]  # type: ignore
            D_g_arr = f["flocs/D_g"][first_indices]  # type: ignore

        D_f_avg_file: np.ndarray = np.full_like(y_left_arr, np.nan)
        D_g_avg_file: np.ndarray = np.full_like(y_left_arr, np.nan)
        D_f_mass_avg_file: np.ndarray = np.full_like(y_left_arr, np.nan)
        D_g_mass_avg_file: np.ndarray = np.full_like(y_left_arr, np.nan)

        inner_D_f_avg_file: np.ndarray = np.full_like(yp_left_arr, np.nan)
        inner_D_g_avg_file: np.ndarray = np.full_like(yp_left_arr, np.nan)
        inner_D_f_mass_avg_file: np.ndarray = np.full_like(yp_left_arr, np.nan)
        inner_D_g_mass_avg_file: np.ndarray = np.full_like(yp_left_arr, np.nan)

        y_mean_file: np.ndarray = np.full_like(y_left_arr, np.nan)
        yp_mean_file: np.ndarray = np.full_like(yp_left_arr, np.nan)

        for i in range(y_left_arr.shape[0]):
            bin_mask: np.ndarray = (y_floc_arr >= y_left_arr[i]) & (
                y_floc_arr < y_right_arr[i]
            )

            if np.sum(bin_mask) > 0:
                D_f_avg_file[i] = np.mean(D_f_arr[bin_mask])
                D_g_avg_file[i] = np.mean(D_g_arr[bin_mask])
                y_mean_file[i] = np.mean(y_floc_arr[bin_mask])

                if np.sum(n_p_arr[bin_mask]) > 0:
                    D_f_mass_avg_file[i] = np.average(
                        D_f_arr[bin_mask], weights=n_p_arr[bin_mask]
                    )
                    D_g_mass_avg_file[i] = np.average(
                        D_g_arr[bin_mask], weights=n_p_arr[bin_mask]
                    )

        for i in range(yp_left_arr.shape[0]):
            bin_mask: np.ndarray = (yp_floc_arr >= yp_left_arr[i]) & (
                yp_floc_arr < yp_right_arr[i]
            )

            if np.sum(bin_mask) > 0:
                inner_D_f_avg_file[i] = np.mean(D_f_arr[bin_mask])
                inner_D_g_avg_file[i] = np.mean(D_g_arr[bin_mask])
                yp_mean_file[i] = np.mean(yp_floc_arr[bin_mask])

                if np.sum(n_p_arr[bin_mask]) > 0:
                    inner_D_f_mass_avg_file[i] = np.average(
                        D_f_arr[bin_mask], weights=n_p_arr[bin_mask]
                    )
                    inner_D_g_mass_avg_file[i] = np.average(
                        D_g_arr[bin_mask], weights=n_p_arr[bin_mask]
                    )

        all_D_f_avg.append(D_f_avg_file)
        all_D_g_avg.append(D_g_avg_file)
        all_D_f_mass_avg.append(D_f_mass_avg_file)
        all_D_g_mass_avg.append(D_g_mass_avg_file)
        all_inner_D_f_avg.append(inner_D_f_avg_file)
        all_inner_D_g_avg.append(inner_D_g_avg_file)
        all_inner_D_f_mass_avg.append(inner_D_f_mass_avg_file)
        all_inner_D_g_mass_avg.append(inner_D_g_mass_avg_file)
        all_y_mean.append(y_mean_file)
        all_yp_mean.append(yp_mean_file)

    all_D_f_avg_arr = np.array(all_D_f_avg)
    all_D_g_avg_arr = np.array(all_D_g_avg)
    all_D_f_mass_avg_arr = np.array(all_D_f_mass_avg)
    all_D_g_mass_avg_arr = np.array(all_D_g_mass_avg)
    all_inner_D_f_avg_arr = np.array(all_inner_D_f_avg)
    all_inner_D_g_avg_arr = np.array(all_inner_D_g_avg)
    all_inner_D_f_mass_avg_arr = np.array(all_inner_D_f_mass_avg)
    all_inner_D_g_mass_avg_arr = np.array(all_inner_D_g_mass_avg)
    all_y_mean_arr = np.array(all_y_mean)
    all_yp_mean_arr = np.array(all_yp_mean)

    D_f_avg: np.ndarray = np.nanmean(all_D_f_avg_arr, axis=0)
    D_g_avg: np.ndarray = np.nanmean(all_D_g_avg_arr, axis=0)
    D_f_mass_avg: np.ndarray = np.nanmean(all_D_f_mass_avg_arr, axis=0)
    D_g_mass_avg: np.ndarray = np.nanmean(all_D_g_mass_avg_arr, axis=0)
    inner_D_f_avg: np.ndarray = np.nanmean(all_inner_D_f_avg_arr, axis=0)
    inner_D_g_avg: np.ndarray = np.nanmean(all_inner_D_g_avg_arr, axis=0)
    inner_D_f_mass_avg: np.ndarray = np.nanmean(all_inner_D_f_mass_avg_arr, axis=0)
    inner_D_g_mass_avg: np.ndarray = np.nanmean(all_inner_D_g_mass_avg_arr, axis=0)
    y_mean: np.ndarray = np.nanmean(all_y_mean_arr, axis=0)
    yp_mean: np.ndarray = np.nanmean(all_yp_mean_arr, axis=0)

    std_D_f_avg: np.ndarray = np.nanstd(all_D_f_avg_arr, axis=0)
    std_D_g_avg: np.ndarray = np.nanstd(all_D_g_avg_arr, axis=0)
    std_D_f_mass_avg: np.ndarray = np.nanstd(all_D_f_mass_avg_arr, axis=0)
    std_D_g_mass_avg: np.ndarray = np.nanstd(all_D_g_mass_avg_arr, axis=0)
    inner_std_D_f_avg: np.ndarray = np.nanstd(all_inner_D_f_avg_arr, axis=0)
    inner_std_D_g_avg: np.ndarray = np.nanstd(all_inner_D_g_avg_arr, axis=0)
    inner_std_D_f_mass_avg: np.ndarray = np.nanstd(all_inner_D_f_mass_avg_arr, axis=0)
    inner_std_D_g_mass_avg: np.ndarray = np.nanstd(all_inner_D_g_mass_avg_arr, axis=0)
    std_y_mean: np.ndarray = np.nanstd(all_y_mean_arr, axis=0)
    std_yp_mean: np.ndarray = np.nanstd(all_yp_mean_arr, axis=0)

    N_flocs_bin: np.ndarray = np.zeros_like(y_left_arr)
    mass_bin: np.ndarray = np.zeros_like(y_left_arr)
    inner_N_flocs_bin: np.ndarray = np.zeros_like(yp_left_arr)
    inner_mass_bin: np.ndarray = np.zeros_like(yp_left_arr)

    for floc_file in floc_files:
        with h5py.File(str(floc_file), "r") as f:
            floc_ids: np.ndarray = f["flocs/floc_id"][:]  # type: ignore
            _, first_indices = np.unique(floc_ids, return_index=True)
            y_floc_arr = f["flocs/y"][first_indices]  # type: ignore
            yp_floc_arr = to_wall_units(y_floc_arr)
            n_p_arr = f["flocs/n_p"][first_indices]  # type: ignore

        for i in range(y_left_arr.shape[0]):
            bin_mask: np.ndarray = (y_floc_arr >= y_left_arr[i]) & (
                y_floc_arr < y_right_arr[i]
            )
            N_flocs_bin[i] += np.sum(bin_mask)
            mass_bin[i] += np.sum(n_p_arr[bin_mask])

        for i in range(yp_left_arr.shape[0]):
            bin_mask: np.ndarray = (yp_floc_arr >= yp_left_arr[i]) & (
                yp_floc_arr < yp_right_arr[i]
            )
            inner_N_flocs_bin[i] += np.sum(bin_mask)
            inner_mass_bin[i] += np.sum(n_p_arr[bin_mask])

    results: Dict[str, Union[int, float, np.ndarray]] = {
        "d": 2 * r_p,
        "nbins": n_bins,
        "N_flocs": N_flocs_bin,
        "y_left": y_left_arr,
        "y_right": y_right_arr,
        "y_center": y_center,
        "y_mean": y_mean,
        "yp_left": yp_left_arr,
        "yp_right": yp_right_arr,
        "yp_center": yp_center,
        "yp_mean": yp_mean,
        "D_f_avg": D_f_avg,
        "D_g_avg": D_g_avg,
        "D_f_mass_avg": D_f_mass_avg,
        "D_g_mass_avg": D_g_mass_avg,
        "inner_D_f_avg": inner_D_f_avg,
        "inner_D_g_avg": inner_D_g_avg,
        "inner_D_f_mass_avg": inner_D_f_mass_avg,
        "inner_D_g_mass_avg": inner_D_g_mass_avg,
        "std_D_f_avg": std_D_f_avg,
        "std_D_g_avg": std_D_g_avg,
        "std_D_f_mass_avg": std_D_f_mass_avg,
        "std_D_g_mass_avg": std_D_g_mass_avg,
        "inner_std_D_f_avg": inner_std_D_f_avg,
        "inner_std_D_g_avg": inner_std_D_g_avg,
        "inner_std_D_f_mass_avg": inner_std_D_f_mass_avg,
        "inner_std_D_g_mass_avg": inner_std_D_g_mass_avg,
        "std_y_mean": std_y_mean,
        "std_yp_mean": std_yp_mean,
    }

    myio.save_to_h5(Path(output_dir) / "avg_floc_diam.h5", results)

    return results
