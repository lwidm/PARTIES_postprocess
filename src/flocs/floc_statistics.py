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
