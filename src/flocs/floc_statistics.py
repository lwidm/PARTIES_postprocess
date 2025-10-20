# -- src/flocs/floc_statistics.py

from pathlib import Path
import h5py  # type: ignore
import numpy as np
from typing import Dict, Union, Tuple, Optional, List
from tqdm import tqdm  # type: ignore
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed


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
    """Calculate the floc felocity."""
    return U.mean(axis=0)


def calc_feret_diam(
    particle_diam: float,
    X_p: np.ndarray,
    X_CoM: np.ndarray,
    shift: np.ndarray,
) -> float:
    """Calcualte the floc Feret diameter."""
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
    """Calculates the floc fractal dimension."""
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
    """Calcualtes the floc pitch.

    The pitch is the angle the floc makes with the x-axis in the y-direction.
    """
    if N_partices < 2:
        return 0
    return np.arctan(orientation[1] / orientation[0])


def calc_theta(orientation: np.ndarray, N_particles: int) -> np.ndarray:
    """Calculates the floc angle.

    Theta is the angle the floc makes with the x-axis in the z-direction.
    """
    if N_particles < 2:
        return np.zeros(3)
    return np.arccos(orientation / np.linalg.norm(orientation))

def _count_floc_PDF_occurences(
    h5_file: Union[str, Path],
) -> Tuple[List[int], List[float], List[float], int]:
    floc_ids: np.ndarray
    n_p: np.ndarray
    D_f: np.ndarray
    D_g: np.ndarray
    with h5py.File(h5_file, "r") as f:
        floc_ids = f["flocs/floc_id"][:]  # type: ignore
        n_p = f["flocs/n_p"][:]  # type: ignore
        D_f = f["flocs/D_f"][:]  # type: ignore
        D_g = f["flocs/D_g"][:]  # type: ignore
    unique_vals, first_indices = np.unique(floc_ids, return_index=True)
    N_flocs_local: int = len(unique_vals)
    n_p_list: List[int] = n_p[first_indices].tolist()
    D_f_list: List[float] = D_f[first_indices].tolist()
    D_g_list: List[float] = D_g[first_indices].tolist()

    return n_p_list, D_f_list, D_g_list, N_flocs_local

def calc_PDF(
    output_dir: Union[str, Path],
    floc_dir: Union[str, Path],
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

    N_flocs: int = 0
    n_p_list: List[int] = []
    D_f_list: List[float] = []
    D_g_list: List[float] = []
    if num_workers is not None:
        if use_threading:
            executor = ThreadPoolExecutor
        else:
            executor = ProcessPoolExecutor
        with executor(max_workers=num_workers) as ex:
            futures = [
                ex.submit(_count_floc_PDF_occurences, h5_file) for h5_file in floc_files
            ]

            for f in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Processing flocs for PDF generation",
            ):
                _n_p_list, _D_f_list, _D_g_list, _N_flocs = f.result()
                N_flocs += _N_flocs
                n_p_list += _n_p_list
                D_f_list += _D_f_list
                D_g_list += _D_g_list

    else:
        N_flocs: int = 0
        n_p_list: List[int] = []
        D_f_list: List[float] = []
        D_g_list: List[float] = []
        for h5_file in tqdm(
            floc_files,
            total=len(floc_files),
            desc="Processing flocs for PDF generation",
        ):
            _n_p_list, _D_f_list, _D_g_list, _N_flocs = _count_floc_PDF_occurences(
                h5_file
            )
            N_flocs += _N_flocs
            n_p_list += _n_p_list
            D_f_list += _D_f_list
            D_g_list += _D_g_list

    if N_flocs == 0:
        raise ValueError("No flocs found (N_flocs == 0)")

    d: float
    with h5py.File(floc_files[0], "r") as f:
        d = f["particles/r"][0] * 2  # type: ignore

    n_p_arr: np.ndarray = np.array(n_p_list)
    D_f_arr: np.ndarray = np.array(D_f_list) / d
    D_g_arr: np.ndarray = np.array(D_g_list) / d

    def edges_from_width(data, width):
        min_val, max_val = min(data)-width/2, max(data)-width/2
        num_bins = int(np.ceil((max_val - min_val) / width))
        return np.linspace(min_val, min_val + num_bins * width, num_bins + 1)

    edges_n_p = edges_from_width(n_p_arr, bin_widths[0])
    edges_D_f = edges_from_width(D_f_arr, bin_widths[1])
    edges_D_g = edges_from_width(D_g_arr, bin_widths[2])

    counts_n_p, edges_n_p = np.histogram(n_p_arr, bins=edges_n_p)
    counts_D_f, edges_D_f = np.histogram(D_f_arr, bins=edges_D_f)
    counts_D_g, edges_D_g = np.histogram(D_g_arr, bins=edges_D_g)

    centers_n_p: np.ndarray = (edges_n_p[:-1] + edges_n_p[1:]) / 2.0
    centers_D_f: np.ndarray = (edges_D_f[:-1] + edges_D_f[1:]) / 2.0
    centers_D_g: np.ndarray = (edges_D_g[:-1] + edges_D_g[1:]) / 2.0

    probab_n_p: np.ndarray = counts_n_p.astype(float) / float(N_flocs)
    probab_D_f: np.ndarray = counts_D_f.astype(float) / float(N_flocs)
    probab_D_g: np.ndarray = counts_D_g.astype(float) / float(N_flocs)

    results: Dict[str, Union[float, np.ndarray]] = {
        "d": d,
        "N_flocs": N_flocs,
        "bin_width_n_p": bin_widths[0],
        "bin_width_D_f": bin_widths[1],
        "bin_width_D_g": bin_widths[2],
        "counts_n_p": counts_n_p,
        "counts_D_f": counts_D_f,
        "counts_D_g": counts_D_g,
        "edges_n_p": edges_n_p,
        "edges_D_f": edges_D_f,
        "edges_D_g": edges_D_g,
        "centers_n_p": centers_n_p,
        "centers_D_f": centers_D_f,
        "centers_D_g": centers_D_g,
        "probab_n_p": probab_n_p,
        "probab_D_f": probab_D_f,
        "probab_D_g": probab_D_g,
    }

    myio.save_to_h5(Path(output_dir) / "floc_PDF.h5", results)

    return results



def CalcAvgDiam(
    output_dir: Union[str, Path],
    floc_dir: Union[str, Path],
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
    with h5py.File(floc_files[0], "r") as f:
        r_p: float = f["particles/r"][0] # type: ignore

    def to_wall_units(y: np.ndarray):
        return Re * y * u_tau

    y: np.ndarray = np.linspace(0.0, channel_half_width, n_bins, endpoint=True)
    yp: np.ndarray = np.linspace(0.0, to_wall_units(y[-1])[0], n_bins_inner, endpoint=True)
    y_left_arr: np.ndarray = y[:-1]
    y_right_arr: np.ndarray = y[1:]
    yp_left_arr: np.ndarray = yp[:-1]
    yp_right_arr: np.ndarray = yp[1:]
    N_flocs_bin: np.ndarray = np.zeros_like(y_left_arr) # total number of flocs in each bin
    inner_N_flocs_bin: np.ndarray = np.zeros_like(yp_left_arr)
    tot_n_p_arr = np.zeros_like(y_left_arr) # total number of particles in each bin
    tot_D_f_arr = np.zeros_like(y_left_arr) # total number of particles in each bin
    tot_D_g_arr = np.zeros_like(y_left_arr) # total number of particles in each bin
    tot_Vol_arr = np.zeros_like(y_left_arr)
    inner_tot_n_p_arr = np.zeros_like(yp_left_arr) # total number of particles in each bin (bin in inner units)
    inner_tot_D_f_arr = np.zeros_like(yp_left_arr)
    inner_tot_D_g_arr = np.zeros_like(yp_left_arr)
    inner_tot_Vol_arr = np.zeros_like(yp_left_arr)

    for floc_file in floc_files:
        y_floc_arr: np.ndarray
        yp_floc_arr: np.ndarray
        n_p_arr: np.ndarray
        D_f_arr: np.ndarray
        D_g_arr: np.ndarray
        with h5py.File(floc_file, "r") as f:
            floc_ids: np.ndarray = f["flocs/floc_id"][:] # type: ignore
            _, first_indices = np.unique(floc_ids, return_index=True)
            y_floc_arr = f["flocs/y"][first_indices] # type: ignore
            yp_floc_arr = to_wall_units(y_floc_arr)
            n_p_arr = f["flocs/n_p"][first_indices] # type: ignore
            D_f_arr = f["flocs/D_f"][first_indices] # type: ignore
            D_g_arr = f["flocs/D_p"][first_indices] # type: ignore

        for i in range(y_left_arr.shape[0]):
            bin_map: np.ndarray = y_floc_arr >= y_left_arr[i] & y_floc_arr > y_left_arr[i]
            N_flocs_bin[i] += np.sum(bin_map)
            tot_n_p_arr[i] += np.sum(n_p_arr[bin_map])
            tot_D_f_arr[i] += np.sum(D_f_arr[bin_map])
            tot_D_g_arr[i] += np.sum(D_g_arr[bin_map])
            tot_Vol_arr[i] += np.sum(n_p_arr[bin_map] * 4.0/3.0 * np.pi *r_p**3)

        for i in range(yp_left_arr.shape[0]):
            bin_map: np.ndarray = yp_floc_arr >= yp_left_arr[i] & yp_floc_arr > yp_left_arr[i]
            inner_N_flocs_bin[i] += np.sum(bin_map)
            inner_tot_n_p_arr[i] += np.sum(n_p_arr[bin_map])
            inner_tot_D_f_arr[i] += np.sum(D_f_arr[bin_map])
            inner_tot_D_g_arr[i] += np.sum(D_g_arr[bin_map])
            inner_tot_Vol_arr[i] += np.sum(n_p_arr[bin_map] * 4.0/3.0 * np.pi *r_p**3)

            # y_floc_arr = y_floc_arr[not bin_map]
            # n_p_arr = n_p_arr[not bin_map]
            # D_f_arr = D_f_arr[not bin_map]
            # D_g_arr = D_g_arr[not bin_map]

    results: Dict[str, Union[int, float, np.ndarray]] = {
        "d": 2*r_p,
        "nbins": n_bins,
        "N_flocs": N_flocs_bin,
        "y_left": y_left_arr,
        "y_right": y_right_arr,
        "y_center": (y_left_arr + y_right_arr)/2,
        "yp_left": yp_left_arr,
        "yp_right": yp_right_arr,
        "yp_center": (yp_left_arr + yp_right_arr)/2,
        "n_p_avg": tot_n_p_arr / N_flocs_bin,
        "D_f_avg": tot_D_f_arr / N_flocs_bin,
        "D_g_avg": tot_D_g_arr / N_flocs_bin,
        "Vol_avg": tot_Vol_arr / N_flocs_bin,
        "inner_n_p_avg": inner_tot_n_p_arr / inner_N_flocs_bin,
        "inner_D_f_avg": inner_tot_D_f_arr / inner_N_flocs_bin,
        "inner_D_g_avg": inner_tot_D_g_arr / inner_N_flocs_bin,
        "inner_Vol_avg": inner_tot_Vol_arr / inner_N_flocs_bin,
    }

    myio.save_to_h5(Path(output_dir) / "avg_floc_diam.h5", results)

    return results
