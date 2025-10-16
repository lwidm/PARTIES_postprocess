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

def _fd_nbins(data: np.ndarray) -> int:
    """Freedman-Diaconis rule to compute number of bins. Returns at least 1."""
    if data.size < 2:
        return 1
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25
    if iqr == 0:
        # degenerate IQR: fallback to Sturges
        return int(np.ceil(np.log2(data.size) + 1))
    bin_width = 2.0 * iqr / (data.size ** (1 / 3))
    if bin_width <= 0:
        return int(np.ceil(np.log2(data.size) + 1))
    nbins = int(np.ceil((data.max() - data.min()) / bin_width))
    return max(1, nbins)

def _unique_value_edges(values: np.ndarray) -> np.ndarray:
    """
    Create bin edges so that each unique value falls into its own bin.
    Edges are midpoints between sorted unique values, with half-width on extremes.
    """
    uq = np.unique(values)
    if uq.size == 1:
        v = float(uq[0])
        # single value: make a narrow bin around it
        return np.array([v - 0.5, v + 0.5])
    # midpoints between adjacent unique values
    diffs = np.diff(uq)
    mids = uq[:-1] + diffs / 2.0
    # first edge and last edge: extend by half the first/last diff
    first_edge = uq[0] - diffs[0] / 2.0
    last_edge = uq[-1] + diffs[-1] / 2.0
    return np.concatenate([[first_edge], mids, [last_edge]])

def calc_PDF(
    output_dir: Union[str, Path],
    floc_dir: Union[str, Path],
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
            futures = [ex.submit(_count_floc_PDF_occurences, h5_file) for h5_file in floc_files]

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
            _n_p_list, _D_f_list, _D_g_list, _N_flocs = count_occurences(h5_file)
            N_flocs += _N_flocs
            n_p_list += _n_p_list
            D_f_list += _D_f_list
            D_g_list += _D_g_list

    if N_flocs == 0:
        raise ValueError("No flocs found (N_flocs == 0)")

    edges_n_p: np.ndarray
    if len(n_p_list) == 0:
        edges_n_p = np.array([0.0, 1.0])
    else:
        nmin: int = min(n_p_list)
        nmax: int = max(n_p_list)
        edges_n_p = np.arange(nmin-0.5, nmax + 0.5 + 1e-8, 1.0)

    edges_D_f: np.ndarray
    if len(D_f_list) == 0:
        edges_D_f = np.array([0.0, 1.0])
    else:
        unique_D_f: np.ndarray = np.unique(D_f_list)
        if unique_D_f.size <= 20:
            # small number of distinct values -> one bin per distinct value
            edges_D_f = _unique_value_edges(np.array(D_f_list))
        else:
            nbins: int = _fd_nbins(np.array(D_f_list))
            nbins = min(max(nbins, 10), 200)
            edges_D_f = np.histogram_bin_edges(D_f_list, bins=nbins)

    edges_D_g: np.ndarray
    if len(D_g_list) == 0:
        edges_D_g = np.array([0.0, 1.0])
    else:
        unique_D_g: np.ndarray = np.unique(D_g_list)
        if unique_D_g.size <= 20:
            # small number of distinct values -> one bin per distinct value
            edges_D_g = _unique_value_edges(np.array(D_g_list))
        else:
            nbins: int = _fd_nbins(np.array(D_g_list))
            nbins = min(max(nbins, 10), 200)
            edges_D_g = np.histogram_bin_edges(D_g_list, bins=nbins)



    counts_n_p, edges_n_p = np.histogram(n_p_list, bins=edges_n_p)
    counts_D_f, edges_D_f = np.histogram(D_f_list, bins=edges_D_f)
    counts_D_g, edges_D_g = np.histogram(D_g_list, bins=edges_D_g)

    centers_n_p: np.ndarray = (edges_n_p[:-1] + edges_n_p[1:]) / 2.0
    centers_D_f: np.ndarray = (edges_D_f[:-1] + edges_D_f[1:]) / 2.0
    centers_D_g: np.ndarray = (edges_D_g[:-1] + edges_D_g[1:]) / 2.0

    probab_n_p: np.ndarray = counts_n_p.astype(float) / float(N_flocs)
    probab_D_f: np.ndarray = counts_D_f.astype(float) / float(N_flocs)
    probab_D_g: np.ndarray = counts_D_g.astype(float) / float(N_flocs)

    results: Dict[str, Union[float, np.ndarray]] = {
        "N_flocs": N_flocs,
        "counts_n_p": counts_n_p,
        "counts_D_f": counts_D_f,
        "counts_D_g": counts_D_g,
        "centers_n_p": centers_n_p,
        "centers_D_f": centers_D_f,
        "centers_D_g": centers_D_g,
        "probab_n_p": probab_n_p,
        "probab_D_f": probab_D_f,
        "probab_D_g": probab_D_g,
    }

    myio.save_to_h5(Path(output_dir) / "floc_PDF.h5", results)

    return results
