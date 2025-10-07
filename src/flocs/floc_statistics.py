# -- flocs/floc_statistics.py

import numpy as np
from typing import Dict, Union, Tuple


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
