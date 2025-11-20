import numpy as np
from typing import Literal
import h5py
from pathlib import Path
from tqdm import tqdm

from src.statistics import (
    Accessor,
    AccessorWithMass,
    AccessorWithoutMass,
    FilterPredicate,
    NoFilterPredicate,
    calc_PDF,
)
from src import myio


class FilterPredicateYpRange(FilterPredicate):
    name_: str = "FilterPredicateYpRange"
    yp_range_: tuple[float | None, float | None]
    Re_tau_: float

    def __init__(
        self, yp_range: tuple[float | None, float | None], Re_tau: float
    ) -> None:
        self.yp_range_ = yp_range
        self.Re_tau_ = Re_tau

    def __call__(self, f: h5py.File | dict) -> np.ndarray:
        yp_particle: np.ndarray = f["y"][:] * self.Re_tau_  # type: ignore
        mask: np.ndarray
        if self.yp_range_[0] is not None and self.yp_range_[1] is not None:
            mask = (yp_particle >= self.yp_range_[0]) & (
                yp_particle <= self.yp_range_[1]
            )
        elif self.yp_range_[0] is not None:
            mask = yp_particle >= self.yp_range_[0]
        elif self.yp_range_[1] is not None:
            mask = yp_particle <= self.yp_range_[1]
        else:
            mask = np.ones(len(yp_particle), dtype=bool)
        mask = mask.astype(bool)
        return mask


class AccessParticleAcceleration(AccessorWithoutMass):
    name_: str = "AccessParticleAcceleration"
    dir_: Literal["x", "y", "z"]
    rho_s_: float

    def __init__(self, dir: Literal["x", "y", "z"], rho_s: float):
        self.dir_ = dir
        self.rho_s_ = rho_s

    def __call__(
        self, f: h5py.File | dict, mask: np.ndarray | None
    ) -> tuple[np.ndarray, None]:
        F_IBM_all: np.ndarray = f["F_IBM_" + self.dir_][:]  # type: ignore
        mask_use: np.ndarray
        if mask is None:
            mask_use = np.ones(len(F_IBM_all), dtype=bool)
        else:
            mask_use = mask

        F_IBM: np.ndarray = F_IBM_all[mask_use]
        F_coll: np.ndarray = f["F_coll_" + self.dir_][mask_use]  # type: ignore
        F_rigid: np.ndarray = f["F_rigid_" + self.dir_][mask_use]  # type: ignore
        r: np.ndarray = f["r"][mask_use]  # type: ignore

        F_tot: np.ndarray = F_IBM + F_coll + F_rigid
        m_p: np.ndarray = self.rho_s_ * 4.0 / 3.0 * np.pi * (r**3)

        return (F_tot / m_p, None)


class AccessUPlus(AccessorWithoutMass):
    name_: str = "AccessUPlus"
    u_tau_: float

    def __init__(self, u_tau: float):
        self.u_tau_ = u_tau

    def __call__(
        self, f: h5py.File | dict, mask: np.ndarray | None
    ) -> tuple[np.ndarray, None]:
        u_all: np.ndarray = f["u"][:]  # type: ignore
        mask_use: np.ndarray
        if mask is None:
            mask_use = np.ones(len(u_all), dtype=bool)
        else:
            mask_use = mask

        u_plus: np.ndarray = u_all[mask_use] / self.u_tau_
        return u_plus, None


def _compute_overall_acceleration_std(
    particle_files: list[Path], rho_s: float
) -> tuple[list[float], list[float]]:

    dir: list[str] = ["x", "y", "z"]

    sums: list[float] = [0.0] * len(dir)
    sums_sq: list[float] = [0.0] * len(dir)
    counts: list[int] = [0] * len(dir)

    for particle_file in tqdm(
        particle_files,
        desc="computing acceleration stats",
        total=len(particle_files),
        unit="Files",
    ):
        with h5py.File(str(particle_file), "r") as f:
            r: np.ndarray = f["r"][:]  # type: ignore
            m_p = rho_s * 4.0 / 3.0 * np.pi * (r**3)

            for i in range(len(dir)):
                F_tot: np.ndarray = (
                    f["F_IBM_" + dir[i]][:]  # type: ignore
                    + f["F_coll_" + dir[i]][:]  # type: ignore
                    + f["F_rigid_" + dir[i]][:]  # type: ignore
                )
                a: np.ndarray = F_tot / m_p

                sums[i] += np.sum(a)
                sums_sq[i] += np.sum(a**2)
                counts[i] += len(a)

    a_mean: list[float] = [sums[i] / counts[i] for i in range(len(dir))]
    a_std: list[float] = [
        np.sqrt(sums_sq[i] / counts[i] - a_mean[i] ** 2) for i in range(len(dir))
    ]

    return a_mean, a_std


def _get_field_accessors(
    particle_files: list[Path],
    metadata_file: Path,
    u_tau: float,
    precomputed_dict: dict,
) -> dict[str, Accessor]:

    rho_s: float
    if "rho_s" in precomputed_dict:
        rho_s = precomputed_dict["rho_s"]
    else:
        metadata: dict[str, dict[str, int | float | str]] = myio.metadata.read_metadata(
            metadata_file
        )
        rho_s = float(metadata["General"]["rho_s"])
        precomputed_dict.update({"rho_s": rho_s})

    a_std_x: float
    a_std_y: float
    a_std_z: float
    if (
        "a_std_x" in precomputed_dict
        and "a_std_y" in precomputed_dict
        and "a_std_z" in precomputed_dict
    ):
        a_std_x = precomputed_dict["a_std_x"]
        a_std_y = precomputed_dict["a_std_y"]
        a_std_z = precomputed_dict["a_std_z"]
    else:
        _, a_std = _compute_overall_acceleration_std(particle_files, rho_s)
        a_std_x = a_std[0]
        a_std_y = a_std[1]
        a_std_z = a_std[2]
        precomputed_dict.update(
            {"a_std_x": a_std_x, "a_std_y": a_std_y, "a_std_z": a_std_z}
        )

    access_particle_acceleration_x = AccessParticleAcceleration("x", rho_s)
    access_particle_acceleration_y = AccessParticleAcceleration("y", rho_s)
    access_particle_acceleration_z = AccessParticleAcceleration("z", rho_s)
    access_u_plus = AccessUPlus(u_tau)

    field_accessors: dict[
        str, Callable[[h5py.File, np.ndarray], tuple[np.ndarray, np.ndarray | None]]
    ] = {
        "a_x": access_particle_acceleration_x,
        "a_y": access_particle_acceleration_z,
        "a_z": access_particle_acceleration_y,
        "u_plus": access_u_plus,
    }
    return field_accessors


def calc_lagrangian_PDF_steadystate(
    floc_dir: Path,
    flow_stats_h5_file: Path,
    metadata_file: Path,
    trn: bool,
    fields: list[str],
    bin_widths: dict[str, float],
    yp_range: tuple[float | None, float | None] | None,
    precomputed_dict: dict = {},
) -> tuple[dict[str, dict[str, np.ndarray | dict[str, np.ndarray]]], dict]:
    """
    Computes lagrangian PDF statistics for a simulation just using
    floc files after precomputed steady state period.

    Args:
        floc_dir: Directory containing floc data HDF5 files.
        flow_stats_h5_file: Path to precomuted flow_stats.h5 file.
                            Used to get Re_tau and u_tau.
        metadata_file: Path to metadata.ini file.
        trn: Whether to use trn files (steady state index for trn files).
        fields: list of all field names for wich pdf should be computed
                (has to exist in _get_process_field_functions(...)).
                These must be the keys in the bin_widths dictionary.
        bin_widths: Dictionary of bin_widths corresponding to each field.
        yp_range: Range of yp of which to compute statistics (None if one
                  wants to compute over entire domain).
        bin_widths: Dictionary of bin_widths corresponding to each field.
        precomputed_dict: Some values (for e.g. overall acceleration std
                          can be precomuted as to avoid recomputation)

    Returns:
        Tuple of dictionary containing lagrangian PDF statistics and a
        dictionary containing precomuted values for another pdf computation
        pass
    """

    particle_files: list[Path] = myio.utils.get_steadystate_particle_files(
        floc_dir, metadata_file, trn
    )

    u_tau: float
    Re_tau: float
    if "u_tau" in precomputed_dict and "Re_tau" in precomputed_dict:
        u_tau = precomputed_dict["u_tau"]
        Re_tau = precomputed_dict["Re_tau"]
    else:
        metadata: dict[str, dict[str, int | float | str]] = myio.metadata.read_metadata(
            metadata_file
        )
        u_tau = float(metadata["Flow"]["u_tau"])
        Re_tau = float(metadata["Flow"]["Re_tau"])
        precomputed_dict.update({"u_tau": u_tau, "Re_tau": Re_tau})

    field_accessors = _get_field_accessors(
        particle_files, metadata_file, u_tau, precomputed_dict
    )

    filter_predicate: FilterPredicate
    if yp_range == None:
        filter_predicate = NoFilterPredicate()
    else:
        filter_predicate = FilterPredicateYpRange(yp_range, Re_tau)

    results: dict[str, dict[str, np.ndarray | dict[str, np.ndarray]]] = {}
    for field in fields:

        results[field] = calc_PDF(
            particle_files,
            bin_widths[field],
            field,
            field_accessors[field],
            filter_predicate,
            mass_weighted=False,
            file_type="h5",
        )
    return results, precomputed_dict
