from pathlib import Path
from typing import TypedDict, Literal
import numpy as np
import h5py
from tqdm import tqdm
import pickle
from scipy.stats import binned_statistic

from src.myio import lwidmer, metadata

from src.statistics import (
    Accessor,
    AccessorWithMass,
    AccessorWithoutMass,
    FilterPredicate,
    NoFilterPredicate,
    calc_PDF,
    get_hist_bins,
)


class FlocData(TypedDict):
    floc_x: dict[int, float]
    floc_y: dict[int, float]
    floc_z: dict[int, float]
    floc_sizes: dict[int, int]
    time: float


class FlocRecord(TypedDict):
    parents: list[int]
    children: list[int]
    parents_sizes: list[int]
    children_sizes: list[int]
    parents_position: list[float]
    children_position: list[float]
    constituents: list[int]
    start_time: float
    end_time: float
    start_file_id: int
    end_file_id: int
    size: int
    x_start: float
    x_end: float
    y_start: float
    y_end: float
    z_start: float
    z_end: float


FamilyTreeType = dict[int, FlocRecord]


class AccessFlocBreakupLocation(AccessorWithMass):
    name_: str = "AccessFlocBreakupLocation"
    t_steady_: float
    dt_: float

    def __init__(self, t_steady: float, dt: float) -> None:
        self.t_steady_ = t_steady
        self.dt_ = dt

    def __call__(
        self, f: h5py.File | dict, mask: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(f, dict):
            y_breakup: list = []
            mass: list = []
            family_tree_dict: FamilyTreeType = f
            for floc_id, floc_record in tqdm(
                family_tree_dict.items(),
                desc=f"Collecting floc breakup data",
                total=len(family_tree_dict),
                unit="Flocs",
            ):
                if (
                    len(floc_record["children"]) > 1
                    and floc_record["start_time"] >= self.t_steady_
                    and floc_record["end_time"] - floc_record["start_time"] >= self.dt_
                ):
                    y_breakup.append(floc_record["y_end"])
                    mass.append(floc_record["size"])

            y_breakup_arr: np.ndarray = np.squeeze(np.asarray(y_breakup))
            mass_arr: np.ndarray = np.squeeze(np.asarray(mass))
            return y_breakup_arr, mass_arr
        else:
            raise ValueError(
                f"must pass a family tree dict into {self.name_}, not a h5fp.File object"
            )


class AccessFlocFormationLocation(AccessorWithMass):
    name_: str = "AccessFlocFormationLocation"
    t_steady_: float
    dt_: float

    def __init__(self, t_steady: float, dt: float) -> None:
        self.t_steady_ = t_steady
        self.dt_ = dt

    def __call__(
        self, f: h5py.File | dict, mask: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(f, dict):
            y_breakup: list = []
            mass: list = []
            family_tree_dict: FamilyTreeType = f
            for floc_id, floc_record in tqdm(
                family_tree_dict.items(),
                desc=f"Collecting floc formation data",
                total=len(family_tree_dict),
                unit="Flocs",
            ):
                if (
                    len(floc_record["parents"]) > 1
                    and floc_record["start_time"] >= self.t_steady_
                    and floc_record["end_time"] - floc_record["start_time"] >= self.dt_
                ):
                    y_breakup.append(floc_record["y_start"])
                    mass.append(floc_record["size"])

            y_breakup_arr: np.ndarray = np.squeeze(np.asarray(y_breakup))
            mass_arr: np.ndarray = np.squeeze(np.asarray(mass))
            return y_breakup_arr, mass_arr
        else:
            raise ValueError(
                f"must pass a family tree dict into {self.name_}, not a h5fp.File object"
            )


class AccessFlocBreakupLocationAdvanced(AccessorWithMass):
    name_: str = "AccessFlocBreakupLocationAdvanced"
    t_steady_: float
    slope_: float
    intersect_: float

    def __init__(self, t_steady: float, slope: float, intersect: float) -> None:
        self.t_steady_ = t_steady
        self.slope_ = slope
        self.intersect_ = intersect

    def __call__(
        self, f: h5py.File | dict, mask: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(f, dict):
            y_breakup: list = []
            mass: list = []
            family_tree_dict: FamilyTreeType = f
            for floc_id, floc_record in tqdm(
                family_tree_dict.items(),
                desc=f"Collecting floc breakup data",
                total=len(family_tree_dict),
                unit="Flocs",
            ):
                if (
                    len(floc_record["children"]) > 1
                    and floc_record["start_time"] >= self.t_steady_
                ):
                    floc_lifetime: float = (
                        floc_record["end_time"] - floc_record["start_time"]
                    )
                    mean_floc_location: float = (
                        floc_record["y_start"] + floc_record["y_end"]
                    ) / 2
                    mean_floc_location = 1 - abs(1 - mean_floc_location)
                    if (
                        mean_floc_location * self.slope_ + self.intersect_
                        < floc_lifetime
                    ):
                        y_breakup.append(floc_record["y_end"])
                        mass.append(floc_record["size"])

            y_breakup_arr: np.ndarray = np.squeeze(np.asarray(y_breakup))
            mass_arr: np.ndarray = np.squeeze(np.asarray(mass))
            return y_breakup_arr, mass_arr
        else:
            raise ValueError(
                f"must pass a family tree dict into {self.name_}, not a h5fp.File object"
            )


class AccessFlocFormationLocationAdvanced(AccessorWithMass):
    name_: str = "AccessFlocFormationLocationAdvanced"
    t_steady_: float
    slope_: float
    intersect_: float

    def __init__(self, t_steady: float, slope: float, intersect: float) -> None:
        self.t_steady_ = t_steady
        self.slope_ = slope
        self.intersect_ = intersect

    def __call__(
        self, f: h5py.File | dict, mask: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(f, dict):
            y_formation: list = []
            mass: list = []
            family_tree_dict: FamilyTreeType = f
            for floc_id, floc_record in tqdm(
                family_tree_dict.items(),
                desc=f"Collecting floc breakup data",
                total=len(family_tree_dict),
                unit="Flocs",
            ):
                if (
                    len(floc_record["parents"]) > 1
                    and floc_record["start_time"] >= self.t_steady_
                ):
                    floc_lifetime: float = (
                        floc_record["end_time"] - floc_record["start_time"]
                    )
                    mean_floc_location: float = (
                        floc_record["y_start"] + floc_record["y_end"]
                    ) / 2
                    mean_floc_location = 1 - abs(1 - mean_floc_location)
                    if (
                        mean_floc_location * self.slope_ + self.intersect_
                        < floc_lifetime
                    ):
                        y_formation.append(floc_record["y_start"])
                        mass.append(floc_record["size"])

            y_formation_arr: np.ndarray = np.squeeze(np.asarray(y_formation))
            mass_arr: np.ndarray = np.squeeze(np.asarray(mass))
            return y_formation_arr, mass_arr
        else:
            raise ValueError(
                f"must pass a family tree dict into {self.name_}, not a h5fp.File object"
            )


def calc_famtree_pdf_steadystate(
    pickle_dir: Path,
    metadata_file: Path,
    fields: list[str],
    bin_widths: dict[str, float],
    U_mean: float,
    L: float,
    d_p: float,
    filter_t_min: float | None,
) -> dict[str, dict[str, np.ndarray | dict[str, np.ndarray]]]:

    metadata_dict: dict[str, dict[str, float | int | str]] = metadata.read_metadata(
        metadata_file
    )
    t_steady: int = int(metadata_dict["Time"]["t_steady"])

    files: list[Path] = [pickle_dir / "family_tree.pkl"]

    poisseulle_u = lambda y: 3 / 2 * U_mean * (1 - ((y - L) / L) ** 2)
    poisseulle_du_dy = lambda y: -3 * U_mean * (y - L) / L**2
    max_poisseulle_du_dy = 3 * U_mean / L

    min_floc_lifetime = 2 * d_p / (d_p * max_poisseulle_du_dy)
    min_floc_lifetime *= 4
    if filter_t_min is not None:
        min_floc_lifetime = filter_t_min
    print(f"Used minimum floc lifetime t_min= {min_floc_lifetime}")

    field_accessors: dict

    # slope = 22.752  # max
    # intersect = 1.013
    slope = 7.913  # 3 sigma
    intersect = 0.025
    if min_floc_lifetime < 0:
        field_accessors = {
            "breakup": AccessFlocBreakupLocationAdvanced(t_steady, slope, intersect),
            "formation": AccessFlocFormationLocationAdvanced(t_steady, slope, intersect),
        }
    else:
        field_accessors = {
            "breakup": AccessFlocBreakupLocation(t_steady, min_floc_lifetime),
            "formation": AccessFlocFormationLocation(t_steady, min_floc_lifetime),
        }

    filter_predicate = NoFilterPredicate()

    results: dict[str, dict[str, np.ndarray | dict[str, np.ndarray]]] = {}
    for field in fields:

        results[field] = calc_PDF(
            files,
            bin_widths[field],
            field,
            field_accessors[field],
            filter_predicate,
            mass_weighted=False,
            file_type="pkl",
        )
    return results


def average_floc_lifetime(
    pickle_dir: Path,
    metadata_file: Path,
    bin_width: float,
    U_mean: float,
    L: float,
    d_p: float,
) -> dict[str, np.ndarray]:

    metadata_dict: dict[str, dict[str, float | int | str]] = metadata.read_metadata(
        metadata_file
    )
    t_steady: int = int(metadata_dict["Time"]["t_steady"])

    poisseulle_u = lambda y: 3 / 2 * U_mean * (1 - ((y - L) / L) ** 2)
    poisseulle_du_dy = lambda y: -3 * U_mean * (y - L) / L**2
    max_poisseulle_du_dy = 3 * U_mean / L

    min_floc_lifetime = 2 * d_p / (d_p * max_poisseulle_du_dy)
    min_floc_lifetime *= 4

    locations: list = []
    lifetimes: list = []

    with open(pickle_dir / "family_tree.pkl", "rb") as file:
        f = pickle.load(file)
        family_tree_dict: FamilyTreeType = f
        for floc_id, floc_record in tqdm(
            family_tree_dict.items(),
            desc=f"Collecting floc breakup data",
            total=len(family_tree_dict),
            unit="Flocs",
        ):
            if (
                len(floc_record["children"]) > 1 or len(floc_record["parents"]) > 1
            ) and floc_record["start_time"] >= t_steady:
                locations.append((floc_record["y_start"] + floc_record["y_end"]) / 2)
                lifetimes.append(floc_record["end_time"] - floc_record["start_time"])

    locations_arr = np.squeeze(np.asarray(locations))
    lifetimes_arr = np.squeeze(np.asarray(lifetimes))

    edges, centers = get_hist_bins([locations_arr], bin_width)

    means, _, _ = binned_statistic(
        locations_arr, lifetimes_arr, statistic="mean", bins=edges
    )
    means_y, _, _ = binned_statistic(
        locations_arr, locations_arr, statistic="mean", bins=edges
    )
    medians, _, _ = binned_statistic(
        locations_arr, lifetimes_arr, statistic="median", bins=edges
    )
    stds, _, _ = binned_statistic(
        locations_arr, lifetimes_arr, statistic="std", bins=edges
    )
    maxs, _, _ = binned_statistic(
        locations_arr, lifetimes_arr, statistic="max", bins=edges
    )

    results: dict[str, np.ndarray] = {
        "y_mean": means_y,
        "y_edges": edges,
        "mean": means,
        "median": medians,
        "std": stds,
        "max": maxs,
    }

    return results
