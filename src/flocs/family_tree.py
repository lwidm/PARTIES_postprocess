from pathlib import Path
from typing import TypedDict, Literal
import numpy as np
import h5py
from tqdm import tqdm
import pickle
from scipy.stats import binned_statistic

from src.myio import metadata
from src import myio

from src.statistics import (
    AccessorWithMass,
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

    def __call__(self, f: h5py.File | dict, _) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(f, dict):
            y_breakup: list = []
            mass: list = []
            family_tree_dict: FamilyTreeType = f
            for _, floc_record in tqdm(
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

    def __call__(self, f: h5py.File | dict, _) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(f, dict):
            y_breakup: list = []
            mass: list = []
            family_tree_dict: FamilyTreeType = f
            for floc_record in tqdm(
                family_tree_dict.values(),
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

    def __call__(self, f: h5py.File | dict, _) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(f, dict):
            y_breakup: list = []
            mass: list = []
            family_tree_dict: FamilyTreeType = f
            for floc_record in tqdm(
                family_tree_dict.values(),
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

    def __call__(self, f: h5py.File | dict, _) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(f, dict):
            y_formation: list = []
            mass: list = []
            family_tree_dict: FamilyTreeType = f
            for floc_record in tqdm(
                family_tree_dict.values(),
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

    # poisseulle_u = lambda y: 3 / 2 * U_mean * (1 - ((y - L) / L) ** 2)
    # poisseulle_du_dy = lambda y: -3 * U_mean * (y - L) / L**2
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
            "formation": AccessFlocFormationLocationAdvanced(
                t_steady, slope, intersect
            ),
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

    # poisseulle_u = lambda y: 3 / 2 * U_mean * (1 - ((y - L) / L) ** 2)
    # poisseulle_du_dy = lambda y: -3 * U_mean * (y - L) / L**2
    max_poisseulle_du_dy = 3 * U_mean / L

    min_floc_lifetime = 2 * d_p / (d_p * max_poisseulle_du_dy)
    min_floc_lifetime *= 4

    locations: list = []
    lifetimes: list = []

    with open(pickle_dir / "family_tree.pkl", "rb") as file:
        f = pickle.load(file)
        family_tree_dict: FamilyTreeType = f
        for floc_record in tqdm(
            family_tree_dict.values(),
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

    edges, _ = get_hist_bins([locations_arr], bin_width)

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


def _get_bin_size(
    size: float, size_list_centers: list[float], size_list_edges: list[float]
) -> tuple[int, float]:
    idx: int = int(np.searchsorted(size_list_edges, size) - 1)
    idx = int(np.clip(idx, 0, len(size_list_centers) - 1))
    size_binned: float = float(size_list_centers[idx])
    return idx, size_binned


# def _get_bin_size_floc_record(
#     floc_record,
#     size_type: Literal["n_p", "D_f", "D_g"],
#     size_list_centers: list[float],
#     size_list_edges: list[float],
# ) -> tuple[int, float]:
#     size: float
#     match size_type:
#         case "n_p":
#             size = float(floc_record["size"])
#         case "D_f":
#             raise NotImplementedError(
#                 "Number Density evolution parameters only works for n_p"
#             )
#         case "D_g":
#             raise NotImplementedError(
#                 "Number Density evolution parameters only works for n_p"
#             )
#     return _get_bin_size(size, size_list_centers, size_list_edges)


def _get_bin_size_floc_file(
    f: h5py.File | h5py.Group,
    size_type: Literal["n_p", "D_f", "D_g"],
    size_list_centers: list[float],
    size_list_edges: list[float],
) -> tuple[list[int], list[float]]:
    size_arr: np.ndarray
    match size_type:
        case "n_p":
            size_arr = f["n_p"][:]  # type: ignore
        case "D_f":
            size_arr = f["D_f"][:]  # type: ignore
        case "D_g":
            size_arr = f["D_g"][:]  # type: ignore
    bin_idx_list: list[int] = []
    binned_size_list: list[float] = []
    for i in range(len(size_arr)):
        idx, binned_size = _get_bin_size(
            size_arr[i], size_list_centers, size_list_edges
        )
        bin_idx_list.append(idx)
        binned_size_list.append(binned_size)

    return bin_idx_list, binned_size_list


def compute_number_density_evolutions_params(
    pickle_dir: Path,
    metadata_file: Path,
    data_dir: Path,
    trn: bool,
    size_type: Literal["n_p", "D_f", "D_g"],
    bin_width: float,
    U_mean: float,
    L: float,
    d_p: float,
) -> dict[str, dict]:

    metadata_dict: dict[str, dict[str, int | float | str]] = metadata.read_metadata(
        metadata_file
    )
    xmin: float = float(metadata_dict["Domain"]["xmin"])
    xmax: float = float(metadata_dict["Domain"]["xmax"])
    ymin: float = float(metadata_dict["Domain"]["ymin"])
    ymax: float = float(metadata_dict["Domain"]["ymax"])
    zmin: float = float(metadata_dict["Domain"]["zmin"])
    zmax: float = float(metadata_dict["Domain"]["zmax"])
    t_steady: float = float(metadata_dict["Time"]["t_steady"])
    t_end: float = float(metadata_dict["Time"]["t_end"])
    delta_t: float = t_end - t_steady

    V: float = (xmax - xmin) * (ymax - ymin) * (zmax - zmin)

    # poisseulle_u = lambda y: 3 / 2 * U_mean * (1 - ((y - L) / L) ** 2)
    # poisseulle_du_dy = lambda y: -3 * U_mean * (y - L) / L**2
    max_poisseulle_du_dy = 3 * U_mean / L

    min_floc_lifetime = 2 * d_p / (d_p * max_poisseulle_du_dy)
    min_floc_lifetime *= 4
    min_floc_lifetime *= 0

    C_count: dict[tuple[int, int], int]
    F_count: dict[int, int]
    nu_count: dict[int, int]
    p_count: dict[tuple[int, int], int]

    size_list_centers: list[float]
    size_list_edges: list[float]
    size_list_centers_idx: list[int]
    size_list_edges_idx: list[int]

    with open(pickle_dir / "family_tree.pkl", "rb") as file:
        fam_tree: FamilyTreeType = pickle.load(file)

        size_max: float = 0.0
        size_min: float = np.inf
        for floc_record in fam_tree.values():
            match size_type:
                case "n_p":
                    if size_max < floc_record["size"]:
                        size_max = floc_record["size"]
                    if size_min > floc_record["size"]:
                        size_min = floc_record["size"]
                case "D_f":
                    raise NotImplementedError(
                        "Number Density evolution parameters only works for n_p"
                    )
                case "D_g":
                    raise NotImplementedError(
                        "Number Density evolution parameters only works for n_p"
                    )

        num_bins: int = int((size_max - size_min) / bin_width)
        bin_width_eff: float = (size_max - size_min) / num_bins
        size_list_centers = np.linspace(
            size_min, size_max, num_bins + 1, dtype=float
        ).tolist()
        size_list_edges = np.linspace(
            size_min - bin_width_eff / 2,
            size_max + bin_width_eff / 2,
            num_bins + 2,
            dtype=float,
        ).tolist()
        size_list_centers_idx = [i for i in range(len(size_list_centers))]
        size_list_edges_idx = [i for i in range(len(size_list_edges))]

        C_count = {
            (i, j): 0 for i in size_list_centers_idx for j in size_list_centers_idx
        }
        F_count = {i: 0 for i in size_list_centers_idx}
        nu_count = {i: 0 for i in size_list_centers_idx}
        p_count = {
            (i, j): 0 for i in size_list_centers_idx for j in size_list_centers_idx
        }

        for floc_record in tqdm(
            fam_tree.values(),
            desc=f"Collecting floc data for K",
            total=len(fam_tree),
            unit=" floc_record",
        ):
            if (
                len(floc_record["parents"]) == 2
                and floc_record["start_time"] >= t_steady
                and floc_record["end_time"] - floc_record["start_time"]
                >= min_floc_lifetime
            ):
                parent_bin_idx_1, _ = _get_bin_size(
                    floc_record["parents_sizes"][0], size_list_centers, size_list_edges
                )
                parent_bin_idx_2, _ = _get_bin_size(
                    floc_record["parents_sizes"][1], size_list_centers, size_list_edges
                )

                C_count[(parent_bin_idx_1, parent_bin_idx_2)] += 1
                C_count[(parent_bin_idx_2, parent_bin_idx_1)] += 1

        for floc_record in tqdm(
            fam_tree.values(),
            desc=f"Collecting floc data for F, nu and p",
            total=len(fam_tree),
            unit=" floc_record",
        ):
            if (
                len(floc_record["children"]) == 2
                and floc_record["start_time"] >= t_steady
                and floc_record["end_time"] - floc_record["start_time"]
                >= min_floc_lifetime
            ):
                size: float = floc_record["size"]
                parent_bin_idx: int
                child_bin_idx_1: int
                child_bin_idx_2: int
                parent_bin_idx, _ = _get_bin_size(
                    size, size_list_centers, size_list_edges
                )
                child_bin_idx_1, _ = _get_bin_size(
                    floc_record["children_sizes"][0], size_list_centers, size_list_edges
                )
                child_bin_idx_2, _ = _get_bin_size(
                    floc_record["children_sizes"][1], size_list_centers, size_list_edges
                )

                F_count[parent_bin_idx] += 1
                p_count[(child_bin_idx_1, parent_bin_idx)] += 1
                p_count[(child_bin_idx_2, parent_bin_idx)] += 1

                nu_count[parent_bin_idx] += 2

    floc_files: list[Path] = myio.utils.get_steadystate_floc_files(
        data_dir / "flocs", metadata_file, trn
    )
    num_files = len(floc_files)

    concentration_count_files: list[dict[int, float]] = []

    for i, floc_file in tqdm(
        enumerate(floc_files),
        desc=f"Computing floc size concentrations",
        total=num_files,
        unit=" floc files",
    ):
        with h5py.File(str(floc_file), "r") as f:
            concentration_count_files.append({c: 0.0 for c in size_list_centers_idx})
            binned_idx_arr: list[int]
            binned_idx_arr, _ = _get_bin_size_floc_file(
                f, size_type, size_list_centers, size_list_edges
            )
            for bin_idx in binned_idx_arr:
                concentration_count_files[i][bin_idx] += 1

    c: dict[int, float] = {i: 0.0 for i in size_list_centers_idx}
    for bin_idx in size_list_centers_idx:
        for file_idx in range(num_files):
            c[bin_idx] += concentration_count_files[file_idx][bin_idx]
        c[bin_idx] /= num_files * V

    K: dict[tuple[int, int], float] = {}
    for i in size_list_centers_idx:
        for j in size_list_centers_idx:
            if c[i] == 0 or c[j] == 0:
                K[(i, j)] = np.nan
                continue
            K[(i, j)] = float(C_count[(i, j)]) / (c[i] * c[j] * V * delta_t)

    F: dict[int, float] = {}
    for i in size_list_centers_idx:
        if c[i] == 0:
            F[i] = np.nan
            continue
        F[i] = float(F_count[i]) / (c[i] * V * delta_t)

    nu: dict[int, float] = {
        i: (float(nu_count[i]) / float(F_count[i]) if F_count[i] != 0 else np.nan)
        for i in size_list_centers_idx
    }

    p: dict[tuple[int, int], float] = {}
    for i in size_list_centers_idx:
        for j in size_list_centers_idx:
            if F_count[j] == 0:
                p[(i, j)] = np.nan
                continue
            p[(i, j)] = float(p_count[(i, j)]) / (
                float(F_count[j]) * nu[j] * bin_width_eff
            )

    bin_info: dict[str, list[int] | list[float]] = {
        "center_sizes": size_list_centers,
        "center_idxs": size_list_centers_idx,
        "edge_sizes": size_list_edges,
        "edge_idxs": size_list_edges_idx,
    }

    return {"K": K, "F": F, "nu": nu, "p": p, "bin_info": bin_info}
