import numpy as np
from pathlib import Path
import h5py
import scipy
from typing import Callable, Literal
from tqdm import tqdm
from abc import ABC, abstractmethod
import pickle

from src import myio


class FilterPredicate(ABC):
    name_: str = "FilterPredicate"

    @abstractmethod
    def __call__(self, f: h5py.File | dict) -> np.ndarray | None:
        pass

    def __str__(self) -> str:
        return self.name_


class Accessor(ABC):
    name_: str = "Accessor"

    @abstractmethod
    def __call__(
        self, f: h5py.File | dict, mask: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        pass

    def __str__(self) -> str:
        return self.name_


class AccessorWithMass(Accessor):
    name_: str = "AccessorWithMass"

    @abstractmethod
    def __call__(
        self, f: h5py.File | dict, mask: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray]:
        pass


class AccessorWithoutMass(Accessor):
    name_: str = "AccessorWithoutMass"

    @abstractmethod
    def __call__(
        self, f: h5py.File | dict, mask: np.ndarray | None
    ) -> tuple[np.ndarray, None]:
        pass


class NoFilterPredicate(FilterPredicate):
    name_: str = "NoFilterPredicate"

    def __call__(self, f: h5py.File | dict) -> None:
        return None


def get_hist_bins(
    data: list[np.ndarray], bin_width: float
) -> tuple[np.ndarray, np.ndarray]:
    global_min: float = min(np.min(d) for d in data)
    global_max: float = max(np.max(d) for d in data)
    edges = np.arange(
        global_min - 0.5 * bin_width, global_max + 0.5 * bin_width, bin_width
    )
    centers = 0.5 * (edges[:-1] + edges[1:])
    return edges, centers


def _compute_hist_stats(
    hists: np.ndarray, bin_means: np.ndarray, bin_width: float
) -> dict[str, np.ndarray]:
    results: dict[str, np.ndarray] = {}
    # counts statistics across files
    results["counts_mean"] = np.nanmean(hists, axis=0)
    non_nan_counts: np.ndarray = np.sum(~np.isnan(hists), axis=0)
    results["counts_err"] = np.nanstd(hists, axis=0) / np.sqrt(non_nan_counts)
    results["bin_means"] = np.nanmean(bin_means, axis=0)
    non_nan_means: np.ndarray = np.sum(~np.isnan(bin_means), axis=0)
    results["bin_means_err"] = np.nanstd(bin_means, axis=0) / np.sqrt(non_nan_means)

    # convert counts to probabilities per file (normalize by number of samples in that file)
    counts_per_file = np.array([h.sum() for h in hists], dtype=float)
    probabs = hists / (counts_per_file[:, None]  * bin_width)

    results["probabs_mean"] = np.nanmean(probabs, axis=0)
    non_nan_probabs: np.ndarray = np.sum(~np.isnan(probabs), axis=0)
    results["probabs_err"] = np.nanstd(probabs, axis=0) / np.sqrt(non_nan_probabs)
    return results


def calc_PDF(
    files: list[Path],
    bin_width: float,
    field_name: str,
    field_accessor: Accessor,
    filter_predicate: FilterPredicate,
    mass_weighted: bool,
    file_type: Literal["h5", "pkl"],
) -> dict[str, dict[str, np.ndarray] | np.ndarray]:
    data: list[np.ndarray] = [np.array([]) for _ in range(len(files))]
    mass: list[np.ndarray] = [np.array([]) for _ in range(len(files))]

    for i, particle_file in tqdm(
        enumerate(files),
        desc=f"Computing PDF for field: {field_name}",
        total=len(files),
        unit="Files",
    ):
        if file_type == "h5":
            with h5py.File(str(particle_file), "r") as f:
                mask: np.ndarray | None = filter_predicate(f)
                data_file: np.ndarray
                mass_file: np.ndarray | None
                data_file, mass_file = field_accessor(f, mask)
                data[i] = data_file
                if mass_weighted:
                    if mass_file is None:
                        raise ValueError(
                            f"field_accessor ({str(field_accessor)}) returned None value for mass entry. It cannot be used for mass weigted statistics!"
                        )
                    else:
                        mass[i] = mass_file
        elif file_type == "pkl":
            with open(particle_file, "rb") as file:
                f = pickle.load(file)
                mask: np.ndarray | None = filter_predicate(f)
                data_file: np.ndarray
                mass_file: np.ndarray | None
                data_file, mass_file = field_accessor(f, mask)
                data[i] = data_file
                if mass_weighted:
                    if mass_file is None:
                        raise ValueError(
                            f"field_accessor ({str(field_accessor)}) returned None value for mass entry. It cannot be used for mass weigted statistics!"
                        )
                    else:
                        mass[i] = mass_file

    edges: np.ndarray
    centers: np.ndarray
    edges, centers = get_hist_bins(data, bin_width)

    # compute histogram (counts) for each row (each file / array)
    if mass_weighted:
        mass_weighted_hists = np.vstack(
            [np.histogram(a, weights=m, bins=edges)[0] for a, m in zip(data, mass)]
        )
    else:
        mass_weighted_hists = None
    unweighted_hists = np.vstack([np.histogram(a, bins=edges)[0] for a in data])

    unweighted_binmeans = np.vstack(
        [scipy.stats.binned_statistic(a, a, "mean", edges)[0] for a in data]
    )

    if mass_weighted:
        mass_weighted_binmeans_list: list[np.ndarray] = []
        for a, m in zip(data, mass):
            sum_w = np.histogram(a, bins=edges, weights=m)[0].astype(float)
            sum_wx = np.histogram(a, bins=edges, weights=m * a)[0].astype(float)
            mass_weighted_binmeans_list.append(sum_wx / sum_w)
        mass_weighted_binmeans = np.vstack(mass_weighted_binmeans_list)
    else:
        mass_weighted_binmeans = None

    if mass_weighted_hists is not None and mass_weighted_binmeans is not None:
        weighted_stats = _compute_hist_stats(
            mass_weighted_hists, mass_weighted_binmeans, bin_width
        )
    else:
        weighted_stats = None
    unweighted_stats = _compute_hist_stats(unweighted_hists, unweighted_binmeans, bin_width)

    results: dict[str, dict[str, np.ndarray] | np.ndarray] = {
        "unweighted": unweighted_stats,
        "centers": centers,
        "edges": edges,
    }
    if weighted_stats is not None:
        results.update({"mass_weighted": weighted_stats})
    return results
