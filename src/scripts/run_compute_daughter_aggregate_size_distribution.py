from typing import Literal
from pathlib import Path

from src import globals
from src.flocs import family_tree
from src import myio


def main():
    data_names: list[str] = globals.data_names
    U_mean: list[float] = [1.0 for _ in data_names]
    L: list[float] = [1.0 for _ in data_names]
    d_p: list[float] = [0.06451612 for _ in data_names]

    data_dir: Path = globals.data_dir
    output_dir: Path = globals.output_dir

    num_bins: int = 14
    cluster_param: float = 1e-4
    filter_bounce: bool = True
    filter_sparse_bins: int = 40
    size_lim: tuple[float | None, float | None] = (20, None)

    for i, data_name in enumerate(data_names):
        dataset_dir: Path = data_dir / data_name
        out_dataset_dir: Path = output_dir / data_name

        result_corrected: dict[str, dict] = (
            family_tree.compute_daughter_aggregate_size_distribution(
                pickle_dir=dataset_dir,
                metadata_file=dataset_dir / "metadata.ini",
                num_bins=num_bins,
                cluster_param=cluster_param,
                U_mean=U_mean[i],
                L=L[i],
                d_p=d_p[i],
                corrected=True,
                filter_bounce=filter_bounce,
                filter_sparse_bins=filter_sparse_bins,
                size_lim=size_lim,
            )
        )
        myio.output.save_to_pickle(
            out_dataset_dir / "daughter_aggregate_size_distribution_corrected.pkl",
            result_corrected,
        )

        result_corrected: dict[str, dict] = (
            family_tree.compute_daughter_aggregate_size_distribution(
                pickle_dir=dataset_dir,
                metadata_file=dataset_dir / "metadata.ini",
                num_bins=num_bins,
                cluster_param=cluster_param,
                U_mean=U_mean[i],
                L=L[i],
                d_p=d_p[i],
                corrected=False,
                filter_bounce=filter_bounce,
                filter_sparse_bins=filter_sparse_bins,
                size_lim=size_lim,
            )
        )
        myio.output.save_to_pickle(
            out_dataset_dir / "daughter_aggregate_size_distribution_uncorrected.pkl",
            result_corrected,
        )
