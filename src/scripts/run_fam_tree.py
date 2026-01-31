from pathlib import Path
from src.flocs import family_tree
import numpy as np
from src import myio
from src import globals
from typing import Literal


def _compute_single_pdf(
    data_dir: Path, out_dir: Path, U_mean: float, L: float, d_p: float
):
    metadata_file: Path = data_dir / "metadata.ini"
    if not data_dir.exists():
        raise ValueError(f'pickle dir ("{data_dir}") does not exsist!')
    if not metadata_file.exists():
        raise ValueError(f'metadata_file ("{metadata_file}") does not exsist!')
    pdf_stats: dict[str, dict[str, np.ndarray | dict[str, np.ndarray]]] = (
        family_tree.calc_famtree_pdf_steadystate(
            pickle_dir=data_dir,
            metadata_file=metadata_file,
            fields=["breakup", "formation"],
            bin_widths={"breakup": 0.04, "formation": 0.04},
            U_mean=U_mean,
            L=L,
            d_p=d_p,
            filter_t_min=None,
        )
    )

    pdf_stats_out = myio.output.prepare_dict_for_h5save(pdf_stats)  # type: ignore
    myio.output.save_to_h5(
        out_dir / "floc_breakup_formation_filtered_pdf.h5", pdf_stats_out
    )
    myio.lwidmer.save_floc_breakup_formation_pdf(
        out_dir, "breakup", pdf_stats, True, None
    )
    myio.lwidmer.save_floc_breakup_formation_pdf(
        out_dir, "formation", pdf_stats, True, None
    )

    pdf_stats: dict[str, dict[str, np.ndarray | dict[str, np.ndarray]]] = (
        family_tree.calc_famtree_pdf_steadystate(
            pickle_dir=data_dir,
            metadata_file=metadata_file,
            fields=["breakup", "formation"],
            bin_widths={"breakup": 0.04, "formation": 0.04},
            U_mean=U_mean,
            L=L,
            d_p=d_p,
            filter_t_min=0.0,
        )
    )

    pdf_stats_out = myio.output.prepare_dict_for_h5save(pdf_stats)  # type: ignore
    myio.output.save_to_h5(
        out_dir / "floc_breakup_formation_unfiltered_pdf.h5", pdf_stats_out
    )
    myio.lwidmer.save_floc_breakup_formation_pdf(
        out_dir, "breakup", pdf_stats, False, None
    )
    myio.lwidmer.save_floc_breakup_formation_pdf(
        out_dir, "formation", pdf_stats, False, None
    )

    pdf_stats: dict[str, dict[str, np.ndarray | dict[str, np.ndarray]]] = (
        family_tree.calc_famtree_pdf_steadystate(
            pickle_dir=data_dir,
            metadata_file=metadata_file,
            fields=["breakup", "formation"],
            bin_widths={"breakup": 0.04, "formation": 0.04},
            U_mean=U_mean,
            L=L,
            d_p=d_p,
            filter_t_min=-1,
        )
    )

    pdf_stats_out = myio.output.prepare_dict_for_h5save(pdf_stats)  # type: ignore
    myio.output.save_to_h5(
        out_dir / "floc_breakup_formation_advanced_filtered_pdf.h5", pdf_stats_out
    )
    myio.lwidmer.save_floc_breakup_formation_pdf(
        out_dir, "breakup", pdf_stats, False, f"floc_breakup_advanced_filtered_pdf.csv"
    )
    myio.lwidmer.save_floc_breakup_formation_pdf(
        out_dir,
        "formation",
        pdf_stats,
        False,
        f"floc_formation_advanced_filtered_pdf.csv",
    )


def main():
    data_names: list[str] = globals.data_names
    trn: list[bool] = globals.has_trn_data
    U_mean: list[float] = [1.0 for _ in data_names]
    L: list[float] = [1.0 for _ in data_names]
    d_p: list[float] = [0.06451612 for _ in data_names]

    data_dir: Path = globals.data_dir
    output_dir: Path = globals.output_dir

    bin_width: float | None = None
    num_bins_list: list[int] = globals.balance_equation_bins
    log_bins: bool = True
    filter_bounce: bool = True
    filter_sparse_bins: int = 30
    nonbinary_treatement: Literal["discount", "as_binary", "corrected"] = "discount"
    size_lim: tuple[float | None, float | None] = (1, None)

    for i, data_name in enumerate(data_names):
        dataset_dir: Path = data_dir / data_name
        out_dataset_dir: Path = output_dir / data_name
        # _compute_single_pdf(dataset_dir, out_dataset_dir, U_mean[i], L[i], d_p[i])

        # result_corrected: dict[str, dict] = (
        #     family_tree.compute_number_density_evolutions_params(
        #         dataset_dir,
        #         dataset_dir / "metadata.ini",
        #         dataset_dir,
        #         trn[i],
        #         "n_p",
        #         _num_bins=num_bins_list[i],
        #         bin_width=bin_width,
        #         log_bins=log_bins,
        #         size_lim=size_lim,
        #         U_mean=U_mean[i],
        #         L=L[i],
        #         d_p=d_p[i],
        #         corrected=True,
        #         filter_bounce=filter_bounce,
        #         filter_sparse_bins=filter_sparse_bins,
        #         nonbinary_treatement=nonbinary_treatement,
        #     )
        # )
        # myio.output.save_to_pickle(
        #     out_dataset_dir / "number_density_evolution_params_corrected.pkl",
        #     result_corrected,
        # )

        result_uncorrected: dict[str, dict] = (
            family_tree.compute_number_density_evolutions_params(
                dataset_dir,
                dataset_dir / "metadata.ini",
                dataset_dir,
                trn[i],
                "n_p",
                _num_bins=num_bins_list[i],
                bin_width=bin_width,
                log_bins=log_bins,
                size_lim=size_lim,
                U_mean=U_mean[i],
                L=L[i],
                d_p=d_p[i],
                corrected=False,
                filter_bounce=filter_bounce,
                filter_sparse_bins=filter_sparse_bins,
                nonbinary_treatement=nonbinary_treatement,
            )
        )
        myio.output.save_to_pickle(
            out_dataset_dir / "number_density_evolution_params_uncorrected.pkl",
            result_uncorrected,
        )

        # result_diff: dict[str, dict] = {}
        # for key in result_corrected.keys():
        #     if key == "bin_info":
        #         result_diff[key] = result_corrected[
        #             key
        #         ]  # bin_info is the same for both
        #     else:
        #         result_diff[key] = {}
        #         for subkey in result_corrected[key].keys():
        #             if isinstance(result_corrected[key][subkey], (int, float)):
        #                 result_diff[key][subkey] = (
        #                     result_corrected[key][subkey]
        #                     - result_uncorrected[key][subkey]
        #                 )
        #             else:
        #                 result_diff[key][subkey] = result_corrected[key][
        #                     subkey
        #                 ]  # Can't subtract non-numeric types
        # myio.output.save_to_pickle(
        #     out_dataset_dir / "number_density_evolution_params_diff.pkl", result_diff
        # )
        #
        # # Compute flocculation balances for corrected version
        # floc_balance_corrected: dict = family_tree.compute_floculation_balances(
        #     params=result_corrected
        # )
        # myio.lwidmer.save_floculation_balance(
        #     out_dataset_dir, floc_balance_corrected, corrected=True
        # )
        #
        # # Compute flocculation balances for uncorrected version
        # floc_balance_uncorrected: dict = family_tree.compute_floculation_balances(
        #     params=result_uncorrected
        # )
        # myio.lwidmer.save_floculation_balance(
        #     out_dataset_dir, floc_balance_uncorrected, corrected=False
        # )
        #
        # # Compute and save difference in flocculation balances
        # floc_balance_diff: dict[str, np.ndarray] = {}
        # for key in floc_balance_corrected.keys():
        #     if key == "center_sizes_arr" or key == "edge_sizes_arr":
        #         floc_balance_diff[key] = floc_balance_corrected[
        #             key
        #         ]  # Size arrays are the same
        #     else:
        #         floc_balance_diff[key] = (
        #             floc_balance_corrected[key] - floc_balance_uncorrected[key]
        #         )
        # myio.lwidmer.save_floculation_balance(
        #     out_dataset_dir, floc_balance_diff, is_difference=True
        # )


if __name__ == "__name__":
    main()
