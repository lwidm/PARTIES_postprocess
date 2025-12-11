from pathlib import Path
from src.flocs import family_tree
import numpy as np
from src import myio


def _compute_single_pdf(
    data_name: str, out_dir: Path, U_mean: float, L: float, d_p: float
):
    pickle_dir: Path = Path("./data") / data_name
    metadata_file: Path = Path("./data") / data_name / "metadata.ini"
    if not pickle_dir.exists():
        raise ValueError(f'pickle dir ("{pickle_dir}") does not exsist!')
    if not metadata_file.exists():
        raise ValueError(f'metadata_file ("{metadata_file}") does not exsist!')
    pdf_stats: dict[str, dict[str, np.ndarray | dict[str, np.ndarray]]] = (
        family_tree.calc_famtree_pdf_steadystate(
            pickle_dir=pickle_dir,
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
            pickle_dir=pickle_dir,
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
            pickle_dir=pickle_dir,
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
    data_names: list[str] = [
        "phi5p0_noCo",
        "phi1p5",
        "phi3p0",
        "phi5p0",
    ]
    trn: list[bool] = [
        False,
        False,
        True,
        True,
    ]
    U_mean: list[float] = [1.0 for _ in data_names]
    L: list[float] = [1.0 for _ in data_names]
    d_p: list[float] = [0.03225806 for _ in data_names]
    corrected: bool = True
    for i, data_name in enumerate(data_names):
        data_dir: Path = Path("./data") / data_name
        out_dir: Path = Path("./data") / data_name
        _compute_single_pdf(data_name, data_dir, U_mean[i], L[i], d_p[i])

        result: dict[str, dict] = family_tree.compute_number_density_evolutions_params(
            data_dir,
            data_dir / "metadata.ini",
            data_dir,
            trn[i],
            "n_p",
            bin_width=1,
            U_mean=U_mean[i],
            L=L[i],
            d_p=d_p[i],
            corrected=corrected
        )
        name: str = "number_density_evolution_params.pkl"
        if corrected:
            name = "number_density_evolution_params_corrected.pkl"
        myio.output.save_to_pickle(out_dir / "number_density_evolution_params.pkl", result)
        floc_balance: dict = family_tree.compute_floculation_balances(params=result)
        myio.lwidmer.save_floculation_balance(out_dir, floc_balance, corrected=corrected)


if __name__ == "__name__":
    main()
