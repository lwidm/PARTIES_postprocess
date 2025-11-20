from pathlib import Path
from src.flocs import family_tree
import numpy as np
from src import myio


def _compute_single_pdf(data_name: str, out_dir: Path):
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
            bin_widths={"breakup": 0.01, "formation": 0.01},
        )
    )

    pdf_stats_out = myio.output.prepare_dict_for_h5save(pdf_stats)  # type: ignore
    myio.output.save_to_h5(out_dir / "lagrangian_acceleration_pdf.h5", pdf_stats_out)
    myio.lwidmer.save_floc_breakup_formation_pdf(out_dir, "breakup", pdf_stats)
    myio.lwidmer.save_floc_breakup_formation_pdf(out_dir, "formation", pdf_stats)


def main():
    data_names: list[str] = ["phi5p0_noCo", "phi5p0", "phi3p0"]
    for data_name in data_names:
        _compute_single_pdf(data_name, Path("./data") / data_name)


if __name__ == "__name__":
    main()
