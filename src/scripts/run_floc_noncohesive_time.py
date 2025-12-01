from pathlib import Path
from src.flocs import family_tree
import numpy as np
from src import myio


def main():
    data_names: list[str] = [
        "phi5p0_noCo",
        # "phi1p5",
        # "phi3p0",
        # "phi5p0",
    ]
    out_dir: Path = Path("./data")

    U_mean: list[float] = [1.0 for _ in data_names]
    L: list[float] = [1.0 for _ in data_names]
    d_p: list[float] = [0.03225806 for _ in data_names]

    for i, data_name in enumerate(data_names):
        pickle_dir: Path = Path("./data") / data_name
        metadata_file: Path = Path("./data") / data_name / "metadata.ini"
        result = family_tree.average_floc_lifetime(
            pickle_dir=pickle_dir,
            metadata_file=metadata_file,
            bin_width=0.04,
            U_mean=U_mean[i],
            L=L[i],
            d_p=d_p[i],
        )

        myio.lwidmer.save_floc_lifetime_stats(
            output_dir=out_dir / data_name, stats=result
        )


if __name__ == "__name__":
    main()
