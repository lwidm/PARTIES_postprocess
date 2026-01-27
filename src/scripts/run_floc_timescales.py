from pathlib import Path
from src.flocs import family_tree
import numpy as np
from src import myio
from src import globals


def _compute_single_pdf(
    data_dir: Path, out_dir: Path, U_mean: float, L: float, d_p: float, filter_t_min: float, name: str | None
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
            bin_widths={"breakup": 0.04, "formation": 0.08},
            U_mean=U_mean,
            L=L,
            d_p=d_p,
            filter_t_min=filter_t_min
        )
    )

    if name is None:
        name = str(filter_t_min)
    subdir: str = "floc_timescale"

    (out_dir / subdir).mkdir(exist_ok=True)
    pdf_stats_out = myio.output.prepare_dict_for_h5save(pdf_stats)  # type: ignore
    myio.output.save_to_h5(out_dir / subdir / f"t_min={name}.h5", pdf_stats_out)
    myio.lwidmer.save_floc_breakup_formation_pdf(out_dir / subdir, "breakup", pdf_stats, True, f"breakup_t_min={name}.csv")
    myio.lwidmer.save_floc_breakup_formation_pdf(out_dir / subdir, "formation", pdf_stats, True, f"formation_t_min={name}.csv")


def main():
    data_names: list[str] = globals.data_names

    data_dir: Path = globals.data_dir
    output_dir: Path = globals.output_dir

    U_mean: list[float] = [1.0 for _ in data_names]
    L: list[float] = [1.0 for _ in data_names]
    d_p: list[float] = [0.03225806 for _ in data_names]



    for i, data_name in enumerate(data_names):
        poisseulle_u = lambda y: 3 / 2 * U_mean[i] * (1 - ((y - L) / L) ** 2)
        poisseulle_du_dy = lambda y: -3 * U_mean[i] * (y - L[i]) / L[i]**2
        max_poisseulle_du_dy = 3 * U_mean[i] / L[i]

        min_floc_lifetime = 2*d_p[i] / (d_p[i] * max_poisseulle_du_dy)
        min_floc_lifetime *= 4

        filter_t_min_list: np.ndarray = np.linspace(0.0, min_floc_lifetime * 2, 10)

        for filter_t_min in filter_t_min_list:
            _compute_single_pdf(
                data_dir /data_name, output_dir / data_name, U_mean[i], L[i], d_p[i], filter_t_min, None
            )
        _compute_single_pdf(
                data_dir / data_name, output_dir / data_name, U_mean[i], L[i], d_p[i], min_floc_lifetime, "poisseulle"
            )



if __name__ == "__name__":
    main()
