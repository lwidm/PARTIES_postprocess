# flocParties/plot_csv.py
import numpy as np
from pathlib import Path
from typing import Literal

from . import utils
from . import output


def save_fluid_mean_data(
    flow_stats_dir: Path,
    output_dir: Path | None,
) -> dict[str, np.ndarray]:

    if not flow_stats_dir.is_dir():
        raise ValueError(f'ERROR: Directory "{flow_stats_dir}" does not exist!')
    flow_stats_path: Path = flow_stats_dir / "flow_stats.h5"
    stats: dict[str, np.ndarray | dict] = utils.read_from_h5(
        flow_stats_path, ["yc", "yv", "U", "V", "W"]
    )

    yc: np.ndarray = stats["yc"]  # type: ignore
    yv: np.ndarray = stats["yv"]  # type: ignore
    U: np.ndarray = stats["U"]  # type: ignore
    V: np.ndarray = stats["V"]  # type: ignore
    W: np.ndarray = stats["W"]  # type: ignore

    header_lines = [
        "==================================================================",
        "Mean flow data in inner units",
        "==================================================================",
        "",
        "Column descriptions:",
        "- yc: Wall-normal coordinates at cell centers",
        "- U: mean velocity in U direction (cell centered coord)",
        "- W: mean velocity in w direction (cell centered coord)",
        "- yv: Wall-normal coordinates at face centers",
        "- V: mean velocity in x direction (face centered coord)",
        "",
    ]

    result: dict[str, np.ndarray] = {
        "yc": yc,
        "U": U,
        "W": W,
        "yv": yv,
        "V": V,
    }

    if output_dir is not None:
        output_dir.mkdir(exist_ok=True)
        csv_file: Path = output_dir / "flow_mean_data.csv"
        output.save_to_csv(csv_file, result, header_lines)

    return result


def save_fluid_fluc_data(
    flow_stats_dir: Path,
    output_dir: Path | None,
) -> dict[str, np.ndarray]:

    if not flow_stats_dir.is_dir():
        raise ValueError(f'ERROR: Directory "{flow_stats_dir}" does not exist!')
    flow_stats_path: Path = flow_stats_dir / "flow_stats.h5"
    stats: dict[str, np.ndarray | dict] = utils.read_from_h5(
        flow_stats_path,
        ["yc", "yv", "uu", "vv", "ww", "k", "uv"],
    )

    yc: np.ndarray = stats["yc"]  # type: ignore
    yv: np.ndarray = stats["yv"]  # type: ignore
    uu: np.ndarray = stats["uu"]  # type: ignore
    vv: np.ndarray = stats["vv"]  # type: ignore
    ww: np.ndarray = stats["ww"]  # type: ignore
    uv: np.ndarray = stats["uv"]  # type: ignore
    k: np.ndarray = stats["k"]  # type: ignore

    header_lines = [
        "==================================================================",
        "Flucuationand data and Reynolds stresses in inner units",
        "==================================================================",
        "",
        "Column descriptions:",
        "- yc: Wall-normal coordinates at cell centers",
        "- u'u': Streamwise stress <u' u'> at cell centers",
        "- w'w': Spanwise stress <w' w'> at cell centers",
        "- u'v': Cross stress <u' v'> at cell centers",
        "- k: TKE k = 2/3 * (<u' u'> + <v' v'> + <w' w'> ) at cell centers",
        "- yv: Wall-normal coordinates at cell faces",
        "- v'v': Wall-normal stress <v' v'> at cell faces",
        "",
        "All quantities in wall units / inner units (normalised by u_tau and Re_tau)",
    ]

    result: dict[str, np.ndarray] = {
        "yc": yc,
        "u'u'": uu,
        "w'w'": ww,
        "u'v'": uv,
        "k": k,
        "yv": yv,
        "v'v'": vv,
    }

    if output_dir is not None:
        output_dir.mkdir(exist_ok=True)
        csv_file: Path = output_dir / "flow_fluctuation_data.csv"
        output.save_to_csv(csv_file, result, header_lines)

    return result


def save_fluid_mean_data_inner_units(
    flow_stats_dir: Path,
    output_dir: Path | None,
) -> dict[str, np.ndarray]:

    if not flow_stats_dir.is_dir():
        raise ValueError(f'ERROR: Directory "{flow_stats_dir}" does not exist!')
    flow_stats_path: Path = flow_stats_dir / "flow_stats.h5"
    stats: dict[str, np.ndarray | dict] = utils.read_from_h5(
        flow_stats_path, ["yc_plus", "yv_plus", "U_plus", "V_plus", "W_plus"]
    )

    yc_plus: np.ndarray = stats["yc_plus"]  # type: ignore
    yv_plus: np.ndarray = stats["yv_plus"]  # type: ignore
    U_plus: np.ndarray = stats["U_plus"]  # type: ignore
    V_plus: np.ndarray = stats["V_plus"]  # type: ignore
    W_plus: np.ndarray = stats["W_plus"]  # type: ignore

    header_lines = [
        "==================================================================",
        "Mean flow data in inner units",
        "==================================================================",
        "",
        "Column descriptions:",
        "- yc^+: Wall-normal coordinates at cell centers",
        "- U: mean velocity in U direction (cell centered coord)",
        "- W: mean velocity in w direction (cell centered coord)",
        "- yv^+: Wall-normal coordinates at face centers",
        "- V: mean velocity in x direction (face centered coord)",
        "",
        "All quantities in wall units / inner units (normalised by u_tau and Re_tau)",
    ]

    result: dict[str, np.ndarray] = {
        "yc^+": yc_plus,
        "U": U_plus,
        "W": W_plus,
        "yv^+": yv_plus,
        "V": V_plus,
    }

    if output_dir is not None:
        output_dir.mkdir(exist_ok=True)
        csv_file: Path = output_dir / "flow_mean_data_inner.csv"
        output.save_to_csv(csv_file, result, header_lines)

    return result


def save_fluid_fluc_data_inner_units(
    flow_stats_dir: Path,
    output_dir: Path | None,
) -> dict[str, np.ndarray]:

    if not flow_stats_dir.is_dir():
        raise ValueError(f'ERROR: Directory "{flow_stats_dir}" does not exist!')
    flow_stats_path: Path = flow_stats_dir / "flow_stats.h5"
    stats: dict[str, np.ndarray | dict] = utils.read_from_h5(
        flow_stats_path,
        ["yc_plus", "yv_plus", "uu_plus", "vv_plus", "ww_plus", "k_plus", "uv_plus"],
    )

    yc_plus: np.ndarray = stats["yc_plus"]  # type: ignore
    yv_plus: np.ndarray = stats["yv_plus"]  # type: ignore
    uu_plus: np.ndarray = stats["uu_plus"]  # type: ignore
    vv_plus: np.ndarray = stats["vv_plus"]  # type: ignore
    ww_plus: np.ndarray = stats["ww_plus"]  # type: ignore
    uv_plus: np.ndarray = stats["uv_plus"]  # type: ignore
    k_plus: np.ndarray = stats["k_plus"]  # type: ignore

    header_lines = [
        "==================================================================",
        "Flucuationand data and Reynolds stresses in inner units",
        "==================================================================",
        "",
        "Column descriptions:",
        "- yc^+: Wall-normal coordinates at cell centers",
        "- u'u': Streamwise stress <u' u'>^+ at cell centers",
        "- w'w': Spanwise stress <w' w'>^+ at cell centers",
        "- u'v': Cross stress <u' v'> at cell centers",
        "- k: TKE k^+ = 2/3 * (<u' u'>^+ + <v' v'>^+ + <w' w'>^+ ) at cell centers",
        "- yv^+: Wall-normal coordinates at cell faces",
        "- v'v': Wall-normal stress <v' v'>^+ at cell faces",
        "",
        "All quantities in wall units / inner units (normalised by u_tau and Re_tau)",
    ]

    result: dict[str, np.ndarray] = {
        "yc^+": yc_plus,
        "u'u'": uu_plus,
        "w'w'": ww_plus,
        "u'v'": uv_plus,
        "k": k_plus,
        "yv^+": yv_plus,
        "v'v'": vv_plus,
    }

    if output_dir is not None:
        output_dir.mkdir(exist_ok=True)
        csv_file: Path = output_dir / "flow_fluctuation_data_inner.csv"
        output.save_to_csv(csv_file, result, header_lines)

    return result


def save_particle_eulerian_stats(
    eulerian_stats_dir: Path,
    output_dir: Path | None,
) -> dict[str, np.ndarray]:

    if not eulerian_stats_dir.is_dir():
        raise ValueError(f'ERROR: Directory "{eulerian_stats_dir}" does not exist!')
    flow_stats_path: Path = eulerian_stats_dir / "particle_eulerian_stats.h5"
    stats: dict[str, np.ndarray | dict] = utils.read_from_h5(
        flow_stats_path,
        [
            "y",
            "Phi",
            "U",
            "V",
            "W",
            "UU",
            "VV",
            "WW",
            "UV",
            "u'u'",
            "v'v'",
            "w'w'",
            "u'v'",
        ],
    )

    y: np.ndarray = stats["y"]  # type: ignore
    Phi: np.ndarray = stats["Phi"]  # type: ignore
    U: np.ndarray = stats["U"]  # type: ignore
    V: np.ndarray = stats["V"]  # type: ignore
    W: np.ndarray = stats["W"]  # type: ignore
    UU: np.ndarray = stats["UU"]  # type: ignore
    VV: np.ndarray = stats["VV"]  # type: ignore
    WW: np.ndarray = stats["WW"]  # type: ignore
    UV: np.ndarray = stats["UV"]  # type: ignore
    uu: np.ndarray = stats["u'u'"]  # type: ignore
    vv: np.ndarray = stats["v'v'"]  # type: ignore
    ww: np.ndarray = stats["w'w'"]  # type: ignore
    uv: np.ndarray = stats["u'v'"]  # type: ignore

    header_lines = [
        "==================================================================",
        "Eulerian Particle data obtained by averaging particle data",
        "weighted by the (analytically computed) volume the particles in",
        "each slice.",
        "==================================================================",
        "",
        "Column descriptions:",
        "- y: Wall-normal coordinates at cell centers",
        "- Phi: Particle volume fraciton at cell centers",
        "- U: mean velocity of particle in U direction (cell centered coord)",
        "- V: mean velocity of particle in V direction (cell centered coord)",
        "- W: mean velocity of particle in W direction (cell centered coord)",
        "- UU: <u u> (cell centered coord)",
        "- VV: <v v> (cell centered coord)",
        "- WW: <w w> (cell centered coord)",
        "- UV: <u v> (cell centered coord)",
        "- u'u': Streamwise stress <u' u'>^+ at cell centers",
        "- v'v': Wall-normal stress <v' v'>^+ at cell faces",
        "- w'w': Spanwise stress <w' w'>^+ at cell centers",
        "- u'v': Cross stress <u' v'> at cell centers",
        "",
    ]

    result: dict[str, np.ndarray] = {
        "y": y,
        "Phi": Phi,
        "U": U,
        "V": V,
        "W": W,
        "UU": UU,
        "VV": VV,
        "WW": WW,
        "UV": UV,
        "u'u'": uu,
        "v'v'": vv,
        "w'w'": ww,
        "u'v'": uv,
    }

    if output_dir is not None:
        output_dir.mkdir(exist_ok=True)
        csv_file: Path = output_dir / "particle_eulerian_stats.csv"
        output.save_to_csv(csv_file, result, header_lines)

    return result


def save_vfu_phi_mean(
    eulerian_stats_dir: Path,
    output_dir: Path | None,
    save_err: bool,
) -> dict[str, np.ndarray]:

    if not eulerian_stats_dir.is_dir():
        raise ValueError(f'ERROR: Directory "{eulerian_stats_dir}" does not exist!')
    flow_stats_path: Path = eulerian_stats_dir / "vfu_phi_mean.h5"
    stats: dict[str, np.ndarray | dict] = utils.read_from_h5(
        flow_stats_path, ["yv", "Phi", "Phi_err"]
    )
    stats_norm: dict[str, np.ndarray | dict] = {}
    if save_err:
        stats_norm = utils.read_from_h5(flow_stats_path, ["Phi_norm", "Phi_err_norm"])

    result: dict[str, np.ndarray] = {
        "y^+": stats["yv"],  # type: ignore
        "Phi": stats["Phi"],  # type: ignore
        "Phi_err": stats["Phi_err"],  # type: ignore
    }
    header_lines: list[str] = [
        "==================================================================",
        "Particle volume fraction data"
        "==================================================================",
        "",
        "Column descriptions:",
        "- y^+: Wall-normal coordinates at face centers",
        "- Phi: Particle volume fraciton at face centers",
        "- Phi err: standard deviation of particle volume fraction",
        "",
    ]
    if save_err:
        header_lines += [
            "- Phi norm: Particle volume fraction",
            "            normalised by total particle volume fraction",
            "- Phi err norm: standard deviation of particle volume fraction",
            "                normalised by total particle volume fraction",
        ]

        result["Phi norm"] = stats_norm["Phi_norm"]  # type: ignore
        result["Phi err norm"] = stats_norm["Phi_err_norm"]  # type: ignore

    if output_dir is not None:
        output_dir.mkdir(exist_ok=True)
        csv_file: Path = output_dir / "vfu_phi_mean.csv"
        output.save_to_csv(csv_file, result, header_lines)

    return result


def save_particle_acceleration_pdf(
    output_dir: Path,
    dir: Literal["x", "y", "z"],
    stats: dict,
) -> dict[str, np.ndarray]:

    a: np.ndarray = stats["a_" + dir]["unweighted"]["bin_means"]  # type: ignore
    PDF: np.ndarray = stats["a_" + dir]["unweighted"]["probabs_mean"]  # type: ignore
    err: np.ndarray = stats["a_" + dir]["unweighted"]["probabs_err"]  # type: ignore

    header_lines = [
        "==================================================================",
        f"The distribution of the particle acceleration in {dir} direction",
        "==================================================================",
        "",
        "Column descriptions:",
        f"- a_{dir} / sigma_{dir}: The mean value withing the histogram bin of the particle accelartion devided by the overall acceleration standard deviation",
        f"- PDF: The PDF of a_{dir} / sigma_{dir}",
        "- err: The standard deviation of the PDF value in each bin over all the files devided by the total number of files",
        "",
    ]

    result: dict[str, np.ndarray] = {
        f"a_{dir} / sigma_{dir}": a,
        "PDF": PDF,
        "err": err,
    }

    if output_dir is not None:
        output_dir.mkdir(exist_ok=True)
        csv_file: Path = output_dir / f"particle_acceleration_pdf_{dir}.csv"
        output.save_to_csv(csv_file, result, header_lines)

    return result


def save_particle_up_pdf(
    output_dir: Path,
    yp: float | None,
    stats: dict,
) -> dict[str, np.ndarray]:

    u_p: np.ndarray = stats["u_plus"]["unweighted"]["bin_means"]  # type: ignore
    PDF: np.ndarray = stats["u_plus"]["unweighted"]["probabs_mean"]  # type: ignore
    err: np.ndarray = stats["u_plus"]["unweighted"]["probabs_err"]  # type: ignore

    header_title: str
    if yp is not None:
        header_title = (
            f"The distribution of particle streamwise velocity at roughly y^+={yp}"
        )
    else:
        header_title = f"The distribution of particle streamwise velocity"

    header_lines = [
        "==================================================================",
        header_title,
        "==================================================================",
        "",
        "Column descriptions:",
        f"- u^+: The mean value withing the histogram bin of the particle streamwise velocities",
        f"- PDF: The PDF of u^+",
        "- err: The standard deviation of the PDF value in each bin over all the files devided by the total number of files",
        "",
    ]

    result: dict[str, np.ndarray] = {
        f"u^+": u_p,
        "PDF": PDF,
        "err": err,
    }

    if output_dir is not None:
        output_dir.mkdir(exist_ok=True)
        csv_file: Path
        if yp is not None:
            csv_file = output_dir / f"particle_u_plus_pdf_{yp}.csv"
        else:
            csv_file = output_dir / f"particle_u_plus_pdf.csv"
        output.save_to_csv(csv_file, result, header_lines)

    return result
