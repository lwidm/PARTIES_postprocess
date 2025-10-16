# -- src/plotting/series.py

from pathlib import Path
from typing import Union, Any, Dict, List, Optional, Tuple
import h5py  # type: ignore

import numpy as np

from src.myio import myio
from src.plotting.tools import (
    PlotSeries,
)

# ------------------------- flocs -------------------------


def floc_count_evolution(
    floc_dir: Union[str, Path],
    colour: str,
    label: Optional[str],
    min_file_index: Optional[int],
    max_file_index: Optional[int],
) -> PlotSeries:

    floc_files: List[Path] = myio.list_parties_data_files(
        floc_dir, "Flocs", min_file_index, max_file_index
    )
    time: np.ndarray = myio.get_time_array(
        "Flocs",
        floc_dir,
        min_file_index,
        max_file_index,
        "time",
    )
    counts: List[int] = []

    for floc_file in floc_files:
        with h5py.File(floc_file, "r") as f:
            if "flocs" in f and "floc_id" in f["flocs"]:  # type: ignore
                floc_count = len(np.unique(f["flocs"]["floc_id"][:]))  # type: ignore
                counts.append(floc_count)

    s: PlotSeries = PlotSeries(
        data={"time": time, "counts": np.asarray(counts)},
        x_key="time",
        y_key="counts",
        label=label,
        plot_method="plot",
        color=colour,
    )

    return s


# ------------------------- u_plus_mean_wall series -------------------------


def u_plus_mean_wall_parties(
    parties_data_path: Union[Path, str],
    label: str,
    colour: str,
    linestyles: Optional[Tuple[str, str]] = None,
) -> List[PlotSeries]:

    stats, _ = myio.load_from_h5(parties_data_path)

    if linestyles is None:
        linestyles = ("-.", ":")

    s_parties = PlotSeries(
        data={
            "x": stats["yc_plus"],
            "y": stats["U_plus"],
            **{"Re": stats.get("Re"), "Re_tau": stats.get("Re_tau")},
        },
        x_key="x",
        y_key="y",
        label=label,
        plot_method="semilogx",
        linestyle=linestyles[0],
        color=colour,
    )
    s_parties_log = PlotSeries(
        data={
            "x": stats["parties_log_yc_plus"],
            "y": stats["parties_log_U_plus"],
            **{"Re": stats.get("Re"), "Re_tau": stats.get("Re_tau")},
        },
        x_key="x",
        y_key="y",
        plot_method="semilogx",
        linestyle=linestyles[1],
        linewidth=0.9,
        label=f"Law of the wall ({label})",
        color=colour,
    )

    return [s_parties, s_parties_log]


def u_plus_mean_wall_utexas(
    utexas_data_path: Union[Path, str],
    colour_map: Dict[str, str] = {},
    linestyle_map: Dict[str, str] = {},
) -> List[PlotSeries]:

    stats, _ = myio.load_from_h5(utexas_data_path)

    if colour_map == {}:
        colour_map = {
            "utexas": "k",
            "utexas_visc": "k",
        }
    if linestyle_map == {}:
        linestyle_map = {
            "utexas": "-.",
            "utexas_visc": ":",
        }

    s_utexas = PlotSeries(
        data={
            "x": stats["utexas_y_plus"],
            "y": stats["utexas_U_plus"],
            **{"Re": stats.get("Re"), "Re_tau": stats.get("Re_tau")},
        },
        x_key="x",
        y_key="y",
        label="utexas",
        linestyle=linestyle_map["utexas"],
        plot_method="semilogx",
        color=colour_map["utexas"],
    )
    s_utexas_visc = PlotSeries(
        data={
            "x": stats["utexas_viscous_y_plus"],
            "y": stats["utexas_viscous_U_plus"],
            **{"Re": stats.get("Re"), "Re_tau": stats.get("Re_tau")},
        },
        x_key="x",
        y_key="y",
        label="Law of the wall (utexas)",
        plot_method="semilogx",
        linestyle=linestyle_map["utexas_visc"],
        linewidth=0.9,
        color=colour_map["utexas_visc"],
    )
    s_utexas_log = PlotSeries(
        data={
            "x": stats["utexas_log_y_plus"],
            "y": stats["utexas_log_U_plus"],
            **{"Re": stats.get("Re"), "Re_tau": stats.get("Re_tau")},
        },
        x_key="x",
        y_key="y",
        plot_method="semilogx",
        linestyle=linestyle_map["utexas_visc"],
        linewidth=0.9,
        color=colour_map["utexas_visc"],
    )
    return [s_utexas, s_utexas_visc, s_utexas_log]


# ------------------------- normal_stress_wall series -------------------------


def normal_stress_wall_parties(
    parties_data_path: Union[Path, str],
    label: Optional[str] = None,
    colour: str = "k",
    linestyle_map: Optional[Dict[str, str]] = None,
) -> List[PlotSeries]:

    stats, _ = myio.load_from_h5(parties_data_path)
    if linestyle_map is None:
        linestyle_map = {"u": "-", "v": "-.", "w": "--", "k": ":"}

    parties_yc_plus = stats["yc_plus"]
    parties_yv_plus = stats["yv_plus"]
    parties_upup_plus = stats["upup_plus"]
    parties_vpvp_plus = stats["vpvp_plus"]
    parties_wpwp_plus = stats["wpwp_plus"]
    parties_k_plus = stats["k_plus"]

    def gen_label(u: str, which: str) -> str:
        base = f"$\\langle {u}^\\prime {u}^\\prime\\rangle / u_\\tau$ ({which})"
        if label:
            return f"{base} ({label})"
        return base

    l_u_part = gen_label("u", "parties")
    l_v_part = gen_label("v", "parties")
    l_w_part = gen_label("w", "parties")
    l_k_part = gen_label("k", "parties")

    s_u_part = PlotSeries(
        data={
            "x": parties_yc_plus,
            "y": parties_upup_plus,
            "Re": stats.get("Re"),
            "Re_tau": stats.get("Re_tau"),
        },
        x_key="x",
        y_key="y",
        label=l_u_part,
        plot_method="plot",
        linestyle=linestyle_map.get("u", "-"),
        color=colour,
    )
    s_v_part = PlotSeries(
        data={
            "x": parties_yv_plus,
            "y": parties_vpvp_plus,
            "Re": stats.get("Re"),
            "Re_tau": stats.get("Re_tau"),
        },
        x_key="x",
        y_key="y",
        label=l_v_part,
        plot_method="plot",
        linestyle=linestyle_map.get("v", "-."),
        color=colour,
    )
    s_w_part = PlotSeries(
        data={
            "x": parties_yc_plus,
            "y": parties_wpwp_plus,
            "Re": stats.get("Re"),
            "Re_tau": stats.get("Re_tau"),
        },
        x_key="x",
        y_key="y",
        label=l_w_part,
        plot_method="plot",
        linestyle=linestyle_map.get("w", "--"),
        color=colour,
    )
    s_k_part = PlotSeries(
        data={
            "x": parties_yc_plus,
            "y": parties_k_plus,
            "Re": stats.get("Re"),
            "Re_tau": stats.get("Re_tau"),
        },
        x_key="x",
        y_key="y",
        label=l_k_part,
        plot_method="plot",
        linestyle=linestyle_map.get("k", ":"),
        color=colour,
    )

    return [s_u_part, s_v_part, s_w_part, s_k_part]


def normal_stress_wall_utexas(
    utexas_data_path: Union[Path, str],
    colour_map: Dict[str, str] = {},
    linestyle_map: Dict[str, str] = {},
    marker_map: Dict[str, str] = {},
) -> List[PlotSeries]:

    stats, _ = myio.load_from_h5(utexas_data_path)

    if colour_map == {}:
        colour_map = {
            "utexas_upup": "k",
            "utexas_vpvp": "k",
            "utexas_wpwp": "k",
            "utexas_k": "k",
        }
    if linestyle_map == {}:
        linestyle_map = {
            "utexas_upup": "None",
            "utexas_vpvp": "None",
            "utexas_wpwp": "None",
            "utexas_k": "None",
        }
    if marker_map == {}:
        marker_map = {
            "utexas_upup": "o",
            "utexas_vpvp": "d",
            "utexas_wpwp": "^",
            "utexas_k": "x",
        }

    utexas_y_plus: np.ndarray = stats["utexas_y_plus"]
    idx: np.ndarray = np.linspace(0, len(utexas_y_plus) - 1, 40, dtype=int)
    idx_upup: np.ndarray = np.linspace(0, len(utexas_y_plus) - 1, 70, dtype=int)

    ux_y: np.ndarray = utexas_y_plus[idx]
    ux_y_upup: np.ndarray = utexas_y_plus[idx_upup]

    ux_upup: np.ndarray = stats["utexas_upup_plus"][idx_upup]
    ux_vpvp: np.ndarray = stats["utexas_vpvp_plus"][idx]
    ux_wpwp: np.ndarray = stats["utexas_wpwp_plus"][idx]
    ux_k: np.ndarray = stats["utexas_k_plus"][idx]

    def create_series(x, y, colour, marker, linestyle, label):
        return PlotSeries(
            data={
                "x": x,
                "y": y,
                "Re": stats.get("Re"),
                "Re_tau": stats.get("Re_tau"),
            },
            x_key="x",
            y_key="y",
            label=label,
            plot_method="plot",
            marker=marker,
            linestyle=linestyle,
            color=colour,
            kwargs={"fillstyle": "none"},
        )

    s_u_tex = create_series(
        ux_y_upup,
        ux_upup,
        colour_map["utexas_upup"],
        marker_map["utexas_upup"],
        linestyle_map["utexas_upup"],
        r"$\langle u^\prime u^\prime \/ u_\tau$ (utexas)",
    )
    s_v_tex = create_series(
        ux_y,
        ux_vpvp,
        colour_map["utexas_vpvp"],
        marker_map["utexas_vpvp"],
        linestyle_map["utexas_vpvp"],
        r"$\langle v^\prime v^\prime \/ u_\tau$ (utexas)",
    )
    s_w_tex = create_series(
        ux_y,
        ux_wpwp,
        colour_map["utexas_wpwp"],
        marker_map["utexas_wpwp"],
        linestyle_map["utexas_wpwp"],
        r"$\langle w^\prime w^\prime \/ u_\tau$ (utexas)",
    )
    s_k_tex = create_series(
        ux_y,
        ux_k,
        colour_map["utexas_k"],
        marker_map["utexas_k"],
        linestyle_map["utexas_k"],
        r"$\langle k \/ u_\tau^2$ (utexas)",
    )

    return [s_u_tex, s_v_tex, s_w_tex, s_k_tex]


# -------------------- Steady state --------------------


def Ekin_evolution(
    h5_path: Union[str, Path],
    colour: str,
    linestyle: str,
    marker: str,
    label: Optional[str],
) -> PlotSeries:

    E_kin: np.ndarray
    time: np.ndarray
    with h5py.File(h5_path, "r") as f:
        E_kin = f["E_kin"][:]  # type: ignore
        time = f["time"][:]  # type: ignore

    return PlotSeries(
        data={
            "x": time,
            "y": E_kin,
        },
        x_key="x",
        y_key="y",
        label=label,
        plot_method="plot",
        linestyle=linestyle,
        marker=marker,
        color=colour,
        kwargs={"fillstyle": "none"},
    )
