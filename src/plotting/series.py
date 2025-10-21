# -- src/plotting/series.py

from pathlib import Path
from typing import Union, Dict, List, Optional, Tuple
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
    normalised: bool,
    reset_time: bool,
) -> PlotSeries:

    print(
        f'Looking for floc files in directory: "{floc_dir}" with min_file_index: {min_file_index} and max_file_index: {max_file_index}'
    )
    floc_files: List[Path] = myio.list_data_files(
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

    if reset_time:
        time = time - time[0]

    for floc_file in floc_files:
        with h5py.File(floc_file, "r") as f:
            floc_count = len(np.unique(f["flocs"]["floc_id"][:]))  # type: ignore
            counts.append(floc_count)

    counts_arr = np.asarray(counts)
    if normalised:
        counts_arr = counts_arr.astype(float)
        N_particles: int
        with h5py.File(floc_files[0], "r") as f:
            N_particles = len(f["particles"]["r"][:])  # type: ignore
        counts_arr /= N_particles

    s: PlotSeries = PlotSeries(
        data={"time": time, "counts": counts_arr},
        x_key="time",
        y_key="counts",
        plot_method="plot",
        kwargs={
            "label": label,
            "color": colour,
        },
    )

    return s


def floc_pdf(
    floc_dir: Union[str, Path],
    labels: List[Optional[str]],
    colours: List[str],
    linestyles: List[str],
    markers: List[str],
) -> Tuple[PlotSeries, PlotSeries, PlotSeries]:

    # edges_n_p: np.ndarray
    # edges_D_f: np.ndarray
    # edges_D_g: np.ndarray
    centers_n_p: np.ndarray
    centers_D_f: np.ndarray
    centers_D_g: np.ndarray

    bin_width_n_p: np.ndarray
    bin_width_D_f: np.ndarray
    bin_width_D_g: np.ndarray

    probab_n_p: np.ndarray
    probab_D_f: np.ndarray
    probab_D_g: np.ndarray
    with h5py.File(Path(floc_dir) / "floc_PDF.h5", "r") as f:
        # edges_n_p = f["edges_n_p"][:]  # type: ignore
        # edges_D_f = f["edges_D_f"][:]  # type: ignore
        # edges_D_g = f["edges_D_g"][:]  # type: ignore
        centers_n_p = f["centers_n_p"][:]  # type: ignore
        centers_D_f = f["centers_D_f"][:]  # type: ignore
        centers_D_g = f["centers_D_g"][:]  # type: ignore

        bin_width_n_p = f["bin_width_n_p"][()]  # type: ignore
        bin_width_D_f = f["bin_width_D_f"][()]  # type: ignore
        bin_width_D_g = f["bin_width_D_g"][()]  # type: ignore
        probab_n_p = f["probab_n_p"][:]  # type: ignore
        probab_D_f = f["probab_D_f"][:]  # type: ignore
        probab_D_g = f["probab_D_g"][:]  # type: ignore

    markeredgewidth: float = 0.5

    s_n_p: PlotSeries = PlotSeries(
        # data={"edges": edges_n_p, "counts": probab_n_p},
        data={"x": centers_n_p, "y": probab_n_p, "bin_width": bin_width_n_p},
        # x_key="edges",
        # y_key="counts",
        x_key="x",
        y_key="y",
        plot_method="semilogy",
        kwargs={
            "label": labels[0],
            "linestyle": linestyles[0],
            "marker": markers[0],
            "markerfacecolor": colours[0],
            "markeredgecolor": "k",
            "markeredgewidth": markeredgewidth,
            "color": "k",
            "fillstyle": "full",
        },
    )
    s_D_f: PlotSeries = PlotSeries(
        # data={"edges": edges_D_f, "counts": probab_D_f},
        data={"x": centers_D_f, "y": probab_D_f, "bin_width": bin_width_D_f},
        # x_key="edges",
        # y_key="counts",
        x_key="x",
        y_key="y",
        plot_method="semilogy",
        kwargs={
            "label": labels[1],
            "linestyle": linestyles[1],
            "marker": markers[1],
            "markerfacecolor": colours[1],
            "markeredgecolor": "k",
            "markeredgewidth": markeredgewidth,
            "color": "k",
            "fillstyle": "full",
        },
    )
    s_D_g: PlotSeries = PlotSeries(
        # data={"edges": edges_D_g, "counts": probab_D_g},
        data={"x": centers_D_g, "y": probab_D_g, "bin_width": bin_width_D_g},
        # x_key="edges",
        # y_key="counts",
        x_key="x",
        y_key="y",
        plot_method="semilogy",
        kwargs={
            "label": labels[2],
            "linestyle": linestyles[2],
            "marker": markers[2],
            "markerfacecolor": colours[2],
            "markeredgecolor": "k",
            "markeredgewidth": markeredgewidth,
            "color": "k",
            "fillstyle": "full",
        },
    )

    return s_n_p, s_D_f, s_D_g


def floc_avg_dir(
    floc_dir: Union[str, Path],
    labels: List[Optional[str]],
    colours: List[str],
    linestyles: List[str],
    markers: List[str],
    inner_units: bool,
) -> Tuple[PlotSeries, PlotSeries, PlotSeries, PlotSeries]:

    x_data: np.ndarray
    D_f_avg: np.ndarray
    D_g_avg: np.ndarray
    D_f_mass_avg: np.ndarray
    D_g_mass_avg: np.ndarray
    with h5py.File(Path(floc_dir) / "avg_floc_diam.h5", "r") as f:
        if inner_units:
            x_data = f["yp_center"][:]  # type: ignore
            D_f_avg = f["inner_D_f_avg"][:]  # type: ignore
            D_g_avg = f["inner_D_g_avg"][:]  # type: ignore
            D_f_mass_avg = f["inner_D_f_mass_avg"][:]  # type: ignore
            D_g_mass_avg = f["inner_D_g_mass_avg"][:]  # type: ignore
        else:
            x_data = f["y_center"][:]  # type: ignore
            D_f_avg = f["D_f_avg"][:]  # type: ignore
            D_g_avg = f["D_g_avg"][:]  # type: ignore
            D_f_mass_avg = f["D_f_mass_avg"][:]  # type: ignore
            D_g_mass_avg = f["D_g_mass_avg"][:]  # type: ignore

    markeredgewidth: float = 0.5

    def create_series(y_data: np.ndarray, idx: int) -> PlotSeries:

        return PlotSeries(
            # data={"edges": edges_n_p, "counts": probab_n_p},
            data={"x": x_data, "y": y_data},
            x_key="x",
            y_key="y",
            plot_method="plot",
            kwargs={
                "label": labels[idx],
                "linestyle": linestyles[idx],
                "marker": markers[idx],
                "markerfacecolor": colours[idx],
                "markeredgecolor": "k",
                "markeredgewidth": markeredgewidth,
                "color": "k",
                "fillstyle": "full",
            },
        )

    s_D_f_avg: PlotSeries = create_series(D_f_avg, 0)
    s_D_g_avg: PlotSeries = create_series(D_g_avg, 1)
    s_D_f_mass_avg: PlotSeries = create_series(D_f_mass_avg, 2)
    s_D_g_mass_avg: PlotSeries = create_series(D_g_mass_avg, 3)

    return s_D_f_avg, s_D_g_avg, s_D_f_mass_avg, s_D_g_mass_avg


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
        plot_method="semilogx",
        kwargs={"label": label, "linestyle": linestyles[0], "color": colour},
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
        kwargs={
            "linestyle": linestyles[1],
            "linewidth": 0.9,
            "label": f"Law of the wall ({label})",
            "color": colour,
        },
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
        plot_method="semilogx",
        kwargs={
            "label": "utexas",
            "linestyle": linestyle_map["utexas"],
            "color": colour_map["utexas"],
        },
    )
    s_utexas_visc = PlotSeries(
        data={
            "x": stats["utexas_viscous_y_plus"],
            "y": stats["utexas_viscous_U_plus"],
            **{"Re": stats.get("Re"), "Re_tau": stats.get("Re_tau")},
        },
        x_key="x",
        y_key="y",
        plot_method="semilogx",
        kwargs={
            "label": "Law of the wall (utexas)",
            "linestyle": linestyle_map["utexas_visc"],
            "linewidth": 0.9,
            "color": colour_map["utexas_visc"],
        },
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
        kwargs={
            "linestyle": linestyle_map["utexas_visc"],
            "linewidth": 0.9,
            "color": colour_map["utexas_visc"],
        },
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
        plot_method="plot",
        kwargs={
            "label": l_u_part,
            "linestyle": linestyle_map.get("u", "-"),
            "color": colour,
        },
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
        plot_method="plot",
        kwargs={
            "label": l_v_part,
            "linestyle": linestyle_map.get("v", "-."),
            "color": colour,
        },
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
        plot_method="plot",
        kwargs={
            "label": l_w_part,
            "linestyle": linestyle_map.get("w", "--"),
            "color": colour,
        },
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
        plot_method="plot",
        kwargs={
            "label": l_k_part,
            "linestyle": linestyle_map.get("k", ":"),
            "color": colour,
        },
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
            plot_method="plot",
            kwargs={
                "label": label,
                "marker": marker,
                "linestyle": linestyle,
                "color": colour,
                "fillstyle": "none",
            },
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
        plot_method="plot",
        kwargs={
            "label": label,
            "linestyle": linestyle,
            "marker": marker,
            "color": colour,
            "fillstyle": "none",
        },
    )
