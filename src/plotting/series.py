# -- src/plotting/series.py

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import h5py

import numpy as np
import scipy.optimize

from src.myio import myio
from src.plotting.tools import (
    PlotSeries,
)

# ------------------------- flocs -------------------------


def floc_count_evolution(
    csv_dir: Path,
    colour: str,
    label: Optional[str],
    normalised: bool,
    reset_time: bool,
) -> PlotSeries:

    csv_file: Path
    if normalised:
        csv_file = csv_dir / "floc_count_norm.csv"
    else:
        csv_file = csv_dir / "floc_count.csv"  
    if not csv_file.exists():
        raise FileNotFoundError(
            f'ERROR: Floc count CSV file not found: "{csv_file}"'
        )

    time, counts = np.loadtxt(
            str(csv_file),
            comments='%',
            delimiter=',',
            usecols=(0, 1),
            skiprows=1,
            unpack=True
    )


    if reset_time and len(time) > 0:
        time = time - time[0]

    s: PlotSeries = PlotSeries(
        data={"time": time, "counts": counts},
        x_key="time",
        y_key="counts",
        plot_method="plot",
        kwargs={
            "label": label,
            "color": colour,
        },
    )

    return s


def floc_count_evolution_fit(
    csv_dir: Path,
    colour: str,
    label: Optional[str],
    normalised: bool,
    reset_time: bool,
) -> PlotSeries:

    floc_count_csv: Path
    if normalised:
        floc_count_csv = csv_dir / "floc_count_norm.csv"
    else:
        floc_count_csv = csv_dir / "floc_count.csv"  
    if not floc_count_csv.exists():
        raise FileNotFoundError(
            f'ERROR: Floc count CSV file not found: "{floc_count_csv}"'
        )
    floc_count_fit_csv: Path = csv_dir / "flocculation_fit_parameters.csv"
    if not floc_count_fit_csv.exists():
        raise FileNotFoundError(
            f'ERROR: Floc count fit CSV file not found: "{floc_count_fit_csv}"'
        )

    time = np.loadtxt(
            str(floc_count_csv),
            comments='%',
            delimiter=',',
            usecols=0,
            skiprows=1,
    )

    fit_data = np.loadtxt(
            str(floc_count_fit_csv),
            comments='%',
            delimiter=',',
            dtype=str
    )
    header_row = fit_data[0, :]
    data_row = fit_data[1, :]
    b: float | None = None
    Nf_eq: int | None= None
    n_particles: int | None = None
    for data, key in zip(data_row, header_row):
        match key:
            case "b":
                b = float(data)
            case "Nf_eq":
                Nf_eq = int(data)
            case "n_particles":
                n_particles = int(data)
            case _:
                raise ValueError(f'ERROR: Unknown column header in "{floc_count_fit_csv}". Expected "b", "Nf_eq" or "n_particles", got "{key}"!')

    if b is None:
        raise ValueError(f'ERROR: Could not obtain "b" from "{floc_count_fit_csv}"!')
    if Nf_eq is None:
        raise ValueError(f'ERROR: Could not obtain "Nf_eq" from "{floc_count_fit_csv}"!')
    if n_particles is None:
        raise ValueError(f'ERROR: Could not obtain "n_particles" from "{floc_count_fit_csv}"!')

    def model(time: np.ndarray, b: float, Nf_eq: float) -> np.ndarray:
        return (float(n_particles) - float(Nf_eq)) * np.exp(
            -b * (time - time[0])
        ) + float(Nf_eq)

    print(f"Floc count evolution fit for {label}: b = {b}, Nf_eq = {Nf_eq}")

    time_fit: np.ndarray = np.linspace(time[0], time[-1], 200)
    counts_fit: np.ndarray = model(time_fit, b, Nf_eq)
    if normalised:
        counts_fit /= n_particles

    if reset_time:
        time_fit = time_fit - time_fit[0]

    if label is not None:
        label += f"; fit $b={b:.4f}$ , $N_{{f,eq}}={Nf_eq:.4f}$"

    s: PlotSeries = PlotSeries(
        data={"time": time_fit, "counts": counts_fit},
        x_key="time",
        y_key="counts",
        plot_method="plot",
        kwargs={
            "label": label,
            "color": colour,
            "linestyle": ":",
        },
    )

    return s


def floc_pdf(
    floc_dir: Path,
    labels: List[Optional[str]],
    colours: List[str],
    markers: List[str],
) -> Tuple[
    PlotSeries,
    PlotSeries,
    PlotSeries,
    PlotSeries,
    PlotSeries,
    PlotSeries,
    PlotSeries,
    PlotSeries,
    PlotSeries,
    PlotSeries,
    PlotSeries,
    PlotSeries,
]:

    means_list: List[np.ndarray] = []
    std_probab_list: List[np.ndarray] = []
    bin_widths_list: List[np.ndarray] = []
    probabs_list: List[np.ndarray] = []
    postfixes: List[str] = ["n_p", "D_f", "D_g"]

    with h5py.File(str(floc_dir / "pdf_stats.h5"), "r") as f:
        for i in range(len(postfixes)):
            bin_widths_list.append(f["bin_width_" + postfixes[i]][()])  # type: ignore

            means_list.append(f["means_" + postfixes[i]][:])  # type: ignore
            std_probab_list.append(f["std_probab_" + postfixes[i]][:])  # type: ignore
            probabs_list.append(f["probab_" + postfixes[i]][:])  # type: ignore

        for i in range(len(postfixes)):
            bin_widths_list.append(f["bin_width_" + postfixes[i]][()])  # type: ignore
            means_list.append(f["mass_means_" + postfixes[i]][:])  # type: ignore
            std_probab_list.append(f["std_mass_probab_" + postfixes[i]][:])  # type: ignore
            probabs_list.append(f["mass_probab_" + postfixes[i]][:])  # type: ignore

    markeredgewidth: float = 0.5

    def create_series(i: int) -> Tuple[PlotSeries, PlotSeries]:
        s: PlotSeries = PlotSeries(
            data={
                "x": means_list[i],
                "y": probabs_list[i],
                "bin_width": bin_widths_list[i],
            },
            x_key="x",
            y_key="y",
            plot_method="semilogy",
            kwargs={
                "label": labels[i],
                "linestyle": "None",
                "marker": markers[i],
                "markerfacecolor": colours[i],
                "markeredgecolor": "k",
                "markeredgewidth": markeredgewidth,
                "color": "k",
                "fillstyle": "full",
            },
        )

        s_err: PlotSeries = PlotSeries(
            data={
                "x": means_list[i],
                "y": probabs_list[i],
                "bin_width": bin_widths_list[i],
                "y_err": std_probab_list[i],
            },
            x_key="x",
            y_key="y",
            plot_method="err_semilogy",
            kwargs={
                "color": "k",
                "linestyle": "None",
                "fmt": "None",
                "ecolor": "k",
                "elinewidth": 0.6,
                "capsize": 2,
                "capthick": 0.8,
                "barsabove": True,
            },
        )

        return s, s_err

    s_n_p, s_n_p_err = create_series(0)
    s_D_f, s_D_f_err = create_series(1)
    s_D_g, s_D_g_err = create_series(2)
    s_mass_n_p, s_mass_n_p_err = create_series(3)
    s_mass_D_f_d_particle, s_mass_D_f_d_particle_err = create_series(4)
    s_mass_D_g_d_particle, s_mass_D_g_d_particle_err = create_series(5)

    return (
        s_n_p,
        s_D_f,
        s_D_g,
        s_n_p_err,
        s_D_f_err,
        s_D_g_err,
        s_mass_n_p,
        s_mass_D_f_d_particle,
        s_mass_D_g_d_particle,
        s_mass_n_p_err,
        s_mass_D_f_d_particle_err,
        s_mass_D_g_d_particle_err,
    )


def floc_avg_dir(
    floc_dir: Path,
    labels: List[Optional[str]],
    colours: List[str],
    markers: List[str],
) -> Tuple[
    PlotSeries,
    PlotSeries,
    PlotSeries,
    PlotSeries,
    PlotSeries,
    PlotSeries,
    PlotSeries,
    PlotSeries,
]:

    x_data: np.ndarray
    D_f_d_particle_avg: np.ndarray
    D_g_d_particle_avg: np.ndarray
    D_f_d_particle_mass_avg: np.ndarray
    D_g_d_particle_mass_avg: np.ndarray
    std_D_f_d_particle_avg: np.ndarray
    std_D_g_d_particle_avg: np.ndarray
    std_D_f_d_particle_mass_avg: np.ndarray
    std_D_g_d_particle_mass_avg: np.ndarray
    with h5py.File(str(floc_dir / "avg_diam_stats.h5"), "r") as f:
        x_data = f["y_mean"][:]  # type: ignore
        D_f_d_particle_avg = f["D_f_avg"][:]  # type: ignore
        D_g_d_particle_avg = f["D_g_avg"][:]  # type: ignore
        D_f_d_particle_mass_avg = f["D_f_mass_avg"][:]  # type: ignore
        D_g_d_particle_mass_avg = f["D_g_mass_avg"][:]  # type: ignore
        std_D_f_d_particle_avg = f["std_D_f_avg"][:]  # type: ignore
        std_D_g_d_particle_avg = f["std_D_g_avg"][:]  # type: ignore
        std_D_f_d_particle_mass_avg = f["std_D_f_mass_avg"][:]  # type: ignore
        std_D_g_d_particle_mass_avg = f["std_D_g_mass_avg"][:]  # type: ignore

    markeredgewidth: float = 0.5

    def create_series(
        y_data: np.ndarray, std_data: np.ndarray, idx: int
    ) -> Tuple[PlotSeries, PlotSeries]:

        s: PlotSeries = PlotSeries(
            # data={"edges": edges_n_p, "counts": probab_n_p},
            data={"x": x_data, "y": y_data},
            x_key="x",
            y_key="y",
            plot_method="plot",
            kwargs={
                "label": labels[idx],
                "linestyle": "None",
                "marker": markers[idx],
                "markerfacecolor": colours[idx],
                "markeredgecolor": "k",
                "markeredgewidth": markeredgewidth,
                "color": "k",
                "fillstyle": "full",
            },
        )

        s_err: PlotSeries = PlotSeries(
            data={
                "x": x_data,
                "y": y_data,
                "y_err": std_data,
            },
            x_key="x",
            y_key="y",
            plot_method="err_plot",
            kwargs={
                "color": "k",
                "linestyle": "None",
                "fmt": "None",
                "ecolor": "k",
                "elinewidth": 0.6,
                "capsize": 2,
                "capthick": 0.8,
                "barsabove": True,
            },
        )
        return s, s_err

    s_D_f_d_particle_avg, s_D_f_d_particle_err = create_series(
        D_f_d_particle_avg, std_D_f_d_particle_avg, 0
    )
    s_D_g_d_particle_avg, s_D_g_d_particle_err = create_series(
        D_g_d_particle_avg, std_D_g_d_particle_avg, 1
    )
    s_D_f_d_particle_mass_avg, s_D_f_d_particle_mass_err = create_series(
        D_f_d_particle_mass_avg, std_D_f_d_particle_mass_avg, 2
    )
    s_D_g_d_particle_mass_avg, s_D_g_d_particle_mass_err = create_series(
        D_g_d_particle_mass_avg, std_D_g_d_particle_mass_avg, 3
    )

    return (
        s_D_f_d_particle_avg,
        s_D_g_d_particle_avg,
        s_D_f_d_particle_mass_avg,
        s_D_g_d_particle_mass_avg,
        s_D_f_d_particle_err,
        s_D_g_d_particle_err,
        s_D_f_d_particle_mass_err,
        s_D_g_d_particle_mass_err,
    )


# ------------------------- u_plus_mean_wall series -------------------------


def u_plus_mean_wall_parties(
    parties_data_path: Path,
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
    utexas_data_path: Path,
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
    csv_dir: Path,
    label: Optional[str] = None,
    colour: str = "k",
    linestyle_map: Optional[Dict[str, str]] = None,
) -> List[PlotSeries]:

    if not csv_dir.is_dir():
        raise ValueError(f'ERROR: "{csv_dir}" is not a directory!')
    csv_path: Path = csv_dir / "reynolds_stress_wall_normal.csv"
    if not csv_dir.exists():
        raise ValueError(f'ERROR: Could not find "{csv_path}" !')
    yc, uu, ww, k, yv, vv, counts = np.loadtxt(
            str(csv_path),
            comments='%',
            delimiter=',',
            skiprows=1,
            unpack=True
    )

    if linestyle_map is None:
        linestyle_map = {"u": "-", "v": "-.", "w": "--", "k": ":"}

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
            "x": yc,
            "y": uu,
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
            "x": yv,
            "y": vv,
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
            "x": yc,
            "y": ww,
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
            "x": yc,
            "y": k,
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
    utexas_data_path: Path,
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
    h5_path: Path,
    colour: str,
    linestyle: str,
    marker: str,
    label: Optional[str],
) -> PlotSeries:

    E_kin: np.ndarray
    time: np.ndarray
    with h5py.File(str(h5_path), "r") as f:
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


# -------------------- Fluid volume fraction --------------------


def phi_eulerian(
    fluid_dir: Path,
    colour: str,
    linestyle: str,
    label: Optional[str],
    normalised: bool,
    show_err: bool,
) -> Tuple[PlotSeries, Optional[PlotSeries]]:

    mean_phi_h5: Path = fluid_dir / "mean_phi.h5"
    yv: np.ndarray
    Phi_mean: np.ndarray
    Phi_mean_err: Optional[np.ndarray] = None
    yv: np.ndarray
    h5_postfix: str = "_norm" if normalised else ""
    with h5py.File(str(mean_phi_h5), "r") as h5_file:
        yv = h5_file["yv"][:]  # type: ignore
        Phi_mean = h5_file["Phi_mean" + h5_postfix][:]  # type: ignore
        if show_err:
            Phi_mean_err = h5_file["Phi_err" + h5_postfix][:]  # type: ignore

    if not normalised:
        Phi_mean *= 100  # convert to %

    s: PlotSeries = PlotSeries(
        data={"x": yv, "y": Phi_mean},
        x_key="x",
        y_key="y",
        plot_method="plot",
        kwargs={
            "label": label,
            "linestyle": linestyle,
            "color": colour,
        },
    )

    if show_err:
        assert Phi_mean_err is not None
        s_err: PlotSeries = PlotSeries(
            data={
                "x": yv,
                "y": Phi_mean,
                "y_err": Phi_mean_err,
            },
            x_key="x",
            y_key="y",
            plot_method="err_plot",
            kwargs={
                "color": "k",
                "linestyle": "None",
                "fmt": "None",
                "ecolor": "k",
                "elinewidth": 0.6,
                "capsize": 2,
                "capthick": 0.8,
                "barsabove": True,
            },
        )
        return s, s_err
    return s, None
