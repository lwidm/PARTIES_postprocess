# -- src/plotting/series.py

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import h5py

import numpy as np
import scipy.optimize

from src.theory import law_of_the_wall as low
from src.myio import myio
from src.myio import new_myio
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
        raise FileNotFoundError(f'ERROR: Floc count CSV file not found: "{csv_file}"')

    time, counts = new_myio.read_csv_columns(csv_file, (0, 1))

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

    (time,) = new_myio.read_csv_columns(floc_count_csv, (0,))

    fit_data = new_myio.read_csv_columns(floc_count_fit_csv, (0, 1, 2))

    b = float(fit_data[0])
    Nf_eq = int(fit_data[1])
    n_particles = int(fit_data[2])

    if b is None:
        raise ValueError(f'ERROR: Could not obtain "b" from "{floc_count_fit_csv}"!')
    if Nf_eq is None:
        raise ValueError(
            f'ERROR: Could not obtain "Nf_eq" from "{floc_count_fit_csv}"!'
        )
    if n_particles is None:
        raise ValueError(
            f'ERROR: Could not obtain "n_particles" from "{floc_count_fit_csv}"!'
        )

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


def u_plus_mean_parties(
    csv_dir: Path,
    label: str,
    colour: str,
    log_fit: bool,
    visc_fit: bool,
    linestyles: Optional[Tuple[str, str, str]],
) -> List[PlotSeries]:
    yc_plus, U = new_myio.read_csv_columns(
        csv_dir / "flow_fluctuation_data_inner.csv", (0, 1)
    )
    return u_plus_mean(yc_plus, U, label, colour, log_fit, visc_fit, linestyles)


def u_plus_mean_utexas(
    csv_dir: Path,
    label: str,
    colour: str,
    log_fit: bool,
    visc_fit: bool,
    linestyles: Optional[Tuple[str, str, str]],
) -> List[PlotSeries]:
    yc_plus, U = new_myio.read_csv_columns(
        csv_dir / "LM_Channel_0180_mean_prof.dat", (1, 2)
    )
    return u_plus_mean(yc_plus, U, label, colour, log_fit, visc_fit, linestyles)


def u_plus_mean(
    yc_plus: np.ndarray,
    U: np.ndarray,
    label: str,
    colour: str,
    log_fit: bool,
    visc_fit: bool,
    linestyles: Optional[Tuple[str, str, str]],
) -> List[PlotSeries]:

    fitted_kappa: float
    fitted_constant: float
    fitted_kappa, fitted_constant = low.fit_parameters(yc_plus, U)
    visc_yc_plus, visc_U, log_yc_plus, log_U = low.generate_profile(
        yc_plus, fitted_kappa, fitted_constant
    )

    if linestyles is None:
        linestyles = ("-.", ":", ":")

    results: List[PlotSeries] = []
    s_parties = PlotSeries(
        data={
            "x": yc_plus,
            "y": U,
        },
        x_key="x",
        y_key="y",
        plot_method="semilogx",
        kwargs={"label": label, "linestyle": linestyles[0], "color": colour},
    )
    results.append(s_parties)
    s_parties_visc: Optional[PlotSeries] = None
    if visc_fit:
        s_parties_visc = PlotSeries(
            data={
                "x": visc_yc_plus,
                "y": visc_U,
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
        results.append(s_parties_visc)
    s_parties_log: Optional[PlotSeries] = None
    if log_fit:
        s_parties_visc = PlotSeries(
            data={
                "x": log_yc_plus,
                "y": log_U,
            },
            x_key="x",
            y_key="y",
            plot_method="semilogx",
            kwargs={
                "linestyle": linestyles[2],
                "linewidth": 0.9,
                "label": f"Law of the wall ({label})",
                "color": colour,
            },
        )
        results.append(s_parties_visc)

    return results


# ------------------------- normal_stress_wall series -------------------------


def normal_stress_wall_parties(
    csv_dir: Path, colour: str, label: str
) -> List[PlotSeries]:
    yc, uu, ww, _, yv, vv, uv = new_myio.read_csv_columns(
        csv_dir / "flow_fluctuation_data_inner.csv", (0, 1, 2, 3, 4, 5, 6)
    )

    stats: dict[str, np.ndarray] = {
        "yc_plus": yc,
        "yv_plus": yv,
        "uu_plus": uu,
        "vv_plus": vv,
        "ww_plus": ww,
        "uv_plus": uv,
    }

    colours: list[str] = [colour for _ in range(4)]
    linestyles: list[str] = ["None" for _ in range(4)]
    markers: list[str] = ["o", "d", "^", "s"]
    return normal_stress_wall(stats, colours, linestyles, markers, label)


def normal_stress_wall_utexas(
    csv_dir: Path,
) -> List[PlotSeries]:
    yp, uu, vv, ww, uv, uw, vw, k = new_myio.read_csv_columns(
        csv_dir / "LM_Channel_0180_vel_fluc_prof.dat", (1, 2, 3, 4, 5, 6, 7, 8)
    )

    stats: dict[str, np.ndarray] = {
        "yc_plus": yp,
        "yv_plus": yp,
        "uu_plus": uu,
        "vv_plus": vv,
        "ww_plus": ww,
        "uv_plus": uv,
    }

    colours: list[str] = ["k" for _ in range(4)]
    linestyles: list[str] = ["-", "-.", "--", ":"]
    markers: list[str] = ["None" for _ in range(4)]
    return normal_stress_wall(stats, colours, linestyles, markers, "utexas")


def normal_stress_wall(
    stats: dict[str, np.ndarray],
    colours: list[str],
    linestyles: list[str],
    markers: list[str],
    label: str,
) -> List[PlotSeries]:

    yc_plus: np.ndarray = stats["yc_plus"]
    yv_plus: np.ndarray = stats["yv_plus"]
    idx: np.ndarray = np.linspace(0, len(yc_plus) - 1, 40, dtype=int)
    idx_v: np.ndarray = np.linspace(0, len(yv_plus) - 1, 40, dtype=int)
    idx_upup: np.ndarray = np.linspace(0, len(yc_plus) - 1, 70, dtype=int)

    yc: np.ndarray = yc_plus[idx]
    yv: np.ndarray = yv_plus[idx_v]
    yc_uu: np.ndarray = yc_plus[idx_upup]

    uu: np.ndarray = stats["uu_plus"][idx_upup]
    vv: np.ndarray = stats["vv_plus"][idx_v]
    ww: np.ndarray = stats["ww_plus"][idx]
    uv: np.ndarray = stats["uv_plus"][idx]

    def create_series(x, y, colour, marker, linestyle, label_local):
        return PlotSeries(
            data={
                "x": x,
                "y": y,
            },
            x_key="x",
            y_key="y",
            plot_method="plot",
            kwargs={
                "label": label_local,
                "marker": marker,
                "linestyle": linestyle,
                "color": colour,
                "fillstyle": "none",
            },
        )

    s_uu = create_series(
        yc_uu,
        uu,
        colours[0],
        markers[0],
        linestyles[0],
        rf"$\langle u^\prime u^\prime \rangle / u_\tau^2$ ({label})",
    )
    s_vv = create_series(
        yv,
        vv,
        colours[1],
        markers[1],
        linestyles[1],
        rf"$\langle v^\prime v^\prime \rangle / u_\tau^2$ ({label})",
    )
    s_ww = create_series(
        yc,
        ww,
        colours[2],
        markers[2],
        linestyles[2],
        rf"$\langle w^\prime w^\prime \rangle / u_\tau^2$ ({label})",
    )
    s_uv = create_series(
        yc,
        uv,
        colours[3],
        markers[3],
        linestyles[3],
        rf"$\langle u^\prime v^\prime \rangle / u_\tau^2$ ({label})",
    )

    return [s_uu, s_vv, s_ww, s_uv]


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
