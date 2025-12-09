# -- src/plotting/series.py

from pathlib import Path
from typing import Literal, Callable
import h5py
from scipy import stats
from scipy.interpolate import RectBivariateSpline

import numpy as np
import pickle

from matplotlib.colors import Colormap, LogNorm
from src.theory import law_of_the_wall as low
from src.myio import output, utils, lwidmer
from src.plotting.tools import (
    PlotSeries,
    _gaussian_filter_2d,
)
from src.flocs.family_tree import CoagulationFragmentationCalculator


def create_proxy_series(
    colour: str | tuple[float, float, float, float],
    colour_face: str | tuple[float, float, float, float],
    fillstyle: str,
    linestyle: str,
    marker: str,
    markeredgewidth: float,
    label: str,
):
    marker_kwargs: dict = {}
    if marker != "None":
        marker_kwargs = {
            "marker": marker,
            "markerfacecolor": colour_face,
            "markeredgecolor": "k",
            "markeredgewidth": markeredgewidth,
            "color": "k",
            "fillstyle": fillstyle,
        }
    plot_kwargs: dict = {
        "label": label,
        "linestyle": linestyle,
        "color": colour,
        "fillstyle": "none",
    }
    plot_kwargs.update(marker_kwargs)
    return PlotSeries(
        data={
            "x": [-2, -1],
            "y": [-2, -1],
        },
        x_key="x",
        y_key="y",
        plot_method="plot",
        kwargs=plot_kwargs,
    )


# ------------------------- flocs -------------------------


def floc_count_evolution(
    csv_dir: Path,
    colour: str | tuple[float, float, float, float],
    label: str | None,
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

    time, counts = lwidmer.read_csv_columns(csv_file, (0, 1), remove_nan=1)

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
    colour: str | tuple[float, float, float, float],
    label: str | None,
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

    (time,) = lwidmer.read_csv_columns(floc_count_csv, (0,), remove_nan=1)

    fit_data = lwidmer.read_csv_columns(floc_count_fit_csv, (0, 1, 2), remove_nan=1)

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
    labels: list[str | None],
    colours: list[str | tuple[float, float, float, float]],
    markers: list[str],
) -> tuple[
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

    means_list: list[np.ndarray] = []
    std_probab_list: list[np.ndarray] = []
    bin_widths_list: list[np.ndarray] = []
    probabs_list: list[np.ndarray] = []
    postfixes: list[str] = ["n_p", "D_f", "D_g"]

    with h5py.File(str(floc_dir / "pdf_stats.h5"), "r") as f:
        for key in ["n_p", "D_f", "D_g"]:
            # means_list.append(f[key]["centers"][:])  # type: ignore
            means_list.append(f[key]["unweighted"]["bin_means"][:])  # type: ignore
            std_probab_list.append(f[key]["unweighted"]["probabs_err"][:])  # type: ignore
            probabs_list.append(f[key]["unweighted"]["probabs_mean"][:])  # type: ignore

        for i in range(len(postfixes)):
            # means_list.append(f[key]["centers"][:])  # type: ignore
            means_list.append(f[key]["mass_weighted"]["bin_means"][:])  # type: ignore
            std_probab_list.append(f[key]["mass_weighted"]["probabs_err"][:])  # type: ignore
            probabs_list.append(f[key]["mass_weighted"]["probabs_mean"][:])  # type: ignore

    markeredgewidth: float = 0.7

    def create_series(i: int) -> tuple[PlotSeries, PlotSeries]:
        s: PlotSeries = PlotSeries(
            data={
                "x": means_list[i],
                "y": probabs_list[i],
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
    labels: list[str | None],
    colours: list[str | tuple[float, float, float, float]],
    markers: list[str],
) -> tuple[
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

    markeredgewidth: float = 0.7

    def create_series(
        y_data: np.ndarray, std_data: np.ndarray, idx: int
    ) -> tuple[PlotSeries, PlotSeries]:

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
    colour: str | tuple[float, float, float, float],
    log_fit: bool,
    visc_fit: bool,
    use_label: bool,
    use_label_log_fit: bool,
    use_label_visc_fit: bool,
    linestyles: tuple[str, str, str],
) -> list[PlotSeries]:
    yc_plus, U = lwidmer.read_csv_columns(
        csv_dir / "flow_mean_data_inner.csv", (0, 1), remove_nan=1
    )

    mask = yc_plus < 180

    return u_plus_mean(
        yc_plus[mask],
        U[mask],
        label,
        colour,
        log_fit,
        visc_fit,
        use_label,
        use_label_log_fit,
        use_label_visc_fit,
        linestyles,
    )


def u_plus_mean_utexas(
    csv_dir: Path,
    label: str,
    colour: str,
    log_fit: bool,
    visc_fit: bool,
    use_label: bool,
    use_label_log_fit: bool,
    use_label_visc_fit: bool,
    linestyles: tuple[str, str, str],
) -> list[PlotSeries]:
    yc_plus, U = lwidmer.read_csv_columns(
        csv_dir / "LM_Channel_0180_mean_prof.dat", (1, 2), remove_nan=1
    )
    return u_plus_mean(
        yc_plus,
        U,
        label,
        colour,
        log_fit,
        visc_fit,
        use_label,
        use_label_log_fit,
        use_label_visc_fit,
        linestyles,
    )


def u_plus_mean(
    yc_plus: np.ndarray,
    U: np.ndarray,
    label: str,
    colour: str | tuple[float, float, float, float],
    log_fit: bool,
    visc_fit: bool,
    use_label: bool,
    use_label_log_fit: bool,
    use_label_visc_fit: bool,
    linestyles: tuple[str, str, str],
) -> list[PlotSeries]:

    label_local: str

    fitted_kappa: float
    fitted_constant: float
    fitted_kappa, fitted_constant = low.fit_parameters(yc_plus, U)
    visc_yc_plus, visc_U, log_yc_plus, log_U = low.generate_profile(
        yc_plus, fitted_kappa, fitted_constant
    )

    results: list[PlotSeries] = []
    if use_label:
        label_local = label
    else:
        label_local = ""
    s_parties = PlotSeries(
        data={
            "x": yc_plus,
            "y": U,
        },
        x_key="x",
        y_key="y",
        plot_method="semilogx",
        kwargs={"label": label_local, "linestyle": linestyles[0], "color": colour},
    )
    results.append(s_parties)

    s_parties_visc: PlotSeries | None = None
    if use_label_visc_fit:
        label_local = f"Law of the wall ({label})"
    else:
        label_local = ""
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
                "label": label_local,
                "color": colour,
            },
        )
        results.append(s_parties_visc)

    s_parties_log: PlotSeries | None = None
    if use_label_log_fit:
        label_local = f"Law of the wall ({label})"
    else:
        label_local = ""
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
                "label": label_local,
                "color": colour,
            },
        )
        results.append(s_parties_visc)

    return results


def u_plus_proxies(
    linestyles: list[str],
    labels: list[str],
    colours: list[str | tuple[float, float, float, float]],
) -> list[PlotSeries]:
    quantities: list[tuple[str, str]] = [
        ("Numerical", linestyles[0]),
        ("Law of the wall", linestyles[1]),
    ]

    cases: list[tuple[str, str | tuple[float, float, float, float]]] = []
    for i in range(len(labels)):
        cases.append((labels[i], colours[i]))

    s_quantities: list[PlotSeries] = []
    s_cases: list[PlotSeries] = []

    for i in range(len(quantities)):
        s_quantities.append(
            create_proxy_series(
                "k",
                "white",
                "none",
                quantities[i][1],
                "none",
                0.5,
                quantities[i][0],
            )
        )
    for case in cases:
        s_cases.append(
            create_proxy_series(case[1], case[1], "full", "None", "s", 0, case[0])
        )

    return s_quantities + s_cases


# ------------------------- normal_stress_wall series -------------------------


def normal_stress_wall_parties(
    csv_dir: Path,
    linestyles: list[str],
    markers: list[str],
    colour: str | tuple[float, float, float, float],
    label: str,
) -> list[PlotSeries]:
    yc, uu, ww, uv, _, yv, vv = lwidmer.read_csv_columns(
        csv_dir / "flow_fluctuation_data_inner.csv", (0, 1, 2, 3, 4, 5, 6), remove_nan=1
    )

    stats: dict[str, np.ndarray] = {
        "yc_plus": yc,
        "yv_plus": yv,
        "uu_plus": uu,
        "vv_plus": vv,
        "ww_plus": ww,
        "uv_plus": uv,
    }

    colours: list[str | tuple[float, float, float, float]] = [colour for _ in range(4)]
    return normal_stress_wall(stats, colours, linestyles, markers, label, False)


def normal_stress_wall_utexas(
    csv_dir: Path,
    linestyles: list[str],
    colour: str,
) -> list[PlotSeries]:
    yp, uu, vv, ww, uv, uw, vw, k = lwidmer.read_csv_columns(
        csv_dir / "LM_Channel_0180_vel_fluc_prof.dat",
        (1, 2, 3, 4, 5, 6, 7, 8),
        remove_nan=1,
    )

    stats: dict[str, np.ndarray] = {
        "yc_plus": yp,
        "yv_plus": yp,
        "uu_plus": uu,
        "vv_plus": vv,
        "ww_plus": ww,
        "uv_plus": uv,
    }

    colours: list[str | tuple[float, float, float, float]] = [colour for _ in range(4)]
    markers: list[str] = ["None" for _ in range(4)]
    return normal_stress_wall(stats, colours, linestyles, markers, "utexas", False)


normal_stress_wall_labels: dict[str, str] = {
    "uu": rf"$\langle u^\prime u^\prime \rangle / u_\tau^2$",
    "vv": rf"$\langle v^\prime v^\prime \rangle / u_\tau^2$",
    "ww": rf"$\langle w^\prime w^\prime \rangle / u_\tau^2$",
    "uv": rf"$\langle u^\prime v^\prime \rangle / u_\tau^2$",
}


def normal_stress_wall(
    stats: dict[str, np.ndarray],
    colours: list[str | tuple[float, float, float, float]],
    linestyles: list[str],
    markers: list[str],
    label: str,
    plot_labels: bool,
) -> list[PlotSeries]:

    yc_plus: np.ndarray = stats["yc_plus"]
    yv_plus: np.ndarray = stats["yv_plus"]
    idx: np.ndarray = np.linspace(0, len(yc_plus) - 1, 40, dtype=int)
    idx_v: np.ndarray = np.linspace(0, len(yv_plus) - 1, 40, dtype=int)
    idx_upup: np.ndarray = np.linspace(0, len(yc_plus) - 1, 70, dtype=int)

    # yc: np.ndarray = yc_plus[idx]
    # yv: np.ndarray = yv_plus[idx_v]
    # yc_uu: np.ndarray = yc_plus[idx_upup]
    yc: np.ndarray = yc_plus
    yv: np.ndarray = yv_plus
    yc_uu: np.ndarray = yc_plus

    # uu: np.ndarray = stats["uu_plus"][idx_upup]
    # vv: np.ndarray = stats["vv_plus"][idx_v]
    # ww: np.ndarray = stats["ww_plus"][idx]
    # uv: np.ndarray = stats["uv_plus"][idx]
    uu: np.ndarray = stats["uu_plus"][:]
    vv: np.ndarray = stats["vv_plus"][:]
    ww: np.ndarray = stats["ww_plus"][:]
    uv: np.ndarray = stats["uv_plus"][:]

    labels: dict[str, str] = normal_stress_wall_labels.copy()
    for key in labels:
        if not plot_labels:
            labels[key] = ""
        else:
            labels[key] = labels[key] + f" ({label})"

    def create_series(x, y, colour, marker, linestyle, label_local):
        markeredgewidth: float = 0.7
        marker_kwargs: dict = {}
        if marker != "None":
            marker_kwargs = {
                "marker": marker,
                "markerfacecolor": colour,
                "markeredgecolor": "k",
                "markeredgewidth": markeredgewidth,
                # "color": "k",
                "fillstyle": "full",
            }
        plot_kwargs: dict = {
            "label": label_local,
            "linestyle": linestyle,
            "color": colour,
            "fillstyle": "none",
        }

        plot_kwargs.update(marker_kwargs)

        return PlotSeries(
            data={
                "x": x,
                "y": y,
            },
            x_key="x",
            y_key="y",
            plot_method="plot",
            kwargs=plot_kwargs,
        )

    s_uu = create_series(
        yc_uu,
        uu,
        colours[0],
        markers[0],
        linestyles[0],
        labels["uu"],
    )
    s_vv = create_series(
        yv,
        vv,
        colours[1],
        markers[1],
        linestyles[1],
        labels["vv"],
    )
    s_ww = create_series(
        yc,
        ww,
        colours[2],
        markers[2],
        linestyles[2],
        labels["ww"],
    )
    s_uv = create_series(
        yc,
        uv,
        colours[3],
        markers[3],
        linestyles[3],
        labels["uv"],
    )

    return [s_uu, s_vv, s_ww, s_uv]


def normal_stress_wall_label_proxies(
    linestyles: list[str],
    markers: list[str],
    labels: list[str],
    colours: list[str | tuple[float, float, float, float]],
    marker_cases: list[str],
    linestyle_cases: list[str],
) -> list[PlotSeries]:
    quantities: dict[str, tuple[str, str, str]] = {}
    for i, key in enumerate(normal_stress_wall_labels):
        quantities[key] = (normal_stress_wall_labels[key], linestyles[i], markers[i])

    cases: list[tuple[str, str, str, str | tuple[float, float, float, float]]] = []
    for i, label in enumerate(labels):
        cases.append((label, linestyle_cases[i], marker_cases[i], colours[i]))

    s_quantities: list[PlotSeries] = []
    s_cases: list[PlotSeries] = []

    for key in quantities:
        s_quantities.append(
            create_proxy_series(
                "k",
                "white",
                "none",
                quantities[key][1],
                quantities[key][2],
                0.5,
                quantities[key][0],
            )
        )
    for case in cases:
        s_cases.append(
            create_proxy_series(case[3], case[3], "full", case[1], case[2], 0, case[0])
        )

    return s_quantities + s_cases


# -------------------- Steady state --------------------


def Ekin_evolution(
    h5_path: Path,
    colour: str,
    linestyle: str,
    marker: str,
    label: str | None,
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


def phi_eulerian_ana(
    csv_dir: Path,
    colour: str,
    linestyle: str,
    label: str | None,
    phi_tot: float | None,
) -> tuple[PlotSeries, PlotSeries | None]:

    y, Phi = lwidmer.read_csv_columns(
        csv_dir / "particle_eulerian_stats.csv", (0, 1), remove_nan=1
    )

    if phi_tot is None:
        Phi *= 100  # convert to %
    else:
        Phi /= phi_tot

    return phi_eulerian(y, Phi, None, colour, linestyle, label)


def phi_eulerian_vfu(
    csv_dir: Path,
    colour: str | tuple[float, float, float, float],
    linestyle: str,
    label: str | None,
    normalised: bool,
    show_err: bool,
) -> tuple[PlotSeries, PlotSeries | None]:

    if normalised:
        y, Phi, Phi_err = lwidmer.read_csv_columns(
            csv_dir / "vfu_phi_mean.csv", (0, 3, 4), remove_nan=1
        )
    else:
        y, Phi, Phi_err = lwidmer.read_csv_columns(
            csv_dir / "vfu_phi_mean.csv", (0, 1, 2), remove_nan=1
        )

    if not normalised:
        Phi *= 100  # convert to %

    if not show_err:
        Phi_err = None
    return phi_eulerian(y, Phi, Phi_err, colour, linestyle, label)


def phi_eulerian(
    y: np.ndarray,
    Phi: np.ndarray,
    Phi_err: np.ndarray | None,
    colour: str | tuple[float, float, float, float],
    linestyle: str,
    label: str | None,
) -> tuple[PlotSeries, PlotSeries | None]:

    s: PlotSeries = PlotSeries(
        data={"x": y, "y": Phi},
        x_key="x",
        y_key="y",
        plot_method="plot",
        kwargs={
            "label": label,
            "linestyle": linestyle,
            "color": colour,
        },
    )

    if Phi_err is not None:
        s_err: PlotSeries = PlotSeries(
            data={
                "x": y,
                "y": Phi,
                "y_err": Phi_err,
            },
            x_key="x",
            y_key="y",
            plot_method="err_plot",
            kwargs={
                "color": "k",
                "linestyle": "None",
                "fmt": "None",
                "ecolor": "k",
            },
        )
        return s, s_err
    return s, None


# -------------------- Lagrangian data pdf --------------------


def lagrangian_acceleration_pdf(
    csv_dir: Path,
    labels: list[str | None],
    colours: list[str | tuple[float, float, float, float]],
    markers: list[str],
    show_legend: bool,
) -> tuple[
    PlotSeries, PlotSeries, PlotSeries, PlotSeries, PlotSeries, PlotSeries, PlotSeries
]:

    a: list[np.ndarray] = [np.array([]), np.array([]), np.array([])]
    PDF: list[np.ndarray] = [np.array([]), np.array([]), np.array([])]
    err: list[np.ndarray] = [np.array([]), np.array([]), np.array([])]
    a[0], PDF[0], err[0] = lwidmer.read_csv_columns(
        csv_dir / f"particle_acceleration_pdf_x.csv", (0, 1, 2), remove_nan=2
    )
    a[1], PDF[1], err[1] = lwidmer.read_csv_columns(
        csv_dir / f"particle_acceleration_pdf_y.csv", (0, 1, 2), remove_nan=2
    )
    a[2], PDF[2], err[2] = lwidmer.read_csv_columns(
        csv_dir / f"particle_acceleration_pdf_z.csv", (0, 1, 2), remove_nan=2
    )

    a_min: float = min([np.nanmin(a_arr) for a_arr in a])
    a_max: float = max([np.nanmax(a_arr) for a_arr in a])
    num: int = int((a_max - a_min) // 0.05)
    a_fit: np.ndarray = np.linspace(a_min, a_max, num, endpoint=True)

    def standard_normal_gaussian(x: np.ndarray) -> np.ndarray:
        return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2)

    PDF_fit: np.ndarray = standard_normal_gaussian(a_fit)

    markeredgewidth: float = 0.7
    dir_labels: list[str] = ["x", "y", "z"]

    def create_series(i: int) -> tuple[PlotSeries, PlotSeries]:
        local_label: str
        if show_legend:
            if labels[i] is None:
                local_label = f"$a_{{p,{dir_labels[i]} }}$"
            else:
                local_label = f"$a_{{p,{dir_labels[i]} }}$ ({labels[i]})"
        else:
            local_label = ""

        s: PlotSeries = PlotSeries(
            data={
                "x": a[i],
                "y": PDF[i],
            },
            x_key="x",
            y_key="y",
            plot_method="semilogy",
            kwargs={
                "label": local_label,
                "linestyle": "None",
                "marker": markers[i],
                # "markerfacecolor": colours[i],
                "markeredgecolor": colours[i],
                "markeredgewidth": markeredgewidth,
                "color": colours[i],
                "fillstyle": "none",
            },
        )

        s_err: PlotSeries = PlotSeries(
            data={
                "x": a[i],
                "y": PDF[i],
                "y_err": err[i],
            },
            x_key="x",
            y_key="y",
            plot_method="err_semilogy",
            kwargs={
                "color": colours[i],
                "linestyle": "None",
                "fmt": "None",
                "ecolor": colours[i],
            },
        )

        return s, s_err

    s_fit: PlotSeries = PlotSeries(
        data={
            "x": a_fit,
            "y": PDF_fit,
        },
        x_key="x",
        y_key="y",
        plot_method="semilogy",
        kwargs={
            "label": "gaussian",
            "linestyle": "None",
            "marker": None,
            "color": "red",
            "linewidth": 0.5,
            "linestyle": "--",
        },
    )

    s_ax, s_ax_err = create_series(0)
    s_ay, s_ay_err = create_series(1)
    s_az, s_az_err = create_series(2)

    return (
        s_ax,
        s_ay,
        s_az,
        s_ax_err,
        s_ay_err,
        s_az_err,
        s_fit,
    )


def lagrangian_u_p_pdf(
    csv_dir: Path,
    yp: float | None,
    label: str | None,
    colour: str | tuple[float, float, float, float],
    marker: str,
    show_legend: bool,
) -> tuple[PlotSeries, PlotSeries]:

    csv_file: Path
    if yp is not None:
        csv_file = csv_dir / f"particle_u_plus_pdf_{yp}.csv"
    else:
        csv_file = csv_dir / f"particle_u_plus_pdf.csv"
    up, PDF, err = lwidmer.read_csv_columns(csv_file, (0, 1, 2), remove_nan=2)

    markeredgewidth: float = 0.7
    dir_labels: list[str] = ["x", "y", "z"]

    local_label: str
    if show_legend:
        if yp is None:
            if label is None:
                raise ValueError("If yp is None then label can not be None")
            else:
                local_label = label
        else:
            if label is None:
                local_label = f"$y^+ = {yp}$"
            else:
                local_label = f"$y^+ = {yp}$ ({label})"
    else:
        local_label = ""

    s_up: PlotSeries = PlotSeries(
        data={
            "x": up,
            "y": PDF,
        },
        x_key="x",
        y_key="y",
        plot_method="plot",
        kwargs={
            "label": local_label,
            "linestyle": "None",
            "marker": marker,
            # "markerfacecolor": colours[i],
            "markeredgecolor": colour,
            "markeredgewidth": markeredgewidth,
            "color": colour,
            "fillstyle": "none",
        },
    )

    s_err: PlotSeries = PlotSeries(
        data={
            "x": up,
            "y": PDF,
            "y_err": err,
        },
        x_key="x",
        y_key="y",
        plot_method="err_plot",
        kwargs={
            "color": colour,
            "linestyle": "None",
            "fmt": "None",
            "ecolor": colour,
        },
    )

    return (
        s_up,
        s_err,
    )


def lagrangian_acceleration_pdf_proxies(
    markers: list[str],
    labels: list[str],
    colours: list[str | tuple[float, float, float, float]],
    marker_cases: list[str],
) -> list[PlotSeries]:
    quantities: list[tuple[str, str]] = [
        ("$a_{p,x}$", markers[0]),
        ("$a_{p,y}$", markers[1]),
        ("$a_{p,z}$", markers[2]),
    ]

    cases: list[tuple[str, str, str | tuple[float, float, float, float]]] = []
    for i, label in enumerate(labels):
        cases.append((label, marker_cases[i], colours[i]))

    s_quantities: list[PlotSeries] = []
    s_cases: list[PlotSeries] = []

    for i in range(len(quantities)):
        s_quantities.append(
            create_proxy_series(
                colour="k",
                colour_face="white",
                fillstyle="none",
                linestyle="None",
                marker=quantities[i][1],
                markeredgewidth=0.5,
                label=quantities[i][0],
            )
        )
    for case in cases:
        s_cases.append(
            create_proxy_series(
                colour=case[2],
                colour_face=case[2],
                fillstyle="full",
                linestyle="None",
                marker=case[1],
                markeredgewidth=0.0,
                label=case[0],
            )
        )

    return s_quantities + s_cases


def lagrangian_up_pdf_proxies(
    markers: list[str],
    labels: list[str],
    yp_list: list[float],
    colours: list[str | tuple[float, float, float, float]],
    marker_cases: list[str],
) -> list[PlotSeries]:
    quantities: list[tuple[str, str]] = []
    for i, yp in enumerate(yp_list):
        quantities.append((f"$y^+ = {yp}$", markers[i]))

    cases: list[tuple[str, str, str | tuple[float, float, float, float]]] = []
    for i, label in enumerate(labels):
        cases.append((label, marker_cases[i], colours[i]))

    s_quantities: list[PlotSeries] = []
    s_cases: list[PlotSeries] = []

    for i in range(len(quantities)):
        s_quantities.append(
            create_proxy_series(
                colour="k",
                colour_face="white",
                fillstyle="none",
                linestyle="None",
                marker=quantities[i][1],
                markeredgewidth=0.5,
                label=quantities[i][0],
            )
        )
    for case in cases:
        s_cases.append(
            create_proxy_series(
                colour=case[2],
                colour_face=case[2],
                fillstyle="full",
                linestyle="None",
                marker=case[1],
                markeredgewidth=0.0,
                label=case[0],
            )
        )

    return s_quantities + s_cases


# -------------------- familiy tree --------------------


def family_tree_breakup_formation_pdf(
    csv_dir: Path,
    label: str | None,
    colour: str | tuple[float, float, float, float],
    marker: str,
    linestyle: str,
    type: Literal["breakup", "formation"],
    filtered_t_min: bool,
    name: str | None,
    show_label: bool,
) -> tuple[PlotSeries, PlotSeries]:

    y: np.ndarray
    PDF: np.ndarray
    if name is None:
        name = (
            f"floc_{type}_filtered_pdf.csv"
            if filtered_t_min
            else f"floc_{type}_non_filtered_pdf.csv"
        )
    y, edges, PDF = lwidmer.read_csv_columns(csv_dir / name, (0, 1, 2), remove_nan=1)
    print(f"len(y)={len(y)}, len(edges)={len(edges)}, len(PDF)={len(PDF)}")

    markeredgewidth: float = 0.5
    dir_labels: list[str] = ["x", "y", "z"]

    local_label: str
    if show_label:
        if label is None:
            local_label = f"{type}"
        else:
            local_label = f"{type} ({label})"
    else:
        local_label = ""

    s_plot: PlotSeries = PlotSeries(
        data={
            "x": y,
            "y": PDF,
        },
        x_key="x",
        y_key="y",
        plot_method="plot",
        kwargs={
            "label": local_label,
            "linestyle": linestyle,
            "marker": marker,
            "markersize": 5,
            "markerfacecolor": colour,
            "markeredgecolor": "k",
            "markeredgewidth": markeredgewidth,
            "color": colour,
            "fillstyle": "full",
        },
    )

    s_bar: PlotSeries = PlotSeries(
        data={
            "edges": edges,
            "counts": PDF,
        },
        x_key="x",
        y_key="y",
        plot_method="bar",
        kwargs={
            "label": local_label,
            "color": colour,
        },
    )
    return s_bar, s_plot


def noncohesive_floc_lifetime(
    csv_dir: Path,
) -> tuple[PlotSeries, PlotSeries, PlotSeries, PlotSeries, PlotSeries, PlotSeries]:

    y, edges, max_vals, std_vals, mean_vals, median_vals = lwidmer.read_csv_columns(
        csv_dir / "floc_lifetime.csv", (0, 1, 2, 3, 4, 5), remove_nan=1
    )

    markeredgewidth: float = 0.5

    s_max: PlotSeries = PlotSeries(
        data={
            "x": y,
            "y": max_vals,
        },
        x_key="x",
        y_key="y",
        plot_method="plot",
        kwargs={
            "label": "max lifetime",
            "linestyle": "None",
            "marker": "x",
            "markersize": 5,
            "markerfacecolor": "k",
            "markeredgecolor": "k",
            "markeredgewidth": markeredgewidth,
            "color": "k",
            "fillstyle": "none",
        },
    )
    s_std: PlotSeries = PlotSeries(
        data={
            "x": y,
            "y": mean_vals,
            "y_err": std_vals,
        },
        x_key="x",
        y_key="y",
        plot_method="err_plot",
        kwargs={
            "label": "standard devation",
            "linestyle": "None",
            "color": "k",
        },
    )

    s_mean: PlotSeries = PlotSeries(
        data={
            "x": y,
            "y": mean_vals,
        },
        x_key="x",
        y_key="y",
        plot_method="plot",
        kwargs={
            "label": "mean lifetime",
            "linestyle": "-",
            "color": "k",
        },
    )
    s_median: PlotSeries = PlotSeries(
        data={
            "x": y,
            "y": median_vals,
        },
        x_key="x",
        y_key="y",
        plot_method="plot",
        kwargs={
            "label": "median lifetime",
            "linestyle": ":",
            "color": "k",
        },
    )
    linregressresult = stats.linregress(y[y <= 1], max_vals[y <= 1])
    max_vals_fit = linregressresult.slope * y[y <= 1] + linregressresult.intercept
    s_max_fit: PlotSeries = PlotSeries(
        data={
            "x": y[y <= 1],
            "y": max_vals_fit,
        },
        x_key="x",
        y_key="y",
        plot_method="plot",
        kwargs={
            "label": f"fit: $y_{{max}} \\Longrightarrow t_{{floc}} = {linregressresult.slope:.3f}y + {linregressresult.intercept:.3f} (R^2={linregressresult.rvalue**2:.3f})$",
            "linestyle": "--",
            "color": "red",
        },
    )

    num_stds = 3

    linregressresult = stats.linregress(
        y[y <= 1], mean_vals[y <= 1] + std_vals[y <= 1] * num_stds
    )
    std_vals_fit = linregressresult.slope * y[y <= 1] + linregressresult.intercept
    s_model_fit: PlotSeries = PlotSeries(
        data={
            "x": y[y <= 1],
            "y": std_vals_fit,
        },
        x_key="x",
        y_key="y",
        plot_method="plot",
        kwargs={
            "label": f"fit: $y_{{mean}} + {num_stds} \\cdot \\sigma_t \\Longrightarrow t_{{floc}} = {linregressresult.slope:.3f} y + {linregressresult.intercept:.3f}$",
            "linestyle": ":",
            "color": "red",
        },
    )

    return s_max, s_std, s_mean, s_median, s_max_fit, s_model_fit


def coagulation_kernel(
    pickle_dir: Path,
    label: str | None,
    cmap: Colormap,
    xlim: tuple[float, float] | None,
    pcolormesh_log_scale: bool,
    contour_log_scale: bool,
    contour_interp_factor: int,
    contour_sigma: float,
    contour_levels: int,
    contour_color: str | None = "black",
    contour_cmap: Colormap | None = None,
) -> tuple[PlotSeries, PlotSeries]:

    ylim = xlim

    with open(pickle_dir / "number_density_evolution_params.pkl", "rb") as file:
        results: dict[str, dict] = pickle.load(file)

    K: dict[tuple[int, int], float] = results["K"]
    x_list: list[float] = results["bin_info"]["center_sizes"]
    x_idx_list: list[int] = results["bin_info"]["center_idxs"]

    x_arr: np.ndarray = np.asarray(x_list, dtype=float)
    X, Y = np.meshgrid(x_arr, x_arr)
    C: np.ndarray = np.zeros_like(X, dtype=float)

    for i, x1_idx in enumerate(x_idx_list):
        for j, x2_idx in enumerate(x_idx_list):
            C[i, j] = K[(x1_idx, x2_idx)]

    x_data_max = np.nanmax(x_arr)
    y_data_max = np.nanmax(x_arr)

    if xlim is not None:
        x_min_plot, x_max_plot = xlim
    else:
        x_min_plot, x_max_plot = 1, x_data_max

    if ylim is not None:
        y_min_plot, y_max_plot = ylim
    else:
        y_min_plot, y_max_plot = 1, y_data_max

    x_max_plot = min(x_max_plot, x_data_max)
    y_max_plot = min(y_max_plot, y_data_max)

    mask: np.ndarray = (
        (X >= x_min_plot) & (X <= x_max_plot) & (Y >= y_min_plot) & (Y <= y_max_plot)
    )
    n_rows = np.sum(mask[:, 0])
    n_cols = np.sum(mask[0, :])
    C_filtered = C[mask].reshape(n_rows, n_cols)

    C_smoothed = _gaussian_filter_2d(C, contour_sigma)
    mask: np.ndarray = (
        (X >= x_min_plot) & (X <= x_max_plot) & (Y >= y_min_plot) & (Y <= y_max_plot)
    )
    n_rows = np.sum(mask[:, 0])
    n_cols = np.sum(mask[0, :])
    X_filtered_smoothed = X[mask].reshape(n_rows, n_cols)
    Y_filtered_smoothed = Y[mask].reshape(n_rows, n_cols)
    C_filtered_smoothed = C_smoothed[mask].reshape(n_rows, n_cols)

    pcolormesh_kwargs = {
        "label": label,
        "cmap": cmap,
        "edgecolors": None,
    }
    if pcolormesh_log_scale:
        vmin = (
            np.nanmin(C_filtered[C_filtered > 0]) if np.any(C_filtered > 0) else 1e-10
        )
        vmax = np.nanmax(C_filtered)
        pcolormesh_kwargs["norm"] = LogNorm(vmin=vmin, vmax=vmax)

    s_pcolormesh: PlotSeries = PlotSeries(
        data={
            "X": X,
            "Y": Y,
            "C": C,
            "xlim": (x_min_plot, x_max_plot),
            "ylim": (y_min_plot, y_max_plot),
        },
        x_key="x",
        y_key="y",
        plot_method="pcolormesh",
        kwargs=pcolormesh_kwargs,
    )

    contour_kwargs = {
        "levels": contour_levels,
        "linewidths": 0.8,
        "interp_factor": contour_interp_factor,
    }
    if contour_log_scale:
        vmin = (
            np.nanmin(C_filtered_smoothed[C_filtered_smoothed > 0])
            if np.any(C_filtered_smoothed > 0)
            else 1e-10
        )
        vmax = np.nanmax(C_filtered_smoothed)
        contour_kwargs["norm"] = LogNorm(vmin=vmin, vmax=vmax)
    if contour_cmap is not None:
        contour_kwargs["cmap"] = contour_cmap
    elif contour_color is not None:
        contour_kwargs["colors"] = contour_color

    s_contour: PlotSeries = PlotSeries(
        data={
            "X": X_filtered_smoothed,
            "Y": Y_filtered_smoothed,
            "C": C_filtered_smoothed,
        },
        x_key="x",
        y_key="y",
        plot_method="contour",
        kwargs=contour_kwargs,
    )

    return s_pcolormesh, s_contour


def fragment_size_distribution(
    pickle_dir: Path,
    label: str | None,
    cmap: Colormap,
    xlim: tuple[float, float] | None,
    contour_sigma: float = 1.5,
    contour_levels: int = 10,
    contour_color: str | None = "black",
    contour_cmap: Colormap | None = None,
    contour_interp_factor: int = 5,
    pcolormesh_log_scale: bool = False,
    contour_log_scale: bool = False,
) -> tuple[PlotSeries, PlotSeries]:

    ylim = xlim

    with open(pickle_dir / "number_density_evolution_params.pkl", "rb") as file:
        results: dict[str, dict] = pickle.load(file)

    p: dict[tuple[int, int], float] = results["p"]
    x_list: list[float] = results["bin_info"]["center_sizes"]
    x_idx_list: list[int] = results["bin_info"]["center_idxs"]

    x_arr: np.ndarray = np.asarray(x_list, dtype=float)
    X, Y = np.meshgrid(x_arr, x_arr)
    C: np.ndarray = np.zeros_like(X, dtype=float)

    for i, x1_idx in enumerate(x_idx_list):
        for j, x2_idx in enumerate(x_idx_list):
            C[i, j] = p[(x1_idx, x2_idx)]

    x_min_default, x_max_default = 1.0, 200.0
    y_min_default, y_max_default = 1.0, 200.0

    x_data_max = np.nanmax(x_arr)
    y_data_max = np.nanmax(x_arr)

    if xlim is not None:
        x_min_plot, x_max_plot = xlim
    else:
        x_min_plot, x_max_plot = 1, None

    if ylim is not None:
        y_min_plot, y_max_plot = ylim
    else:
        y_min_plot, y_max_plot = 1, None

    if x_max_plot is None or x_data_max < x_max_plot:
        x_max_plot = x_data_max
    if y_max_plot is None or y_data_max < y_max_plot:
        y_max_plot = y_data_max

    mask: np.ndarray = (
        (X >= x_min_plot) & (X <= x_max_plot) & (Y >= y_min_plot) & (Y <= y_max_plot)
    )

    n_rows = np.sum(mask[:, 0])
    n_cols = np.sum(mask[0, :])

    X_filtered: np.ndarray = X[mask].reshape(n_rows, n_cols)
    Y_filtered: np.ndarray = Y[mask].reshape(n_rows, n_cols)
    C_filtered = C[mask].reshape(n_rows, n_cols)

    if contour_sigma > 0:
        C_smoothed = _gaussian_filter_2d(C_filtered, contour_sigma)
    else:
        C_smoothed = C_filtered

    pcolormesh_kwargs = {
        "label": label,
        "cmap": cmap,
        "edgecolors": None,
    }
    if pcolormesh_log_scale:
        # vmin = np.nanmin(C[C > 0]) if np.any(C > 0) else 1e-10
        vmin = np.nanmin(C[2 < X[:, 0] < 4][C > 0]) if np.any(C > 0) else 1e-10
        vmax = np.nanmax(C)
        pcolormesh_kwargs["norm"] = LogNorm(vmin=vmin, vmax=vmax)

    s_pcolormesh: PlotSeries = PlotSeries(
        data={
            "X": X,
            "Y": Y,
            "C": C,
            "xlim": (x_min_plot, x_max_plot),
            "ylim": (y_min_plot, y_max_plot),
        },
        x_key="x",
        y_key="y",
        plot_method="pcolormesh",
        kwargs=pcolormesh_kwargs,
    )

    contour_kwargs = {
        "levels": contour_levels,
        "linewidths": 0.8,
        "interp_factor": contour_interp_factor,
    }
    if contour_log_scale:
        # vmin = np.nanmin(C_smoothed[C_smoothed > 0]) if np.any(C_smoothed > 0) else 1e-10
        vmin = (
            np.nanmin(C_smoothed[2 < X[:, 0] < 4][C_smoothed > 0])
            if np.any(C_smoothed > 0)
            else 1e-10
        )
        vmax = np.nanmax(C_smoothed)
        contour_kwargs["norm"] = LogNorm(vmin=vmin, vmax=vmax)
    if contour_cmap is not None:
        contour_kwargs["cmap"] = contour_cmap
    elif contour_color is not None:
        contour_kwargs["colors"] = contour_color

    s_contour: PlotSeries = PlotSeries(
        data={
            "X": X_filtered,
            "Y": Y_filtered,
            "C": C_smoothed,
        },
        x_key="x",
        y_key="y",
        plot_method="contour",
        kwargs=contour_kwargs,
    )

    return s_pcolormesh, s_contour


def breakage_rate(
    pickle_dir: Path,
    colour: str | tuple[float, float, float, float],
    linestyle: str,
    label: str | None,
) -> PlotSeries:

    with open(pickle_dir / "number_density_evolution_params.pkl", "rb") as file:
        results: dict[str, dict] = pickle.load(file)

    F: dict[int, float] = results["F"]
    x: list[float] = results["bin_info"]["center_sizes"]
    x_idx_list: list[int] = results["bin_info"]["center_idxs"]

    x_arr: np.ndarray = np.asarray(x, dtype=float)
    F_arr: np.ndarray = np.zeros_like(x_arr, dtype=float)

    for i, x_idx in enumerate(x_idx_list):
        F_arr[i] = F[x_idx]

    F_arr: np.ndarray = np.zeros(len(F))
    for i, x_idx in enumerate(x_idx_list):
        F_arr[i] = F[x_idx]

    s: PlotSeries = PlotSeries(
        data={"x": x_arr, "y": F_arr},
        x_key="x",
        y_key="y",
        plot_method="plot",
        kwargs={
            "label": label,
            "color": colour,
            "linestyle": linestyle,
        },
    )

    return s


def number_density_evo_sink_source(
    data_dir: Path,
    data_names: list[str],
    labels: list[str],
    colours: list[str | tuple[float, float, float, float]],
    linestyles: list[str],
    markers: list[str],
    mass_weighted: bool,
    separate_plots: bool,
) -> tuple[
    list[PlotSeries],
    list[PlotSeries],
    list[PlotSeries],
    list[PlotSeries],
    list[PlotSeries],
    list[PlotSeries],
    list[PlotSeries],
    PlotSeries,
]:

    s_list_gain_coag: list[PlotSeries] = []
    s_list_loss_coag: list[PlotSeries] = []
    s_list_gain_frag: list[PlotSeries] = []
    s_list_loss_frag: list[PlotSeries] = []
    s_list_dn_dt: list[PlotSeries] = []

    quantities: list[str] = [
        "gain by coagulation",
        "loss by coagulation",
        "gain by fragmentation",
        "loss by fragmentation",
        r"$\frac{\partial n(n_p)}{\partial t}$",
    ]

    for data_idx, data_name in enumerate(data_names):
        with open(
            data_dir / data_name / "number_density_evolution_params.pkl", "rb"
        ) as file:
            results: dict[str, dict] = pickle.load(file)

        K_dict: dict[tuple[int, int], float] = results["K"]
        p_dict: dict[tuple[int, int], float] = results["p"]
        F_dict: dict[int, float] = results["F"]
        nu_dict: dict[int, float] = results["nu"]
        n_dict: dict[int, float] = results["n"]

        x_list: list[float] = results["bin_info"]["center_sizes"]
        x_edges_list: list[float] = results["bin_info"]["edge_sizes"]
        x_idx_list: list[int] = results["bin_info"]["center_idxs"]
        x_arr: np.ndarray = np.asarray(x_list, dtype=float)
        x_edges_arr: np.ndarray = np.asarray(x_edges_list, dtype=float)
        bin_widths: np.ndarray = x_edges_arr[1:] - x_edges_arr[:-1]
        x_idx_arr: np.ndarray = np.asarray(x_idx_list, dtype=int)
        X, _ = np.meshgrid(x_arr, x_arr)
        # X_idx, Y_idx = np.meshgrid(x_idx_list, x_arr)

        K: np.ndarray = np.zeros_like(X, dtype=float)
        for i, x1_idx in enumerate(x_idx_list):
            for j, x2_idx in enumerate(x_idx_list):
                K[i, j] = K_dict[(x1_idx, x2_idx)]
        K = np.nan_to_num(K, nan=0.0)

        p: np.ndarray = np.zeros_like(X, dtype=float)
        for i, x1_idx in enumerate(x_idx_list):
            for j, x2_idx in enumerate(x_idx_list):
                p[i, j] = p_dict[(x1_idx, x2_idx)]
        p = np.nan_to_num(p, nan=0.0)

        F: np.ndarray = np.zeros_like(x_arr, dtype=float)
        for i, x_idx in enumerate(x_idx_list):
            F[i] = F_dict[x_idx]
        F = np.nan_to_num(F, nan=0.0)

        nu: np.ndarray = np.zeros_like(x_arr, dtype=float)
        for i, x_idx in enumerate(x_idx_list):
            nu[i] = nu_dict[x_idx]
        nu = np.nan_to_num(nu, nan=2.0)

        n: np.ndarray = np.zeros_like(x_arr, dtype=float)
        for i, x_idx in enumerate(x_idx_list):
            n[i] = n_dict[x_idx]
        n = np.nan_to_num(n, nan=0.0)

        calculator = CoagulationFragmentationCalculator(
            K=K,
            F=F,
            nu=nu,
            p=p,
            n=n,
            x_arr=x_arr,
            x_edges_arr=x_edges_arr,
            x_idx_arr=x_idx_arr,
        )

        def dn_dt(
            gain_coag: np.ndarray,
            loss_coag: np.ndarray,
            gain_frag: np.ndarray,
            loss_frag: np.ndarray,
        ) -> np.ndarray:
            return gain_coag + loss_coag + gain_frag + loss_frag

        def weight_by_mass(data: np.ndarray) -> np.ndarray:
            return data * x_arr

        y_gain_coag: np.ndarray = calculator.gain_coag()
        y_loss_coag: np.ndarray = calculator.loss_coag()
        y_gain_frag: np.ndarray = calculator.gain_frag()
        y_loss_frag: np.ndarray = calculator.loss_frag()
        y_dn_dt: np.ndarray = dn_dt(y_gain_coag, y_loss_coag, y_gain_frag, y_loss_frag)

        if mass_weighted:
            y_gain_coag = weight_by_mass(y_gain_coag)
            y_loss_coag = weight_by_mass(y_loss_coag)
            y_gain_frag = weight_by_mass(y_gain_frag)
            y_loss_frag = weight_by_mass(y_loss_frag)
            y_dn_dt = weight_by_mass(y_dn_dt)

        def create_series(idx: int, y_data: np.ndarray) -> PlotSeries:
            labels_local: list[str] = ["" for _ in range(len(quantities))]
            colours_local: list[str | tuple[float, float, float, float]] = [
                colours[data_idx] for _ in range(len(quantities))
            ]
            if separate_plots:
                labels_local = [f"{quantity} ({labels[data_idx]})" for quantity in quantities]
                colours_local = colours
            return PlotSeries(
                data={"x": x_arr, "y": y_data},
                x_key="x",
                y_key="y",
                plot_method="plot",
                kwargs={
                    "label": labels_local[idx],
                    "color": colours_local[idx],
                    "linestyle": linestyles[idx],
                    "marker": markers[idx],
                },
            )

        s_list_gain_coag.append(create_series(0, y_gain_coag))
        s_list_loss_coag.append(create_series(1, y_loss_coag))
        s_list_gain_frag.append(create_series(2, y_gain_frag))
        s_list_loss_frag.append(create_series(3, y_loss_frag))
        s_list_dn_dt.append(create_series(4, y_dn_dt))

    s_quantities: list[PlotSeries] = []
    s_cases: list[PlotSeries] = []
    if not separate_plots:
        for i in range(len(quantities)): s_quantities.append( create_proxy_series(
                    "k",
                    "white",
                    "none",
                    linestyles[i],
                    "none",
                    0.5,
                    quantities[i],
                )
            )
        for i in range(len(labels)):
            s_cases.append(
                create_proxy_series(
                    colours[i], colours[i], "full", "None", "s", 0, labels[i]
                )
            )

    s_hline_zero = PlotSeries(
        data={"y": 0},
        x_key=None,
        y_key=None,
        plot_method="hline",
        kwargs={
            "color": "black",
            "linestyle": "--",
            "linewidth": 0.8,
            "alpha": 0.7,
        },
    )

    return (
        s_list_gain_coag,
        s_list_loss_coag,
        s_list_gain_frag,
        s_list_loss_frag,
        s_list_dn_dt,
        s_cases,
        s_quantities,
        s_hline_zero,
    )
