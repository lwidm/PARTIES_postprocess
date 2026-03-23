# -- src/plotting/series.py

from pathlib import Path
from typing import Literal
import h5py
from scipy import stats
from scipy.optimize import curve_fit

import numpy as np
import pickle

from matplotlib.colors import Colormap, LogNorm, TwoSlopeNorm
from matplotlib import pyplot as plt
from src.theory import law_of_the_wall as low
from src.myio import lwidmer
from src.plotting.tools import (
    PlotSeries,
    _gaussian_filter_2d,
    kFontScale,
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

    b = float(fit_data[0][0])
    Nf_eq = int(fit_data[1][0])
    n_particles = int(fit_data[2][0])

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

        for key in ["n_p", "D_f", "D_g"]:
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
    n_p_avg: np.ndarray
    n_p_mass_avg: np.ndarray
    std_D_f_d_particle_avg: np.ndarray
    std_D_g_d_particle_avg: np.ndarray
    std_D_f_d_particle_mass_avg: np.ndarray
    std_D_g_d_particle_mass_avg: np.ndarray
    std_n_p_avg: np.ndarray
    std_n_p_mass_avg: np.ndarray
    with h5py.File(str(floc_dir / "avg_diam_stats.h5"), "r") as f:
        x_data = f["y_mean"][:]  # type: ignore
        D_f_d_particle_avg = f["D_f_avg"][:]  # type: ignore
        D_g_d_particle_avg = f["D_g_avg"][:]  # type: ignore
        D_f_d_particle_mass_avg = f["D_f_mass_avg"][:]  # type: ignore
        D_g_d_particle_mass_avg = f["D_g_mass_avg"][:]  # type: ignore
        n_p_avg = f["n_p_avg"][:]  # type: ignore
        n_p_mass_avg = f["n_p_mass_avg"][:]  # type: ignore
        std_D_f_d_particle_avg = f["std_D_f_avg"][:]  # type: ignore
        std_D_g_d_particle_avg = f["std_D_g_avg"][:]  # type: ignore
        std_D_f_d_particle_mass_avg = f["std_D_f_mass_avg"][:]  # type: ignore
        std_D_g_d_particle_mass_avg = f["std_D_g_mass_avg"][:]  # type: ignore
        std_n_p_avg = f["std_n_p_avg"][:]  # type: ignore
        std_n_p_mass_avg = f["std_n_p_mass_avg"][:]  # type: ignore

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
    s_n_p_avg, s_n_p_err = create_series(n_p_avg, std_n_p_avg, 0)
    s_n_p_mass_avg, s_n_p_mass_err = create_series(n_p_mass_avg, std_n_p_mass_avg, 0)

    return (
        s_D_f_d_particle_avg,
        s_D_g_d_particle_avg,
        s_D_f_d_particle_mass_avg,
        s_D_g_d_particle_mass_avg,
        s_n_p_avg,
        s_n_p_mass_avg,
        s_D_f_d_particle_err,
        s_D_g_d_particle_err,
        s_D_f_d_particle_mass_err,
        s_D_g_d_particle_mass_err,
        s_n_p_err,
        s_n_p_mass_err,
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
    font_scale: float,
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
        font_scale,
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
    font_scale: float,
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
        font_scale,
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
    font_scale: float,
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
                "linewidth": 0.9 * font_scale,
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
                "linewidth": 0.9 * font_scale,
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

    dir_labels: list[str] = ["x", "y", "z"]

    def create_series(i: int) -> tuple[PlotSeries, PlotSeries, PlotSeries]:
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
                "markeredgecolor": colours[i],
                "color": colours[i],
                "fillstyle": "none",
            },
        )

        s_dot: PlotSeries = PlotSeries(
            data={
                "x": a[i],
                "y": PDF[i],
            },
            x_key="x",
            y_key="y",
            plot_method="semilogy",
            kwargs={
                "label": "_nolegend_",
                "linestyle": "None",
                "marker": ".",
                "color": colours[i],
                "markersize": 1.5 * kFontScale,
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

        return s, s_dot, s_err

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
            "marker": None,
            "color": "red",
            "linewidth": 0.9 * kFontScale,
            "linestyle": "--",
        },
    )

    s_ax, s_ax_dot, s_ax_err = create_series(0)
    s_ay, s_ay_dot, s_ay_err = create_series(1)
    s_az, s_az_dot, s_az_err = create_series(2)

    return (
        s_ax,
        s_ay,
        s_az,
        s_ax_dot,
        s_ay_dot,
        s_az_dot,
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
) -> tuple[PlotSeries, PlotSeries, PlotSeries]:

    csv_file: Path
    if yp is not None:
        csv_file = csv_dir / f"particle_u_plus_pdf_{yp}.csv"
    else:
        csv_file = csv_dir / f"particle_u_plus_pdf.csv"
    up, PDF, err = lwidmer.read_csv_columns(csv_file, (0, 1, 2), remove_nan=2)

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
            "markeredgecolor": colour,
            "color": colour,
            "fillstyle": "none",
        },
    )

    s_dot: PlotSeries = PlotSeries(
        data={
            "x": up,
            "y": PDF,
        },
        x_key="x",
        y_key="y",
        plot_method="plot",
        kwargs={
            "label": "_nolegend_",
            "linestyle": "None",
            "marker": ".",
            "color": colour,
            "markersize": 1.5 * kFontScale,
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
        s_dot,
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
                markeredgewidth=0.7 * kFontScale,
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
                markeredgewidth=0.7 * kFontScale,
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


def breakup_formation_pdf_proxies_separate(
    colours: list[str | tuple[float, float, float, float]],
    show_filter_proxies: bool,
) -> list[PlotSeries]:
    type_proxies: list[PlotSeries] = []
    for type_label, colour in zip(["formation", "breakup"], colours):
        type_proxies.append(
            create_proxy_series(
                colour=colour,
                colour_face=colour,
                fillstyle="full",
                linestyle="None",
                marker="s",
                markeredgewidth=0.0,
                label=type_label,
            )
        )

    filter_proxies: list[PlotSeries] = []
    if show_filter_proxies:
        filter_proxies = [
            PlotSeries(
                data={
                    "counts": np.array([-1.0]),
                    "edges": np.array([-2.0, -1.0]),
                },
                x_key="x",
                y_key="y",
                plot_method="bar",
                kwargs={"label": "filtered", "color": "k"},
            ),
            create_proxy_series(
                colour="k",
                colour_face="k",
                fillstyle="none",
                linestyle="-",
                marker="None",
                markeredgewidth=0.0,
                label="unfiltered",
            ),
        ]

    return type_proxies + filter_proxies


def breakup_formation_pdf_proxies(
    linestyles: list[str],
    labels: list[str],
    colours: list[str | tuple[float, float, float, float]],
) -> list[PlotSeries]:
    type_proxies: list[PlotSeries] = []
    for type_label, linestyle in zip(["formation", "breakup"], linestyles):
        type_proxies.append(
            create_proxy_series(
                colour="k",
                colour_face="k",
                fillstyle="none",
                linestyle=linestyle,
                marker="None",
                markeredgewidth=0.0,
                label=type_label,
            )
        )

    case_proxies: list[PlotSeries] = []
    for label, colour in zip(labels, colours):
        case_proxies.append(
            create_proxy_series(
                colour=colour,
                colour_face=colour,
                fillstyle="full",
                linestyle="-",
                marker="None",
                markeredgewidth=0.0,
                label=label,
            )
        )

    return type_proxies + case_proxies


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
    show_filter_in_label: bool,
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

    local_label: str
    if show_label:
        filter_str = " (filtered)" if filtered_t_min else " (unfiltered)"
        if label is None:
            local_label = f"{type}{filter_str if show_filter_in_label else ''}"
        else:
            local_label = f"{type} ({label}){filter_str if show_filter_in_label else ''}"
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
            "label": r"standard deviation, $\sigma$",
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
    print(f"[noncohesive_floc_lifetime] fit y_max: t_floc = {linregressresult.slope:.4f}*y + {linregressresult.intercept:.4f}  (R^2={linregressresult.rvalue**2:.4f})")
    s_max_fit: PlotSeries = PlotSeries(
        data={
            "x": y[y <= 1],
            "y": max_vals_fit,
        },
        x_key="x",
        y_key="y",
        plot_method="plot",
        kwargs={
            "label": r"fit: $y_{max}$",
            "linestyle": "--",
            "color": "red",
        },
    )

    num_stds = 3

    linregressresult = stats.linregress(
        y[y <= 1], mean_vals[y <= 1] + std_vals[y <= 1] * num_stds
    )
    std_vals_fit = linregressresult.slope * y[y <= 1] + linregressresult.intercept
    print(f"[noncohesive_floc_lifetime] fit y_mean + {num_stds}*sigma_t: t_floc = {linregressresult.slope:.4f}*y + {linregressresult.intercept:.4f}  (R^2={linregressresult.rvalue**2:.4f})")
    s_model_fit: PlotSeries = PlotSeries(
        data={
            "x": y[y <= 1],
            "y": std_vals_fit,
        },
        x_key="x",
        y_key="y",
        plot_method="plot",
        kwargs={
            "label": rf"fit: $\langle t_{{floc}} \rangle + {num_stds}\sigma$",
            "linestyle": ":",
            "color": "red",
        },
    )

    return s_max, s_mean, s_median, s_std, s_max_fit, s_model_fit


def coagulation_kernel(
    pickle_dir: Path,
    label: str | None,
    cmap: Colormap,
    xlim: tuple[float, float] | None,
    size_filter: tuple[float | None, float | None],
    pcolormesh_log_scale: bool,
    contour_log_scale: bool,
    contour_sigma: float,
    contour_levels: int,
    corrected: bool,
    x_axis_value: Literal["np", "D", "DD"],
    contour_color: str | None = "black",
    contour_cmap: Colormap | None = None,
) -> tuple[PlotSeries, PlotSeries]:

    ylim = xlim

    pickle_file = (
        "number_density_evolution_params_corrected.pkl"
        if corrected
        else "number_density_evolution_params_uncorrected.pkl"
    )
    with open(pickle_dir / pickle_file, "rb") as file:
        results: dict[str, dict] = pickle.load(file)

    K: dict[tuple[int, int], float] = results["K"]
    x_list: list[float] = results["bin_info"]["center_sizes"]
    x_idx_list: list[int] = results["bin_info"]["center_idxs"]

    K_filtered: dict[tuple[int, int], float] = {}
    x_idx_list_filtered = set()

    for x1_idx in x_idx_list:
        for x2_idx in x_idx_list:
            min_child_size = 0
            max_child_size = np.inf
            if size_filter[0] is not None:
                min_child_size = size_filter[0]
            if size_filter[1] is not None:
                max_child_size = size_filter[1]
            if (x_list[x1_idx] < min_child_size) or (x_list[x2_idx] < min_child_size):
                continue
            if (x_list[x1_idx] <= max_child_size) or (x_list[x2_idx] <= max_child_size):
                K_filtered[(x1_idx, x2_idx)] = K[(x1_idx, x2_idx)]
            else:
                K_filtered[(x1_idx, x2_idx)] = np.nan
            x_idx_list_filtered |= set([x1_idx, x2_idx])

    x_idx_list_filtered = list(x_idx_list_filtered)

    x_idx_list_filtered = sorted(x_idx_list_filtered)
    x_arr: np.ndarray = np.asarray(
        [x_list[idx] for idx in x_idx_list_filtered], dtype=float
    )
    if x_axis_value == "D":
        x_arr = np.pow(x_arr, 1 / 3)
    if x_axis_value == "DD":
        x_arr = np.pow(x_arr, 2 / 3)
    X, Y = np.meshgrid(x_arr, x_arr)
    C: np.ndarray = np.zeros_like(X, dtype=float)

    for i, x1_idx in enumerate(x_idx_list_filtered):
        for j, x2_idx in enumerate(x_idx_list_filtered):
            C[i, j] = K_filtered[(x1_idx, x2_idx)]

    x_data_max = np.nanmax(x_arr)
    y_data_max = np.nanmax(x_arr)

    column_has_data = np.any(~np.isnan(C), axis=0)
    valid_columns = np.where(column_has_data)[0]
    max_nonnan_column = valid_columns[-1] if len(valid_columns) > 0 else -1

    row_has_data = np.any(~np.isnan(C), axis=1)
    valid_rows = np.where(row_has_data)[0]
    max_nonnan_row = valid_rows[-1] if len(valid_rows) > 0 else -1

    if x_data_max > x_arr[max_nonnan_column]:
        x_data_max = x_arr[max_nonnan_column]
    if y_data_max > x_arr[max_nonnan_row]:
        y_data_max = x_arr[max_nonnan_row]

    if xlim is not None:
        x_min_plot, x_max_plot = xlim
    else:
        x_min_plot, x_max_plot = 1, x_data_max

    if ylim is not None:
        y_min_plot, y_max_plot = ylim
    else:
        y_min_plot, y_max_plot = 1, y_data_max

    # x_max_plot = min(x_max_plot, x_data_max)
    # y_max_plot = min(y_max_plot, y_data_max)

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
    ylim: tuple[float, float] | None,
    corrected: bool,
    contour_sigma: float,
    contour_levels: int,
    contour_color: str | None,
    contour_cmap: Colormap | None,
    pcolormesh_log_scale: bool,
    contour_log_scale: bool,
    normalised: bool,
) -> tuple[PlotSeries, PlotSeries]:

    xlim: tuple[float, float] | None = None
    if normalised:
        xlim = (0.0, 1.0)
    elif ylim is not None:
        xlim = ylim

    pickle_file = (
        "number_density_evolution_params_corrected.pkl"
        if corrected
        else "number_density_evolution_params_uncorrected.pkl"
    )
    with open(pickle_dir / pickle_file, "rb") as file:
        results: dict[str, dict] = pickle.load(file)

    p: dict[tuple[int, int], float] = results["p"]
    x_list: list[float] = results["bin_info"]["center_sizes"]
    x_idx_list: list[int] = results["bin_info"]["center_idxs"]

    x_arr: np.ndarray = np.asarray(x_list, dtype=float)
    n: int = len(x_list)
    X = np.zeros((n, n), dtype=float)
    Y = np.zeros((n, n), dtype=float)
    C = np.zeros((n, n), dtype=float)

    if normalised:
        x_uniform = np.linspace(0, 1, n)
        for i, y_idx in enumerate(x_idx_list):
            y: float = x_arr[y_idx]
            if y <= 0:
                continue

            valid_v: list[float] = []
            valid_c: list[float] = []
            for j, x_idx in enumerate(x_idx_list):
                x = x_arr[x_idx]
                if x <= y:
                    valid_v.append(x / y)
                    valid_c.append(p[(x_idx, y_idx)])

            v_arr = np.array(valid_v)
            c_arr = np.array(valid_c)

            edges = np.empty(len(v_arr) + 1)
            edges[0] = 0.0
            edges[-1] = 1.0
            for m in range(1, len(v_arr)):
                edges[m] = (v_arr[m - 1] + v_arr[m]) / 2

            bin_indices = np.clip(np.digitize(x_uniform, edges) - 1, 0, len(c_arr) - 1)

            X[i, :] = x_uniform
            Y[i, :] = y
            C[i, :] = c_arr[bin_indices]

            row_integral = np.trapezoid(C[i, :], x_uniform)
            if row_integral > 0:
                C[i, :] /= row_integral
    else:
        for i, y_idx in enumerate(x_idx_list):
            y: float = x_arr[y_idx]
            for j, x_idx in enumerate(x_idx_list):
                x = x_arr[x_idx]
                X[i, j] = x
                Y[i, j] = y
                C[i, j] = p[(x_idx, y_idx)]

    x_data_max = np.nanmax(X)
    y_data_max = np.nanmax(Y)

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

    row_mask = mask.any(axis=1)
    col_mask = mask.any(axis=0)
    n_rows = int(row_mask.sum())
    n_cols = int(col_mask.sum())

    X_filtered: np.ndarray = X[np.ix_(row_mask, col_mask)]
    Y_filtered: np.ndarray = Y[np.ix_(row_mask, col_mask)]
    C_filtered = C[np.ix_(row_mask, col_mask)]

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

    display_xlim = (-0.01, 1.01) if normalised else (x_min_plot, x_max_plot)

    s_pcolormesh: PlotSeries = PlotSeries(
        data={
            "X": X,
            "Y": Y,
            "C": C,
            "xlim": display_xlim,
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


def fragment_size_distribution_normalised(
    pickle_dir: Path,
    label: str | None,
    cmap: Colormap,
    ylim: tuple[float, float] | None,
    corrected: bool,
    x_axis_value: Literal["np", "D", "DD", "DD+D"],
) -> PlotSeries:

    pickle_file = (
        "daughter_aggregate_size_distribution_corrected.pkl"
        if corrected
        else "daughter_aggregate_size_distribution_uncorrected.pkl"
    )
    with open(pickle_dir / pickle_file, "rb") as file:
        results: dict[str, dict] = pickle.load(file)

    p: dict[tuple[int, int], float] = results["p_2d"]
    sizes: np.ndarray = results["bin_info"]["center_sizes"]
    sizes_idx_list: list[int] = results["bin_info"]["center_idxs"]
    sizes_parents: np.ndarray = results["parent_bin_info"]["center_sizes"]
    sizes_parents_idx_list: list[int] = results["parent_bin_info"]["center_idxs"]

    X, Y = np.meshgrid(sizes, sizes_parents)
    C: np.ndarray = np.zeros_like(X, dtype=float)
    for i in sizes_idx_list:
        for j in sizes_parents_idx_list:
            C[j, i] = p[(i, j)]

    pcolormesh_kwargs = {
        "label": label,
        "cmap": cmap,
        "edgecolors": None,
    }

    s_pcolormesh: PlotSeries = PlotSeries(
        data={
            "X": X,
            "Y": Y,
            "C": C,
            "xlim": (0.01, 1.01),
            "ylim": ylim,
        },
        x_key="x",
        y_key="y",
        plot_method="pcolormesh",
        kwargs=pcolormesh_kwargs,
    )

    return s_pcolormesh


def breakage_agglomeration_rate(
    linestyle: str,
    marker: str,
    colour: str | tuple[float, float, float, float],
    label: str | None,
    x_axis_value: Literal["np", "D", "DD", "DD+D"],
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    only_base_legend: bool,
) -> tuple[PlotSeries, PlotSeries | None]:

    plot_method: str
    if x_axis_value == "np":
        plot_method = "plot"
    else:
        plot_method = "loglog"

    s: PlotSeries = PlotSeries(
        data={"x": x_arr, "y": y_arr},
        x_key="x",
        y_key="y",
        plot_method=plot_method,
        kwargs={
            "label": label,
            "color": colour,
            "linestyle": linestyle,
            "marker": marker,
        },
    )

    s_fit: PlotSeries | None
    if x_axis_value == "np":
        s_fit = None
    else:
        x_fit_min: float = 1.5
        x_fit_max: float = 4.5

        mask: np.ndarray = (
            (x_arr >= x_fit_min)
            & (x_arr <= x_fit_max)
            & np.isfinite(x_arr)
            & np.isfinite(y_arr)
            & (y_arr != 0)
        )

        log_x = np.log(x_arr[mask])
        log_y = np.log(y_arr[mask])
        coeffs = np.polyfit(log_x, log_y, 1)
        a = coeffs[0]
        b = np.exp(coeffs[1])

        x_fit: np.ndarray = np.geomspace(x_fit_min, x_fit_max, 100)
        y_fit: np.ndarray = b * x_fit**a
        fit_label: str = ""
        if not only_base_legend:
            # fit_label =  f"${b:.3g}\\cdot x^{{{a:.3g}}}$"
            fit_label = f"$\\sim x^{{{a:.3g}}}$"

        s_fit = PlotSeries(
            data={"x": x_fit, "y": y_fit},
            x_key="x",
            y_key="y",
            plot_method=plot_method,
            kwargs={
                "label": fit_label,
                "color": colour,
                "linestyle": "--",
                "marker": "None",
            },
        )

    return s, s_fit


def breakage_rate(
    pickle_dir: Path,
    linestyle: str,
    marker: str,
    colour: str | tuple[float, float, float, float],
    label: str | None,
    corrected: bool,
    x_axis_value: Literal["np", "D"],
    only_base_legend: bool,
) -> tuple[PlotSeries, PlotSeries | None]:

    pickle_file = (
        "number_density_evolution_params_corrected.pkl"
        if corrected
        else "number_density_evolution_params_uncorrected.pkl"
    )
    with open(pickle_dir / pickle_file, "rb") as file:
        results: dict[str, dict] = pickle.load(file)

    F: dict[int, float] = results["F"]
    x: np.ndarray = results["bin_info"]["center_sizes"]
    x_idx_list: list[int] = results["bin_info"]["center_idxs"]

    x_arr: np.ndarray = np.asarray(x, dtype=float)
    F_arr: np.ndarray = np.zeros_like(x_arr, dtype=float)

    for i, x_idx in enumerate(x_idx_list):
        F_arr[i] = F[x_idx]

    if x_axis_value == "D":
        x_arr = np.pow(x_arr, 1 / 3)

    s, s_fit = breakage_agglomeration_rate(
        linestyle=linestyle,
        marker=marker,
        colour=colour,
        label=label,
        x_axis_value=x_axis_value,
        x_arr=x_arr,
        y_arr=F_arr,
        only_base_legend=only_base_legend,
    )

    return s, s_fit


def coalescence_kernel_coletti(
    pickle_dir: Path,
    size_filter: tuple[float | None, float | None],
    linestyle: str,
    marker: str,
    colour: str | tuple[float, float, float, float],
    label: str | None,
    corrected: bool,
    x_axis_value: Literal["np", "D", "DD", "DD+D"],
    only_base_legend: bool,
) -> tuple[PlotSeries, PlotSeries | None]:

    pickle_file = (
        "number_density_evolution_params_corrected.pkl"
        if corrected
        else "number_density_evolution_params_uncorrected.pkl"
    )
    with open(pickle_dir / pickle_file, "rb") as file:
        results: dict[str, dict] = pickle.load(file)

    K: dict[tuple[int, int], float] = results["K"]
    x: np.ndarray = results["bin_info"]["center_sizes"]
    x_idx_list: list[int] = results["bin_info"]["center_idxs"]

    xx_list: list[float] = []
    y_list: list[float] = []
    for x1_idx in x_idx_list:
        for x2_idx in x_idx_list:
            min_child_size = 0
            max_child_size = np.inf
            if size_filter[0] is not None:
                min_child_size = size_filter[0]
            if size_filter[1] is not None:
                max_child_size = size_filter[1]
            if x[x1_idx] < min_child_size or x[x2_idx] < min_child_size:
                continue
            if x_axis_value == "np":
                xx_list.append(x[x1_idx] + x[x2_idx])
            elif x_axis_value == "D":
                xx_list.append(np.pow(x[x1_idx], 1 / 3) + np.pow(x[x2_idx], 1 / 3))
            elif x_axis_value == "DD":
                xx_list.append(np.pow(x[x1_idx], 2 / 3) + np.pow(x[x2_idx], 2 / 3))
            elif x_axis_value == "DD+D":
                xx_list.append(
                    (np.pow(x[x1_idx], 2 / 3) + np.pow(x[x2_idx], 2 / 3))
                    * (np.pow(x[x1_idx], 1 / 3) + np.pow(x[x2_idx], 1 / 3))
                )
            else:
                raise NotImplementedError

            if (x[x1_idx] <= max_child_size) or (x[x2_idx] <= max_child_size):
                y_list.append(K[(x1_idx, x2_idx)])
            else:
                y_list.append(np.nan)

    s, s_fit = breakage_agglomeration_rate(
        linestyle=linestyle,
        marker=marker,
        colour=colour,
        label=label,
        x_axis_value=x_axis_value,
        x_arr=np.asarray(xx_list),
        y_arr=np.asarray(y_list),
        only_base_legend=only_base_legend,
    )

    return s, s_fit


def daughter_aggregate_size_distribution(
    pickle_dir: Path,
    linestyle: str,
    marker: str,
    colour: str | tuple[float, float, float, float],
    label: str | None,
    corrected: bool,
) -> PlotSeries:

    pickle_file = (
        "daughter_aggregate_size_distribution_corrected.pkl"
        if corrected
        else "daughter_aggregate_size_distribution_uncorrected.pkl"
    )
    with open(pickle_dir / pickle_file, "rb") as file:
        results: dict[str, dict] = pickle.load(file)

    p: dict[int, float] = results["p"]
    sizes: np.ndarray = results["bin_info"]["center_sizes"]
    sizes_idx_list: list[int] = results["bin_info"]["center_idxs"]

    y_list: list[float] = []
    for i in sizes_idx_list:
        y_list.append(p[i])

    s: PlotSeries = PlotSeries(
        data={"x": sizes, "y": np.asarray(y_list)},
        x_key="x",
        y_key="y",
        plot_method="plot",
        kwargs={
            "label": label,
            "color": colour,
            "linestyle": linestyle,
            "marker": marker,
        },
    )

    return s


def daughter_aggregate_size_distribution_fit(s_list: list[PlotSeries]):
    x_data: np.ndarray = np.array([])
    y_data: np.ndarray = np.array([])

    for s in s_list:
        x_data = np.concatenate((x_data, np.asarray(s.data[s.x_key])))  # type: ignore
        y_data = np.concatenate((y_data, np.asarray(s.data[s.y_key])))  # type: ignore

    mask = np.isfinite(x_data) & np.isfinite(y_data)
    x_data = x_data[mask]
    y_data = y_data[mask]

    def fit_func(x: np.ndarray, a: float, b: float) -> np.ndarray:
        return a * (x ** (-1) + (1 - x) ** (-1)) + b

    params, _ = curve_fit(fit_func, x_data, y_data)
    a_fit, b_fit = params

    x_fit: np.ndarray = np.linspace(0.01, 0.99, 200)
    y_fit: np.ndarray = fit_func(x_fit, a_fit, b_fit)
    s_fit: PlotSeries = PlotSeries(
        data={"x": x_fit, "y": y_fit},
        x_key="x",
        y_key="y",
        plot_method="plot",
        kwargs={
            "label": "fit",
            "color": "red",
            "linestyle": "--",
            "marker": "None",
        },
    )
    return s_fit


def number_density_evo_sink_source(
    data_dir: Path,
    data_names: list[str],
    labels: list[str],
    colours: list[str | tuple[float, float, float, float]],
    linestyles: list[str],
    markers: list[str],
    mass_weighted: bool,
    separate_plots: bool,
    corrected: bool,
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

    pickle_file = (
        "number_density_evolution_params_corrected.pkl"
        if corrected
        else "number_density_evolution_params_uncorrected.pkl"
    )
    for data_idx, data_name in enumerate(data_names):
        with open(data_dir / data_name / pickle_file, "rb") as file:
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
                labels_local = [
                    f"{quantity} ({labels[data_idx]})" for quantity in quantities
                ]
                colours_local = colours
            return PlotSeries(
                data={"x": x_arr, "y": y_data},
                x_key="x",
                y_key="y",
                plot_method="semilogx",
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
        for i in range(len(quantities)):
            s_quantities.append(
                create_proxy_series(
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


def cumulative_floculation_balance(
    data_dir: Path,
    data_names: list[str],
    labels: list[str],
    colours: list[str | tuple[float, float, float, float]],
    linestyles: list[str],
    markers: list[str],
    mass_weighted: bool,
    corrected: bool,
    separate_plots: bool,
    plot_dn_dt: bool,
) -> tuple[
    list[PlotSeries],
    list[PlotSeries],
    list[PlotSeries],
    list[PlotSeries],
    list[PlotSeries],
    PlotSeries,
]:

    csv_filename = (
        "floculation_balance_corrected.csv" if corrected else "floculation_balance.csv"
    )

    s_list_coag: list[PlotSeries] = []
    s_list_frag: list[PlotSeries] = []
    s_list_dn_dt: list[PlotSeries] = []

    for data_idx, data_name in enumerate(data_names):
        csv_path = data_dir / data_name / csv_filename

        if mass_weighted:
            n_p, T_coag_cumsum, T_frag_cumsum, dn_dt_cumsum = lwidmer.read_csv_columns(
                csv_path, (0, 10, 11, 12), remove_nan=1
            )
        else:
            n_p, T_coag_cumsum, T_frag_cumsum, dn_dt_cumsum = lwidmer.read_csv_columns(
                csv_path, (0, 7, 8, 9), remove_nan=1
            )

        labels_local_coag: str = ""
        labels_local_frag: str = ""
        labels_local_dn_dt: str = ""
        if separate_plots:
            labels_local_coag = f"$T_{{coag}}$ ({labels[data_idx]})"
            labels_local_frag = f"$T_{{frag}}$ ({labels[data_idx]})"
            labels_local_dn_dt = f"$dn/dt$ ({labels[data_idx]})"

        s_coag = PlotSeries(
            data={"x": n_p, "y": T_coag_cumsum},
            x_key="x",
            y_key="y",
            plot_method="semilogx",
            kwargs={
                "label": labels_local_coag,
                "color": colours[data_idx],
                "linestyle": linestyles[0],
                "marker": markers[data_idx] if markers[data_idx] else "",
            },
        )
        s_list_coag.append(s_coag)

        s_frag = PlotSeries(
            data={"x": n_p, "y": T_frag_cumsum},
            x_key="x",
            y_key="y",
            plot_method="plot",
            kwargs={
                "label": labels_local_frag,
                "color": colours[data_idx],
                "linestyle": linestyles[1],
                "marker": markers[data_idx] if markers[data_idx] else "",
            },
        )
        s_list_frag.append(s_frag)

        if plot_dn_dt:
            s_dn_dt = PlotSeries(
                data={"x": n_p, "y": dn_dt_cumsum},
                x_key="x",
                y_key="y",
                plot_method="plot",
                kwargs={
                    "label": labels_local_dn_dt,
                    "color": colours[data_idx],
                    "linestyle": linestyles[2] if len(linestyles) > 2 else "-",
                    "marker": markers[data_idx] if markers[data_idx] else "",
                },
            )
            s_list_dn_dt.append(s_dn_dt)

    s_quantities: list[PlotSeries] = []
    s_cases: list[PlotSeries] = []
    if not separate_plots:
        quantities: list[tuple[str, str]] = [
            (r"$T_{coag}$", linestyles[0]),
            (r"$T_{frag}$", linestyles[1]),
        ]
        if plot_dn_dt:
            quantities.append(
                (r"$dn/dt$", linestyles[2] if len(linestyles) > 2 else "-")
            )
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

    return s_list_coag, s_list_frag, s_list_dn_dt, s_cases, s_quantities, s_hline_zero


# ==================== DIFFERENCE PLOTTING FUNCTIONS ====================


def coagulation_kernel_diff(
    pickle_dir: Path,
    label: str | None,
    xlim: tuple[float, float] | None,
) -> PlotSeries:

    ylim = xlim

    # Load corrected data
    with open(
        pickle_dir / "number_density_evolution_params_corrected.pkl", "rb"
    ) as file:
        results_corr: dict[str, dict] = pickle.load(file)

    # Load uncorrected data
    with open(
        pickle_dir / "number_density_evolution_params_uncorrected.pkl", "rb"
    ) as file:
        results_uncorr: dict[str, dict] = pickle.load(file)

    K_corr: dict[tuple[int, int], float] = results_corr["K"]
    K_uncorr: dict[tuple[int, int], float] = results_uncorr["K"]
    x_list: list[float] = results_corr["bin_info"]["center_sizes"]
    x_idx_list: list[int] = results_corr["bin_info"]["center_idxs"]

    x_arr: np.ndarray = np.asarray(x_list, dtype=float)
    X, Y = np.meshgrid(x_arr, x_arr)
    C_corr: np.ndarray = np.zeros_like(X, dtype=float)
    C_uncorr: np.ndarray = np.zeros_like(X, dtype=float)

    for i, x1_idx in enumerate(x_idx_list):
        for j, x2_idx in enumerate(x_idx_list):
            C_corr[i, j] = K_corr[(x1_idx, x2_idx)]
            C_uncorr[i, j] = K_uncorr[(x1_idx, x2_idx)]

    # Compute difference
    C = C_corr - C_uncorr
    C = np.nan_to_num(C, nan=0.0)

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

    # Use diverging colormap centered at zero
    cmap = plt.cm.RdBu_r
    vmax = np.nanmax(np.abs(C))
    vmin = -vmax

    pcolormesh_kwargs = {
        "label": label,
        "cmap": cmap,
        "edgecolors": None,
        "norm": TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax),
    }

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

    return s_pcolormesh


def fragment_size_distribution_diff(
    pickle_dir: Path,
    label: str | None,
    xlim: tuple[float, float] | None,
) -> PlotSeries:

    ylim = xlim

    # Load corrected data
    with open(
        pickle_dir / "number_density_evolution_params_corrected.pkl", "rb"
    ) as file:
        results_corr: dict[str, dict] = pickle.load(file)

    # Load uncorrected data
    with open(
        pickle_dir / "number_density_evolution_params_uncorrected.pkl", "rb"
    ) as file:
        results_uncorr: dict[str, dict] = pickle.load(file)

    p_corr: dict[tuple[int, int], float] = results_corr["p"]
    p_uncorr: dict[tuple[int, int], float] = results_uncorr["p"]
    x_list: list[float] = results_corr["bin_info"]["center_sizes"]
    x_idx_list: list[int] = results_corr["bin_info"]["center_idxs"]

    x_arr: np.ndarray = np.asarray(x_list, dtype=float)
    X, Y = np.meshgrid(x_arr, x_arr)
    C_corr: np.ndarray = np.zeros_like(X, dtype=float)
    C_uncorr: np.ndarray = np.zeros_like(X, dtype=float)

    for i, x1_idx in enumerate(x_idx_list):
        for j, x2_idx in enumerate(x_idx_list):
            C_corr[j, i] = p_corr[(x1_idx, x2_idx)]
            C_uncorr[j, i] = p_uncorr[(x1_idx, x2_idx)]

    # Compute difference
    C = C_corr - C_uncorr
    C = np.nan_to_num(C, nan=0.0)

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

    # Use diverging colormap centered at zero
    cmap = plt.cm.RdBu_r
    vmax = np.nanmax(np.abs(C))
    vmin = -vmax

    pcolormesh_kwargs = {
        "label": label,
        "cmap": cmap,
        "edgecolors": None,
        "norm": TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax),
    }

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

    return s_pcolormesh


def breakage_rate_diff(
    pickle_dir: Path,
    colour: str | tuple[float, float, float, float],
    linestyle: str,
    label: str | None,
) -> PlotSeries:
    # Load corrected data
    with open(
        pickle_dir / "number_density_evolution_params_corrected.pkl", "rb"
    ) as file:
        results_corr: dict[str, dict] = pickle.load(file)

    # Load uncorrected data
    with open(
        pickle_dir / "number_density_evolution_params_uncorrected.pkl", "rb"
    ) as file:
        results_uncorr: dict[str, dict] = pickle.load(file)

    F_corr: dict[int, float] = results_corr["F"]
    F_uncorr: dict[int, float] = results_uncorr["F"]
    x: list[float] = results_corr["bin_info"]["center_sizes"]
    x_idx_list: list[int] = results_corr["bin_info"]["center_idxs"]

    x_arr: np.ndarray = np.asarray(x, dtype=float)
    F_arr_corr: np.ndarray = np.zeros_like(x_arr, dtype=float)
    F_arr_uncorr: np.ndarray = np.zeros_like(x_arr, dtype=float)

    for i, x_idx in enumerate(x_idx_list):
        F_arr_corr[i] = F_corr[x_idx]
        F_arr_uncorr[i] = F_uncorr[x_idx]

    # Compute difference
    F_arr_diff: np.ndarray = F_arr_corr - F_arr_uncorr
    F_arr_diff = np.nan_to_num(F_arr_diff, nan=0.0)

    s: PlotSeries = PlotSeries(
        data={"x": x_arr, "y": F_arr_diff},
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


def number_density_evo_sink_source_diff(
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
        "gain by coagulation (diff)",
        "loss by coagulation (diff)",
        "gain by fragmentation (diff)",
        "loss by fragmentation (diff)",
        r"$\frac{\partial n(n_p)}{\partial t}$ (diff)",
    ]

    for data_idx, data_name in enumerate(data_names):
        # Load corrected data
        with open(
            data_dir / data_name / "number_density_evolution_params_corrected.pkl", "rb"
        ) as file:
            results_corr: dict[str, dict] = pickle.load(file)

        # Load uncorrected data
        with open(
            data_dir / data_name / "number_density_evolution_params_uncorrected.pkl",
            "rb",
        ) as file:
            results_uncorr: dict[str, dict] = pickle.load(file)

        # Extract corrected data
        K_dict_corr: dict[tuple[int, int], float] = results_corr["K"]
        p_dict_corr: dict[tuple[int, int], float] = results_corr["p"]
        F_dict_corr: dict[int, float] = results_corr["F"]
        nu_dict_corr: dict[int, float] = results_corr["nu"]
        n_dict_corr: dict[int, float] = results_corr["n"]

        # Extract uncorrected data
        K_dict_uncorr: dict[tuple[int, int], float] = results_uncorr["K"]
        p_dict_uncorr: dict[tuple[int, int], float] = results_uncorr["p"]
        F_dict_uncorr: dict[int, float] = results_uncorr["F"]
        nu_dict_uncorr: dict[int, float] = results_uncorr["nu"]
        n_dict_uncorr: dict[int, float] = results_uncorr["n"]

        x_list: list[float] = results_corr["bin_info"]["center_sizes"]
        x_edges_list: list[float] = results_corr["bin_info"]["edge_sizes"]
        x_idx_list: list[int] = results_corr["bin_info"]["center_idxs"]
        x_arr: np.ndarray = np.asarray(x_list, dtype=float)
        x_edges_arr: np.ndarray = np.asarray(x_edges_list, dtype=float)
        bin_widths: np.ndarray = x_edges_arr[1:] - x_edges_arr[:-1]
        x_idx_arr: np.ndarray = np.asarray(x_idx_list, dtype=int)
        X, _ = np.meshgrid(x_arr, x_arr)

        # Build arrays for corrected data
        K_corr: np.ndarray = np.zeros_like(X, dtype=float)
        for i, x1_idx in enumerate(x_idx_list):
            for j, x2_idx in enumerate(x_idx_list):
                K_corr[i, j] = K_dict_corr[(x1_idx, x2_idx)]
        K_corr = np.nan_to_num(K_corr, nan=0.0)

        p_corr: np.ndarray = np.zeros_like(X, dtype=float)
        for i, x1_idx in enumerate(x_idx_list):
            for j, x2_idx in enumerate(x_idx_list):
                p_corr[i, j] = p_dict_corr[(x1_idx, x2_idx)]
        p_corr = np.nan_to_num(p_corr, nan=0.0)

        F_corr: np.ndarray = np.zeros_like(x_arr, dtype=float)
        for i, x_idx in enumerate(x_idx_list):
            F_corr[i] = F_dict_corr[x_idx]
        F_corr = np.nan_to_num(F_corr, nan=0.0)

        nu_corr: np.ndarray = np.zeros_like(x_arr, dtype=float)
        for i, x_idx in enumerate(x_idx_list):
            nu_corr[i] = nu_dict_corr[x_idx]
        nu_corr = np.nan_to_num(nu_corr, nan=2.0)

        n_corr: np.ndarray = np.zeros_like(x_arr, dtype=float)
        for i, x_idx in enumerate(x_idx_list):
            n_corr[i] = n_dict_corr[x_idx]
        n_corr = np.nan_to_num(n_corr, nan=0.0)

        # Build arrays for uncorrected data
        K_uncorr: np.ndarray = np.zeros_like(X, dtype=float)
        for i, x1_idx in enumerate(x_idx_list):
            for j, x2_idx in enumerate(x_idx_list):
                K_uncorr[i, j] = K_dict_uncorr[(x1_idx, x2_idx)]
        K_uncorr = np.nan_to_num(K_uncorr, nan=0.0)

        p_uncorr: np.ndarray = np.zeros_like(X, dtype=float)
        for i, x1_idx in enumerate(x_idx_list):
            for j, x2_idx in enumerate(x_idx_list):
                p_uncorr[i, j] = p_dict_uncorr[(x1_idx, x2_idx)]
        p_uncorr = np.nan_to_num(p_uncorr, nan=0.0)

        F_uncorr: np.ndarray = np.zeros_like(x_arr, dtype=float)
        for i, x_idx in enumerate(x_idx_list):
            F_uncorr[i] = F_dict_uncorr[x_idx]
        F_uncorr = np.nan_to_num(F_uncorr, nan=0.0)

        nu_uncorr: np.ndarray = np.zeros_like(x_arr, dtype=float)
        for i, x_idx in enumerate(x_idx_list):
            nu_uncorr[i] = nu_dict_uncorr[x_idx]
        nu_uncorr = np.nan_to_num(nu_uncorr, nan=2.0)

        n_uncorr: np.ndarray = np.zeros_like(x_arr, dtype=float)
        for i, x_idx in enumerate(x_idx_list):
            n_uncorr[i] = n_dict_uncorr[x_idx]
        n_uncorr = np.nan_to_num(n_uncorr, nan=0.0)

        # Calculate corrected values
        calculator_corr = CoagulationFragmentationCalculator(
            K=K_corr,
            F=F_corr,
            nu=nu_corr,
            p=p_corr,
            n=n_corr,
            x_arr=x_arr,
            x_edges_arr=x_edges_arr,
            x_idx_arr=x_idx_arr,
        )

        # Calculate uncorrected values
        calculator_uncorr = CoagulationFragmentationCalculator(
            K=K_uncorr,
            F=F_uncorr,
            nu=nu_uncorr,
            p=p_uncorr,
            n=n_uncorr,
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

        # Compute differences
        y_gain_coag_diff: np.ndarray = (
            calculator_corr.gain_coag() - calculator_uncorr.gain_coag()
        )
        y_loss_coag_diff: np.ndarray = (
            calculator_corr.loss_coag() - calculator_uncorr.loss_coag()
        )
        y_gain_frag_diff: np.ndarray = (
            calculator_corr.gain_frag() - calculator_uncorr.gain_frag()
        )
        y_loss_frag_diff: np.ndarray = (
            calculator_corr.loss_frag() - calculator_uncorr.loss_frag()
        )
        y_dn_dt_diff: np.ndarray = dn_dt(
            y_gain_coag_diff, y_loss_coag_diff, y_gain_frag_diff, y_loss_frag_diff
        )

        if mass_weighted:
            y_gain_coag_diff = weight_by_mass(y_gain_coag_diff)
            y_loss_coag_diff = weight_by_mass(y_loss_coag_diff)
            y_gain_frag_diff = weight_by_mass(y_gain_frag_diff)
            y_loss_frag_diff = weight_by_mass(y_loss_frag_diff)
            y_dn_dt_diff = weight_by_mass(y_dn_dt_diff)

        def create_series(idx: int, y_data: np.ndarray) -> PlotSeries:
            labels_local: list[str] = ["" for _ in range(len(quantities))]
            colours_local: list[str | tuple[float, float, float, float]] = [
                colours[data_idx] for _ in range(len(quantities))
            ]
            if separate_plots:
                labels_local = [
                    f"{quantity} ({labels[data_idx]})" for quantity in quantities
                ]
                colours_local = colours

            return PlotSeries(
                data={"x": x_arr, "y": y_data},
                x_key="x",
                y_key="y",
                plot_method="plot",
                kwargs={
                    "label": labels_local[idx],
                    "color": colours_local[idx],
                    "linestyle": (
                        linestyles[idx] if not separate_plots else linestyles[0]
                    ),
                    "marker": markers[data_idx] if markers[data_idx] else "",
                },
            )

        s_list_gain_coag.append(create_series(0, y_gain_coag_diff))
        s_list_loss_coag.append(create_series(1, y_loss_coag_diff))
        s_list_gain_frag.append(create_series(2, y_gain_frag_diff))
        s_list_loss_frag.append(create_series(3, y_loss_frag_diff))
        s_list_dn_dt.append(create_series(4, y_dn_dt_diff))

    s_quantities: list[PlotSeries] = []
    s_cases: list[PlotSeries] = []
    if not separate_plots:
        for i in range(len(quantities)):
            s_quantities.append(
                create_proxy_series(
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


def cumulative_floculation_balance_diff(
    data_dir: Path,
    data_names: list[str],
    labels: list[str],
    colours: list[str | tuple[float, float, float, float]],
    linestyles: list[str],
    markers: list[str],
    mass_weighted: bool,
    separate_plots: bool,
    plot_dn_dt: bool,
) -> tuple[
    list[PlotSeries],
    list[PlotSeries],
    list[PlotSeries],
    list[PlotSeries],
    list[PlotSeries],
    PlotSeries,
]:
    s_list_coag: list[PlotSeries] = []
    s_list_frag: list[PlotSeries] = []
    s_list_dn_dt: list[PlotSeries] = []

    for data_idx, data_name in enumerate(data_names):
        csv_path_corr = data_dir / data_name / "floculation_balance_corrected.csv"
        csv_path_uncorr = data_dir / data_name / "floculation_balance.csv"

        if mass_weighted:
            n_p_corr, T_coag_cumsum_corr, T_frag_cumsum_corr, dn_dt_cumsum_corr = (
                lwidmer.read_csv_columns(csv_path_corr, (0, 10, 11, 12), remove_nan=1)
            )
            (
                n_p_uncorr,
                T_coag_cumsum_uncorr,
                T_frag_cumsum_uncorr,
                dn_dt_cumsum_uncorr,
            ) = lwidmer.read_csv_columns(csv_path_uncorr, (0, 10, 11, 12), remove_nan=1)
        else:
            n_p_corr, T_coag_cumsum_corr, T_frag_cumsum_corr, dn_dt_cumsum_corr = (
                lwidmer.read_csv_columns(csv_path_corr, (0, 7, 8, 9), remove_nan=1)
            )
            (
                n_p_uncorr,
                T_coag_cumsum_uncorr,
                T_frag_cumsum_uncorr,
                dn_dt_cumsum_uncorr,
            ) = lwidmer.read_csv_columns(csv_path_uncorr, (0, 7, 8, 9), remove_nan=1)

        # Compute differences
        T_coag_cumsum_diff: np.ndarray = T_coag_cumsum_corr - T_coag_cumsum_uncorr
        T_frag_cumsum_diff: np.ndarray = T_frag_cumsum_corr - T_frag_cumsum_uncorr
        dn_dt_cumsum_diff = dn_dt_cumsum_corr - dn_dt_cumsum_uncorr
        T_coag_cumsum_diff = np.nan_to_num(T_coag_cumsum_diff, nan=0.0)
        T_frag_cumsum_diff = np.nan_to_num(T_frag_cumsum_diff, nan=0.0)
        dn_dt_cumsum_diff = np.nan_to_num(dn_dt_cumsum_diff, nan=0.0)

        labels_local_coag: str = ""
        labels_local_frag: str = ""
        labels_local_dn_dt: str = ""
        if separate_plots:
            labels_local_coag = f"$T_{{coag}}$ diff ({labels[data_idx]})"
            labels_local_frag = f"$T_{{frag}}$ diff ({labels[data_idx]})"
            labels_local_dn_dt = f"$dn/dt$ diff ({labels[data_idx]})"

        s_coag = PlotSeries(
            data={"x": n_p_corr, "y": T_coag_cumsum_diff},
            x_key="x",
            y_key="y",
            plot_method="semilogx",
            kwargs={
                "label": labels_local_coag,
                "color": colours[data_idx],
                "linestyle": linestyles[0],
                "marker": markers[data_idx] if markers[data_idx] else "",
            },
        )
        s_list_coag.append(s_coag)

        s_frag = PlotSeries(
            data={"x": n_p_corr, "y": T_frag_cumsum_diff},
            x_key="x",
            y_key="y",
            plot_method="plot",
            kwargs={
                "label": labels_local_frag,
                "color": colours[data_idx],
                "linestyle": linestyles[1],
                "marker": markers[data_idx] if markers[data_idx] else "",
            },
        )
        s_list_frag.append(s_frag)

        if plot_dn_dt:
            s_dn_dt = PlotSeries(
                data={"x": n_p_corr, "y": dn_dt_cumsum_diff},
                x_key="x",
                y_key="y",
                plot_method="plot",
                kwargs={
                    "label": labels_local_dn_dt,
                    "color": colours[data_idx],
                    "linestyle": linestyles[2] if len(linestyles) > 2 else "-",
                    "marker": markers[data_idx] if markers[data_idx] else "",
                },
            )
            s_list_dn_dt.append(s_dn_dt)

    s_quantities: list[PlotSeries] = []
    s_cases: list[PlotSeries] = []
    if not separate_plots:
        quantities: list[tuple[str, str]] = [
            (r"$T_{coag}$ diff", linestyles[0]),
            (r"$T_{frag}$ diff", linestyles[1]),
        ]
        if plot_dn_dt:
            quantities.append(
                (r"$dn/dt$ diff", linestyles[2] if len(linestyles) > 2 else "-")
            )
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

    return s_list_coag, s_list_frag, s_list_dn_dt, s_cases, s_quantities, s_hline_zero


def total_frequencies(
    data_dir: Path,
    data_names_sets: list[list[str]],
    phi_values_sets: list[list[float]],
    labels: list[str | None],
    colours: list[str | tuple[float, float, float, float]],
    floc_markers: list[str],
    break_markers: list[str],
    corrected: bool = True,
) -> tuple[list[PlotSeries], list[PlotSeries]]:
    import pickle

    markeredgewidth: float = 2

    s_list_floc: list[PlotSeries] = []
    s_list_break: list[PlotSeries] = []

    for set_idx, (data_names, phi_values) in enumerate(
        zip(data_names_sets, phi_values_sets)
    ):
        phi_list: list[float] = []
        floc_freq_list: list[float] = []
        break_freq_list: list[float] = []

        for data_name, phi in zip(data_names, phi_values):
            pickle_file: Path
            if corrected:
                pickle_file = (
                    data_dir
                    / data_name
                    / "number_density_evolution_params_corrected.pkl"
                )
            else:
                pickle_file = (
                    data_dir
                    / data_name
                    / "number_density_evolution_params_uncorrected.pkl"
                )

            if not pickle_file.exists():
                continue

            with open(pickle_file, "rb") as f:
                params: dict = pickle.load(f)

            K_dict: dict[tuple[int, int], float] = params["K"]
            F_dict: dict[int, float] = params["F"]
            n_dict: dict[int, float] = params["n"]
            bin_info: dict[str, list[int] | list[float]] = params["bin_info"]

            center_idxs_list: list[int] = bin_info["center_idxs"]
            center_sizes_list: list[float] = bin_info["center_sizes"]
            edge_sizes_list: list[float] = bin_info["edge_sizes"]

            bin_width_eff: float = edge_sizes_list[1] - edge_sizes_list[0]

            center_idxs_arr: np.ndarray = np.asarray(center_idxs_list, dtype=int)
            center_sizes_arr: np.ndarray = np.asarray(center_sizes_list, dtype=float)

            X, _ = np.meshgrid(center_sizes_arr, center_sizes_arr)
            K: np.ndarray = np.zeros_like(X, dtype=float)
            for i, x1_idx in enumerate(center_idxs_list):
                for j, x2_idx in enumerate(center_idxs_list):
                    K[i, j] = K_dict[(x1_idx, x2_idx)]
            K = np.nan_to_num(K, nan=0.0)

            F: np.ndarray = np.zeros_like(center_sizes_arr, dtype=float)
            for i, x_idx in enumerate(center_idxs_list):
                F[i] = F_dict[x_idx]
            F = np.nan_to_num(F, nan=0.0)

            n: np.ndarray = np.zeros_like(center_sizes_arr, dtype=float)
            for i, x_idx in enumerate(center_idxs_list):
                n[i] = n_dict[x_idx]
            n = np.nan_to_num(n, nan=0.0)

            floc_freq: float = 0.5 * float(
                np.sum(K * n[:, None] * n[None, :] * bin_width_eff**2)
            )
            break_freq: float = float(np.sum(F * n * bin_width_eff))

            phi_list.append(phi)
            floc_freq_list.append(floc_freq)
            break_freq_list.append(break_freq)

        s_floc = PlotSeries(
            data={"x": phi_list, "y": floc_freq_list},
            x_key="x",
            y_key="y",
            plot_method="plot",
            kwargs={
                # "label": labels[set_idx] if labels[set_idx] else f"Set {set_idx}",
                "label": "Aggregation rate",
                "linestyle": "None",
                "marker": floc_markers[set_idx],
                "markerfacecolor": colours[set_idx],
                "markersize": 10,
                "markeredgecolor": "k",
                "markeredgewidth": markeredgewidth,
                "color": colours[set_idx],
                "fillstyle": "full",
            },
        )
        s_list_floc.append(s_floc)

        s_break = PlotSeries(
            data={"x": phi_list, "y": break_freq_list},
            x_key="x",
            y_key="y",
            plot_method="plot",
            kwargs={
                # "label": labels[set_idx] if labels[set_idx] else f"Set {set_idx}",
                "label": "Breakup rate",
                "linestyle": "None",
                "marker": break_markers[set_idx],
                "markerfacecolor": colours[set_idx],
                "markersize": 10,
                "markeredgecolor": "k",
                "markeredgewidth": markeredgewidth,
                "color": colours[set_idx],
                "fillstyle": "full",
            },
        )
        s_list_break.append(s_break)

    return s_list_floc, s_list_break
