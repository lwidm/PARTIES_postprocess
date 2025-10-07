import h5py  # type: ignore
import numpy as np
import matplotlib
from matplotlib.axes import Axes
from pathlib import Path
import os
from typing import Optional, List, Tuple, Dict, Literal, Any

from myio import myio
import theory
from fluid import flow_statistics
from flocs.find_flocs import find_flocs
import scripts

# =============================================================================
# CONFIGURATION AND CONSTANTS
# =============================================================================

ON_ANVIL: bool = os.getenv("MY_MACHINE", "") == "anvil"

if not ON_ANVIL:
    matplotlib.use("Qt5Agg")
from matplotlib import pyplot as plt  # noqa: E402

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern"],
        "text.latex.preamble": r"\usepackage{amsmath}",
    }
)

BLAS_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

Re: float = 2800.0
output_dir: str = "./learning/output"
parties_data_dir: str = "./learning/data"
utexas_data_dir: str = "./learning/data"

num_workers_single_component: Optional[int] = 5
num_workers_cross_component: Optional[int] = 2
min_file_index: Optional[int] = 233

if ON_ANVIL:
    output_dir = "/home/x-lwidmer/Documents/PARTIES_postprocess/learning/output"
    utexas_data_dir = "/home/x-lwidmer/Documents/PARTIES_postprocess/learning/data"
    parties_data_dir = "/anvil/scratch/x-lwidmer/RUN5"
    num_workers_single_component = 8
    num_workers_cross_component = 4
    min_file_index = 250

# =============================================================================
# FlOW STATISTICS AND REYNOLDS STRESSES
# =============================================================================

# ########## Computation ##########

def compute_all_reynolds_stresses_ADLeonelli(
    min_file_index: Optional[int] = None,
    max_file_index: Optional[int] = None,
    num_workers_single_component: Optional[int] = None,
    num_workers_cross_component: Optional[int] = None,
    use_threads: bool = False,
    save_intermediates: bool = True,
) -> Dict[str, Any]:
    """
    Compute all Reynolds stresses using ADLeonelli's code

    This function processes components one at a time, saves intermediate results,
    and computes Reynolds stresses while minimizing memory usage.

    Args:
        min_file_index: Minimum file index to process
        max_file_index: Maximum file index to process
        num_workers: Number of parallel workers
        use_threads: Use threads instead of processes
        save_intermediates: Whether to save intermediate results

    Returns:
        Dictionary containing all final results including wall units
    """
    print("Starting Reynolds stress computation...")

    data_files: List[Path] = myio.list_parties_data_files(
        parties_data_dir, "Data", min_file_index, max_file_index
    )
    if not data_files:
        return {}

    grid: Dict[str, np.ndarray] = flow_statistics.get_grid(data_files[0])

    results = flow_statistics.process_mean_flow(data_files, grid)
    results = results | flow_statistics.process_fluctuations(data_files, results, grid)

    tau_w, u_tau = flow_statistics.calc_friction_velocity(results, grid, Re)

    results = results | flow_statistics.get_wall_units(results, grid, Re, tau_w, u_tau)

    myio.save_to_h5(
        f"{output_dir}/reynolds_stresses.h5",
        results,
        {
            "min_index": min_file_index,
            "max_index": max_file_index,
            "num_files_processed": len(data_files),
        },
    )

    return results


def collect_flow_statistics(
    # processing_method: Literal["step_by_step", "saved", "ADLeonelli"],
    processing_method: Literal["saved", "ADLeonelli"],
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    float,
    float,
]:
    """
    Main data retrieval function that coordinates data processing and theory.

    Args:
        processing_method: One of "step_by_step" or "saved"

    Returns:
        Comprehensive tuple containing all processed data for plotting
    """
    # Load reference data from utexas
    utexas_mean_data_file: str = f"{utexas_data_dir}/LM_Channel_0180_mean_prof.dat"
    utexas_fluc_data_file: str = f"{utexas_data_dir}/LM_Channel_0180_vel_fluc_prof.dat"
    (
        utexas_y_delta,
        utexas_y_plus,
        utexas_u_plus,
        utexas_velocity_gradient,
        utexas_w,
        utexas_p,
    ) = myio.load_columns_from_txt_numpy(utexas_mean_data_file)
    (
        _,
        _,
        utexas_upup_plus,
        utexas_vpvp_plus,
        utexas_wpwp_plus,
        utexas_upvp_plus,
        utexas_upwp_plus,
        utexas_vpwp_plus,
        utexas_k_plus,
    ) = myio.load_columns_from_txt_numpy(utexas_fluc_data_file)

    parties_results: Dict[str, Any]
    if processing_method == "ADLeonelli":
        parties_results = compute_all_reynolds_stresses_ADLeonelli(
            min_file_index=min_file_index,
            num_workers_single_component=num_workers_single_component,
            num_workers_cross_component=num_workers_cross_component,
            use_threads=False,
        )
    # elif processing_method == "step_by_step":
    #     parties_results = compute_all_reynolds_stresses_step_by_step(
    #         min_file_index=min_file_index,
    #         num_workers_single_component=num_workers_single_component,
    #         num_workers_cross_component=num_workers_cross_component,
    #         use_threads=False,
    #     )
    elif processing_method == "saved":
        parties_results, _ = myio.load_from_h5(f"{output_dir}/reynolds_stresses.h5")
    else:
        raise ValueError(
            f'processing_method must be one of ["step_by_step", "saved", "ADLeonelli"]. Got: {processing_method}'
        )

    # Extract required values from results dictionary
    parties_yc_plus = parties_results["yc_plus"]
    parties_yv_plus = parties_results["yv_plus"]
    parties_u_plus = parties_results["U_plus"]
    parties_upup_plus = parties_results["upup_plus"]
    parties_vpvp_plus = parties_results["vpvp_plus"]
    parties_wpwp_plus = parties_results["wpwp_plus"]
    parties_k_plus = parties_results["k_plus"]
    u_tau = parties_results["u_tau"]
    tau_w = parties_results["tau_w"]
    Re_tau = parties_results["Re_tau"]

    print(f"u_tau: {u_tau}, tau_w: {tau_w}, Re_tau: {Re_tau}")

    # Fit law of the wall parameters to both datasets
    utexas_kappa, utexas_constant = theory.law_of_the_wall.fit_parameters(
        utexas_y_plus, utexas_u_plus
    )
    parties_kappa, parties_constant = theory.law_of_the_wall.fit_parameters(
        parties_yc_plus, parties_u_plus
    )

    # Compute law of the wall for both datasets
    (
        utexas_viscous_y_plus,
        utexas_viscous_u_plus,
        utexas_log_y_plus,
        utexas_log_u_plus,
    ) = theory.law_of_the_wall.generate_profile(
        utexas_y_plus, utexas_kappa, utexas_constant
    )

    (
        parties_viscous_yc_plus,
        parties_viscous_u_plus,
        parties_log_yc_plus,
        parties_log_u_plus,
    ) = theory.law_of_the_wall.generate_profile(
        parties_yc_plus, parties_kappa, parties_constant
    )

    print(
        f"Law of the wall parameters (utexas):  κ={utexas_kappa:.3f}, C+={utexas_constant:.3f}\n"
        f"Law of the wall parameters (PARTIES): κ={parties_kappa:.3f}, C+={parties_constant:.3f}"
    )

    return (
        utexas_y_plus,
        utexas_u_plus,
        utexas_viscous_y_plus,
        utexas_viscous_u_plus,
        utexas_log_y_plus,
        utexas_log_u_plus,
        utexas_upup_plus,
        utexas_vpvp_plus,
        utexas_wpwp_plus,
        utexas_k_plus,
        parties_yc_plus,
        parties_yv_plus,
        parties_u_plus,
        parties_viscous_yc_plus,
        parties_viscous_u_plus,
        parties_log_yc_plus,
        parties_log_u_plus,
        parties_upup_plus,
        parties_vpvp_plus,
        parties_wpwp_plus,
        parties_k_plus,
        Re,
        Re_tau,
        u_tau,
    )

# ########## Plotting ##########

def format_plot_axes(axes: Axes) -> Axes:
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    axes.spines["left"].set_linewidth(1.2)
    axes.spines["bottom"].set_linewidth(1.0)
    axes.tick_params(axis="both", which="both", direction="out", labelsize=12)
    axes.legend(frameon=False, fontsize=12)
    plt.tight_layout()
    return axes


def create_velocity_profile_plot(
    utexas_y_plus: np.ndarray,
    utexas_u_plus: np.ndarray,
    utexas_viscous_y_plus: np.ndarray,
    utexas_viscous_u_plus: np.ndarray,
    utexas_log_y_plus: np.ndarray,
    utexas_log_u_plus: np.ndarray,
    parties_yc_plus: np.ndarray,
    parties_u_plus: np.ndarray,
    parties_viscous_yc_plus: np.ndarray,
    parties_viscous_u_plus: np.ndarray,
    parties_log_yc_plus: np.ndarray,
    parties_log_u_plus: np.ndarray,
    Re: float,
    Re_tau: float,
) -> None:
    figure, axes = plt.subplots(figsize=(6.5, 5.5))

    axes.set_xlim(1.0, min(max(np.max(utexas_y_plus), np.max(parties_yc_plus)), 1e2))
    axes.set_ylim(0.0, 1.1 * max(np.max(utexas_u_plus), np.max(parties_u_plus)))

    axes.semilogx(utexas_y_plus, utexas_u_plus, "-k", label="utexas data")
    axes.semilogx(parties_yc_plus, parties_u_plus, "-.k", label="PARTIES data")
    axes.semilogx(
        utexas_viscous_y_plus,
        utexas_viscous_u_plus,
        "--k",
        linewidth=0.9,
        label="Law of the wall (utexas)",
    )
    axes.semilogx(utexas_log_y_plus, utexas_log_u_plus, "--k", linewidth=0.8)
    axes.semilogx(
        parties_log_yc_plus,
        parties_log_u_plus,
        ":k",
        linewidth=0.9,
        label="Law of the wall (PARTIES)",
    )

    viscous_sublayer_boundary = 5.0
    buffer_layer_boundary = 30.0

    for boundary_position in (viscous_sublayer_boundary, buffer_layer_boundary):
        axes.axvline(
            x=boundary_position,
            color="0.25",
            linewidth=0.8,
            linestyle=":",
            alpha=0.7,
            zorder=0,
        )

    x_max = axes.get_xlim()[1]
    y_max = axes.get_ylim()[1]
    label_y_position = 0.99 * y_max

    viscous_center = np.sqrt(1.0 * viscous_sublayer_boundary)
    buffer_center = np.sqrt(viscous_sublayer_boundary * buffer_layer_boundary)
    log_center = np.sqrt(buffer_layer_boundary * x_max)

    label_style: Dict[str, Any] = {
        "ha": "center",
        "va": "top",
        "fontsize": 12,
        "bbox": {"facecolor": "white", "edgecolor": "none", "alpha": 0.0},
    }

    axes.text(
        viscous_center, label_y_position, "Viscous sublayer\n$y^+<5$", **label_style
    )
    axes.text(
        buffer_center, label_y_position, "Buffer layer\n$5<y^+<30$", **label_style
    )
    axes.text(log_center, label_y_position, "Log-law region\n$30<y^+$", **label_style)

    axes.set_xlabel(r"$y^+$", fontsize=14)
    axes.set_ylabel(r"$u^+$", fontsize=14)
    axes.legend(loc="lower right", bbox_to_anchor=(1.0, 0.20))
    axes = format_plot_axes(axes)

    plot_filename = f"{output_dir}/Re={Re:.0f}_Re_tau={Re_tau:.0f}-y+_u+.png"
    plt.savefig(plot_filename, dpi=300)

    if not ON_ANVIL:
        plt.show()

    plt.close(figure)


def create_normal_stress_plot(
    utexas_y_plus: np.ndarray,
    utexas_upup_plus: np.ndarray,
    utexas_vpvp_plus: np.ndarray,
    utexas_wpwp_plus: np.ndarray,
    utexas_k_plus: np.ndarray,
    parties_yc_plus: np.ndarray,
    parties_yv_plus: np.ndarray,
    parties_upup_plus: np.ndarray,
    parties_vpvp_plus: np.ndarray,
    parties_wpwp_plus: np.ndarray,
    parties_k_plus: np.ndarray,
    Re: float,
    Re_tau: float,
    u_tau: float,
) -> None:
    figure, axes = plt.subplots(figsize=(6.5, 5.5))

    # downsample utexas points to 30
    idx: np.ndarray = np.linspace(0, len(utexas_y_plus) - 1, 40, dtype=int)
    idx_upup: np.ndarray = np.linspace(0, len(utexas_y_plus) - 1, 70, dtype=int)
    ux_y: np.ndarray = utexas_y_plus[idx]
    ux_y_upup: np.ndarray = utexas_y_plus[idx_upup]
    ux_upup: np.ndarray = utexas_upup_plus[idx_upup]
    ux_vpvp: np.ndarray = utexas_vpvp_plus[idx]
    ux_wpwp: np.ndarray = utexas_wpwp_plus[idx]
    ux_k: np.ndarray = utexas_k_plus[idx]

    axes.plot(
        ux_y_upup,
        ux_upup,
        "o",
        fillstyle="none",
        color="k",
        label=r"$\langle u^{\prime}u^{\prime}\rangle / u_{\tau}$ (utexas)",
    )
    axes.plot(
        ux_y,
        ux_vpvp,
        "d",
        fillstyle="none",
        color="k",
        label=r"$\langle v^{\prime}v^{\prime}\rangle / u_{\tau}$ (utexas)",
    )
    axes.plot(
        ux_y,
        ux_wpwp,
        "^",
        fillstyle="none",
        color="k",
        label=r"$\langle w^{\prime}w^{\prime}\rangle / u_{\tau}$ (utexas)",
    )
    axes.plot(
        ux_y,
        ux_k,
        "x",
        fillstyle="none",
        color="k",
        label=r"$\langle k\rangle / u_{\tau}$ (utexas)",
    )

    axes.plot(
        parties_yc_plus,
        parties_upup_plus,
        "-k",
        label=r"$\langle u^{\prime}u^{\prime}\rangle / u_{\tau}$ (PARTIES)",
    )
    axes.plot(
        parties_yv_plus,
        parties_vpvp_plus,
        "-.k",
        label=r"$\langle v^{\prime}v^{\prime}\rangle / u_{\tau}$ (PARTIES)",
    )
    axes.plot(
        parties_yc_plus,
        parties_wpwp_plus,
        "--k",
        label=r"$\langle w^{\prime}w^{\prime}\rangle / u_{\tau}$ (PARTIES)",
    )
    axes.plot(
        parties_yc_plus,
        parties_k_plus,
        ":k",
        label=r"$\langle k\rangle / u_{\tau}$ (PARTIES)",
    )

    axes.set_xlabel(r"$y^+$", fontsize=14)
    axes.set_ylabel(
        r"$\left\{\langle u^\prime u^\prime \rangle, \langle v^\prime v^\prime \rangle, \langle w^\prime w^\prime \rangle, \langle k \rangle\right\} / u_\tau^2$",
        fontsize=14,
    )
    axes.set_xlim(
        0.0,
        min(
            max(
                np.max(utexas_y_plus),
                np.max(parties_yc_plus),
                np.max(parties_yv_plus),
            ),
            80,
        ),
    )
    axes.set_ylim(
        0.0,
        min(
            1.1
            * max(
                np.max(utexas_upup_plus),
                np.max(utexas_vpvp_plus),
                np.max(utexas_wpwp_plus),
                np.max(utexas_k_plus),
                np.max(parties_upup_plus),
                np.max(parties_vpvp_plus),
                np.max(parties_wpwp_plus),
                np.max(parties_k_plus),
            ),
            8.0,
        ),
    )
    axes.legend(loc="lower right", bbox_to_anchor=(1.0, 0.70))
    axes = format_plot_axes(axes)

    plot_filename = f"{output_dir}/Re={Re:.0f}_Re_tau={Re_tau:.0f}-u'u'.png"
    plt.savefig(plot_filename, dpi=300)

    if not ON_ANVIL:
        plt.show()

    plt.close(figure)


# =============================================================================
# FlOCULATION
# =============================================================================

# ########## Computation ##########



# ########## Plotting ##########

def create_particle_slice_plot(
    Re: float,
    Re_tau: float,
    min_file_index: Optional[int] = None,
    max_file_index: Optional[int] = None,
):
    data_files: List[Path] = myio.list_parties_data_files(
        parties_data_dir, "Data", min_file_index, max_file_index
    )
    data_file: Path = data_files[-1]
    print(f"Showing slice of {data_file}")

    vfu: np.ndarray
    u: np.ndarray
    xu: np.ndarray
    yc: np.ndarray
    with h5py.File(data_file, "r") as h5_file:
        vfu = h5_file["vfu"][:]  # type: ignore
        u = h5_file["u"][:]  # type: ignore
        xu = h5_file["grid"]["xu"][:-1]  # type: ignore
        yc = h5_file["grid"]["yc"][:-1]  # type: ignore

    u = u[:-1, :-1, :-1]
    vfu = vfu[:-1, :-1, :-1]

    k: int = u.shape[0] // 2
    u = np.squeeze(u[k, :, :])
    particle_search_width: int = 5
    vfu_2d: np.ndarray = np.zeros_like(u)
    idx = k - particle_search_width // 2
    while idx < k - particle_search_width // 2 + particle_search_width:
        vfu_2d += vfu[idx, :, :]
        idx += 1
    np.clip(vfu_2d, 0, 1, out=vfu_2d)

    X, Y = np.meshgrid(xu, yc)
    fig, ax = plt.subplots(figsize=(12, 4))
    pcm = ax.pcolormesh(X, Y, u, shading="auto", cmap="rainbow")

    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label("u")

    eps = 1e-12
    mask_interior = np.isclose(vfu_2d, 1.0, atol=eps)
    mask_interface = (vfu_2d > 0.0 + eps) & (vfu_2d < 1.0 - eps)

    interior_plot = np.where(mask_interior, 0.5, np.nan)
    ax.pcolormesh(X, Y, interior_plot, shading="auto", cmap="Greys", vmin=0, vmax=1)

    interface_plot = np.where(mask_interface, 1.0, np.nan)
    ax.pcolormesh(X, Y, interface_plot, shading="auto", cmap="Greys", vmin=0, vmax=1)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")

    plot_filename = f"{output_dir}/Re={Re:.0f}_Re_tau={Re_tau:.0f}-particles.png"
    plt.savefig(plot_filename, dpi=300)

    if not ON_ANVIL:
        plt.show()

    plt.close(fig)

# =============================================================================
# MAIN EXECUTION
# =============================================================================


def main() -> None:
    # method: Literal["step_by_step", "saved", "ADLeonelli"] = "step_by_step"
    # method: Literal["step_by_step", "saved", "ADLeonelli"] = "saved"
    # method: Literal["step_by_step", "saved", "ADLeonelli"] = "ADLeonelli"
    # method: Literal["saved", "ADLeonelli"] = "saved"
    method: Literal["saved", "ADLeonelli"] = "ADLeonelli"
    (
        utexas_y_plus,
        utexas_u_plus,
        utexas_viscous_y_plus,
        utexas_viscous_u_plus,
        utexas_log_y_plus,
        utexas_log_u_plus,
        utexas_upup_plus,
        utexas_vpvp_plus,
        utexas_wpwp_plus,
        utexas_k_plus,
        parties_yc_plus,
        parties_yv_plus,
        parties_u_plus,
        parties_viscous_yc_plus,
        parties_viscous_u_plus,
        parties_log_yc_plus,
        parties_log_u_plus,
        parties_upup_plus,
        parties_vpvp_plus,
        parties_wpwp_plus,
        parties_k_plus,
        Re,
        Re_tau,
        u_tau,
    ) = collect_flow_statistics(method)

    create_velocity_profile_plot(
        utexas_y_plus,
        utexas_u_plus,
        utexas_viscous_y_plus,
        utexas_viscous_u_plus,
        utexas_log_y_plus,
        utexas_log_u_plus,
        parties_yc_plus,
        parties_u_plus,
        parties_viscous_yc_plus,
        parties_viscous_u_plus,
        parties_log_yc_plus,
        parties_log_u_plus,
        Re,
        Re_tau,
    )

    create_normal_stress_plot(
        utexas_y_plus,
        utexas_upup_plus,
        utexas_vpvp_plus,
        utexas_wpwp_plus,
        utexas_k_plus,
        parties_yc_plus,
        parties_yv_plus,
        parties_upup_plus,
        parties_vpvp_plus,
        parties_wpwp_plus,
        parties_k_plus,
        Re,
        Re_tau,
        u_tau,
    )

    create_particle_slice_plot(Re, Re_tau)


    data_files: List[Path] = myio.list_parties_data_files(
        parties_data_dir, "Data"
    )
    data_file: Path = data_files[-1]
    phi = flow_statistics.calc_tot_vol_frac(data_file)
    print(f"Total volume fraction is {phi*100} %")


if __name__ == "__main__":
    scripts.run_floc_analysis.main()
    # main()
