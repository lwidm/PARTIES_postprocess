# -- src/plotting/templates.py

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from typing import Dict, Union, Optional, Any, List
from pathlib import Path
import h5py  # type: ignore

from .tools import format_plot_axes
from src import globals
from src.myio import myio


def create_velocity_profile_wall_plot(
    output_dir: Union[str, Path],
    stats: Dict[str, Union[np.ndarray, float]],
) -> None:
    """
    Create/save (and optionally show) the y+ / u+ velocity-profile plot.

    Args:
        output_dir: directory or path to save the figure into
        stats: dictionary produced by collect_flow_statistics containing keys:
            'utexas_y_plus', 'utexas_U_plus', 'utexas_viscous_y_plus',
            'utexas_viscous_U_plus', 'utexas_log_y_plus', 'utexas_log_U_plus',
            'yc_plus', 'U_plus', 'parties_log_yc_plus',
            'parties_log_U_plus', 'Re', 'Re_tau'
    """
    # required arrays / scalars (will KeyError if absent)
    utexas_y_plus: np.ndarray = stats["utexas_y_plus"]  # type: ignore
    utexas_U_plus: np.ndarray = stats["utexas_U_plus"]  # type: ignore
    utexas_viscous_y_plus: np.ndarray = stats["utexas_viscous_y_plus"]  # type: ignore
    utexas_viscous_U_plus: np.ndarray = stats["utexas_viscous_U_plus"]  # type: ignore
    utexas_log_y_plus: np.ndarray = stats["utexas_log_y_plus"]  # type: ignore
    utexas_log_U_plus: np.ndarray = stats["utexas_log_U_plus"]  # type: ignore

    parties_yc_plus: np.ndarray = stats["yc_plus"]  # type: ignore
    parties_U_plus: np.ndarray = stats["U_plus"]  # type: ignore
    parties_log_yc_plus: np.ndarray = stats["parties_log_yc_plus"]  # type: ignore
    parties_log_U_plus: np.ndarray = stats["parties_log_U_plus"]  # type: ignore
    Re: float = stats["Re"]  # type: ignore
    Re_tau: float = stats["Re"]  # type: ignore

    figure: Figure
    axes: Axes
    figure, axes = plt.subplots(figsize=(6.5, 5.5))

    axes.set_xlim(1.0, min(max(np.max(utexas_y_plus), np.max(parties_yc_plus)), 1e2))
    axes.set_ylim(0.0, 1.1 * max(np.max(utexas_U_plus), np.max(parties_U_plus)))

    axes.semilogx(utexas_y_plus, utexas_U_plus, "-k", label="utexas data")
    axes.semilogx(parties_yc_plus, parties_U_plus, "-.k", label="PARTIES data")
    axes.semilogx(
        utexas_viscous_y_plus,
        utexas_viscous_U_plus,
        "--k",
        linewidth=0.9,
        label="Law of the wall (utexas)",
    )
    axes.semilogx(utexas_log_y_plus, utexas_log_U_plus, "--k", linewidth=0.8)
    axes.semilogx(
        parties_log_yc_plus,
        parties_log_U_plus,
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

    viscous_center: float = np.sqrt(1.0 * viscous_sublayer_boundary)
    buffer_center: float = np.sqrt(viscous_sublayer_boundary * buffer_layer_boundary)
    log_center: float = np.sqrt(buffer_layer_boundary * x_max)

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

    out_path = Path(output_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot_filename = out_path / f"Re={Re:.0f}_Re_tau={Re_tau:.0f}-y+_u+.png"
    plt.savefig(str(plot_filename), dpi=300)

    if not globals.on_anvil:
        plt.show()

    plt.close(figure)


def create_normal_stress_wall_plot(
    output_dir: Union[str, Path],
    stats: Dict[str, Union[np.ndarray, float]],
) -> None:
    """
    Create/save (and optionally show) the normal-stress velocity-profile plot.

    Args:
        output_dir: directory or path to save the figure into
        stats: dictionary produced by collect_flow_statistics containing keys:
            'utexas_y_plus', 'utexas_upup_plus', 'utexas_vpvp_plus',
            'utexas_wpwp_plus', 'utexas_k_plus', 'yc_plus','yv_plus',
            'upup_plus', 'vpvp_plus','wpwp_plus', 'k_plus', 'Re', 'Re_tau',
            'u_tau'
    """
    # pull arrays / scalars from dict
    utexas_y_plus: np.ndarray = stats["utexas_y_plus"]  # type: ignore
    utexas_upup_plus: np.ndarray = stats["utexas_upup_plus"]  # type: ignore
    utexas_vpvp_plus: np.ndarray = stats["utexas_vpvp_plus"]  # type: ignore
    utexas_wpwp_plus: np.ndarray = stats["utexas_wpwp_plus"]  # type: ignore
    utexas_k_plus: np.ndarray = stats["utexas_k_plus"]  # type: ignore

    parties_yc_plus: np.ndarray = stats["yc_plus"]  # type: ignore
    parties_yv_plus: np.ndarray = stats["yv_plus"]  # type: ignore
    parties_upup_plus: np.ndarray = stats["upup_plus"]  # type: ignore
    parties_vpvp_plus: np.ndarray = stats["vpvp_plus"]  # type: ignore
    parties_wpwp_plus: np.ndarray = stats["wpwp_plus"]  # type: ignore
    parties_k_plus: np.ndarray = stats["k_plus"]  # type: ignore
    Re: float = stats["Re"]  # type: ignore
    Re_tau: float = stats["Re_tau"]  # type: ignore

    figure: Figure
    axes: Axes
    figure, axes = plt.subplots(figsize=(6.5, 5.5))

    # downsample utexas points
    idx: np.ndarray = np.linspace(0, len(utexas_y_plus) - 1, 40, dtype=int)
    idx_upup: np.ndarray = np.linspace(0, len(utexas_y_plus) - 1, 70, dtype=int)
    ux_y: np.ndarray = utexas_y_plus[idx]
    ux_y_upup: np.ndarray = utexas_y_plus[idx_upup]
    ux_upup: np.ndarray = utexas_upup_plus[idx_upup]
    ux_vpvp: np.ndarray = utexas_vpvp_plus[idx]
    ux_wpwp: np.ndarray = utexas_wpwp_plus[idx]
    ux_k: np.ndarray = utexas_k_plus[idx]

    def generate_label(u: str, which: str) -> str:
        return f"$\\langle {u}^\\prime {u}^\\prime\\rangle / u_\\tau$ ({which})"

    l_u_tex: str = generate_label("u", "utexas")
    l_v_tex: str = generate_label("v", "utexas")
    l_w_tex: str = generate_label("w", "utexas")
    l_k_tex: str = generate_label("k", "utexas")
    l_u_part: str = generate_label("u", "parties")
    l_v_part: str = generate_label("v", "parties")
    l_w_part: str = generate_label("w", "parties")
    l_k_part: str = generate_label("k", "parties")

    axes.plot(
        ux_y_upup,
        ux_upup,
        "o",
        fillstyle="none",
        color="k",
        label=l_u_tex,
    )
    axes.plot(
        ux_y,
        ux_vpvp,
        "d",
        fillstyle="none",
        color="k",
        label=l_v_tex,
    )
    axes.plot(
        ux_y,
        ux_wpwp,
        "^",
        fillstyle="none",
        color="k",
        label=l_w_tex,
    )
    axes.plot(
        ux_y,
        ux_k,
        "x",
        fillstyle="none",
        color="k",
        label=l_k_tex,
    )

    axes.plot(
        parties_yc_plus,
        parties_upup_plus,
        "-k",
        label=l_u_part,
    )
    axes.plot(
        parties_yv_plus,
        parties_vpvp_plus,
        "-.k",
        label=l_v_part,
    )
    axes.plot(
        parties_yc_plus,
        parties_wpwp_plus,
        "--k",
        label=l_w_part,
    )
    axes.plot(
        parties_yc_plus,
        parties_k_plus,
        ":k",
        label=l_k_part,
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

    out_path = Path(output_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot_filename = out_path / f"Re={Re:.0f}_Re_tau={Re_tau:.0f}-u'u'.png"
    plt.savefig(str(plot_filename), dpi=300)

    if not globals.on_anvil:
        plt.show()

    plt.close(figure)


def create_particle_slice_plot(
    parties_data_dir: Union[str, Path],
    output_dir: Union[str, Path],
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
        vfu = h5_file["vfu"][:-1, :-1, :-1]  # type: ignore
        u = h5_file["u"][:-1, :-1, :-1]  # type: ignore
        xu = h5_file["grid"]["xu"][:-1]  # type: ignore
        yc = h5_file["grid"]["yc"][:-1]  # type: ignore

    k: int = u.shape[0] // 2
    u = np.squeeze(u[k, :, :])
    particle_search_width: int = 5
    vfu_2d: np.ndarray = np.zeros_like(u)
    idx = k - particle_search_width // 2
    while idx < k - particle_search_width // 2 + particle_search_width:
        vfu_2d += vfu[idx, :, :]
        idx += 1
    np.clip(vfu_2d, 0, 1, out=vfu_2d)

    X: np.ndarray
    Y: np.ndarray
    X, Y = np.meshgrid(xu, yc)
    fig, ax = plt.subplots(figsize=(12, 4))
    pcm = ax.pcolormesh(X, Y, u, shading="auto", cmap="rainbow")

    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label("u")

    eps: float = 1e-12
    mask_interior: np.ndarray
    mask_interior = np.isclose(vfu_2d, 1.0, atol=eps)
    mask_interface = (vfu_2d > 0.0 + eps) & (vfu_2d < 1.0 - eps)

    interior_plot: np.ndarray = np.where(mask_interior, 0.5, np.nan)
    ax.pcolormesh(X, Y, interior_plot, shading="auto", cmap="Greys", vmin=0, vmax=1)

    interface_plot: np.ndarray = np.where(mask_interface, 1.0, np.nan)
    ax.pcolormesh(X, Y, interface_plot, shading="auto", cmap="Greys", vmin=0, vmax=1)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")

    plot_filename = f"{output_dir}/Re={Re:.0f}_Re_tau={Re_tau:.0f}-particles.png"
    plt.savefig(plot_filename, dpi=300)

    if not globals.on_anvil:
        plt.show()

    plt.close(fig)
