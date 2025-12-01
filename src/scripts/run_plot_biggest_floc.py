from pathlib import Path
import h5py
from typing import List, Tuple
import numpy as np
from matplotlib import pyplot as plt
import configparser
from typing import Any

from src import myio

# parent_dir: Path = Path("/media/usb/UCSB/")
parent_dir: Path = Path("./")


def get_top_flocs(
    floc_files: List[Path], n_flocs: int
) -> List[Tuple[float, int, int, Path, float]]:
    all_flocs: List[Tuple[float, int, int, Path, float]] = []

    for file_idx, floc_file in enumerate(floc_files):
        with h5py.File(str(floc_file), "r") as f:
            Df_arr: np.ndarray = f["D_f"][:]  # type: ignore
            floc_id_arr: np.ndarray = f["floc_id"][:]  # type: ignore
            time_val: float = f["time"][()][0]  # type: ignore

            for i in range(len(Df_arr)):
                all_flocs.append((Df_arr[i], floc_id_arr[i], file_idx, floc_file, time_val))
    all_flocs.sort(key=lambda x: x[0])
    all_flocs = all_flocs[::-1]
    _, first_indices = np.unique([floc[1] for floc in all_flocs], return_index=True)
    first_indices = np.sort(first_indices)

    unique_flocs: List[Tuple[float, int, int, Path, float]] = []
    unique_flocs = [all_flocs[i] for i in first_indices]
    unique_flocs.sort(key=lambda x: x[0])
    unique_flocs = unique_flocs[::-1]

    # unique_flocs = all_flocs

    return unique_flocs[:n_flocs]


def get_floc_particles(
    floc_file: Path, floc_id: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(str(floc_file), "r") as f:
        x_all: np.ndarray = f["x"][:]  # type: ignore
        y_all: np.ndarray = f["y"][:]  # type: ignore
        z_all: np.ndarray = f["z"][:]  # type: ignore
        r_all: np.ndarray = f["r"][:]  # type: ignore
        id_all: np.ndarray = f["floc_id"][:]  # type: ignore

        mask = id_all == floc_id
        x = x_all[mask]
        y = y_all[mask]
        z = z_all[mask]
        r = r_all[mask]

    return x, y, z, r


def main() -> None:
    plot_dir: Path = parent_dir / "output" / "plots"
    name: str = "phi5p0"
    metadata_path: Path = parent_dir / "data" / name / "metadata.ini"
    floc_path: Path = parent_dir / "data" / name / "flocs"

    # ========== get domain info =========
    metadata = myio.metadata.read_metadata(metadata_path)
    geom: dict[str, Any] = dict(metadata["Domain"])
    for key in geom:
        geom[key] = float(geom[key])

    # ========== get top N flocs =========
    floc_files: List[Path] = myio.utils.find_data_files(floc_path, "Particles_*")


    N = 11

    top_flocs = get_top_flocs(floc_files, N)

    print(f"Top {N} flocs by Df:")
    for i, (Df, floc_id, _, file_path, time_val) in enumerate(top_flocs):
        print(f"  {i+1}. Df={Df:.4f}, ID={floc_id}, Time={time_val:.2f}, File={file_path.name}")

    # ========== create three subplots =========
    all_plots = True
    show_legend = True
    if all_plots:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        ax_xy = axes[0, 0]  # Top left: XY slice
        ax_zy = axes[0, 1]  # Top right: YZ slice  
        ax_xz = axes[1, 0]  # Bottom left: XZ slice
        ax_legend = axes[1, 1]
    else:
        fig, axes = plt.subplots(1, 1, figsize=(8, 4))
        ax_xy = axes
        ax_zy = axes
        ax_xz = axes
        ax_legend = axes

    colors = plt.cm.viridis(np.linspace(0, 1, N))

    all_handles = []
    all_labels = []


    for i, (Df, floc_id, _, file_path, time_val) in enumerate(top_flocs):
        x, y, z, r = get_floc_particles(file_path, floc_id)

        print(f"Floc {floc_id}: Df={Df:.4f}, Time={time_val:.2f}, {len(x)} particles")

        label = f"Floc {floc_id} (Df={Df:.3f}, t={time_val:.2f})"

        for j in range(len(x)):
            circle_xy = plt.Circle(
                (x[j], y[j]),
                r[j],
                fill=True,
                alpha=0.7,
                linewidth=0.5,
                facecolor=colors[i],
                label=label if j == 0 else "",
            )
            if all_plots:
                ax_xy.add_patch(circle_xy)

        for j in range(len(y)):
            circle_yz = plt.Circle(
                (z[j], y[j]),
                r[j],
                fill=True,
                alpha=0.7,
                linewidth=0.5,
                facecolor=colors[i],
                label=label if j == 0 else "",  # Label only once per floc
            )
            if all_plots:
                ax_zy.add_patch(circle_yz)

        for j in range(len(x)):
            circle_xz = plt.Circle(
                (x[j], z[j]),
                r[j],
                fill=True,
                alpha=0.7,
                linewidth=0.5,
                facecolor=colors[i],
                label=label if j == 0 else "",  # Label only once per floc
            )
            ax_xz.add_patch(circle_xz)
        all_handles.append(plt.Circle((0, 0), 1, fill=True, alpha=0.7, facecolor=colors[i]))
        all_labels.append(label)

    if all_plots:
        ax_xy.set_aspect(1)
        ax_xy.set_xlim(geom["xmin"], geom["xmax"])
        ax_xy.set_ylim(geom["ymin"], geom["ymax"])
        ax_xy.set_xlabel("X")
        ax_xy.set_ylabel("Y")
        ax_xy.set_title("XY Slice")

        ax_zy.set_aspect(1)
        ax_zy.set_xlim(geom["zmin"], geom["zmax"])
        ax_zy.set_ylim(geom["ymin"], geom["ymax"])
        ax_zy.set_xlabel("Z")
        ax_zy.set_ylabel("Y")
        ax_zy.set_title("ZY Slice")

    ax_xz.set_aspect(1)
    ax_xz.set_xlim(geom["xmin"], geom["xmax"])
    ax_xz.set_ylim(geom["zmin"], geom["zmax"])
    ax_xz.set_xlabel("X")
    ax_xz.set_ylabel("Z")
    ax_xz.set_title("XZ Slice")

    handles_xy, labels_xy = ax_xz.get_legend_handles_labels()
    by_label_xy = dict(zip(labels_xy, handles_xy))

    if show_legend:
        if all_plots:
            ax_legend.axis('off')
            ax_legend.legend(all_handles, all_labels, loc='center')
        else:
            plt.legend(all_handles, all_labels, loc='best')

    plt.tight_layout()
    plt.show()
    if all_plots:
        fig.savefig(str(plot_dir / f"max_flocsize_all"))
    else:
        fig.savefig(str(plot_dir / f"max_flocsize"))


if __name__ == "__main__":
    main()
