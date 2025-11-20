import h5py
from pathlib import Path
from src.myio import utils
from typing import List
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as colors

parent_dir: Path = Path("/media/usb/UCSB")


def main() -> None:

    floc_dir: Path = parent_dir / "output" / "phi5p0_new" / "flocs_new"
    floc_files: List[Path] = utils.find_data_files(floc_dir, "Flocs_*")
    particle_files: List[Path] = utils.find_data_files(floc_dir, "Particles_*")

    xmin: int
    xmax: int
    ymin: int
    ymax: int
    zmin: int
    zmax: int
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    n_p: np.ndarray
    floc_ids: np.ndarray
    D_f: np.ndarray  # Floc size data
    with h5py.File(str(floc_files[-1]), "r") as f:
        xmin = f["domain/xmin"][()][0]  # type: ignore
        xmax = f["domain/xmax"][()][0]  # type: ignore
        ymin = f["domain/ymin"][()][0]  # type: ignore
        ymax = f["domain/ymax"][()][0]  # type: ignore
        zmin = f["domain/zmin"][()][0]  # type: ignore
        zmax = f["domain/zmax"][()][0]  # type: ignore

        x = f["x"][:]  # type: ignore
        y = f["y"][:]  # type: ignore
        z = f["z"][:]  # type: ignore
        n_p = f["n_p"][:]  # type: ignore
        floc_ids = f["floc_id"][:]  # type: ignore
        D_f = f["D_f"][:]  # type: ignore

    N_slices: int = 4

    width_divisor: float = N_slices / 2
    x_width: float = (float(xmax) - float(xmin)) / width_divisor
    y_width: float = (float(ymax) - float(ymin)) / width_divisor
    z_width: float = (float(zmax) - float(zmin)) / width_divisor

    x_step: float = (float(xmax) - float(xmin)) / 2 / N_slices
    y_step: float = (float(ymax) - float(ymin)) / 2 / N_slices
    z_step: float = (float(zmax) - float(zmin)) / 2 / N_slices

    num_rows: int = int(N_slices * 2 / 3)
    num_cols: int = 1
    while (num_cols * num_rows) < N_slices:
        num_cols = num_cols + 1

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(14, 8))
    axes = axes.flatten()

    xp: np.ndarray
    yp: np.ndarray
    zp: np.ndarray
    r: np.ndarray
    particle_floc_ids: np.ndarray
    with h5py.File(str(particle_files[-1]), "r") as f:
        xp = f["x_p"][:]  # type: ignore
        yp = f["y_p"][:]  # type: ignore
        zp = f["z_p"][:]  # type: ignore
        r = f["r"][:]  # type: ignore
        particle_floc_ids = f["floc_id"][:]  # type: ignore

    all_D_f = D_f[n_p > 1]
    if len(all_D_f) > 0:
        vmin = np.min(all_D_f)
        vmax = np.max(all_D_f)
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.cm.viridis
    else:
        norm = None
        cmap = plt.cm.viridis

    for slice_idx in range(N_slices):
        x_center: float = float(xmin) + x_step * (slice_idx + 1)
        x_left: float = x_center - x_width / 2
        x_right: float = x_center + x_width / 2

        y_center: float = float(ymin) + y_step * (slice_idx + 1)
        y_left: float = y_center - y_width / 2
        y_right: float = y_center + y_width / 2

        z_center: float = float(zmin) + z_step * (slice_idx + 1)
        z_left: float = z_center - z_width / 2
        z_right: float = z_center + z_width / 2

        mask: np.ndarray = (y > y_left) & (y < y_right) & (n_p > 1)

        slice_floc_ids = floc_ids[mask]
        slice_D_f = D_f[mask]

        print(f"Number of flocs found (slice_idx {slice_idx}): {len(slice_floc_ids)}")

        for i, floc_id in enumerate(slice_floc_ids):
            particle_mask: np.ndarray = particle_floc_ids == floc_id

            if not np.any(particle_mask):
                continue

            x_local = xp[particle_mask]
            z_local = zp[particle_mask]
            r_local = r[particle_mask]

            if norm is not None:
                color = cmap(norm(slice_D_f[i]))
            else:
                color = cmap(0.5)

            for j in range(len(x_local)):
                circle = plt.Circle(
                    (x_local[j], z_local[j]),
                    r_local[j],
                    fill=True,
                    alpha=0.7,
                    linewidth=0.5,
                    facecolor=color,
                )
                axes[slice_idx].add_patch(circle)

        axes[slice_idx].set_aspect(1)
        axes[slice_idx].set_xlim([xmin, xmax])
        axes[slice_idx].set_ylim([zmin, zmax])
        axes[slice_idx].set_title(f"y={y_center:.2f}")
        axes[slice_idx].set_xlabel("X")
        axes[slice_idx].set_ylabel("Z")

    for idx in range(N_slices, len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
