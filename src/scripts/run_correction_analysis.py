from typing import Literal, TypedDict
from pathlib import Path
import h5py
import numpy as np
import pickle
import random

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Circle
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
import seaborn as sns
from tqdm import tqdm

from src.flocs.family_tree import FlocRecord, FamilyTreeType
from src.myio import metadata, utils
from src import globals


class EventSnapshot(TypedDict):
    file_id: int
    time: float
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    r: np.ndarray
    floc_ids: np.ndarray


def get_event_tree(
    floc_dir: Path,
    floc_id: int,
    parent_ids: list[int],
    child_ids: list[int],
    file_id: int,
    n_before: int,
    n_after: int,
) -> list[EventSnapshot]:

    floc_files: list[Path] = utils.find_data_files(floc_dir, "Particles_*")

    prepre_event_file: Path = floc_files[file_id - 1]
    pre_event_file: Path = floc_files[file_id]
    post_event_file: Path = floc_files[file_id + 1]

    selected: list[Path] = [
        floc_files[i] for i in range(file_id - n_before, file_id + n_after + 1)
    ]

    relevant_particles: set[int] = set()

    def update_relevent_particles(
        h5_file: h5py.File, file_floc_ids: np.ndarray, floc_id: int
    ) -> None:
        mask: np.ndarray = file_floc_ids == floc_id
        local_relevant_particles: np.ndarray = h5_file["id"][mask]  # type: ignore
        relevant_particles.update(set(local_relevant_particles))

    with h5py.File(str(prepre_event_file), "r") as h5_file:
        file_floc_ids: np.ndarray = h5_file["floc_id"][:]  # type: ignore
        update_relevent_particles(h5_file, file_floc_ids, floc_id)
        for child_id in child_ids:
            update_relevent_particles(h5_file, file_floc_ids, child_id)
        for parent_id in parent_ids:
            update_relevent_particles(h5_file, file_floc_ids, parent_id)

    with h5py.File(str(pre_event_file), "r") as h5_file:
        file_floc_ids: np.ndarray = h5_file["floc_id"][:]  # type: ignore
        update_relevent_particles(h5_file, file_floc_ids, floc_id)
        for child_id in child_ids:
            update_relevent_particles(h5_file, file_floc_ids, child_id)
        for parent_id in parent_ids:
            update_relevent_particles(h5_file, file_floc_ids, parent_id)

    with h5py.File(str(post_event_file), "r") as h5_file:
        file_floc_ids: np.ndarray = h5_file["floc_id"][:]  # type: ignore
        update_relevent_particles(h5_file, file_floc_ids, floc_id)
        for child_id in child_ids:
            update_relevent_particles(h5_file, file_floc_ids, child_id)
        for parent_id in parent_ids:
            update_relevent_particles(h5_file, file_floc_ids, parent_id)

    relevant_flocs: set[int] = set()

    def update_relevent_flocs(
        h5_file: h5py.File, file_particle_ids: np.ndarray, particle_id: int
    ) -> None:
        mask: np.ndarray = file_particle_ids == particle_id
        local_relevant_flocs: np.ndarray = h5_file["floc_id"][mask]  # type: ignore
        relevant_flocs.update(set(local_relevant_flocs))

    for floc_file in selected:
        with h5py.File(str(floc_file), "r") as h5_file:
            file_particle_ids: np.ndarray = h5_file["id"][:]  # type: ignore
            for particle_id in relevant_particles:
                update_relevent_flocs(h5_file, file_particle_ids, particle_id)

    events: list[EventSnapshot] = []

    for floc_file in selected:
        with h5py.File(str(floc_file), "r") as h5_file:
            time: float = float(h5_file["time"][0])  # type: ignore
            local_file_id: int = int(floc_file.stem.split("_")[1])

            file_floc_ids: np.ndarray = h5_file["floc_id"][:]  # type: ignore
            mask: np.ndarray = np.zeros_like(file_floc_ids, dtype=bool)
            for floc_id in relevant_flocs:
                mask |= file_floc_ids == floc_id

            x: np.ndarray = h5_file["x"][mask]  # type: ignore
            y: np.ndarray = h5_file["y"][mask]  # type: ignore
            z: np.ndarray = h5_file["z"][mask]  # type: ignore
            r: np.ndarray = h5_file["r"][mask]  # type: ignore
            fids: np.ndarray = file_floc_ids[mask]

        events.append(
            {
                "time": time,
                "file_id": local_file_id,
                "x": x,
                "y": y,
                "z": z,
                "r": r,
                "floc_ids": fids,
            }
        )

    return events


def find_nonbinary_events(
    family_tree: FamilyTreeType,
    nonbinary_type: Literal[
        "breakup", "agglomeration", "simultaneous", "mass_conservation", "collision"
    ],
    t_min: float,
    size_min: int,
) -> dict[str, int | list[int] | list[list[int]]]:
    nonbinary_flocs: list[int] = []
    start_file_ids: list[int] = []
    end_file_ids: list[int] = []
    child_ids: list[list[int]] = []
    parent_ids: list[list[int]] = []

    floc_id: int
    floc_record: FlocRecord
    for floc_id, floc_record in family_tree.items():
        start_file_id: int = floc_record["start_file_id"]
        end_file_id: int = floc_record["end_file_id"]
        if t_min > floc_record["start_time"]:
            continue
        if size_min > floc_record["size"]:
            continue
        if nonbinary_type == "breakup":
            if floc_record["children"]:
                local_child_ids: list[int] = floc_record["children"]
                if len(local_child_ids) > 2:
                    nonbinary_flocs.append(floc_id)
                    start_file_ids.append(start_file_id)
                    end_file_ids.append(end_file_id)
                    child_ids.append(local_child_ids)
                    parent_ids.append([])

        elif nonbinary_type == "agglomeration":
            if floc_record["parents"]:
                local_parent_ids: list[int] = floc_record["parents"]
                if len(local_parent_ids) > 2:
                    nonbinary_flocs.append(floc_id)
                    start_file_ids.append(start_file_id)
                    end_file_ids.append(end_file_id)
                    child_ids.append([])
                    parent_ids.append(local_parent_ids)

        elif nonbinary_type == "mass_conservation":
            if floc_record["children"]:
                if len(floc_record["children"]) > 1:
                    current_size = floc_record["size"]
                    children_sizes = floc_record["children_sizes"]
                    if current_size != sum(children_sizes):
                        local_child_ids: list[int] = floc_record["children"]
                        all_parents: set[int] = set()
                        for cid in local_child_ids:
                            all_parents.update(family_tree[cid]["parents"])
                        local_parent_ids: list[int] = list(all_parents)
                        nonbinary_flocs.append(floc_id)
                        start_file_ids.append(start_file_id)
                        end_file_ids.append(end_file_id)
                        child_ids.append(local_child_ids)
                        parent_ids.append(local_parent_ids)
        elif nonbinary_type == "simultaneous":
            if floc_record["children"]:
                if len(floc_record["children"]) > 1:
                    local_child_ids: list[int] = floc_record["children"]
                    all_parents: set[int] = set()
                    for cid in local_child_ids:
                        all_parents.update(family_tree[cid]["parents"])
                    local_parent_ids: list[int] = list(all_parents)
                    current_constituents: list[int] = floc_record["constituents"]
                    child1_id: int = local_child_ids[0]
                    child2_id: int = local_child_ids[1]
                    child1_constituents: list[int] = family_tree[child1_id][
                        "constituents"
                    ]
                    child2_constituents: list[int] = family_tree[child2_id][
                        "constituents"
                    ]
                    if set(current_constituents) != set(
                        child1_constituents + child2_constituents
                    ):
                        nonbinary_flocs.append(floc_id)
                        start_file_ids.append(start_file_id)
                        end_file_ids.append(end_file_id)
                        child_ids.append(local_child_ids)
                        parent_ids.append(local_parent_ids)

        else:
            raise NotImplementedError

    return {
        "count": len(nonbinary_flocs),
        "floc_ids": nonbinary_flocs,
        "start_file_ids": start_file_ids,
        "end_file_ids": end_file_ids,
        "child_ids": child_ids,
        "parent_ids": parent_ids,
    }


def draw_particle(
    ax: Axes,
    x: float,
    y: float,
    radius: float,
    color: tuple[float, float, float] = (0.5, 0.5, 0.5),
    label: str = "",
    label_fontsize: float = 4,
    lw: float = 0.3,
    zorder: float = 2,
) -> None:
    n = 64
    lin = np.linspace(-1, 1, n)
    xx, yy = np.meshgrid(lin, lin)

    r_sq = xx**2 + yy**2
    inside = r_sq <= 1.0

    edge_factor = np.sqrt(np.clip(1.0 - r_sq, 0, 1))
    hl_dist = np.sqrt((xx + 0.35) ** 2 + (yy - 0.35) ** 2)
    highlight = np.exp(-(hl_dist**2) * 1.5)

    brightness = 0.35 * edge_factor + 0.45 * highlight * edge_factor
    brightness = np.clip(brightness, 0.12, 0.92)

    rgba = np.zeros((n, n, 4))
    rgba[:, :, 0] = np.clip(color[0] * brightness * 2.0, 0, 1)
    rgba[:, :, 1] = np.clip(color[1] * brightness * 2.0, 0, 1)
    rgba[:, :, 2] = np.clip(color[2] * brightness * 2.0, 0, 1)
    rgba[:, :, 3] = inside.astype(float)

    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    ax.imshow(
        rgba,
        extent=(x - radius, x + radius, y - radius, y + radius),
        origin="lower",
        interpolation="bilinear",
        aspect="auto",
        zorder=zorder,
    )
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    edge = Circle(
        (x, y),
        radius,
        fill=False,
        edgecolor="#404040",
        lw=lw,
        zorder=zorder + 1,
    )
    ax.add_patch(edge)

    if label:
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=label_fontsize,
            color="black",
            zorder=zorder + 2,
        )


def unwrap_periodic(
    events: list[EventSnapshot],
    key: str,
    domain_min: float,
    domain_max: float,
) -> None:
    L: float = domain_max - domain_min

    all_vals: np.ndarray = np.concatenate([e[key] for e in events])
    if len(all_vals) == 0:
        return
    theta: np.ndarray = 2.0 * np.pi * (all_vals - domain_min) / L
    ref_angle: float = float(np.arctan2(np.mean(np.sin(theta)), np.mean(np.cos(theta))))
    ref: float = ref_angle * L / (2.0 * np.pi) + domain_min

    for e in events:
        arr: np.ndarray = e[key]
        e[key] = ref + ((arr - ref + L / 2) % L) - L / 2


def build_floc_color_map(
    all_event_trees: list[list[EventSnapshot]],
) -> dict[int, tuple[float, float, float]]:
    all_fids: np.ndarray = np.concatenate(
        [e["floc_ids"] for tree in all_event_trees for e in tree]
    )
    unique_flocs: np.ndarray = np.unique(all_fids)
    n_flocs: int = len(unique_flocs)
    colours: list[tuple[float, float, float]] = list(
        sns.color_palette("colorblind", n_colors=max(n_flocs, 5))
    )
    return {int(fid): colours[i % len(colours)] for i, fid in enumerate(unique_flocs)}


def plot_event(
    events: list[EventSnapshot],
    plane: Literal["xy", "yz", "xz"],
    nonbinary_type: Literal[
        "breakup", "agglomeration", "simultaneous", "mass_conservation", "collision"
    ],
    seed: int,
    output_dir: Path,
    mode: Literal["save", "show"],
    time_scale: float,
    floc_color_map: dict[int, tuple[float, float, float]],
    x_periodic: tuple[float, float] | None = None,
) -> None:
    axis_map: dict[str, tuple[str, str, str]] = {
        "xy": ("x", "y", "z"),
        "yz": ("y", "z", "x"),
        "xz": ("x", "z", "y"),
    }
    h_key, v_key, depth_key = axis_map[plane]

    if x_periodic is not None:
        unwrap_periodic(events, "x", x_periodic[0], x_periodic[1])

    all_h: np.ndarray = np.concatenate([e[h_key] for e in events])
    all_v: np.ndarray = np.concatenate([e[v_key] for e in events])
    all_r: np.ndarray = np.concatenate([e["r"] for e in events])

    if len(all_h) == 0:
        return

    r_max: float = float(np.max(all_r))

    h_min: float = float(np.min(all_h)) - r_max
    h_max: float = float(np.max(all_h)) + r_max
    v_min: float = float(np.min(all_v)) - r_max
    v_max: float = float(np.max(all_v)) + r_max

    h_pad: float = (h_max - h_min) * 0.05
    v_pad: float = (v_max - v_min) * 0.05
    h_min -= h_pad
    h_max += h_pad
    v_min -= v_pad
    v_max += v_pad

    def draw_frame(fig: Figure, ax: Axes, event: EventSnapshot) -> None:
        ax.clear()
        ax.set_xlim(h_min, h_max)
        ax.set_ylim(v_min, v_max)
        ax.set_aspect("equal", adjustable="box")

        h_vals: np.ndarray = event[h_key]
        v_vals: np.ndarray = event[v_key]
        radii: np.ndarray = event["r"]
        fids: np.ndarray = event["floc_ids"]

        unique_fids, counts = np.unique(fids, return_counts=True)
        floc_size: dict[int, int] = dict(zip(unique_fids.tolist(), counts.tolist()))
        sort_idx = np.argsort([-floc_size[int(f)] for f in fids])

        for idx, i in enumerate(sort_idx):
            color_rgb: tuple[float, float, float] = floc_color_map[int(fids[i])]
            draw_particle(
                ax,
                float(h_vals[i]),
                float(v_vals[i]),
                float(radii[i]),
                color=color_rgb,
                zorder=2 + idx,
            )

        present_fids: set[int] = set(fids.tolist())
        for fid in sorted(present_fids):
            ax.plot(
                [],
                [],
                "o",
                color=floc_color_map[fid],
                label=f"floc {fid}",
                markersize=5,
            )

        ax.set_xlabel(f"${h_key}$", fontsize=14)
        ax.set_ylabel(f"${v_key}$", fontsize=14)
        ax.set_title(f"$t = {event['time']:.4f}$, file {event['file_id']}", fontsize=12)

    if mode == "save":
        output_dir.mkdir(parents=True, exist_ok=True)
        for frame_idx, event in enumerate(events):
            fig, ax = plt.subplots(figsize=(6.5, 5.5))
            ax.set_aspect("equal", adjustable="box")
            draw_frame(fig, ax, event)
            plt.tight_layout()
            fig.savefig(
                output_dir / f"{nonbinary_type}_seed{seed:04d}_{frame_idx:04d}.png",
                dpi=300,
            )
            plt.close(fig)

    elif mode == "show":
        plt.ion()
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        ax.set_aspect("equal", adjustable="box")
        for frame_idx, event in enumerate(events):
            draw_frame(fig, ax, event)
            ax.set_aspect("equal", adjustable="box")
            fig.tight_layout()
            fig.canvas.draw()
            fig.canvas.flush_events()

            if frame_idx < len(events) - 1:
                dt = events[frame_idx + 1]["time"] - event["time"]
                plt.pause(max(dt * time_scale, 0.01))

        plt.ioff()
        plt.show()


def compute_event(
    data_name: str,
    output_dir: Path,
    corrected: bool,
    seed: int,
    nonbinary_type: Literal[
        "breakup", "agglomeration", "simultaneous", "mass_conservation", "collision"
    ],
) -> list[EventSnapshot] | None:
    rng = random.Random(seed)

    pickle_dir: Path = output_dir / data_name
    pickle_file: str = "family_tree.pkl"
    if corrected:
        pickle_file = "family_tree_corrected.pkl"

    fam_tree: FamilyTreeType
    with open(pickle_dir / pickle_file, "rb") as file:
        fam_tree = pickle.load(file)

    metadata_path: Path = output_dir / data_name / "metadata.ini"
    metadata_dict: dict = metadata.read_metadata(metadata_path)
    t_steady: float = metadata_dict["Time"]["t_steady"]
    t_end: float = metadata_dict["Time"]["t_end"]
    t_min: float = t_steady + 2 / 3 * (t_end - t_steady)

    nonbinary_events: dict = find_nonbinary_events(fam_tree, nonbinary_type, t_min, 20)
    n_events: int = nonbinary_events["count"]
    if n_events == 0:
        print(f"No nonbinary_events found of nonbinary_type {nonbinary_type}")
        return None
    i: int = rng.randint(0, n_events - 1)

    file_id: int
    n_before: int
    n_after: int
    if nonbinary_type == "breakup":
        file_id = nonbinary_events["end_file_ids"][i]
        n_before = 0
        n_after = 1
    elif nonbinary_type == "agglomeration":
        file_id = nonbinary_events["start_file_ids"][i]
        n_before = 1
        n_after = 0
    elif nonbinary_type == "simultaneous":
        file_id = nonbinary_events["end_file_ids"][i]
        n_before = 0
        n_after = 1
    elif nonbinary_type == "mass_conservation":
        file_id = nonbinary_events["end_file_ids"][i]
        n_before = 0
        n_after = 1
    else:
        raise NotImplementedError

    return get_event_tree(
        floc_dir=output_dir / data_name / "flocs",
        floc_id=nonbinary_events["floc_ids"][i],
        parent_ids=nonbinary_events["parent_ids"][i],
        child_ids=nonbinary_events["child_ids"][i],
        file_id=file_id,
        n_before=n_before,
        n_after=n_after,
    )


def main() -> None:
    corrected: bool = False
    seeds: np.ndarray = np.asarray([1, 2, 3, 4, 5]) * 19 + 1
    nonbinary_types: list[
        Literal["breakup", "agglomeration", "simultaneous", "mass_conservation"]
    ] = ["breakup", "agglomeration", "simultaneous", "mass_conservation"]

    data_names: list[str] = globals.data_names

    plot_dir: Path = globals.plot_dir
    output_dir: Path = globals.output_dir

    for nonbinary_type in tqdm(
        nonbinary_types,
        desc=f"creating plots for every nonbinary type",
        total=len(nonbinary_types),
        unit="types",
    ):
        all_event_trees: list[
            tuple[str, int, list[EventSnapshot], tuple[float, float]]
        ] = []

        for data_name in tqdm(
            data_names,
            desc=f"creating plots for every dataset",
            total=len(data_names),
            unit="datasets",
        ):
            metadata_path: Path = output_dir / data_name / "metadata.ini"
            metadata_dict: dict = metadata.read_metadata(metadata_path)
            x_periodic: tuple[float, float] = (
                metadata_dict["Domain"]["xmin"],
                metadata_dict["Domain"]["xmax"],
            )
            for seed in tqdm(
                seeds,
                total=len(seeds),
                unit="seeds",
            ):
                event_tree = compute_event(
                    data_name, output_dir, corrected, int(seed), nonbinary_type
                )
                if event_tree is not None:
                    all_event_trees.append(
                        (data_name, int(seed), event_tree, x_periodic)
                    )

        if not all_event_trees:
            continue

        floc_color_map = build_floc_color_map(
            [tree for _, _, tree, _ in all_event_trees]
        )

        for data_name, seed, event_tree, x_periodic in all_event_trees:
            plot_event(
                events=event_tree,
                plane="xy",
                nonbinary_type=nonbinary_type,
                seed=int(seed),
                output_dir=plot_dir / "nonbinary_event_frames" / data_name,
                mode="save",
                time_scale=10.0,
                floc_color_map=floc_color_map,
                x_periodic=x_periodic,
            )


if __name__ == "__main__":
    main()
