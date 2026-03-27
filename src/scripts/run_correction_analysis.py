from typing import Literal, TypedDict
from pathlib import Path
import h5py
import numpy as np
import pickle
import random

import shutil

import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm

from src.flocs.family_tree import FlocRecord, FamilyTreeType
from src.myio import metadata, utils
from src import globals
from src.plotting import templates as plt_templ


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

    # Step 1: Find all relevant particles by reading floc_id and id arrays once per file
    target_floc_ids = np.array([floc_id] + child_ids + parent_ids, dtype=int)
    relevant_particles: set[int] = set()

    for f in [prepre_event_file, pre_event_file, post_event_file]:
        with h5py.File(str(f), "r") as h5_file:
            file_floc_ids: np.ndarray = h5_file["floc_id"][:]  # type: ignore
            file_ids: np.ndarray = h5_file["id"][:]  # type: ignore
            mask = np.isin(file_floc_ids, target_floc_ids)
            relevant_particles.update(file_ids[mask].tolist())

    # Step 2: Find all relevant flocs + extract event data in a single pass
    relevant_particles_arr = np.array(list(relevant_particles), dtype=int)
    relevant_flocs: set[int] = set()
    cached_data: dict[str, dict] = {}

    for floc_file in selected:
        with h5py.File(str(floc_file), "r") as h5_file:
            file_particle_ids: np.ndarray = h5_file["id"][:]  # type: ignore
            file_floc_ids: np.ndarray = h5_file["floc_id"][:]  # type: ignore
            particle_mask = np.isin(file_particle_ids, relevant_particles_arr)
            relevant_flocs.update(file_floc_ids[particle_mask].tolist())
            cached_data[str(floc_file)] = {
                "time": float(h5_file["time"][0]),  # type: ignore
                "file_id": int(floc_file.stem.split("_")[1]),
                "floc_ids": file_floc_ids,
                "x": h5_file["x"][:],  # type: ignore
                "y": h5_file["y"][:],  # type: ignore
                "z": h5_file["z"][:],  # type: ignore
                "r": h5_file["r"][:],  # type: ignore
            }

    # Step 3: Filter cached data by relevant flocs
    relevant_flocs_arr = np.array(list(relevant_flocs), dtype=int)
    events: list[EventSnapshot] = []

    for floc_file in selected:
        d = cached_data[str(floc_file)]
        mask = np.isin(d["floc_ids"], relevant_flocs_arr)
        events.append(
            {
                "time": d["time"],
                "file_id": d["file_id"],
                "x": d["x"][mask],
                "y": d["y"][mask],
                "z": d["z"][mask],
                "r": d["r"][mask],
                "floc_ids": d["floc_ids"][mask],
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
    floc_color_map: dict[int, tuple[float, float, float]],
    x_periodic: tuple[float, float] | None,
    show_seed: bool,
) -> None:
    axis_map: dict[str, tuple[str, str, str]] = {
        "xy": ("x", "y", "z"),
        "yz": ("y", "z", "x"),
        "xz": ("x", "z", "y"),
    }
    h_key, v_key, depth_key = axis_map[plane]

    if x_periodic is not None:
        unwrap_periodic(events, "x", x_periodic[0], x_periodic[1])

    if all(len(e[h_key]) == 0 for e in events):
        return

    title = f"seed {seed}" if show_seed else None

    for frame_idx, event in enumerate(events):
        h_vals: np.ndarray = event[h_key]
        v_vals: np.ndarray = event[v_key]
        radii: np.ndarray = event["r"]

        if len(h_vals) == 0:
            continue

        r_max: float = float(np.max(radii))
        h_min = float(np.min(h_vals)) - r_max
        h_max = float(np.max(h_vals)) + r_max
        v_min = float(np.min(v_vals)) - r_max
        v_max = float(np.max(v_vals)) + r_max

        # Make the bounding box square so particles aren't distorted
        h_span = h_max - h_min
        v_span = v_max - v_min
        max_span = max(h_span, v_span)
        h_center = (h_min + h_max) / 2
        v_center = (v_min + v_max) / 2
        h_min = h_center - max_span / 2
        h_max = h_center + max_span / 2
        v_min = v_center - max_span / 2
        v_max = v_center + max_span / 2

        pad = max_span * 0.05
        h_min -= pad
        h_max += pad
        v_min -= pad
        v_max += pad

        plt_templ.event_tree_frame(
            output_dir=output_dir,
            h_vals=h_vals,
            v_vals=v_vals,
            radii=radii,
            floc_ids=event["floc_ids"],
            floc_color_map=floc_color_map,
            xlim=(h_min, h_max),
            ylim=(v_min, v_max),
            name=f"{nonbinary_type}_seed{seed:04d}_frame{frame_idx:04d}",
            title=title,
        )


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
    # seeds: np.ndarray = np.asarray([1, 2, 3, 4, 5]) * 19 + 1
    seeds: np.ndarray = np.asarray([128])
    nonbinary_types: list[
        Literal["breakup", "agglomeration", "simultaneous", "mass_conservation"]
    ] = ["breakup"]
    # ] = ["breakup", "agglomeration", "simultaneous", "mass_conservation"]

    data_names: list[str] = globals.data_names

    plot_dir: Path = globals.plot_dir
    output_dir: Path = globals.output_dir

    # Clear old event frame plots
    event_frames_dir = plot_dir / "nonbinary_event_frames"
    if event_frames_dir.exists():
        shutil.rmtree(event_frames_dir)

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
                floc_color_map=floc_color_map,
                x_periodic=x_periodic,
                show_seed=True,
            )

    if not globals.on_anvil:
        plt.show()


if __name__ == "__main__":
    main()
