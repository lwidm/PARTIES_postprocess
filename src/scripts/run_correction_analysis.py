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

NonbinaryEventResult = dict[str, int | list[int] | list[list[int]]]


class EventSnapshot(TypedDict):
    file_id: int
    time: float
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    r: np.ndarray
    floc_ids: np.ndarray


def get_event_tree(
    floc_files: list[Path],
    floc_id: int,
    parent_ids: list[int],
    child_ids: list[int],
    file_id: int,
    n_before: int,
    n_after: int,
) -> list[EventSnapshot]:

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


def _empty_event_result() -> NonbinaryEventResult:
    return {
        "count": 0,
        "floc_ids": [],
        "start_file_ids": [],
        "end_file_ids": [],
        "child_ids": [],
        "parent_ids": [],
    }


def _append_event(
    result: NonbinaryEventResult,
    floc_id: int,
    start_file_id: int,
    end_file_id: int,
    child_ids: list[int],
    parent_ids: list[int],
) -> None:
    result["floc_ids"].append(floc_id)  # type: ignore
    result["start_file_ids"].append(start_file_id)  # type: ignore
    result["end_file_ids"].append(end_file_id)  # type: ignore
    result["child_ids"].append(child_ids)  # type: ignore
    result["parent_ids"].append(parent_ids)  # type: ignore
    result["count"] += 1  # type: ignore


def find_all_nonbinary_events(
    family_tree: FamilyTreeType,
    t_min: float,
    size_min: int,
) -> dict[str, NonbinaryEventResult]:
    results: dict[str, NonbinaryEventResult] = {
        "breakup": _empty_event_result(),
        "agglomeration": _empty_event_result(),
        "simultaneous": _empty_event_result(),
        "mass_conservation": _empty_event_result(),
    }

    for floc_id, floc_record in family_tree.items():
        start_file_id: int = floc_record["start_file_id"]
        end_file_id: int = floc_record["end_file_id"]
        if t_min > floc_record["start_time"]:
            continue
        if size_min > floc_record["size"]:
            continue

        # breakup
        if floc_record["children"]:
            local_child_ids: list[int] = floc_record["children"]
            if len(local_child_ids) > 2:
                _append_event(
                    results["breakup"],
                    floc_id,
                    start_file_id,
                    end_file_id,
                    local_child_ids,
                    [],
                )

        # agglomeration
        if floc_record["parents"]:
            local_parent_ids: list[int] = floc_record["parents"]
            if len(local_parent_ids) > 2:
                _append_event(
                    results["agglomeration"],
                    floc_id,
                    start_file_id,
                    end_file_id,
                    [],
                    local_parent_ids,
                )

        # mass_conservation and simultaneous
        if floc_record["children"] and len(floc_record["children"]) > 1:
            local_child_ids = floc_record["children"]
            all_parents: set[int] = set()
            for cid in local_child_ids:
                all_parents.update(family_tree[cid]["parents"])
            local_parent_ids = list(all_parents)

            # mass_conservation
            current_size = floc_record["size"]
            children_sizes = floc_record["children_sizes"]
            if current_size != sum(children_sizes):
                _append_event(
                    results["mass_conservation"],
                    floc_id,
                    start_file_id,
                    end_file_id,
                    local_child_ids,
                    local_parent_ids,
                )

            # simultaneous
            if len(local_child_ids) >= 2:
                child1_constituents: list[int] = family_tree[local_child_ids[0]][
                    "constituents"
                ]
                child2_constituents: list[int] = family_tree[local_child_ids[1]][
                    "constituents"
                ]
                if set(floc_record["constituents"]) != set(
                    child1_constituents + child2_constituents
                ):
                    _append_event(
                        results["simultaneous"],
                        floc_id,
                        start_file_id,
                        end_file_id,
                        local_child_ids,
                        local_parent_ids,
                    )

    return results


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
    data_name: str,
    seed: int,
    output_dir: Path,
    floc_color_map: dict[int, tuple[float, float, float]],
    x_periodic: tuple[float, float] | None,
    show_title: bool,
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

    title = f"{data_name}, {nonbinary_type}, seed {seed}" if show_title else None

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
    nonbinary_events: NonbinaryEventResult,
    nonbinary_type: Literal[
        "breakup", "agglomeration", "simultaneous", "mass_conservation"
    ],
    floc_files: list[Path],
    seed: int,
) -> list[EventSnapshot] | None:
    n_events: int = nonbinary_events["count"]  # type: ignore
    if n_events == 0:
        return None

    rng = random.Random(seed)
    i: int = rng.randint(0, n_events - 1)

    file_id: int
    n_before: int
    n_after: int
    if nonbinary_type == "breakup":
        file_id = nonbinary_events["end_file_ids"][i]  # type: ignore
        n_before = 0
        n_after = 1
    elif nonbinary_type == "agglomeration":
        file_id = nonbinary_events["start_file_ids"][i]  # type: ignore
        n_before = 1
        n_after = 0
    elif nonbinary_type == "simultaneous":
        file_id = nonbinary_events["end_file_ids"][i]  # type: ignore
        n_before = 0
        n_after = 1
    elif nonbinary_type == "mass_conservation":
        file_id = nonbinary_events["end_file_ids"][i]  # type: ignore
        n_before = 0
        n_after = 1
    else:
        raise NotImplementedError

    return get_event_tree(
        floc_files=floc_files,
        floc_id=nonbinary_events["floc_ids"][i],  # type: ignore
        parent_ids=nonbinary_events["parent_ids"][i],  # type: ignore
        child_ids=nonbinary_events["child_ids"][i],  # type: ignore
        file_id=file_id,
        n_before=n_before,
        n_after=n_after,
    )


def main() -> None:
    corrected: bool = False
    show_title: bool = False
    default_seeds: list[int] = (
        (np.asarray([1, 2, 3, 4, 5]) * 19 + 1).astype(int).tolist()
    )
    # default_seeds: list[int] = [128]
    seeds_per_type: dict[str, dict[str, list[int]]] = {
        nbt: {dn: default_seeds for dn in globals.data_names}
        for nbt in ["breakup", "agglomeration", "simultaneous", "mass_conservation"]
    }
    seeds_per_type: dict[str, dict[str, list[int]]] = {
        "breakup": {
            "phi1p5": [],
            "phi3p0": [96],
            "phi5p0_new": [],
        },
        "agglomeration": {
            "phi1p5": [],
            "phi3p0": [39],
            "phi5p0_new": [],
        },
        "simultaneous": {
            "phi1p5": [],
            "phi3p0": [20],
            "phi5p0_new": [],
        },
        "mass_conservation": {
            "phi1p5": [],
            "phi3p0": [20],
            # "phi3p0": [],
            "phi5p0_new": [],
        },
    }
    nonbinary_types: list[
        Literal["breakup", "agglomeration", "simultaneous", "mass_conservation"]
    # ] = ["breakup"]
    ] = ["breakup", "agglomeration", "simultaneous", "mass_conservation"]

    data_names: list[str] = globals.data_names

    plot_dir: Path = globals.plot_dir
    output_dir: Path = globals.output_dir

    # Clear old event frame plots
    event_frames_dir = plot_dir / "nonbinary_event_frames"
    event_frames_dir.mkdir(parents=True, exist_ok=True)
    if event_frames_dir.exists():
        shutil.rmtree(event_frames_dir)

    # Pre-load per-dataset: family tree, metadata, file list
    dataset_cache: dict[str, dict] = {}
    for data_name in tqdm(data_names, desc="loading datasets", unit="datasets"):
        pickle_dir: Path = output_dir / data_name
        pickle_file = "family_tree_corrected.pkl" if corrected else "family_tree.pkl"
        with open(pickle_dir / pickle_file, "rb") as file:
            fam_tree: FamilyTreeType = pickle.load(file)

        metadata_path: Path = output_dir / data_name / "metadata.ini"
        metadata_dict: dict = metadata.read_metadata(metadata_path)
        t_steady: float = metadata_dict["Time"]["t_steady"]
        t_end: float = metadata_dict["Time"]["t_end"]
        t_min: float = t_steady + 2 / 3 * (t_end - t_steady)

        floc_files: list[Path] = utils.find_data_files(
            output_dir / data_name / "flocs", "Particles_*"
        )

        all_events = find_all_nonbinary_events(fam_tree, t_min, 20)

        dataset_cache[data_name] = {
            "x_periodic": (
                metadata_dict["Domain"]["xmin"],
                metadata_dict["Domain"]["xmax"],
            ),
            "floc_files": floc_files,
            "all_events": all_events,
        }

    # Build flat work list and compute all event trees
    work_items: list[tuple[str, str, int]] = []
    for nonbinary_type in nonbinary_types:
        for data_name in data_names:
            for seed in seeds_per_type[nonbinary_type][data_name]:
                work_items.append((nonbinary_type, data_name, seed))

    all_event_trees_by_type: dict[
        str, list[tuple[str, int, list[EventSnapshot], tuple[float, float]]]
    ] = {t: [] for t in nonbinary_types}

    for nonbinary_type, data_name, seed in tqdm(
        work_items, desc="computing and plotting events", unit="events"
    ):
        cache = dataset_cache[data_name]
        event_tree = compute_event(
            nonbinary_events=cache["all_events"][nonbinary_type],
            nonbinary_type=nonbinary_type,
            floc_files=cache["floc_files"],
            seed=seed,
        )
        if event_tree is not None:
            all_event_trees_by_type[nonbinary_type].append(
                (data_name, seed, event_tree, cache["x_periodic"])
            )

    for nonbinary_type in nonbinary_types:
        all_event_trees = all_event_trees_by_type[nonbinary_type]
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
                data_name=data_name,
                seed=seed,
                output_dir=plot_dir / "nonbinary_event_frames" / data_name,
                floc_color_map=floc_color_map,
                x_periodic=x_periodic,
                show_title=show_title,
            )

    if not globals.on_anvil:
        plt.show()


if __name__ == "__main__":
    main()
