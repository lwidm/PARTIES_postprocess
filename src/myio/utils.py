# flocParties/io/utils.py
"""Useful I/O untilities."""
from pathlib import Path
import warnings
import numpy as np
import h5py

from natsort import natsorted

from .metadata import read_metadata


def find_data_files(data_dir: Path, wildcard: str) -> list[Path]:
    """Return all files in a directory that mach a wildcard, naturally sorted."""
    return natsorted(data_dir.glob(wildcard), key=lambda f: f.stem)


def filter_files_by_index(
    data_files: list[Path], idx_min: int | None, idx_max: int | None
) -> list[Path]:
    """Return files with indices in the specified range [idx_min, idx_max].

    Filters a list of files, keeping only those files whose
    index (derived from filename stem) falls within the inclusive range
    [idx_min, idx_max].

    Args:
        data_files: List of naturally sorted file paths
        idx_min: Minimum index to include (inclusive)
        idx_max: Maximum index to include (inclusive)

    Returns:
        Filtered list of files within the specified index range
    """

    if idx_min is None and idx_max is None:
        warnings.warn(
            "WARNING: in filter_files_by_index(...) both idx_min and idx_max are None - data_files list left unchanged!"
        )
        return data_files

    filtered_data_files: list[Path] = []
    for data_file in data_files:
        idx: int = int(data_file.stem.split("_")[-1])
        if (idx_min is None or idx >= idx_min) and (idx_max is None or idx <= idx_max):
            filtered_data_files.append(data_file)
    return filtered_data_files


def _get_steadystate_files(
    file_dir: Path, wildcard: str, metadata_file: Path, trn: bool
) -> list[Path]:
    if not file_dir.is_dir():
        raise ValueError(f'ERROR: Directory "{file_dir}" does not exist!')

    if not metadata_file.exists():
        raise ValueError(f'ERROR: Could not find metadata file at "{metadata_file}"')
    metadata: dict[str, dict[str, float | int | str]] = read_metadata(metadata_file)
    idx_steady_key: str = "idx_trn_steady" if trn else "idx_steady"
    idx_steady: int = int(metadata["Time"][idx_steady_key])
    files: list[Path] = find_data_files(file_dir, wildcard)
    if not files:
        raise ValueError(f'ERROR: No files found in "{file_dir}" matching "{wildcard}"')
    files = filter_files_by_index(files, idx_steady, None)
    return files


def get_steadystate_floc_files(
    file_dir: Path, metadata_file: Path, trn: bool
) -> list[Path]:
    wildcard: str = "Flocs_*.h5"
    return _get_steadystate_files(file_dir, wildcard, metadata_file, trn)


def get_steadystate_particle_files(
    file_dir: Path, metadata_file: Path, trn: bool
) -> list[Path]:
    wildcard: str = "Particles_*.h5"
    return _get_steadystate_files(file_dir, wildcard, metadata_file, trn)


def get_steadystate_fluid_files(file_dir: Path, metadata_file: Path) -> list[Path]:
    wildcard: str = "Data_*.h5"
    return _get_steadystate_files(file_dir, wildcard, metadata_file, trn=False)


def read_from_h5(
    h5_file: Path,
    groups_to_read: list[str] | None,
) -> dict[str, np.ndarray | dict]:

    if not h5_file.exists():
        raise ValueError(f'ERROR: File "{h5_file}" not found.')

    def _read_h5_recursive(
        h5_obj: h5py.Group | h5py.Dataset, current_path: str = ""
    ) -> np.ndarray | dict[str, np.ndarray | dict]:
        data = h5_obj[()]
        if isinstance(data, np.ndarray):
            return data
        if isinstance(h5_obj, h5py.Group):
            result: dict[str, np.ndarray | dict] = {}
            for key in h5_obj.keys():
                result[key] = _read_h5_recursive(h5_obj[key], f"{current_path}/{key}")  # type: ignore
            return result
        else:
            raise ValueError(
                f'ERROR: Unexpected HDF5 object type at "{current_path}": {type(h5_obj)}'
            )

    with h5py.File(str(h5_file), "r") as f:
        result_dict: dict[str, np.ndarray | dict] = {}
        if groups_to_read is not None:
            for group_name in groups_to_read:
                if group_name in f:
                    group: h5py.Group | h5py.Dataset = f[group_name]  # type: ignore
                    result_dict[group_name] = _read_h5_recursive(group, group_name)
                else:
                    raise ValueError(
                        f'ERROR: Group "{group_name}" not found in file "{h5_file}"!'
                    )
        else:
            for key in f.keys():
                group: h5py.Group | h5py.Dataset = f[key]  # type: ignore
                result_dict[key] = _read_h5_recursive(group, key)
    return result_dict
