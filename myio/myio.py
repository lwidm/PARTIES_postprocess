# -- myio/myio.py

import numpy as np
from typing import Optional, List, Dict, Union, Any, Tuple
from natsort import natsorted
from pathlib import Path
import io
import re
import glob
import warnings
import h5py  # type: ignore
import configparser
import ast
import sys

sys.setrecursionlimit(int(1e9))


def save_to_h5(
    output_path: Union[str, Path],
    data_dict: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save a nested dictionary to HDF5 file with proper path handling and recursive structure.

    Args:
        data_dict: Dictionary containing data to save. Can have nested dictionaries.
        output_path: Path where the HDF5 file should be saved.
        metadata: Optional dictionary of metadata to store as attributes at root level.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _save_nested_dict(h5_group: h5py.Group, data: Dict[str, Any]) -> None:
        key: str
        value: Any
        for key, value in data.items():
            if isinstance(value, dict):
                subgroup = h5_group.create_group(key)
                _save_nested_dict(subgroup, value)
            # Handle numpy arrays
            elif isinstance(value, np.ndarray):
                h5_group.create_dataset(
                    key, data=value, compression="gzip", compression_opts=6
                )
            # Handle scalar values
            elif isinstance(value, (int, float, complex)):
                h5_group.create_dataset(key, data=value)
            # Handle strings
            elif isinstance(value, str):
                h5_group.create_dataset(key, data=np.str_(value))
            # Handle booleans
            elif isinstance(value, bool):
                h5_group.create_dataset(key, data=int(value))
            # Handle lists and tuples (convert to numpy arrays)
            elif isinstance(value, (list, tuple)):
                try:
                    array_value = np.array(value)
                    h5_group.create_dataset(
                        key, data=array_value, compression="gzip", compression_opts=6
                    )
                except (ValueError, TypeError):
                    warnings.warn(
                        'when saving to h5 failed to convert tuple or list "{value}" with key "{key}" to numpy array. Converting to np.str_ instead',
                        category=UserWarning,
                    )
                    # If conversion fails, save as string representation
                    h5_group.create_dataset(key, data=np.str_(str(value)))
            else:
                warnings.warn(
                    f'Failed to recognise type of value with key "{key}". Converting to np.str_ instead. (Value is "{value}")',
                    category=UserWarning,
                )
                # Fallback: convert to string
                h5_group.create_dataset(key, data=np.str_(str(value)))

    with h5py.File(output_path, "w") as h5_file:
        _save_nested_dict(h5_file, data_dict)

        if metadata:
            key: str
            value: Any
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    h5_file.attrs[key] = value
                elif isinstance(value, (list, tuple)) and all(
                    isinstance(x, (str, int, float, bool)) for x in value
                ):
                    h5_file.attrs[key] = value
                else:
                    h5_file.attrs[key] = str(value)


def load_from_h5(
    input_path: Union[str, Path],
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """
    Load a nested dictionary from HDF5 file.

    Args:
        input_path: Path to the HDF5 file to load.

    Returns:
        Nested dictionary containing all data and structure from the HDF5 file.
    """
    input_path = Path(input_path)

    def _load_nested_dict(h5_group: h5py.Group) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        key: str
        item: Any
        for key, item in h5_group.items():
            if isinstance(item, h5py.Group):
                result[key] = _load_nested_dict(item)
            elif isinstance(item, h5py.Dataset):
                data: Any = item[()]
                if isinstance(data, bytes):
                    # Convert bytes back to string
                    result[key] = data.decode("utf-8")
                elif data.shape == ():  # Scalar
                    result[key] = data.item()
                else:
                    result[key] = data
        return result

    metadata: Optional[Dict[str, Any]] = None
    with h5py.File(input_path, "r") as h5_file:
        data_dict = _load_nested_dict(h5_file)

        # Load metadata from attributes
        if h5_file.attrs:
            data_dict["_metadata"] = dict(h5_file.attrs)

    return data_dict, metadata


def load_columns_from_txt_numpy(
    path: str, split_chars: str = ",;", comment_chars: str = "#%"
) -> List[np.ndarray]:
    """
    Load numeric columns from a text file and return a list of NumPy 1-D arrays.

    Args:
        path: path to file
        split_chars: characters treated as extra delimiters (each replaced by whitespace)
        comment_chars: characters that start comments (full-line or inline)

    Returns:
        List of 1-D numpy arrays, one per column.
    """
    esc_split: Optional[str] = re.escape(split_chars) if split_chars else ""
    esc_comment: Optional[str] = re.escape(comment_chars) if comment_chars else ""
    pattern_split: Optional[str] = f"[{esc_split}]+" if esc_split else None
    pattern_comment: Optional[str] = f"[{esc_comment}].*" if esc_comment else None

    lines: list[str] = []
    with open(path, "r", encoding="utf-8") as file:
        line: str
        for line in file:
            line = line.strip()
            if not line:
                continue
            # remove comments
            if pattern_comment is not None:
                line = re.sub(pattern_comment, "", line, count=1).strip()
                if line == "":
                    continue
            # split into columns
            if pattern_split is not None:
                line = re.sub(pattern_split, " ", line)
            line = " ".join(line.split())
            if line != "":
                lines.append(line)

    if len(lines) == 0:
        raise ValueError("no data found")

    data: np.ndarray = np.genfromtxt(
        io.StringIO("\n".join(lines)), dtype=np.float64, delimiter=None
    )
    if data.size == 0:
        raise ValueError("no numeric data parsed")
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    return [data[:, i] for i in range(data.shape[1])]


def list_parties_data_files(
    path: Union[str, Path],
    base_name: str,
    min_index: Optional[int] = None,
    max_index: Optional[int] = None,
) -> List[Path]:
    """
    Find HDF5 data files matching the pattern and filter by index range.

    Args:
        base_name: Base pattern for file names (e.g., "Data" for Data_*.h5)
        min_index: Minimum file index to include (inclusive)
        max_index: Maximum file index to include (inclusive)

    Returns:
        Sorted list of file paths matching the criteria
    """
    file_pattern: str = f"{path}/{base_name}_*.h5"
    all_files: List[Path] = [Path(f) for f in glob.glob(file_pattern)]
    all_files = natsorted(all_files, key=lambda f: f.stem)

    if min_index is None and max_index is None:
        return all_files

    filtered_files: List[Path] = []
    file_path: Path
    for file_path in all_files:
        try:
            # Extract index from filename (assumes pattern: base_index.h5)
            file_name: str = file_path.stem
            file_index = int(file_name.split("_")[-1].split(".")[0])
            if (min_index is None or file_index >= min_index) and (
                max_index is None or file_index <= max_index
            ):
                filtered_files.append(file_path)
        except (ValueError, IndexError):
            continue

    return filtered_files


def read_coh_range(data_dir: Path, particle_path: Path) -> float:
    """Read the cohesive range from 'parties.inp' if not, return 0.05 * D_p."""
    try:
        params: Dict[str, Union[np.ndarray, int, float]] = _read_inp(
            data_dir / "parties.inp"
        )
        dy: float = float(params["ymax"] - params["ymin"]) / float(params["NYM"])
        coh_range: float = params["coh_range"] # type: ignore
        return coh_range * dy
    except KeyError:
        warnings.warn(r"parties.inp not found, using coh_range = 0.05 * D_p.")
        with h5py.File(particle_path, "r") as f:
            return 0.1 * f["mobile/R"][0] # type: ignore


def read_domain_info(path: Path) -> Dict[str, Union[int, float]]:
    """Returns a dict containing the domain size and periodicity in each direction."""
    with h5py.File(path, "r") as f:
        domain_data: h5py.Group = f["domain"]  # type: ignore
        Lx: int = domain_data["xmax"][0]  # type: ignore
        Ly: int = domain_data["ymax"][0]  # type: ignore
        Lz: int = domain_data["zmax"][0]  # type: ignore
        x_periodic: int = domain_data["periodic"][0]  # type: ignore
        y_periodic: int = domain_data["periodic"][1]  # type: ignore
        z_periodic: int = domain_data["periodic"][2]  # type: ignore
        domain: Dict[str, Union[int, float]] = {
            "Lx": Lx,
            "Ly": Ly,
            "Lz": Lz,
            "x_periodic": x_periodic,
            "y_periodic": y_periodic,
            "z_periodic": z_periodic,
        }
    return domain


def read_particle_data(path: Union[str, Path]) -> Dict[str, np.ndarray]:
    with h5py.File(path, "r") as f:
        mobile_data: h5py.Group = f["mobile"]  # type: ignore
        id: np.ndarray = np.arange(mobile_data["R"].shape[0])  # type: ignore
        x: np.ndarray = mobile_data["X"][:, 0]  # type: ignore
        y: np.ndarray = mobile_data["X"][:, 1]  # type: ignore
        z: np.ndarray = mobile_data["X"][:, 1]  # type: ignore
        r: np.ndarray = mobile_data["R"][:, 0]  # type: ignore
        u: np.ndarray = mobile_data["U"][:, 0]  # type: ignore
        v: np.ndarray = mobile_data["U"][:, 1]  # type: ignore
        w: np.ndarray = mobile_data["U"][:, 2]  # type: ignore

        particle_data: Dict[str, np.ndarray] = {
            "id": id,
            "x": x,
            "y": y,
            "z": z,
            "r": r,
            "u": u,
            "v": v,
            "w": w,
        }

    # TODO :
    # particle_data_fields: List[str] = ["F_IBM", "F_rigid", "F_coll"]
    particle_data_fields: List[str] = []
    field: str
    for field in particle_data_fields:
        particle_data = _load_3_component_field(particle_data, mobile_data, field)
    return particle_data


def _load_3_component_field(
    particle_data: Dict[str, np.ndarray], mobile_data: h5py.Group, fieldname: str
):
    """Load a data field with 3 components (x, y, z)."""
    i: int
    dir: str
    for i, dir in enumerate(["x", "y", "z"]):
        particle_data[fieldname + f"_{dir}"] = mobile_data[fieldname][:, i]  # type: ignore
    return particle_data


def combine_dicts(dict_list: List[Dict], scalar: bool = False) -> Dict:
    """Combine a list of dictionaries with the same keys along their columns."""
    if not dict_list:
        return {}

    if scalar:
        return {k: np.array([d[k] for d in dict_list]) for k in dict_list[0]}
    return {k: np.concatenate([d[k] for d in dict_list]) for k in dict_list[0]}


def _read_inp(inp_file: Union[Path, str]) -> Dict[str, Union[np.ndarray, int, float]]:
    """Return a dict of all parameters in a config file."""
    config_parser = configparser.ConfigParser(inline_comment_prefixes="#")
    def _optionxform(option: str) -> str:
        return option
    config_parser.optionxform = _optionxform # type: ignore
    config_parser.read(inp_file)
    config_dicts: List[Dict[str, Union[np.ndarray, int, float]]] = [dict(config_parser[s]) for s in config_parser.sections()]  # type: ignore
    config_raw: Dict[str, str] = _merge_dicts(config_dicts)
    config_raw = {k: v.replace("{", "[").replace("}", "]") for k, v in config_raw.items()}
    config_list:  Dict[str, Union[List, int, float]] = {k: ast.literal_eval(v) for k, v in config_raw.items()}
    config_np: Dict[str, Union[np.ndarray, int, float]] = {k: np.array(v) if isinstance(v, list) else v for k, v in config_list.items()}
    return config_np


def _merge_dicts(dict_list: list[dict]) -> dict:
    """Merge a list of dicts into a single dict."""
    merged: dict = {}
    for d in dict_list:
        merged |= d
    return merged
