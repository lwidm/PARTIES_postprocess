# -- src/myio/myio.py

import numpy as np
from typing import Optional, List, Dict, Union, Any, Tuple, Literal
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
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _save_nested_dict(h5_group: h5py.Group, data: Dict[str, Any]) -> None:
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
            elif isinstance(value, (int, float, complex, np.floating, np.integer)):
                h5_group.create_dataset(key, data=value)
            # Handle strings
            elif isinstance(value, str):
                str_dtype = h5py.string_dtype(encoding="utf-8")
                h5_group.create_dataset(key, data=value, dtype=str_dtype)
            # Handle booleans
            elif isinstance(value, (bool, np.bool_)):
                h5_group.create_dataset(key, data=np.bool_(value))
            # Handle lists and tuples (convert to numpy arrays)
            elif isinstance(value, (list, tuple)):
                try:
                    array_value = np.array(value)
                    h5_group.create_dataset(
                        key, data=array_value, compression="gzip", compression_opts=6
                    )
                except (ValueError, TypeError):
                    warnings.warn(
                        f'when saving to h5 failed to convert tuple or list "{value}" with key "{key}" to numpy array. Converting to string instead',
                        category=UserWarning,
                    )
                    str_dtype = h5py.string_dtype(encoding="utf-8")
                    h5_group.create_dataset(key, data=str(value), dtype=str_dtype)

            else:
                warnings.warn(
                    f'Failed to recognise type of value with key "{key}". Converting to string. (Value is "{value}")',
                    category=UserWarning,
                )
                str_dtype = h5py.string_dtype(encoding="utf-8")
                h5_group.create_dataset(key, data=str(value), dtype=str_dtype)

    with h5py.File(output_path, "w") as h5_file:
        _save_nested_dict(h5_file, data_dict)
        if metadata:
            for mkey, mval in metadata.items():
                try:
                    h5_file.attrs[mkey] = mval
                except Exception:
                    h5_file.attrs[mkey] = str(mval)


def load_from_h5(
    input_path: Union[str, Path],
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    input_path = Path(input_path)

    def _load_nested_dict(h5_group: h5py.Group) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, item in h5_group.items():
            if isinstance(item, h5py.Group):
                result[key] = _load_nested_dict(item)
            elif isinstance(item, h5py.Dataset):
                data: Any = item[()]
                if isinstance(data, (bytes, bytearray)):
                    try:
                        result[key] = data.decode("utf-8")
                    except Exception:
                        result[key] = data
                # numpy scalar (0-d) -> convert to Python scalar
                elif np.asarray(data).shape == ():
                    result[key] = np.asarray(data).item()
                else:
                    result[key] = data
        return result

    metadata: Optional[Dict[str, Any]] = None
    with h5py.File(input_path, "r") as h5_file:
        data_dict = _load_nested_dict(h5_file)
        metadata = dict(h5_file.attrs) if h5_file.attrs else None

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


def list_data_files(
    path: Union[str, Path],
    base_name: str,
    min_file_index: Optional[int] = None,
    max_file_index: Optional[int] = None,
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

    if min_file_index is None and max_file_index is None:
        return all_files

    filtered_files: List[Path] = []
    file_path: Path
    for file_path in all_files:
        try:
            # Extract index from filename (assumes pattern: base_index.h5)
            file_name: str = file_path.stem
            file_index = int(file_name.split("_")[-1].split(".")[0])
            if (min_file_index is None or file_index >= min_file_index) and (
                max_file_index is None or file_index <= max_file_index
            ):
                filtered_files.append(file_path)
        except (ValueError, IndexError):
            continue

    return filtered_files


def read_coh_range(
    data_dir: Union[str, Path], particle_path: Union[str, Path]
) -> float:
    """Read the cohesive range from 'parties.inp' if not, return 0.05 * D_p."""
    data_dir = Path(data_dir)
    particle_path = Path(particle_path)
    try:
        params: Dict[str, Union[np.ndarray, int, float]] = _read_inp(
            data_dir / "parties.inp"
        )
        dy: float = float(params["ymax"] - params["ymin"]) / float(params["NYM"])
        coh_range: float = params["coh_range"]  # type: ignore
        return coh_range * dy
    except KeyError:
        warnings.warn(r"parties.inp not found, using coh_range = 0.05 * D_p.")
        with h5py.File(particle_path, "r") as f:
            return 0.1 * f["mobile/R"][0]  # type: ignore


def read_Re(data_dir: Union[str, Path]) -> float:
    """Read the cohesive range from 'parties.inp' if not, return 0.05 * D_p."""
    data_dir = Path(data_dir)
    try:
        params: Dict[str, Union[np.ndarray, int, float]] = _read_inp(
            data_dir / "parties.inp"
        )
        Re: float = params["Re"]  # type: ignore
        return Re
    except KeyError:
        raise KeyError(r"Either parties.inp not found, or Re not found in parties.inp")


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


def read_particle_data(path: Union[str, Path]) -> Dict[str, Union[np.ndarray, float]]:
    with h5py.File(path, "r") as f:
        mobile_data: h5py.Group = f["mobile"]  # type: ignore
        id: np.ndarray = np.arange(mobile_data["R"].shape[0])  # type: ignore
        x: np.ndarray = mobile_data["X"][:, 0]  # type: ignore
        y: np.ndarray = mobile_data["X"][:, 1]  # type: ignore
        z: np.ndarray = mobile_data["X"][:, 2]  # type: ignore
        r: np.ndarray = mobile_data["R"][:, 0]  # type: ignore
        u: np.ndarray = mobile_data["U"][:, 0]  # type: ignore
        v: np.ndarray = mobile_data["U"][:, 1]  # type: ignore
        w: np.ndarray = mobile_data["U"][:, 2]  # type: ignore
        time: float = f["time"][()]  # type: ignore

        particle_data: Dict[str, Union[np.ndarray, float]] = {
            "id": id,
            "x": x,
            "y": y,
            "z": z,
            "r": r,
            "u": u,
            "v": v,
            "w": w,
            "time": time,
        }

        particle_data_fields: List[str] = ["F_IBM", "F_rigid", "F_coll"]
        # BUG :
        particle_data_fields: List[str] = []
        field: str
        for field in particle_data_fields:
            particle_data = _load_3_component_field(particle_data, mobile_data, field)
    return particle_data


def _load_3_component_field(
    particle_data: Dict[str, Union[np.ndarray, float]],
    mobile_data: h5py.Group,
    fieldname: str,
):
    """Load a data field with 3 components (x, y, z)."""
    i: int
    dir: str
    for i, dir in enumerate(["x", "y", "z"]):
        particle_data[fieldname + f"_{dir}"] = mobile_data[fieldname][:, i]  # type: ignore
    return particle_data


def combine_dicts(dict_list: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    if not dict_list:
        return {}

    keys = dict_list[0].keys()
    out: Dict[str, np.ndarray] = {}

    for k in keys:
        vals = [d[k] for d in dict_list]

        is_scalar_list = [
            np.isscalar(v) or (isinstance(v, np.ndarray) and v.ndim == 0) for v in vals
        ]

        if all(is_scalar_list):
            out[k] = np.array(
                [np.asarray(v).item() if isinstance(v, np.ndarray) else v for v in vals]
            )
        else:
            arrs = [
                (
                    np.atleast_1d(v)
                    if not (
                        np.isscalar(v) or (isinstance(v, np.ndarray) and v.ndim == 0)
                    )
                    else np.atleast_1d(np.asarray(v))
                )
                for v in vals
            ]
            out[k] = np.concatenate(arrs, axis=0)

    return out


def _read_inp(inp_file: Union[Path, str]) -> Dict[str, Union[np.ndarray, int, float]]:
    """Return a dict of all parameters in a config file."""
    config_parser = configparser.ConfigParser(inline_comment_prefixes="#")

    def _optionxform(option: str) -> str:
        return option

    config_parser.optionxform = _optionxform  # type: ignore
    config_parser.read(inp_file)
    config_dicts: List[Dict[str, Union[np.ndarray, int, float]]] = [dict(config_parser[s]) for s in config_parser.sections()]  # type: ignore
    config_raw: Dict[str, str] = _merge_dicts(config_dicts)
    config_raw = {
        k: v.replace("{", "[").replace("}", "]") for k, v in config_raw.items()
    }
    config_list: Dict[str, Union[List, int, float]] = {
        k: ast.literal_eval(v) for k, v in config_raw.items()
    }
    config_np: Dict[str, Union[np.ndarray, int, float]] = {
        k: np.array(v) if isinstance(v, list) else v for k, v in config_list.items()
    }
    return config_np


def _merge_dicts(dict_list: list[dict]) -> dict:
    """Merge a list of dicts into a single dict."""
    merged: dict = {}
    for d in dict_list:
        merged |= d
    return merged


def get_time_array(
    file_prefix: str,
    parties_data_dir: Union[str, Path],
    min_file_index: Optional[int],
    max_file_index: Optional[int],
    key: Optional[str] = None,
) -> np.ndarray:

    print("Obtaining time array of data hdf5 files")
    print(
        f'Looking for datafile in directory: "{parties_data_dir}" with min_file_index: {min_file_index} and max_file_index: {max_file_index}'
    )
    if key == None:
        key = "time"
    data_files: List[Path] = list_data_files(
        parties_data_dir, file_prefix, min_file_index, max_file_index
    )

    t_arr: np.ndarray = np.zeros(len(data_files))

    for i, data_file in enumerate(data_files):
        with h5py.File(data_file, "r") as h5_file:
            t_arr[i] = h5_file[key][()]  # type: ignore

    return t_arr
