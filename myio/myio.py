# -- myio/myio.py

import numpy as np
from typing import Optional, List, Dict, Union, Any, Tuple
from natsort import natsorted
from pathlib import Path
import io
import re
import glob
import warnings
import h5py # type: ignore


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
                    warnings.warn('when saving to h5 failed to convert tuple or list "{value}" with key "{key}" to numpy array. Converting to np.str_ instead', category=UserWarning)
                    # If conversion fails, save as string representation
                    h5_group.create_dataset(key, data=np.str_(str(value)))
            else:
                warnings.warn(f'Failed to recognise type of value with key "{key}". Converting to np.str_ instead. (Value is "{value}")', category=UserWarning)
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

    data: np.ndarray = np.genfromtxt(io.StringIO("\n".join(lines)), dtype=np.float64, delimiter=None)
    if data.size == 0:
        raise ValueError("no numeric data parsed")
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    return [data[:, i] for i in range(data.shape[1])]


def list_parties_data_files(
    path: str,
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
