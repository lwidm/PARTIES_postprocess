# flocParties/myio/output.py
"""Methods for saving postprocessing results."""
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def save_to_h5(
    out_file: Path,
    data: dict[str, np.ndarray | dict],
    src_file: Path | None = None,
    src_groups: list[str] | None = None,
):
    """Write data to the output h5 file."""

    if not out_file.parent.is_dir():
        raise ValueError(
            f'ERROR: save_to_h5(...) - Could not find directory at "{out_file.parent}"'
        )
    if src_file is not None:
        if not src_file.exists():
            raise ValueError(
                f'ERROR: save_to_h5(...) - Source file "{src_file}" not found.'
            )
        if src_groups is None:
            raise ValueError(
                "ERROR: save_to_h5(...) - src_groups must be specified when src_file is provided."
            )
        if not src_groups:
            raise ValueError(
                "ERROR: save_to_h5(...) - src_groups cannot be empty when src_file is provided."
            )

    def _save_dict_recursive(
        f_out: h5py.File | h5py.Group, key: str, items: np.ndarray | Any
    ):
        if isinstance(items, np.ndarray):
            f_out.create_dataset(key, data=items)
        elif isinstance(items, dict):
            group = f_out.create_group(key)
            for sub_key in items:
                _save_dict_recursive(group, sub_key, items[sub_key])
        else:
            raise ValueError(
                f"ERROR: save_to_h5(...) - Data at {key} does not have a valid type (should be dict or np.ndarray, got {type(items)}"
            )

    with h5py.File(str(out_file), "w") as f_out:
        for key, items in data.items():
            _save_dict_recursive(f_out, key, items)
        if src_file is not None and src_groups:
            with h5py.File(str(src_file), "r") as f_in:
                for group_name in src_groups:
                    if group_name in f_in:
                        f_in.copy(group_name, f_out)  # type: ignore
                    else:
                        raise ValueError(
                            f'ERROR: save_to_h5(...) - Group "{group_name}" not found in source file "{src_file}"'
                        )


def prepare_dict_for_h5save(
    data: dict[str, Any],
) -> dict[str, np.ndarray | dict]:
    """Ensure all datafields are numpy arrays (converts all scalars) such that save_to_h5(...) doesn't fail"""
    new_dict: dict[str, np.ndarray | dict] = {}
    for key, val in data.items():
        if isinstance(val, np.ndarray):
            new_dict[key] = val.copy()
        elif isinstance(val, (int, float, np.integer, np.floating)):
            new_dict[key] = np.array([val])
        elif isinstance(val, dict):
            new_dict[key] = prepare_dict_for_h5save(val).copy()
        else:
            raise ValueError(
                f"ERROR: prepare_dict_for_h5save(...) failed - all values in dict must be either int, float, np.ndarray or dict (recursive). Instead got {type(val)} at key {key}"
            )
    return new_dict


def save_to_pickle(out_file: Path, obj) -> None:
    with open(out_file, "wb") as f:
        pickle.dump(obj, f)


def _merge_dicts(dict_list: list[dict]) -> dict:
    """Merge a list of dicts into a single dict."""
    merged: dict = {}
    for d in dict_list:
        merged |= d
    return merged


# def make_header(filepath: Path, header_data: dict[str, str]):
#     header = [f"% {key} : {value}\n%\n" for key, value in header_data.items()]
#     header += ["% ----- End of header -----\n"]
#     return header
#
#
# def save_to_csv(out_file: Path, data: dict[str, np.ndarray], header: dict[str, str]):
#     header.update({"Last modified": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
#     with out_file.open("w") as f:
#         f.writelines(make_header(out_file, header))
#         pd.DataFrame(data).to_csv(f, index=False)

def make_header(header_lines: list[str], column_headers: list[str]) -> list[str]:
    header: list[str] = [f"% {line}\n" for line in header_lines]
    header += [
        "%\n",
        "% ----- End of header -----\n",
        "%\n",
    ]
    column_header_line: str = "".join(f"    {header:^21}" for header in column_headers)
    header += ["%" + column_header_line[1:] + "\n"]
    return header


def _format_data(data: dict[str, np.ndarray]) -> list[str]:
    column_headers: list[str] = list(data.keys())
    max_length = max(len(arr) for arr in data.values())

    lines: list[str] = []

    total_width = len(column_headers) * 24 + 2
    lines.append("% " + "-" * total_width + "\n")

    for i in range(max_length):
        row_data: list[str] = []
        for col in column_headers:
            value: float
            if i < len(data[col]):
                value = float(data[col][i])
            else:
                value = float(np.nan)

            formatted: str
            if np.isnan(value):
                formatted = f"    {str(value):^21s}"
            elif value >= 0 or np.isnan(value):
                formatted = f"    {value:17.15e}"
            else:
                formatted = f"   {value:17.15e}"  # One less space before
            row_data.append(formatted)

        line = "".join(row_data) + "\n"
        lines.append(line)

    return lines


def save_to_csv(
    out_file: Path,
    data: dict[str, np.ndarray],
    header_lines: list[str],
):
    """
    Save data to CSV file with formatted header and scientific notation.
    If arrays have different lengths, they will be padded with np.nan.

    Args:
        out_file: Output file path
        data: Dictionary where keys are column names and values are numpy arrays
        header_lines: List of header comment lines
    """
    # Pad arrays to the same length
    max_length = max(len(arr) for arr in data.values())
    padded_data = {}

    for key, arr in data.items():
        if len(arr) < max_length:
            # Create a new array with the same length, padded with nan
            padded_arr = np.full(max_length, np.nan)
            padded_arr[: len(arr)] = arr
            padded_data[key] = padded_arr
        else:
            padded_data[key] = arr


    full_header: list[str] = [
        f"This file was last updated on {datetime.now().strftime('%-m/%-d/%Y')}."
    ] + header_lines

    column_headers = list(padded_data.keys())
    header_content: list[str] = make_header(full_header, column_headers)

    formatted_data: list[str] = _format_data(padded_data)

    with out_file.open("w") as f:
        for line in header_content:
            f.write(line)
        for line in formatted_data:
            f.write(line)
