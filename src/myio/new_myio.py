from pathlib import Path
from typing import Tuple
import numpy as np


def read_csv_columns(
    csv_file: Path, cols: Tuple[int, ...], remove_nan: bool = True
) -> Tuple[np.ndarray, ...]:
    """
    Read columns from CSV file, optionally removing NaN values.

    Args:
        csv_file: Path to CSV file
        cols: Tuple of column indices to read
        remove_nan: If True, remove rows containing NaN in any of the selected columns

    Returns:
        Tuple of numpy arrays, with NaN values removed if remove_nan=True
    """
    if not csv_file.exists():
        raise FileNotFoundError(f'CSV file not found: "{csv_file}"')

    data = np.loadtxt(
        csv_file,
        skiprows=0,
        comments="%",
        delimiter=None,
        dtype=float,
        usecols=cols,
        unpack=False,
        ndmin=2,
    )

    if len(cols) == 1:
        columns_data = [data[:, 0]]
    else:
        columns_data = [data[:, i] for i in range(len(cols))]

    if remove_nan and len(columns_data) > 0:
        valid_mask = ~np.any(np.isnan(np.column_stack(columns_data)), axis=1)
        cleaned_columns = [col[valid_mask] for col in columns_data]
        return tuple(cleaned_columns)
    else:
        return tuple(columns_data)
