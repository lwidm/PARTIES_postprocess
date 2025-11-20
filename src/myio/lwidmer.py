from pathlib import Path
import numpy as np
import pickle
from typing import Any, Literal
from src.myio import output


def read_csv_columns(
    csv_file: Path, cols: tuple[int, ...], remove_nan: bool = True
) -> tuple[np.ndarray, ...]:
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


def load_from_pickle(file_path: Path) -> Any:
    with open(str(file_path), "rb") as f:
        data: Any = pickle.load(f)
    return data


def save_floc_breakup_formation_pdf(
    output_dir: Path,
    type: Literal["breakup", "formation"],
    stats: dict,
) -> dict[str, np.ndarray]:

    y: np.ndarray = stats[type]["unweighted"]["bin_means"]  # type: ignore
    PDF: np.ndarray = stats[type]["unweighted"]["probabs_mean"]  # type: ignore
    # err: np.ndarray = stats[type]["unweighted"]["probabs_err"]  # type: ignore

    header_lines = [
        "==================================================================",
        f"The wall normal distribution of the floc {type}",
        "==================================================================",
        "",
        "Column descriptions:",
        f"- y/L: The mean value withing the histogram bin of the floc {type} location (wall normal coordinate normalised by charachteristic length)",
        f"- PDF: The PDF of the floc {type}",
        # "- err: The standard deviation of the PDF value in each bin over all the files devided by the total number of files",
        "",
    ]

    result: dict[str, np.ndarray] = {
        f"y/L": y,
        "PDF": PDF,
        # "err": err,
    }

    if output_dir is not None:
        output_dir.mkdir(exist_ok=True)
        csv_file: Path = output_dir / f"floc_{type}_pdf.csv"
        output.save_to_csv(csv_file, result, header_lines)

    return result
