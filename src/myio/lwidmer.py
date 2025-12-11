from pathlib import Path
import numpy as np
import pickle
from typing import Any, Literal
from src.myio import output


def read_csv_columns(
    csv_file: Path, cols: tuple[int, ...], remove_nan: Literal[0, 1, 2]
) -> tuple[np.ndarray, ...]:
    """
    Read columns from CSV file with flexible NaN-removal modes.

    Args:
        csv_file: Path to CSV file
        cols: Tuple of column indices to read
        remove_nan:
            - 0 : don't remove any NaNs (return as-is)
            - 1 : remove trailing NaNs independently per column
                  (removes padding at the bottom only)
            - 2 : remove rows that contain NaN in any selected column
                  (row-wise removal across all returned columns)

    Returns:
        Tuple of numpy arrays (one array per requested column).
    """
    if not csv_file.exists():
        raise FileNotFoundError(f'CSV file not found: "{csv_file}"')

    # load the requested columns
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

    if data.ndim == 1:
        data = data.reshape(-1, 1)

    n_requested = len(cols)
    if data.shape[1] != n_requested:
        data = data[:, :n_requested]

    columns_data = [data[:, i].copy() for i in range(data.shape[1])]

    if remove_nan == 0:
        return tuple(columns_data)

    if remove_nan == 1:
        cleaned = []
        for col in columns_data:
            if col.size == 0:
                cleaned.append(col.copy())
                continue
            non_nan_idx = np.where(~np.isnan(col))[0]
            if non_nan_idx.size == 0:
                cleaned.append(np.empty(0, dtype=col.dtype))
            else:
                last_valid = non_nan_idx[-1]
                cleaned.append(col[: last_valid + 1].copy())
        return tuple(cleaned)

    if remove_nan == 2:
        stacked = np.column_stack(columns_data) if len(columns_data) > 1 else columns_data[0].reshape(-1, 1)
        valid_mask = ~np.any(np.isnan(stacked), axis=1)
        cleaned_columns = [col[valid_mask].copy() for col in columns_data]
        return tuple(cleaned_columns)


def load_from_pickle(file_path: Path) -> Any:
    with open(str(file_path), "rb") as f:
        data: Any = pickle.load(f)
    return data


def save_floc_breakup_formation_pdf(
    output_dir: Path,
    type: Literal["breakup", "formation"],
    stats: dict,
    filtered_t_min: bool,
    name: str | None,
) -> dict[str, np.ndarray]:

    y: np.ndarray = stats[type]["unweighted"]["bin_means"]  # type: ignore
    edges: np.ndarray = stats[type]["edges"]  # type: ignore
    PDF: np.ndarray = stats[type]["unweighted"]["probabs_mean"]  # type: ignore
    # err: np.ndarray = stats[type]["unweighted"]["probabs_err"]  # type: ignore

    filtered_line = ""
    if filtered_t_min:
        filtered_line = ", where a flocs that live longer then the maximum predicted contact time between two non cohesive particles in a poisseuille flow"

    header_lines = [
        "==================================================================",
        f"The wall normal distribution of the floc {type}{filtered_line}",
        "==================================================================",
        "",
        "Column descriptions:",
        f"- y/L: The mean value withing the histogram bin of the floc {type} location (wall normal coordinate normalised by charachteristic length)",
        f"- y/L edges: The pdf histogram edges of each bin for the floc {type} location (wall normal coordinate normalised by charachteristic length)",
        f"- PDF: The PDF of the floc {type}",
        # "- err: The standard deviation of the PDF value in each bin over all the files devided by the total number of files",
        "",
    ]

    result: dict[str, np.ndarray] = {
        f"y/L": y,
        f"y/L edges": edges,
        "PDF": PDF,
        # "err": err,
    }

    if output_dir is not None:
        if name is None:
            name = f"floc_{type}_filtered_pdf.csv" if filtered_t_min else f"floc_{type}_non_filtered_pdf.csv"
        output_dir.mkdir(exist_ok=True)
        csv_file: Path = output_dir / name
        output.save_to_csv(csv_file, result, header_lines)

    return result

def save_floc_lifetime_stats(
    output_dir: Path,
    stats: dict,
) -> dict[str, np.ndarray]:

    y_mean: np.ndarray = stats["y_mean"]  # type: ignore
    y_edges: np.ndarray = stats["y_edges"]  # type: ignore
    mean: np.ndarray = stats["mean"]  # type: ignore
    median: np.ndarray = stats["median"]  # type: ignore
    std: np.ndarray = stats["std"]  # type: ignore
    max: np.ndarray = stats["max"]  # type: ignore

    header_lines = [
        "==================================================================",
        f"The floc lifetime statistics as a funcion of wall nomral coord",
        "==================================================================",
        "",
        "Column descriptions:",
        f"- y/L: The mean value withing the histogram bin of the floc location. (formation location + breakup location ) / 2",
        f"       (wall normal coordinate normalised by charachteristic length)",
        f"- y/L edges: The pdf histogram edges of each bin of the floc location. (formation location + breakup location ) / 2",
        f"             (wall normal coordinate normalised by charachteristic length)",
        f"- max(t_floc): maximum floc lifetime in each bin",
        f"- std(t_floc): standard deviation of the floc lifetime in each bin",
        f"- mean(t_floc): mean floc lifetime in each bin",
        f"- median(t_floc): median floc lifetime in each bin",
        # "- err: The standard deviation of the PDF value in each bin over all the files devided by the total number of files",
        "",
    ]

    result: dict[str, np.ndarray] = {
        f"y/L": y_mean,
        f"y/L edges": y_edges,
        f"max(t_floc)": max,
        f"std(t_floc)": std,
        f"mean(t_floc)": mean,
        f"median(t_floc)": median,
    }

    if output_dir is not None:
        name = f"floc_lifetime.csv"
        output_dir.mkdir(exist_ok=True)
        csv_file: Path = output_dir / name
        output.save_to_csv(csv_file, result, header_lines)

    return result

def save_floculation_balance(
    output_dir: Path,
    stats: dict,
    corrected: bool,
) -> dict[str, np.ndarray]:

    n_p: np.ndarray = stats["center_sizes_arr"]
    T_coag: np.ndarray =   stats["T_coag"]
    T_frag: np.ndarray =   stats["T_frag"]
    dn_dt: np.ndarray =   stats["dn_dt"]
    T_coag_mass: np.ndarray =   stats["T_coag_mass"]
    T_frag_mass: np.ndarray =   stats["T_frag_mass"]
    dn_dt_mass: np.ndarray =   stats["dn_dt_mass"]
    T_coag_cumsum: np.ndarray =   stats["T_coag_cumsum"]
    T_frag_cumsum: np.ndarray =   stats["T_frag_cumsum"]
    dn_dt_cumsum: np.ndarray =   stats["dn_dt_cumsum"]
    T_coag_mass_cumsum: np.ndarray =   stats["T_coag_mass_cumsum"]
    T_frag_mass_cumsum: np.ndarray =   stats["T_frag_mass_cumsum"]
    dn_dt_mass_cumsum: np.ndarray =   stats["dn_dt_mass_cumsum"]

    extra_line: str = "Data computed using uncorrected family tree"
    if corrected:
        extra_line = "Data computed using corrected family tree"

    header_lines = [
        "==================================================================",
        f"The floculation balance (sink and source terms due to aggregation and fragmentation)",
        extra_line,
        "==================================================================",
        "",
        "Column descriptions:",
        f"- n_p: Number of particles in floc",
        f"- T_coag: sink + source terms due to coagulation at floc size n_p",
        f"- T_frag: sink + source terms due to fragmenation at floc size n_p",
        f"- dn_dt: sum of T_coag and T_frac",
        f"- T_coag_mass: mass weighted equivalent (i.e. T_coag * n_p)",
        f"- T_frag_mass: mass weighted equivalent (i.e. T_frag * n_p)",
        f"- dn_dt_mass: sum of T_coag_mass and T_frac_mass",
        f"- T_coag_cumsum: cumulative sum from n_p=1 up to n_p of sink + source terms due to coagulation",
        f"- T_frag_cumsum: cumulative sum from n_p=1 up to n_p of sink + source terms due to fragmentation",
        f"- dn_dt_cumsum: cumulative sum of dn_dt",
        f"- T_coag_mass_cumsum: mass weighted equivalent (i.e. T_coag * n_p)",
        f"- T_frag_mass_cumsum: mass weighted equivalent (i.e. T_frag * n_p)",
        f"- dn_dt_mass_cumsum: cumulative sum of dn_dt_mass",
        "",
    ]

    result: dict[str, np.ndarray] = {
        f"n_p": n_p,
        f"T_coag": T_coag,
        f"T_frag": T_frag,
        f"dn_dt": dn_dt,
        f"T_coag_mass": T_coag_mass,
        f"T_frag_mass": T_frag_mass,
        f"dn_dt_mass": dn_dt_mass,
        f"T_coag_cumsum": T_coag_cumsum,
        f"T_frag_cumsum": T_frag_cumsum,
        f"dn_dt_cumsum": dn_dt_cumsum,
        f"T_coag_mass_cumsum": T_coag_mass_cumsum,
        f"T_frag_mass_cumsum": T_frag_mass_cumsum,
        f"dn_dt_mass_cumsum": dn_dt_mass_cumsum,
    }

    name: str = f"floculation_balance.csv"
    if corrected:
        name = f"floculation_balance_corrected.csv"
    if output_dir is not None:
        output_dir.mkdir(exist_ok=True)
        csv_file: Path = output_dir / name
        output.save_to_csv(csv_file, result, header_lines)

    return result
