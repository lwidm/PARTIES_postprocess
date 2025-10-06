import h5py  # type: ignore
import numpy as np
import matplotlib
from matplotlib.axes import Axes
import glob
import os
import gc
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Optional, List, Tuple, Dict, Literal, Any

from numpy._core.numeric import ndarray

# =============================================================================
# CONFIGURATION AND CONSTANTS
# =============================================================================

ON_ANVIL: bool = os.getenv("MY_MACHINE", "") == "anvil"

if not ON_ANVIL:
    matplotlib.use("Qt5Agg")
from matplotlib import pyplot as plt  # noqa: E402

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern"],
        "text.latex.preamble": r"\usepackage{amsmath}",
    }
)

BLAS_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

Re: float = 2800.0
output_dir: str = "output"
DATA_DIRECTORY: str = "data"

NUM_WORKERS: Optional[int] = 4
MIN_FILE_INDEX: Optional[int] = None

if ON_ANVIL:
    DATA_DIRECTORY = "."
    NUM_WORKERS = 8
    MIN_FILE_INDEX = 180

# =============================================================================
# FILE I/O AND DATA LOADING UTILITIES
# =============================================================================


def load_data_with_comments(
    filename: str, comment_chars: str = "#%"
) -> List[np.ndarray]:
    """
    Load data from text file, skipping lines that start with comment characters.

    Args:
        filename: Path to the data file
        comment_chars: Characters that indicate comment lines

    Returns:
        List of numpy arrays for each column in the data file
    """
    with open(filename, "r") as file:
        lines = file.readlines()

    data_lines = [
        line.strip()
        for line in lines
        if line.strip()
        and not any(line.strip().startswith(char) for char in comment_chars)
    ]

    data_array = np.genfromtxt(data_lines, dtype=np.float64)
    return [data_array[:, i] for i in range(data_array.shape[1])]


def find_data_files(
    base_name: str, min_index: Optional[int] = None, max_index: Optional[int] = None
) -> List[str]:
    """
    Find HDF5 data files matching the pattern and filter by index range.

    Args:
        base_name: Base pattern for file names (e.g., "Data" for Data_*.h5)
        min_index: Minimum file index to include (inclusive)
        max_index: Maximum file index to include (inclusive)

    Returns:
        Sorted list of file paths matching the criteria
    """
    file_pattern = f"./{DATA_DIRECTORY}/{base_name}_*.h5"
    all_files = sorted(glob.glob(file_pattern))

    if min_index is None and max_index is None:
        return all_files

    filtered_files = []
    for file_path in all_files:
        try:
            # Extract index from filename (assumes pattern: base_index.h5)
            file_index = int(file_path.split("_")[-1].split(".")[0])
            if (min_index is None or file_index >= min_index) and (
                max_index is None or file_index <= max_index
            ):
                filtered_files.append(file_path)
        except (ValueError, IndexError):
            continue

    return filtered_files


def load_yc_coordinates(file_path: str) -> np.ndarray:
    """
    Load and process y-coordinates from HDF5 file.

    Args:
        file_path: Path to HDF5 file containing grid data

    Returns:
        Processed y-coordinate array (half-channel due to symmetry)
    """
    with h5py.File(file_path, "r") as h5_file:
        y: np.ndarray = h5_file["grid"]["yc"][:-1]  # type: ignore

    return y[: y.shape[0] // 2]


def save_processed_results(
    results: Tuple[np.ndarray, np.ndarray, np.ndarray, float, float],
    output_path: str,
    metadata: Optional[Dict] = None,
) -> None:
    """
    Save processing results to HDF5 file with optional metadata.

    Args:
        results: Tuple containing (y_plus, U_plus, upup, u_tau, tau_w)
        output_path: Path where results should be saved
        metadata: Optional dictionary of metadata to store as attributes
    """
    y_plus, U_plus, upup, u_tau, tau_w = results

    # Create output directory if it doesn't exist
    output_dir: str = (
        os.path.dirname(output_path) if os.path.dirname(output_path) else "."
    )
    os.makedirs(output_dir, exist_ok=True)

    with h5py.File(output_path, "w") as h5_file:
        h5_file.create_dataset("y_plus", data=y_plus)
        h5_file.create_dataset("U_plus", data=U_plus)
        h5_file.create_dataset("upup", data=upup)
        h5_file.create_dataset("u_tau", data=u_tau)
        h5_file.create_dataset("tau_w", data=tau_w)

        # Save metadata as attributes
        if metadata:
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    h5_file.attrs[key] = value


def load_saved_results(
    file_path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Load previously saved processing results from HDF5 file.

    Args:
        file_path: Path to saved results file

    Returns:
        Tuple containing (y_plus, U_plus, upup, u_tau, tau_w)

    Raises:
        FileNotFoundError: If the results file doesn't exist
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Saved data file not found: {file_path}")

    y_plus: np.ndarray
    U_plus: np.ndarray
    upup: np.ndarray
    u_tau: float
    tau_w: float
    with h5py.File(file_path, "r") as h5_file:
        y_plus = h5_file["y_plus"][:]  # type: ignore
        U_plus = h5_file["U_plus"][:]  # type: ignore
        upup = h5_file["upup"][:]  # type: ignore
        u_tau = h5_file["u_tau"][()]  # type: ignore
        tau_w = h5_file["tau_w"][()]  # type: ignore

    return y_plus, U_plus, upup, u_tau, tau_w


# =============================================================================
# CORE DATA PROCESSING FUNCTIONS
# =============================================================================


def compute_flow_statistics_single_file(
    file_path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute flow statistics for a single HDF5 data file.

    MEMORY CRITICAL: This function is called in parallel workers and must be
    extremely careful with memory usage. Large arrays are deleted immediately
    after use and garbage collection is forced.

    Args:
        file_path: Path to HDF5 file containing velocity data

    Returns:
        Tuple of (U_mean, UU_mean, upup) statistics arrays
    """
    print(f"Processing {file_path} ...")

    u: np.ndarray
    vfu: np.ndarray
    with h5py.File(file_path, "r") as h5_file:
        u = h5_file["u"][:]  # type: ignore
        vfu = h5_file["vfu"][:]  # type: ignore

    # Remove boundary points and ensure even grid
    u = u[:-1, :-1, :-1]
    vfu = vfu[:-1, :-1, :-1]

    Ny = u.shape[1]
    if Ny % 2 != 0:
        # Remove center point to make grid symmetric
        u = np.delete(u, Ny // 2, axis=1)
        vfu = np.delete(vfu, Ny // 2, axis=1)
        Ny -= 1

    half_grid_idx = Ny // 2

    # Compute statistics using symmetry - process both halves simultaneously
    # This avoids creating temporary arrays for the flipped data
    u_front_half = u[:, :half_grid_idx, :]
    u_back_half = np.flip(u[:, half_grid_idx:, :], axis=1)
    del u
    gc.collect()

    vfu_front_half = vfu[:, :half_grid_idx, :]
    vfu_back_half = np.flip(vfu[:, half_grid_idx:, :], axis=1)
    del vfu
    gc.collect()

    # Compute weighted sums (memory efficient operations)
    sum_u = (u_front_half * vfu_front_half + u_back_half * vfu_back_half).sum(
        axis=(0, 2)
    )

    sum_uu = (
        u_front_half * u_front_half * vfu_front_half
        + u_back_half * u_back_half * vfu_back_half
    ).sum(axis=(0, 2))

    # Explicitly free large arrays immediately
    del u_front_half, u_back_half
    gc.collect()

    sum_vfu = (vfu_front_half + vfu_back_half).sum(axis=(0, 2))

    del vfu_front_half, vfu_back_half
    gc.collect()

    # Avoid division by zero with small epsilon
    epsilon = 1e-10
    sum_vfu = np.where(sum_vfu > epsilon, sum_vfu, epsilon)

    # Compute volume-weighted averages
    U_mean = sum_u / sum_vfu
    UU_mean = sum_uu / sum_vfu

    # Free intermediate arrays
    del sum_u, sum_uu, sum_vfu
    gc.collect()

    upup = UU_mean - U_mean * U_mean

    return U_mean, UU_mean, upup


def finalize_statistical_results(
    accumulated_U_mean: np.ndarray,
    accumulated_upup: np.ndarray,
    yc: np.ndarray,
    file_count: int,
    save_output: bool,
    min_file_index: Optional[int],
    max_file_index: Optional[int],
    processing_function_name: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Finalize results by computing derived quantities and saving if requested.

    Args:
        accumulated_U_mean: Sum of mean velocities from all files
        accumulated_upup: Sum of velocity fluctuations from all files
        yx: y-coordinate (centerd) array
        file_count: Number of files processed
        save_output: Whether to save results to file
        min_file_index: Minimum file index processed
        max_file_index: Maximum file index processed
        processing_function_name: Name of the processing function used

    Returns:
        Tuple of (y_plus, U_plus, upup, u_tau, tau_w)
    """
    # ensemble_U_mean: np.ndarray = np.concatenate(
    #     ([0.0], np.squeeze(accumulated_U_mean) / float(file_count))
    # )
    ensemble_U_mean: np.ndarray = np.squeeze(accumulated_U_mean) / float(file_count)
    del accumulated_U_mean
    gc.collect()
    # ensemble_upup = np.concatenate(
    #     ([0.0], np.squeeze(accumulated_upup) / float(file_count))
    # )
    ensemble_upup: np.ndarray = np.squeeze(accumulated_upup) / float(file_count)
    del accumulated_upup
    gc.collect()

    # du_dy: np.ndarray = (ensemble_U_mean[1:] - ensemble_U_mean[:-1]) / (
    #     y[1:] - y[:-1]
    # )
    du_dy_0: float = (ensemble_U_mean[0] - (-ensemble_U_mean[0])) / (yc[0] * 2)

    tau_w: float = 1.0 / Re * du_dy_0
    u_tau: float = float(np.sqrt(tau_w))

    y_plus: np.ndarray = Re * yc * u_tau
    U_plus: np.ndarray = ensemble_U_mean / u_tau

    results = (
        y_plus,
        U_plus,
        ensemble_upup,
        u_tau,
        tau_w,
    )

    if save_output:
        metadata = {
            "min_index": min_file_index,
            "max_index": max_file_index,
            "num_files_processed": file_count,
            "function": processing_function_name,
        }
        save_processed_results(results, f"{output_dir}/processed_data.h5", metadata)

    return results


# =============================================================================
# PARALLEL PROCESSING IMPLEMENTATIONS
# =============================================================================


def process_data_parallel(
    min_file_index: Optional[int] = None,
    max_file_index: Optional[int] = None,
    should_save_output: bool = True,
    num_workers: Optional[int] = None,
    use_threads: bool = False,
    set_blas_threads_to_one: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Process data files in parallel using ProcessPoolExecutor or ThreadPoolExecutor.

    MEMORY CRITICAL: This function manages multiple worker processes/threads.
    Each worker loads its own copy of data, so total memory usage scales with
    number of workers.

    Args:
        min_file_index: Minimum file index to process
        max_file_index: Maximum file index to process
        should_save_output: Whether to save results to file
        num_workers: Number of parallel workers (default: min(files, cpu_count))
        use_threads: Use threads instead of processes (shared memory)
        set_blas_threads_to_one: Limit BLAS threads to prevent oversubscription

    Returns:
        Tuple of (y_plus, U_plus, upup, u_tau, tau_w) or empty arrays on error
    """
    # Initialize accumulation arrays
    accumulated_U_mean: Optional[np.ndarray] = None
    accumulated_upup: Optional[np.ndarray] = None
    processed_file_count: int = 0

    data_files: List[str] = find_data_files("Data", min_file_index, max_file_index)

    if not data_files:
        return np.array([]), np.array([]), np.array([]), 0.0, 0.0

    yc: np.ndarray = load_yc_coordinates(data_files[0])

    if num_workers is None:
        num_workers = min(len(data_files), (os.cpu_count() or 1))
    else:
        num_workers = min(len(data_files), (os.cpu_count() or 1), num_workers)

    # Limit BLAS threads to prevent CPU oversubscription in parallel workers
    if set_blas_threads_to_one:
        for env_var in BLAS_THREAD_ENV_VARS:
            if os.environ.get(env_var) is None:
                os.environ[env_var] = "1"

    ExecutorClass = ThreadPoolExecutor if use_threads else ProcessPoolExecutor

    # Process files in parallel
    with ExecutorClass(max_workers=num_workers) as executor:
        U_mean: np.ndarray
        # UU_mean: np.ndarray
        upup: np.ndarray
        for U_mean, _, upup in executor.map(
            compute_flow_statistics_single_file, data_files
        ):
            if accumulated_U_mean is None:
                accumulated_U_mean = np.zeros_like(U_mean, dtype=np.float64)
                accumulated_upup = np.zeros_like(upup, dtype=np.float64)

            # Accumulate results
            accumulated_U_mean += U_mean
            accumulated_upup += upup
            processed_file_count += 1

    # Check for successful processing
    if (
        accumulated_U_mean is None
        or accumulated_upup is None
        or yc is None
        or processed_file_count == 0
    ):
        return np.array([]), np.array([]), np.array([]), 0.0, 0.0

    return finalize_statistical_results(
        accumulated_U_mean,
        accumulated_upup,
        yc,
        processed_file_count,
        should_save_output,
        min_file_index,
        max_file_index,
        "process_data_parallel",
    )


def process_data_serial(
    min_file_index: Optional[int] = None,
    max_file_index: Optional[int] = None,
    should_save_output: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Process data files serially (single-threaded).

    MEMORY EFFICIENT: Only one file is processed at a time, minimizing peak memory usage.
    Use this when memory is extremely constrained.

    Args:
        min_file_index: Minimum file index to process
        max_file_index: Maximum file index to process
        should_save_output: Whether to save results to file

    Returns:
        Tuple of (y_plus, U_plus, upup, u_tau, tau_w) or empty arrays on error
    """
    accumulated_U_mean: Optional[np.ndarray] = None
    accumulated_upup: Optional[np.ndarray] = None
    processed_file_count: int = 0

    data_files: List[str] = find_data_files("Data", min_file_index, max_file_index)

    if not data_files:
        return np.array([]), np.array([]), np.array([]), 0.0, 0.0

    yc: np.ndarray = load_yc_coordinates(data_files[0])

    # Process files one by one
    for file_path in data_files:
        U_mean: np.ndarray
        # UU_mean: np.ndarray
        upup: np.ndarray
        U_mean, _, upup = compute_flow_statistics_single_file(file_path)

        if accumulated_U_mean is None:
            accumulated_U_mean = np.zeros_like(U_mean, dtype=np.float64)
            accumulated_upup = np.zeros_like(upup, dtype=np.float64)

        accumulated_U_mean += U_mean
        accumulated_upup += upup
        processed_file_count += 1

    if (
        accumulated_U_mean is None
        or accumulated_upup is None
        or yc is None
        or processed_file_count == 0
    ):
        return np.array([]), np.array([]), np.array([]), 0.0, 0.0

    return finalize_statistical_results(
        accumulated_U_mean,
        accumulated_upup,
        yc,
        processed_file_count,
        should_save_output,
        min_file_index,
        max_file_index,
        "process_data_serial",
    )


# =============================================================================
# THEORETICAL FUNCTIONS (LAW OF THE WALL)
# =============================================================================


def viscous_sublayer_velocity(y_plus: np.ndarray) -> np.ndarray:
    """
    Compute velocity in viscous sublayer (y+ < 5).

    Args:
        y_plus: y+ coordinates

    Returns:
        u+ velocity in viscous sublayer (u+ = y+)
    """
    return y_plus


def log_law_velocity(
    y_plus: np.ndarray,
    von_karman_constant: float = 0.41,
    log_law_constant: float = 5.0,
) -> np.ndarray:
    """
    Compute velocity according to log law (y+ > 30).

    Args:
        y_plus: y+ coordinates
        von_karman_constant: κ (typically 0.41)
        log_law_constant: C+ (typically 5.0)

    Returns:
        u+ velocity according to log law (u+ = (1/κ) ln(y+) + C+)
    """
    return 1.0 / von_karman_constant * np.log(y_plus) + log_law_constant


def law_of_the_wall(
    y_plus: np.ndarray,
    von_karman_constant: float = 0.41,
    log_law_constant: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute law of the wall velocities for viscous sublayer and log law regions.

    Args:
        y_plus: y+ coordinates
        von_karman_constant: κ constant for log law
        log_law_constant: C+ constant for log law

    Returns:
        Tuple of (y+_viscous, u+_viscous, y+_log, u+_log)
    """
    viscous_buffer: float = 10.0
    log_buffer: float = 24.0

    viscous_region_mask: np.ndarray = y_plus < (5.0 + viscous_buffer)
    log_region_mask: np.ndarray = y_plus > (30.0 - log_buffer)

    viscous_y_plus: np.ndarray = y_plus[viscous_region_mask]
    log_y_plus: np.ndarray = y_plus[log_region_mask]

    viscous_u_plus: np.ndarray = viscous_sublayer_velocity(viscous_y_plus)
    log_u_plus: np.ndarray = log_law_velocity(
        log_y_plus, von_karman_constant, log_law_constant
    )

    return viscous_y_plus, viscous_u_plus, log_y_plus, log_u_plus


def fit_law_of_the_wall_parameters(
    experimental_plus_y: np.ndarray, experimental_plus_velocity: np.ndarray
) -> Tuple[float, float]:
    """
    Fit von Karman constant and log law constant to experimental data.

    Args:
        experimental_plus_y: Experimental y+ values
        experimental_plus_velocity: Experimental u+ values

    Returns:
        Tuple of (fitted_κ, fitted_C+)
    """
    from scipy.optimize import curve_fit  # type: ignore

    # Only use log law region (y+ > 30) for fitting
    log_region_mask: np.ndarray = experimental_plus_y > 30
    log_region_y_plus: np.ndarray = experimental_plus_y[log_region_mask]
    log_region_velocity: np.ndarray = experimental_plus_velocity[log_region_mask]

    initial_guess: Tuple[float, float] = (0.41, 5.0)
    fitted_parameters: Tuple[float, float]
    fitted_kappa: float
    fitted_constant: float

    try:
        fitted_parameters, _ = curve_fit(
            log_law_velocity, log_region_y_plus, log_region_velocity, p0=initial_guess
        )
        fitted_kappa, fitted_constant = fitted_parameters
        return fitted_kappa, fitted_constant
    except Exception as error:
        print(f"Fitting of log law parameters failed: {error}")
        return 0.41, 5.0  # Return default values on failure


# =============================================================================
# DATA COORDINATION AND PLOTTING FUNCTIONS
# =============================================================================


def get_processed_data(
    processing_method: Literal["serial", "parallel", "saved"],
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    float,
]:
    """
    Main data retrieval function that coordinates data processing and theory.

    Args:
        processing_method: One of "serial", "parallel", or "saved"

    Returns:
        Comprehensive tuple containing all processed data for plotting
    """
    # Load reference data from UT Austin
    utexas_data_file = "./data/LM_Channel_0180_mean_prof.dat"
    (
        utexas_y_delta,
        utexas_y_plus,
        utexas_u_plus,
        utexas_velocity_gradient,
        utexas_w,
        utexas_p,
    ) = load_data_with_comments(utexas_data_file)

    parties_y_plus: np.ndarray
    parties_u_plus: np.ndarray
    parties_upup: np.ndarray
    u_tau: float
    tau_w: float
    if processing_method == "serial":
        (
            parties_y_plus,
            parties_u_plus,
            parties_upup,
            u_tau,
            tau_w,
        ) = process_data_serial(min_file_index=MIN_FILE_INDEX)
    elif processing_method == "parallel":
        (
            parties_y_plus,
            parties_u_plus,
            parties_upup,
            u_tau,
            tau_w,
        ) = process_data_parallel(
            num_workers=NUM_WORKERS, use_threads=False, min_file_index=MIN_FILE_INDEX
        )
    elif processing_method == "saved":
        (
            parties_y_plus,
            parties_u_plus,
            parties_upup,
            u_tau,
            tau_w,
        ) = load_saved_results(f"{output_dir}/processed_data.h5")
    else:
        raise ValueError(
            f'processing_method must be one of ["serial", "parallel", "saved"]. Got: {processing_method}'
        )

    Re_tau: float = Re * u_tau
    print(f"u_tau: {u_tau}, tau_w: {tau_w}, Re_tau: {Re_tau}")

    # Fit law of the wall parameters to both datasets
    utexas_kappa, utexas_constant = fit_law_of_the_wall_parameters(
        utexas_y_plus, utexas_u_plus
    )
    parties_kappa, parties_constant = fit_law_of_the_wall_parameters(
        parties_y_plus, parties_u_plus
    )

    # Compute law of the wall for both datasets
    (
        utexas_viscous_y_plus,
        utexas_viscous_u_plus,
        utexas_log_y_plus,
        utexas_log_u_plus,
    ) = law_of_the_wall(utexas_y_plus, utexas_kappa, utexas_constant)

    (
        parties_viscous_y_plus,
        parties_viscous_u_plus,
        parties_log_y_plus,
        parties_log_u_plus,
    ) = law_of_the_wall(parties_y_plus, parties_kappa, parties_constant)

    print(
        f"Law of the wall parameters (utexas):  κ={utexas_kappa:.3f}, C+={utexas_constant:.3f}\n"
        f"Law of the wall parameters (PARTIES): κ={parties_kappa:.3f}, C+={parties_constant:.3f}"
    )

    return (
        utexas_y_plus,
        utexas_u_plus,
        utexas_viscous_y_plus,
        utexas_viscous_u_plus,
        utexas_log_y_plus,
        utexas_log_u_plus,
        parties_y_plus,
        parties_u_plus,
        parties_viscous_y_plus,
        parties_viscous_u_plus,
        parties_log_y_plus,
        parties_log_u_plus,
        Re,
        Re_tau,
    )


def format_plot_axes(axes: Axes) -> Axes:
    """
    Apply consistent formatting to plot axes.

    Args:
        axes: Matplotlib axes object to format

    Returns:
        Formatted axes object
    """
    axes.set_xlabel(r"$y^+$", fontsize=14)
    axes.set_ylabel(r"$u^+$", fontsize=14)

    # Clean up axes appearance
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    axes.spines["left"].set_linewidth(1.2)
    axes.spines["bottom"].set_linewidth(1.0)

    axes.tick_params(axis="both", which="both", direction="out", labelsize=12)
    axes.legend(frameon=False, fontsize=12)

    plt.tight_layout()
    return axes


def create_velocity_profile_plot(
    utexas_y_plus: np.ndarray,
    utexas_u_plus: np.ndarray,
    utexas_viscous_y_plus: np.ndarray,
    utexas_viscous_u_plus: np.ndarray,
    utexas_log_y_plus: np.ndarray,
    utexas_log_u_plus: np.ndarray,
    parties_y_plus: np.ndarray,
    parties_u_plus: np.ndarray,
    parties_viscous_y_plus: np.ndarray,
    parties_viscous_u_plus: np.ndarray,
    parties_log_y_plus: np.ndarray,
    parties_log_u_plus: np.ndarray,
    Re: float,
    Re_tau: float,
) -> None:
    """
    Create comprehensive velocity profile plot comparing data and theory.

    Args:
        All the data arrays returned by get_processed_data()
        reynolds_number: Bulk Reynolds number
        friction_reynolds_number: Friction Reynolds number
    """
    figure, axes = plt.subplots(figsize=(6.5, 5.5))

    # Plot experimental and computational data
    axes.semilogx(utexas_y_plus, utexas_u_plus, "-k", label="UT Austin data")
    axes.semilogx(parties_y_plus, parties_u_plus, "-.k", label="PARTIES data")

    # Plot law of the wall approximations
    axes.semilogx(
        utexas_viscous_y_plus,
        utexas_viscous_u_plus,
        "--k",
        linewidth=0.9,
        label="Law of the wall (UT Austin)",
    )
    axes.semilogx(utexas_log_y_plus, utexas_log_u_plus, "--k", linewidth=0.8)

    axes.semilogx(
        parties_log_y_plus,
        parties_log_u_plus,
        ":k",
        linewidth=0.9,
        label="Law of the wall (PARTIES)",
    )

    # Add vertical lines marking region boundaries
    viscous_sublayer_boundary = 5.0
    buffer_layer_boundary = 30.0

    for boundary_position in (viscous_sublayer_boundary, buffer_layer_boundary):
        axes.axvline(
            x=boundary_position,
            color="0.25",
            linewidth=0.8,
            linestyle=":",
            alpha=0.7,
            zorder=0,
        )

    # Add region labels
    x_max = axes.get_xlim()[1]
    y_max = axes.get_ylim()[1]
    label_y_position = 0.99 * y_max

    # Calculate label positions (geometric means of regions)
    viscous_center = np.sqrt(1.0 * viscous_sublayer_boundary)
    buffer_center = np.sqrt(viscous_sublayer_boundary * buffer_layer_boundary)
    log_center = np.sqrt(buffer_layer_boundary * x_max)

    label_style: Dict[str, Any] = {
        "ha": "center",
        "va": "top",
        "fontsize": 12,
        "bbox": {"facecolor": "white", "edgecolor": "none", "alpha": 0.0},
    }

    axes.text(
        viscous_center, label_y_position, "Viscous sublayer\n$y^+<5$", **label_style
    )
    axes.text(
        buffer_center, label_y_position, "Buffer layer\n$5<y^+<30$", **label_style
    )
    axes.text(log_center, label_y_position, "Log-law region\n$30<y^+$", **label_style)

    # Finalize plot appearance
    axes.set_xlim(1.0, max(np.max(utexas_y_plus), np.max(parties_y_plus)))
    axes.set_ylim(0.0, 0.98*max(np.max(utexas_u_plus), np.max(parties_u_plus)))
    axes = format_plot_axes(axes)
    axes.legend(loc="lower right", bbox_to_anchor=(1.0, 0.20))

    plot_filename = f"{output_dir}/Re={Re:.0f}_Re_tau={Re_tau:.0f}-y+_u+.png"
    plt.savefig(plot_filename, dpi=300)

    if not ON_ANVIL:
        plt.show()

    plt.close(figure)


# =============================================================================
# MAIN EXECUTION
# =============================================================================


def main() -> None:
    (
        utexas_y_plus,
        utexas_u_plus,
        utexas_viscous_y,
        utexas_viscous_u,
        utexas_log_y,
        utexas_log_u,
        parties_y_plus,
        parties_u_plus,
        parties_viscous_y,
        parties_viscous_u,
        parties_log_y,
        parties_log_u,
        reynolds_number,
        friction_reynolds_number,
    ) = get_processed_data("saved")

    create_velocity_profile_plot(
        utexas_y_plus,
        utexas_u_plus,
        utexas_viscous_y,
        utexas_viscous_u,
        utexas_log_y,
        utexas_log_u,
        parties_y_plus,
        parties_u_plus,
        parties_viscous_y,
        parties_viscous_u,
        parties_log_y,
        parties_log_u,
        reynolds_number,
        friction_reynolds_number,
    )


if __name__ == "__main__":
    main()
