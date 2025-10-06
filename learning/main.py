import h5py  # type: ignore
import numpy as np
import matplotlib
from matplotlib.axes import Axes
import glob
import os
import gc
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Optional, List, Tuple, Dict, Literal, Any

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
data_dir: str = "data"

num_workers_single_component: Optional[int] = 5
num_workers_cross_component: Optional[int] = 2
min_file_index: Optional[int] = None

if ON_ANVIL:
    data_dir = "."
    num_workers_single_component = 8
    num_workers_cross_component = 4
    min_file_index = 180

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
    file_pattern = f"./{data_dir}/{base_name}_*.h5"
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


def load_y_coordinates(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and process y-coordinates from HDF5 file.

    Args:
        file_path: Path to HDF5 file containing grid data

    Returns:
        Processed y-coordinate (yc, yv) array (half-channel due to symmetry)
    """
    with h5py.File(file_path, "r") as h5_file:
        yc: np.ndarray = h5_file["grid"]["yc"][:-1]  # type: ignore
        yv: np.ndarray = h5_file["grid"]["yv"][:]  # type: ignore

    results = (yc[: yc.shape[0] // 2], yv[: yv.shape[0] // 2])
    return results


def save_intermediate_results(
    results: Dict[str, np.ndarray],
    component: str,
    output_path: str,
    metadata: Optional[Dict] = None,
) -> None:
    """
    Save intermediate processing results to HDF5 file.

    Args:
        results: Dictionary containing intermediate results
        component: Velocity component name (e.g., "u", "v", "w")
        output_path: Path where results should be saved
        metadata: Optional dictionary of metadata to store as attributes
    """
    # Create output directory if it doesn't exist
    output_dir_path: str = (
        os.path.dirname(output_path) if os.path.dirname(output_path) else "."
    )
    os.makedirs(output_dir_path, exist_ok=True)

    with h5py.File(output_path, "w") as h5_file:
        # Save all results with component prefix
        for key, value in results.items():
            dataset_name: str = f"{component}_{key}"
            h5_file.create_dataset(dataset_name, data=value)

        # Save metadata as attributes
        if metadata:
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    h5_file.attrs[key] = value


def load_intermediate_results(file_path: str, component: str) -> Dict[str, np.ndarray]:
    """
    Load intermediate processing results from HDF5 file.

    Args:
        file_path: Path to saved results file
        component: Velocity component name to load

    Returns:
        Dictionary containing intermediate results for the component

    Raises:
        FileNotFoundError: If the results file doesn't exist
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Intermediate data file not found: {file_path}")

    results: Dict = {}
    with h5py.File(file_path, "r") as h5_file:
        prefix = f"{component}_"
        for key in h5_file.keys():
            if key.startswith(prefix):
                # Remove component prefix from key
                clean_key = key[len(prefix) :]
                results[clean_key] = h5_file[key][:]  # type: ignore

    return results


def save_final_results(
    results: Dict[str, Any],
    output_path: str,
    metadata: Optional[Dict] = None,
) -> None:
    """
    Save final processing results to HDF5 file.

    Args:
        results: Dictionary containing all final results
        output_path: Path where results should be saved
        metadata: Optional dictionary of metadata to store as attributes
    """
    output_dir_path: str = (
        os.path.dirname(output_path) if os.path.dirname(output_path) else "."
    )
    os.makedirs(output_dir_path, exist_ok=True)

    with h5py.File(output_path, "w") as h5_file:
        # Save all results
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                h5_file.create_dataset(key, data=value)
            elif isinstance(value, (int, float)):
                h5_file.create_dataset(key, data=value)

        # Save metadata as attributes
        if metadata:
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    h5_file.attrs[key] = value


def load_final_results(file_path: str) -> Dict[str, Any]:
    """
    Load final processing results from HDF5 file.

    Args:
        file_path: Path to saved results file

    Returns:
        Dictionary containing all final results

    Raises:
        FileNotFoundError: If the results file doesn't exist
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Final data file not found: {file_path}")

    results = {}
    with h5py.File(file_path, "r") as h5_file:
        for key in h5_file.keys():
            results[key] = h5_file[key][:] if h5_file[key].shape else h5_file[key][()]  # type: ignore

    return results


# =============================================================================
# CORE DATA PROCESSING FUNCTIONS
# =============================================================================


def compute_single_component_statistics(
    file_path: str, component: Literal["u", "v", "w"]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute statistics for a single velocity component from HDF5 file.

    MEMORY CRITICAL: This function processes one component at a time to minimize memory usage.

    Args:
        file_path: Path to HDF5 file containing velocity data
        component: Velocity component to process ("u", "v", or "w")

    Returns:
        Tuple of (component_mean, component_squared_mean)
    """
    print(f"Processing {component} from {file_path} ...")

    # Load only the required component and volume fraction
    velocity_data: np.ndarray
    volume_fraction_data: np.ndarray
    with h5py.File(file_path, "r") as h5_file:
        velocity_data = h5_file[component][:]  # type: ignore
        volume_fraction_data = h5_file["vf" + component][:]  # type: ignore

    # Remove boundary points and ensure even grid
    if component != "v":
        velocity_data = velocity_data[:-1, :-1, :-1]
        volume_fraction_data = volume_fraction_data[:-1, :-1, :-1]
    else:
        velocity_data = velocity_data[:-1, :, :-1]
        volume_fraction_data = volume_fraction_data[:-1, :, :-1]

    Ny = velocity_data.shape[1]
    if Ny % 2 != 0:
        # Remove center point to make grid symmetric
        velocity_data = np.delete(velocity_data, Ny // 2, axis=1)
        volume_fraction_data = np.delete(volume_fraction_data, Ny // 2, axis=1)
        Ny -= 1

    half_grid_idx = Ny // 2

    # Compute statistics using symmetry
    velocity_front = velocity_data[:, :half_grid_idx, :]
    velocity_back = np.flip(velocity_data[:, half_grid_idx:, :], axis=1)

    volume_front = volume_fraction_data[:, :half_grid_idx, :]
    volume_back = np.flip(volume_fraction_data[:, half_grid_idx:, :], axis=1)

    # Free memory immediately
    del velocity_data, volume_fraction_data
    gc.collect()

    # Compute weighted sums
    sum_velocity = (velocity_front * volume_front + velocity_back * volume_back).sum(
        axis=(0, 2)
    )
    sum_velocity_squared = (
        velocity_front * velocity_front * volume_front
        + velocity_back * velocity_back * volume_back
    ).sum(axis=(0, 2))
    sum_volume_fraction = (volume_front + volume_back).sum(axis=(0, 2))

    # Free more memory
    del velocity_front, velocity_back, volume_front, volume_back
    gc.collect()

    # Avoid division by zero
    epsilon = 1e-10
    sum_volume_fraction = np.where(
        sum_volume_fraction > epsilon, sum_volume_fraction, epsilon
    )

    # Compute volume-weighted averages
    velocity_mean = sum_velocity / sum_volume_fraction
    velocity_squared_mean = sum_velocity_squared / sum_volume_fraction

    # Free intermediate arrays
    del sum_velocity, sum_velocity_squared, sum_volume_fraction
    gc.collect()

    return velocity_mean, velocity_squared_mean


def compute_cross_component_statistics(
    file_path: str,
    component1: Literal["u", "v", "w"],
    component2: Literal["u", "v", "w"],
) -> np.ndarray:
    """
    Compute cross-component statistics from HDF5 file.

    Args:
        file_path: Path to HDF5 file containing velocity data
        component1: First velocity component
        component2: Second velocity component

    Returns:
        Cross-component mean (e.g., UV_mean for components "u" and "v")
    """
    print(f"Processing {component1}{component2} from {file_path} ...")

    # Load both components and volume fraction
    velocity1: np.ndarray
    velocity2: np.ndarray
    volume_fraction: np.ndarray
    volume_fraction2: np.ndarray
    with h5py.File(file_path, "r") as h5_file:

        # interpolate components to center
        def u_v_w_interp(
            component: Literal["u", "v", "w"],
        ) -> Tuple[np.ndarray, np.ndarray]:

            velocity: np.ndarray = h5_file[component][:]  # type: ignore
            volume_fraction: np.ndarray = h5_file["vf" + component1][:]  # type: ignore
            match component:
                case "u":
                    velocity = (velocity[:-1, :-1, 1:] + velocity[:-1, :-1, :-1]) / 2
                    volume_fraction = (
                        volume_fraction[:-1, :-1, 1:] + volume_fraction[:-1, :-1, :-1]
                    ) / 2
                case "v":
                    velocity = (velocity[:-1, 1:, :-1] + velocity[:-1, :-1, :-1]) / 2
                    volume_fraction = (
                        volume_fraction[:-1, 1:, :-1] + volume_fraction[:-1, :-1, :-1]
                    ) / 2
                case "w":
                    velocity = (velocity[1:, :-1, :-1] + velocity[:-1, :-1, :-1]) / 2
                    volume_fraction = (
                        volume_fraction[1:, :-1, :-1] + volume_fraction[:-1, :-1, :-1]
                    ) / 2
            return velocity, volume_fraction

        velocity1, volume_fraction = u_v_w_interp(component1)

        velocity2, volume_fraction2 = u_v_w_interp(component2)

    volume_fraction[volume_fraction2 != volume_fraction] = 0.0

    Ny = velocity1.shape[1]
    if Ny % 2 != 0:
        velocity1 = np.delete(velocity1, Ny // 2, axis=1)
        velocity2 = np.delete(velocity2, Ny // 2, axis=1)
        volume_fraction = np.delete(volume_fraction, Ny // 2, axis=1)
        Ny -= 1

    half_grid_idx = Ny // 2

    # Compute cross-component statistics using symmetry
    vel1_front = velocity1[:, :half_grid_idx, :]
    vel1_back = np.flip(velocity1[:, half_grid_idx:, :], axis=1)

    vel2_front = velocity2[:, :half_grid_idx, :]
    vel2_back = np.flip(velocity2[:, half_grid_idx:, :], axis=1)

    volume_front = volume_fraction[:, :half_grid_idx, :]
    volume_back = np.flip(volume_fraction[:, half_grid_idx:, :], axis=1)

    # Free memory
    del velocity1, velocity2, volume_fraction
    gc.collect()

    # Compute weighted cross sum
    sum_cross = (
        vel1_front * vel2_front * volume_front + vel1_back * vel2_back * volume_back
    ).sum(axis=(0, 2))
    sum_volume = (volume_front + volume_back).sum(axis=(0, 2))

    # Free more memory
    del vel1_front, vel1_back, vel2_front, vel2_back, volume_front, volume_back
    gc.collect()

    # Avoid division by zero
    epsilon = 1e-10
    sum_volume = np.where(sum_volume > epsilon, sum_volume, epsilon)

    # Compute volume-weighted cross mean
    cross_mean = sum_cross / sum_volume

    # Free intermediate arrays
    del sum_cross, sum_volume
    gc.collect()

    return cross_mean


def process_single_component(
    component: Literal["u", "v", "w"],
    min_file_index: Optional[int] = None,
    max_file_index: Optional[int] = None,
    num_workers: Optional[int] = None,
    use_threads: bool = False,
    set_blas_threads_to_one: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process a single velocity component across all files and save intermediate results.

    Args:
        component: Velocity component to process
        min_file_index: Minimum file index to process
        max_file_index: Maximum file index to process
        num_workers: Number of parallel workers
        use_threads: Use threads instead of processes
        set_blas_threads_to_one: Limit BLAS threads

    Returns:
        Dictionary containing component_mean and component_squared_mean
    """
    # Initialize accumulation arrays
    accumulated_mean_velocity: Optional[np.ndarray] = None
    accumulated_mean_velocity_squared: Optional[np.ndarray] = None
    processed_file_count: int = 0

    data_files: List[str] = find_data_files("Data", min_file_index, max_file_index)

    if not data_files:
        return (np.array([]), np.array([]))

    # Determine number of workers
    if num_workers is None:
        num_workers = min(len(data_files), (os.cpu_count() or 1))
    else:
        num_workers = min(len(data_files), (os.cpu_count() or 1), num_workers)

    # Limit BLAS threads if requested
    if set_blas_threads_to_one:
        for env_var in BLAS_THREAD_ENV_VARS:
            if os.environ.get(env_var) is None:
                os.environ[env_var] = "1"

    ExecutorClass = ThreadPoolExecutor if use_threads else ProcessPoolExecutor

    # Process files in parallel
    with ExecutorClass(max_workers=num_workers) as executor:
        # Create partial function for this component
        from functools import partial

        process_func = partial(compute_single_component_statistics, component=component)

        mean_velocity: np.ndarray
        mean_velocity_squared: np.ndarray
        for mean_velocity, mean_velocity_squared in executor.map(
            process_func, data_files
        ):
            if accumulated_mean_velocity is None:
                accumulated_mean_velocity = np.zeros_like(
                    mean_velocity, dtype=np.float64
                )
                accumulated_mean_velocity_squared = np.zeros_like(
                    mean_velocity_squared, dtype=np.float64
                )

            accumulated_mean_velocity += mean_velocity
            accumulated_mean_velocity_squared += mean_velocity_squared
            processed_file_count += 1

    if accumulated_mean_velocity is None or accumulated_mean_velocity_squared is None:
        return (np.array([]), np.array([]))

    ensemble_mean_velocity = accumulated_mean_velocity / float(processed_file_count)
    ensemble_mean_velocity_squared = accumulated_mean_velocity_squared / float(
        processed_file_count
    )

    del accumulated_mean_velocity, accumulated_mean_velocity_squared
    gc.collect()

    return (ensemble_mean_velocity, ensemble_mean_velocity_squared)


def process_cross_components(
    component1: Literal["u", "v", "w"],
    component2: Literal["u", "v", "w"],
    min_file_index: Optional[int] = None,
    max_file_index: Optional[int] = None,
    num_workers: Optional[int] = None,
    use_threads: bool = False,
    set_blas_threads_to_one: bool = True,
) -> np.ndarray:
    """
    Process cross-component statistics across all files.

    Args:
        component1: First velocity component
        component2: Second velocity component
        min_file_index: Minimum file index to process
        max_file_index: Maximum file index to process
        num_workers: Number of parallel workers
        use_threads: Use threads instead of processes
        set_blas_threads_to_one: Limit BLAS threads

    Returns:
        Cross-component mean array
    """
    accumulated_cross_mean_velocity: Optional[np.ndarray] = None
    processed_file_count: int = 0

    data_files: List[str] = find_data_files("Data", min_file_index, max_file_index)

    if not data_files:
        return np.array([])

    # Determine number of workers
    if num_workers is None:
        num_workers = min(len(data_files), (os.cpu_count() or 1))
    else:
        num_workers = min(len(data_files), (os.cpu_count() or 1), num_workers)

    # Limit BLAS threads if requested
    if set_blas_threads_to_one:
        for env_var in BLAS_THREAD_ENV_VARS:
            if os.environ.get(env_var) is None:
                os.environ[env_var] = "1"

    ExecutorClass = ThreadPoolExecutor if use_threads else ProcessPoolExecutor

    # Process files in parallel
    with ExecutorClass(max_workers=num_workers) as executor:
        # Create partial function for these components
        from functools import partial

        process_func = partial(
            compute_cross_component_statistics,
            component1=component1,
            component2=component2,
        )

        cross_mean_velocity: np.ndarray
        for cross_mean_velocity in executor.map(process_func, data_files):
            if accumulated_cross_mean_velocity is None:
                accumulated_cross_mean_velocity = np.zeros_like(
                    cross_mean_velocity, dtype=np.float64
                )

            accumulated_cross_mean_velocity += cross_mean_velocity
            processed_file_count += 1

    if accumulated_cross_mean_velocity is None:
        return np.array([])

    ensemble_cross_mean_velocity = accumulated_cross_mean_velocity / float(
        processed_file_count
    )

    del accumulated_cross_mean_velocity
    gc.collect()

    return ensemble_cross_mean_velocity


# =============================================================================
# REYNOLDS STRESS COMPUTATION
# =============================================================================


def compute_all_reynolds_stresses_step_by_step(
    min_file_index: Optional[int] = None,
    max_file_index: Optional[int] = None,
    num_workers_single_component: Optional[int] = None,
    num_workers_cross_component: Optional[int] = None,
    use_threads: bool = False,
    save_intermediates: bool = True,
) -> Dict[str, Any]:
    """
    Compute all Reynolds stresses in a memory-efficient step-by-step pipeline.

    This function processes components one at a time, saves intermediate results,
    and computes Reynolds stresses while minimizing memory usage.

    Args:
        min_file_index: Minimum file index to process
        max_file_index: Maximum file index to process
        num_workers: Number of parallel workers
        use_threads: Use threads instead of processes
        save_intermediates: Whether to save intermediate results

    Returns:
        Dictionary containing all final results including wall units
    """
    print("Starting Reynolds stress computation...")

    data_files = find_data_files("Data", min_file_index, max_file_index)
    if not data_files:
        return {}

    yc: np.ndarray
    yv: np.ndarray
    yc, yv = load_y_coordinates(data_files[0])
    final_results: Dict[str, Any] = {"yc": yc, "yv": yv}

    # STEP 1: Process u-component and compute upup
    print("Step 1: Processing u-component...")
    U_mean: np.ndarray
    UU_mean: np.ndarray
    U_mean, UU_mean = process_single_component(
        "u", min_file_index, max_file_index, num_workers_single_component, use_threads
    )

    upup = UU_mean - U_mean * U_mean
    final_results["U_mean"] = U_mean
    final_results["UU_mean"] = UU_mean
    final_results["upup"] = upup

    # Free u-component memory
    del UU_mean, upup
    gc.collect()
    print("Step 1 complete: upup computed")

    # STEP 2: Process v-component and compute vpvp
    print("Step 2: Processing v-component...")
    V_mean: np.ndarray
    VV_mean: np.ndarray
    V_mean, VV_mean = process_single_component(
        "v", min_file_index, max_file_index, num_workers_single_component, use_threads
    )

    vpvp = VV_mean - V_mean * V_mean
    final_results["V_mean"] = V_mean
    final_results["VV_mean"] = VV_mean
    final_results["vpvp"] = vpvp

    # Free v-component memory
    del VV_mean, vpvp
    gc.collect()
    print("Step 2 complete: vpvp computed")

    # STEP 3: Process uv cross-component and compute upvp
    print("Step 3: Processing uv cross-component...")
    UV_mean = process_cross_components(
        "u",
        "v",
        min_file_index,
        max_file_index,
        num_workers_cross_component,
        use_threads,
    )

    upvp = UV_mean - U_mean * V_mean
    final_results["UV_mean"] = UV_mean
    final_results["upvp"] = upvp

    # Free UV memory and U_mean (no longer needed)
    del UV_mean, U_mean
    gc.collect()
    print("Step 3 complete: upvp computed")

    # STEP 4: Process w-component and compute wpwp
    print("Step 4: Processing w-component...")
    W_mean: np.ndarray
    WW_mean: np.ndarray
    W_mean, WW_mean = process_single_component(
        "w", min_file_index, max_file_index, num_workers_single_component, use_threads
    )

    wpwp = WW_mean - W_mean * W_mean
    final_results["W_mean"] = W_mean
    final_results["WW_mean"] = WW_mean
    final_results["wpwp"] = wpwp

    # Free w-component memory
    del WW_mean, wpwp
    gc.collect()
    print("Step 4 complete: wpwp computed")

    # STEP 5: Process remaining cross-components
    print("Step 5: Processing remaining cross-components...")

    # Process uw
    UW_mean = process_cross_components(
        "u",
        "w",
        min_file_index,
        max_file_index,
        num_workers_cross_component,
        use_threads,
    )
    if UW_mean.size > 0:
        upwp = UW_mean - final_results["U_mean"] * W_mean
        final_results["UW_mean"] = UW_mean
        final_results["upwp"] = upwp
        del UW_mean
        gc.collect()
        print("Step 5a complete: upwp computed")

    # Process vw
    VW_mean = process_cross_components(
        "v",
        "w",
        min_file_index,
        max_file_index,
        num_workers_cross_component,
        use_threads,
    )
    if VW_mean.size > 0:
        vpwp = VW_mean - V_mean * W_mean
        final_results["VW_mean"] = VW_mean
        final_results["vpwp"] = vpwp
        del VW_mean, V_mean, W_mean
        gc.collect()
        print("Step 5b complete: vpwp computed")

    # STEP 6: Compute friction velocity and convert to wall units
    print("Step 6: Computing wall units...")
    du_dy_0: float = (final_results["U_mean"][0] - (-final_results["U_mean"][0])) / (
        yc[0] * 2
    )
    tau_w: float = 1.0 / Re * du_dy_0
    u_tau: float = float(np.sqrt(tau_w))

    yc_plus: np.ndarray = Re * yc * u_tau
    yv_plus: np.ndarray = Re * yv * u_tau
    final_results["u_tau"] = u_tau
    final_results["tau_w"] = tau_w
    final_results["yc_plus"] = yc_plus
    final_results["yv_plus"] = yv_plus
    final_results["Re_tau"] = Re * u_tau
    final_results["k"] = 0.5 * (
        final_results["upup"] + final_results["vpvp"] + final_results["wpwp"]
    )

    # Convert to wall units
    final_results["U_plus"] = final_results["U_mean"] / u_tau
    final_results["upup_plus"] = final_results["upup"] / (u_tau * u_tau)
    final_results["vpvp_plus"] = final_results["vpvp"] / (u_tau * u_tau)
    final_results["wpwp_plus"] = final_results["wpwp"] / (u_tau * u_tau)
    final_results["upvp_plus"] = final_results["upvp"] / (u_tau * u_tau)
    final_results["k_plus"] = final_results["k"] / (u_tau * u_tau)

    if "upwp" in final_results:
        final_results["upwp_plus"] = final_results["upwp"] / (u_tau * u_tau)
    if "vpwp" in final_results:
        final_results["vpwp_plus"] = final_results["vpwp"] / (u_tau * u_tau)

    print("Step 6 complete: All quantities converted to wall units")

    # Save final results
    save_final_results(
        final_results,
        f"{output_dir}/reynolds_stresses.h5",
        {
            "min_index": min_file_index,
            "max_index": max_file_index,
            "num_files_processed": len(data_files),
        },
    )

    print("Reynolds stress computation completed successfully!")
    return final_results


# =============================================================================
# THEORETICAL FUNCTIONS (LAW OF THE WALL)
# =============================================================================


def viscous_sublayer_velocity(y_plus: np.ndarray) -> np.ndarray:
    return y_plus


def log_law_velocity(
    y_plus: np.ndarray,
    von_karman_constant: float = 0.41,
    log_law_constant: float = 5.0,
) -> np.ndarray:
    return 1.0 / von_karman_constant * np.log(y_plus) + log_law_constant


def law_of_the_wall(
    y_plus: np.ndarray,
    von_karman_constant: float = 0.41,
    log_law_constant: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    experimental_yc_plus: np.ndarray, experimental_plus_velocity: np.ndarray
) -> Tuple[float, float]:
    from scipy.optimize import curve_fit  # type: ignore

    log_region_mask: np.ndarray = experimental_yc_plus > 30
    log_region_yc_plus: np.ndarray = experimental_yc_plus[log_region_mask]
    log_region_velocity: np.ndarray = experimental_plus_velocity[log_region_mask]

    initial_guess: Tuple[float, float] = (0.41, 5.0)

    try:
        fitted_parameters, _ = curve_fit(
            log_law_velocity, log_region_yc_plus, log_region_velocity, p0=initial_guess
        )
        fitted_kappa, fitted_constant = fitted_parameters
        return fitted_kappa, fitted_constant
    except Exception as error:
        print(f"Fitting of log law parameters failed: {error}")
        return 0.41, 5.0


# =============================================================================
# DATA COORDINATION AND PLOTTING FUNCTIONS
# =============================================================================


def get_processed_data(
    processing_method: Literal["step_by_step", "saved"],
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
    float,
]:
    """
    Main data retrieval function that coordinates data processing and theory.

    Args:
        processing_method: One of "step_by_step" or "saved"

    Returns:
        Comprehensive tuple containing all processed data for plotting
    """
    # Load reference data from utexas
    utexas_mean_data_file: str = f"./{data_dir}/LM_Channel_0180_mean_prof.dat"
    utexas_fluc_data_file: str = f"./{data_dir}/LM_Channel_0180_vel_fluc_prof.dat"
    (
        utexas_y_delta,
        utexas_y_plus,
        utexas_u_plus,
        utexas_velocity_gradient,
        utexas_w,
        utexas_p,
    ) = load_data_with_comments(utexas_mean_data_file)
    (
        _,
        _,
        utexas_upup_plus,
        utexas_vpvp_plus,
        utexas_wpwp_plus,
        utexas_upvp_plus,
        utexas_upwp_plus,
        utexas_vpwp_plus,
        utexas_k_plus,
    ) = load_data_with_comments(utexas_fluc_data_file)

    parties_results: Dict[str, Any]
    if processing_method == "step_by_step":
        parties_results = compute_all_reynolds_stresses_step_by_step(
            min_file_index=min_file_index,
            num_workers_single_component=num_workers_single_component,
            num_workers_cross_component=num_workers_cross_component,
            use_threads=False,
        )
    elif processing_method == "saved":
        parties_results = load_final_results(f"{output_dir}/reynolds_stresses.h5")
    else:
        raise ValueError(
            f'processing_method must be one of ["step_by_step", "saved"]. Got: {processing_method}'
        )

    # Extract required values from results dictionary
    parties_yc_plus = parties_results["yc_plus"]
    parties_yv_plus = parties_results["yv_plus"]
    parties_u_plus = parties_results["U_plus"]
    parties_upup_plus = parties_results["upup_plus"]
    parties_vpvp_plus = parties_results["vpvp_plus"]
    parties_wpwp_plus = parties_results["wpwp_plus"]
    parties_k_plus = parties_results["k_plus"]
    u_tau = parties_results["u_tau"]
    tau_w = parties_results["tau_w"]
    Re_tau = parties_results["Re_tau"]

    print(f"u_tau: {u_tau}, tau_w: {tau_w}, Re_tau: {Re_tau}")

    # Fit law of the wall parameters to both datasets
    utexas_kappa, utexas_constant = fit_law_of_the_wall_parameters(
        utexas_y_plus, utexas_u_plus
    )
    parties_kappa, parties_constant = fit_law_of_the_wall_parameters(
        parties_yc_plus, parties_u_plus
    )

    # Compute law of the wall for both datasets
    (
        utexas_viscous_y_plus,
        utexas_viscous_u_plus,
        utexas_log_y_plus,
        utexas_log_u_plus,
    ) = law_of_the_wall(utexas_y_plus, utexas_kappa, utexas_constant)

    (
        parties_viscous_yc_plus,
        parties_viscous_u_plus,
        parties_log_yc_plus,
        parties_log_u_plus,
    ) = law_of_the_wall(parties_yc_plus, parties_kappa, parties_constant)

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
        utexas_upup_plus,
        utexas_vpvp_plus,
        utexas_wpwp_plus,
        utexas_k_plus,
        parties_yc_plus,
        parties_yv_plus,
        parties_u_plus,
        parties_viscous_yc_plus,
        parties_viscous_u_plus,
        parties_log_yc_plus,
        parties_log_u_plus,
        parties_upup_plus,
        parties_vpvp_plus,
        parties_wpwp_plus,
        parties_k_plus,
        Re,
        Re_tau,
        u_tau,
    )


def format_plot_axes(axes: Axes) -> Axes:
    axes.set_xlabel(r"$y^+$", fontsize=14)
    axes.set_ylabel(r"$u^+$", fontsize=14)
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
    parties_yc_plus: np.ndarray,
    parties_u_plus: np.ndarray,
    parties_viscous_yc_plus: np.ndarray,
    parties_viscous_u_plus: np.ndarray,
    parties_log_yc_plus: np.ndarray,
    parties_log_u_plus: np.ndarray,
    Re: float,
    Re_tau: float,
) -> None:
    figure, axes = plt.subplots(figsize=(6.5, 5.5))

    axes.semilogx(utexas_y_plus, utexas_u_plus, "-k", label="utexas data")
    axes.semilogx(parties_yc_plus, parties_u_plus, "-.k", label="PARTIES data")
    axes.semilogx(
        utexas_viscous_y_plus,
        utexas_viscous_u_plus,
        "--k",
        linewidth=0.9,
        label="Law of the wall (utexas)",
    )
    axes.semilogx(utexas_log_y_plus, utexas_log_u_plus, "--k", linewidth=0.8)
    axes.semilogx(
        parties_log_yc_plus,
        parties_log_u_plus,
        ":k",
        linewidth=0.9,
        label="Law of the wall (PARTIES)",
    )

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

    x_max = axes.get_xlim()[1]
    y_max = axes.get_ylim()[1]
    label_y_position = 0.99 * y_max

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

    axes.set_xlim(1.0, max(np.max(utexas_y_plus), np.max(parties_yc_plus)))
    axes.set_ylim(0.0, 1.1 * max(np.max(utexas_u_plus), np.max(parties_u_plus)))
    axes = format_plot_axes(axes)
    axes.legend(loc="lower right", bbox_to_anchor=(1.0, 0.20))

    plot_filename = f"{output_dir}/Re={Re:.0f}_Re_tau={Re_tau:.0f}-y+_u+.png"
    plt.savefig(plot_filename, dpi=300)

    if not ON_ANVIL:
        plt.show()

    plt.close(figure)


def create_normal_stress_plot(
    utexas_y_plus: np.ndarray,
    utexas_upup_plus: np.ndarray,
    utexas_vpvp_plus: np.ndarray,
    utexas_wpwp_plus: np.ndarray,
    utexas_k_plus: np.ndarray,
    parties_yc_plus: np.ndarray,
    parties_yv_plus: np.ndarray,
    parties_upup_plus: np.ndarray,
    parties_vpvp_plus: np.ndarray,
    parties_wpwp_plus: np.ndarray,
    parties_k_plus: np.ndarray,
    Re: float,
    Re_tau: float,
    u_tau: float,
) -> None:
    figure, axes = plt.subplots(figsize=(6.5, 5.5))

    axes.plot(
        utexas_y_plus,
        utexas_upup_plus,
        "ok",
        label=r"$\langle u^{\prime}u^{\prime}\rangle / u_{\tau}$ (utexas)",
    )
    axes.plot(
        utexas_y_plus,
        utexas_vpvp_plus,
        "dk",
        label=r"$\langle v^{\prime}v^{\prime}\rangle / u_{\tau}$ (utexas)",
    )
    axes.plot(
        utexas_y_plus,
        utexas_wpwp_plus,
        "^k",
        label=r"$\langle w^{\prime}w^{\prime}\rangle / u_{\tau}$ (utexas)",
    )
    axes.plot(
        utexas_y_plus,
        utexas_k_plus,
        "*k",
        label=r"$\langle k\rangle / u_{\tau}$ (utexas)",
    )

    axes.plot(
        parties_yv_plus,
        parties_upup_plus,
        "-k",
        label=r"$\langle u^{\prime}u^{\prime}\rangle / u_{\tau}$ (PARTIES)",
    )
    axes.plot(
        parties_yv_plus,
        parties_vpvp_plus,
        "-.k",
        label=r"$\langle v^{\prime}v^{\prime}\rangle / u_{\tau}$ (PARTIES)",
    )
    axes.plot(
        parties_yc_plus,
        parties_wpwp_plus,
        "--k",
        label=r"$\langle w^{\prime}w^{\prime}\rangle / u_{\tau}$ (PARTIES)",
    )
    axes.plot(
        parties_yc_plus,
        parties_k_plus,
        ":k",
        label=r"$\langle k\rangle / u_{\tau}$ (PARTIES)",
    )

    axes.set_xlim(
        0.0,
        min(
            max(
                np.max(utexas_y_plus),
                np.max(parties_yc_plus),
                np.max(parties_yv_plus),
            ),
            180,
        ),
    )
    axes.set_ylim(
        0.0,
        1.1
        * max(
            np.max(utexas_upup_plus),
            np.max(utexas_vpvp_plus),
            np.max(utexas_vpvp_plus),
            np.max(utexas_k_plus),
            np.max(parties_upup_plus),
            np.max(parties_vpvp_plus),
            np.max(parties_vpvp_plus),
            np.max(parties_k_plus),
        ),
    )
    axes = format_plot_axes(axes)
    axes.legend(loc="lower right", bbox_to_anchor=(1.0, 0.20))

    plot_filename = f"{output_dir}/Re={Re:.0f}_Re_tau={Re_tau:.0f}-u'u'.png"
    plt.savefig(plot_filename, dpi=300)

    if not ON_ANVIL:
        plt.show()

    plt.close(figure)


# =============================================================================
# MAIN EXECUTION
# =============================================================================


def main() -> None:
    # method: Literal["step_by_step", "saved"] = "step_by_step"
    method: Literal["step_by_step", "saved"] = "saved"
    (
        utexas_y_plus,
        utexas_u_plus,
        utexas_viscous_y_plus,
        utexas_viscous_u_plus,
        utexas_log_y_plus,
        utexas_log_u_plus,
        utexas_upup_plus,
        utexas_vpvp_plus,
        utexas_wpwp_plus,
        utexas_k_plus,
        parties_yc_plus,
        parties_yv_plus,
        parties_u_plus,
        parties_viscous_yc_plus,
        parties_viscous_u_plus,
        parties_log_yc_plus,
        parties_log_u_plus,
        parties_upup_plus,
        parties_vpvp_plus,
        parties_wpwp_plus,
        parties_k_plus,
        Re,
        Re_tau,
        u_tau,
    ) = get_processed_data(method)

    create_velocity_profile_plot(
        utexas_y_plus,
        utexas_u_plus,
        utexas_viscous_y_plus,
        utexas_viscous_u_plus,
        utexas_log_y_plus,
        utexas_log_u_plus,
        parties_yc_plus,
        parties_u_plus,
        parties_viscous_yc_plus,
        parties_viscous_u_plus,
        parties_log_yc_plus,
        parties_log_u_plus,
        Re,
        Re_tau,
    )

    create_normal_stress_plot(
        utexas_y_plus,
        utexas_upup_plus,
        utexas_vpvp_plus,
        utexas_wpwp_plus,
        utexas_k_plus,
        parties_yc_plus,
        parties_yv_plus,
        parties_upup_plus,
        parties_vpvp_plus,
        parties_wpwp_plus,
        parties_k_plus,
        Re,
        Re_tau,
        u_tau,
    )


if __name__ == "__main__":
    main()
