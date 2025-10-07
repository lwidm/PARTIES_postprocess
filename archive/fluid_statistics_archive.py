# =============================================================================
# FILE I/O AND DATA LOADING UTILITIES
# =============================================================================


def load_y_coordinates(file_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
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
        volume_fraction_data = 1.0 - h5_file["vf" + component][:]  # type: ignore

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
        Cross-component mean (e.g., UV for components "u" and "v")
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
            volume_fraction: np.ndarray = 1.0 - h5_file["vf" + component1][:]  # type: ignore
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

    data_files: List[Path] = myio.list_parties_data_files(
        parties_data_dir, "Data", min_file_index, max_file_index
    )

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

    data_files: List[Path] = myio.list_parties_data_files(
        parties_data_dir, "Data", min_file_index, max_file_index
    )

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

    data_files: List[Path] = myio.list_parties_data_files(
        parties_data_dir, "Data", min_file_index, max_file_index
    )
    if not data_files:
        return {}

    yc: np.ndarray
    yv: np.ndarray
    yc, yv = load_y_coordinates(data_files[0])
    final_results: Dict[str, Any] = {"yc": yc, "yv": yv}

    # STEP 1: Process u-component and compute upup
    print("Step 1: Processing u-component...")
    U: np.ndarray
    UU: np.ndarray
    U, UU = process_single_component(
        "u", min_file_index, max_file_index, num_workers_single_component, use_threads
    )

    upup = UU - U * U
    final_results["U"] = U
    final_results["UU"] = UU
    final_results["upup"] = upup

    # Free u-component memory
    del UU, upup
    gc.collect()
    print("Step 1 complete: upup computed")

    # STEP 2: Process v-component and compute vpvp
    print("Step 2: Processing v-component...")
    V: np.ndarray
    VV: np.ndarray
    V, VV = process_single_component(
        "v", min_file_index, max_file_index, num_workers_single_component, use_threads
    )

    vpvp = VV - V * V
    final_results["V"] = V
    final_results["VV"] = VV
    final_results["vpvp"] = vpvp

    # Free v-component memory
    del VV, vpvp
    gc.collect()
    print("Step 2 complete: vpvp computed")

    # STEP 3: Process uv cross-component and compute upvp
    print("Step 3: Processing uv cross-component...")
    UV = process_cross_components(
        "u",
        "v",
        min_file_index,
        max_file_index,
        num_workers_cross_component,
        use_threads,
    )

    upvp = UV - U * V
    final_results["UV"] = UV
    final_results["upvp"] = upvp

    # Free UV memory and U (no longer needed)
    del UV, U
    gc.collect()
    print("Step 3 complete: upvp computed")

    # STEP 4: Process w-component and compute wpwp
    print("Step 4: Processing w-component...")
    W: np.ndarray
    WW: np.ndarray
    W, WW = process_single_component(
        "w", min_file_index, max_file_index, num_workers_single_component, use_threads
    )

    wpwp = WW - W * W
    final_results["W"] = W
    final_results["WW"] = WW
    final_results["wpwp"] = wpwp

    # Free w-component memory
    del WW, wpwp
    gc.collect()
    print("Step 4 complete: wpwp computed")

    # STEP 5: Process remaining cross-components
    print("Step 5: Processing remaining cross-components...")

    # Process uw
    UW = process_cross_components(
        "u",
        "w",
        min_file_index,
        max_file_index,
        num_workers_cross_component,
        use_threads,
    )
    if UW.size > 0:
        upwp = UW - final_results["U"] * W
        final_results["UW"] = UW
        final_results["upwp"] = upwp
        del UW
        gc.collect()
        print("Step 5a complete: upwp computed")

    # Process vw
    VW = process_cross_components(
        "v",
        "w",
        min_file_index,
        max_file_index,
        num_workers_cross_component,
        use_threads,
    )
    if VW.size > 0:
        vpwp = VW - V * W
        final_results["VW"] = VW
        final_results["vpwp"] = vpwp
        del VW, V, W
        gc.collect()
        print("Step 5b complete: vpwp computed")

    final_results["k"] = 0.5 * (
        final_results["upup"] + final_results["vpvp"] + final_results["wpwp"]
    )

    # STEP 6: Compute friction velocity and convert to wall units
    print("Step 6: Computing wall units...")
    tmp_grid: Dict[str, np.ndarray] = {"yc": yc, "yv": yv}  # type: ignore
    tau_w, u_tau = flow_statistics.calc_friction_velocity(final_results, tmp_grid, Re)

    wall_results: Dict[str, Any] = flow_statistics.get_wall_units(
        final_results, tmp_grid, Re, tau_w, u_tau
    )

    final_results = final_results | wall_results
    del wall_results
    gc.collect()

    print("Step 6 complete: All quantities converted to wall units")

    # Save final results
    myio.save_to_h5(
        f"{output_dir}/reynolds_stresses.h5",
        final_results,
        {
            "min_index": min_file_index,
            "max_index": max_file_index,
            "num_files_processed": len(data_files),
        },
    )

    print("Reynolds stress computation completed successfully!")
    return final_results
