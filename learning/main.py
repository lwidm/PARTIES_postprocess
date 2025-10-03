import h5py  # type: ignore
import numpy as np
import matplotlib
import glob
import os

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Optional, List, Tuple, Dict

matplotlib.use("Qt5Agg")
from matplotlib import pyplot as plt

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
    "MKD_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

Re: float = 2800.0
output_dir: str = "output"
data_dir: str = "data"


def load_data_with_comments(filename, comment_chars="#%") -> List[np.ndarray]:
    lines: List[str]
    with open(filename, "r") as f:
        lines = f.readlines()

    data_lines: List[str] = []
    line: str
    for line in lines:
        stripped: str = line.strip()
        if stripped and not any(stripped.startswith(char) for char in comment_chars):
            data_lines.append(stripped)

    data: np.ndarray = np.genfromtxt(data_lines, dtype=np.float64)

    return [data[:, i] for i in range(data.shape[1])]


# y+ < 5
def viscous_sublayer(y_plus: np.ndarray) -> np.ndarray:
    u_plus: np.ndarray = y_plus
    return u_plus


# y+ > 30
def log_law(y_plus: np.ndarray, kappa: float = 0.41, C_plus: float = 5.0):
    u_plus: np.ndarray = 1 / kappa * np.log(y_plus) + C_plus
    return u_plus


def law_of_the_wall(
    y_plus: np.ndarray, kappa: float = 0.41, C_plus: float = 5.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    viscous_buffer: float = 10.0
    log_buffer: float = 24.0
    viscous_mask: np.ndarray = y_plus < (5.0 + viscous_buffer)
    log_mask: np.ndarray = y_plus > (30.0 - log_buffer)

    y_plus_viscous: np.ndarray = y_plus[viscous_mask]
    y_plus_log: np.ndarray = y_plus[log_mask]

    U_plus_viscous: np.ndarray = viscous_sublayer(y_plus_viscous)
    U_plus_log: np.ndarray = log_law(y_plus_log, kappa, C_plus)

    return (y_plus_viscous, U_plus_viscous, y_plus_log, U_plus_log)


def fit_law_of_the_wall_parameters(
    y_plus_experimental: np.ndarray, U_plus_experimental: np.ndarray
) -> Tuple[float, float]:
    from scipy.optimize import curve_fit  # type: ignore

    log_mask: np.ndarray = y_plus_experimental > 30
    y_plus_log: np.ndarray = y_plus_experimental[log_mask]
    U_plus_log: np.ndarray = U_plus_experimental[log_mask]

    initial_guess: Tuple[float, float] = (0.41, 5.0)

    try:
        popt: Tuple[float, float]
        popt, _ = curve_fit(log_law, y_plus_log, U_plus_log, p0=initial_guess)
        kappa_fit: float
        C_plus_fit: float
        kappa_fit, C_plus_fit = popt
        return kappa_fit, C_plus_fit
    except Exception as e:
        print(f"Fitting of log law parameters failed: {e}")
        return 0.41, 5.0


def _process_file(path: str) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    import h5py
    import numpy as np

    with h5py.File(path, "r") as f:
        u_raw: np.ndarray = f["u"][:]  # type: ignore
        u: np.ndarray = u_raw[:-1, :-1, :-1]
        print(f'Processing file "{path}" ... ')

        Ny: int = u.shape[1]

        if Ny % 2 != 0:
            u = np.delete(u, Ny // 2, axis=1)
            Ny = Ny - 1

        mid: int = Ny // 2
        sum_u: np.ndarray = np.zeros((mid,), dtype=u.dtype)
        sum_uu: np.ndarray = np.zeros((mid,), dtype=u.dtype)
        u_top: np.ndarray = u[:, :mid, :]
        u_bottom: np.ndarray = np.flip(u[:, mid:, :], axis=1)
        sum_u += u_top.sum(axis=(0, 2))
        sum_uu += (u_top * u_top).sum(axis=(0, 2))
        sum_u += u_bottom.sum(axis=(0, 2))
        sum_uu += (u_bottom * u_bottom).sum(axis=(0, 2))

        y_raw: np.ndarray = f["grid"]["yc"]  # type: ignore
        y = y_raw[:-1]
        Ny = u.shape[1]
        if Ny % 2 != 0:
            y = np.delete(y, Ny // 2)
            Ny = Ny - 1
        y = y[:mid]

        Nz: int = u.shape[0]
        Nx: int = u_top.shape[2] + u_bottom.shape[2]
        U_mean_single: np.ndarray = sum_u / (Nx * Nz)

        UU_mean_single: np.ndarray = sum_uu / (Nx * Nz)
        upup_single: np.ndarray = UU_mean_single - U_mean_single * U_mean_single

        return U_mean_single.astype(np.float64), upup_single.astype(np.float64), y

def save_results_to_h5(results: Tuple, output_path: str, metadata: Optional[Dict] = None) -> None:
    y_plus: np.ndarray = results[0]
    U_plus: np.ndarray = results[1]
    upup: np.ndarray = results[2]
    u_tau: np.ndarray = results[3]
    tau_w: np.ndarray = results[4]

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    with h5py.File(output_path, 'w') as f:
        f.create_dataset('y_plus', data=y_plus)
        f.create_dataset('U_plus', data=U_plus)
        f.create_dataset('upup', data=upup)
        f.create_dataset('u_tau', data=u_tau)
        f.create_dataset('tau_w', data=tau_w)

        if metadata:
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    f.attrs[key] = value

def get_nuerical_data_concurrent(
    min_index: Optional[int] = None,
    max_index: Optional[int] = None,
    save_output: bool = True,
    num_workers: Optional[int] = None,
    use_threads: bool = False,
    set_blas_threads_to_1: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    sum_U_mean: Optional[np.ndarray] = None
    sum_upup: Optional[np.ndarray] = None
    count: int = 0
    y: Optional[np.ndarray] = None

    files: List[str] = sorted(glob.glob(f"./{data_dir}/Data_*.h5"))
    if min_index is not None or max_index is not None:
        filtered_files: List[str] = []
        for file in files:
            try:
                file_index: int = int(file.split("_")[-1].split(".")[0])
                if (min_index is None or file_index >= min_index) and (
                    max_index is None or file_index <= max_index
                ):
                    filtered_files.append(file)
            except (ValueError, IndexError):
                continue
        files = filtered_files
    if len(files) == 0:
        return np.array([]), np.array([]), np.array([]), 0.0, 0.0

    if num_workers is None:
        num_workers = min(len(files), (os.cpu_count() or 1))

    if set_blas_threads_to_1:
        for var in BLAS_THREAD_ENV_VARS:
            if os.environ.get(var) is None:
                os.environ[var] = "1"

    Executor = ThreadPoolExecutor if use_threads else ProcessPoolExecutor

    with Executor(max_workers=num_workers) as ex:
        for U_mean_single, upup_single, y_single in ex.map(_process_file, files):
            if y is None and y_single is not None:
                y = np.asarray(y_single, dtype=np.float64).copy()

            if sum_U_mean is None:
                sum_U_mean = np.zeros_like(U_mean_single, dtype=np.float64)
                sum_upup = np.zeros_like(upup_single, dtype=np.float64)

            sum_U_mean += U_mean_single
            sum_upup += upup_single
            count += 1

        if sum_U_mean is None or sum_upup is None or y is None or count == 0:
            return np.array([]), np.array([]), np.array([]), 0.0, 0.0

    U_mean: np.ndarray = np.concatenate(
        (np.array([0]), np.squeeze(sum_U_mean / float(count)))
    )
    upup: np.ndarray = np.concatenate(
        (np.array([0]), np.squeeze(sum_upup / float(count)))
    )
    y = np.concatenate((np.array([0]), np.squeeze(y)))

    du_dy: np.ndarray = (U_mean[1:] - U_mean[:-1]) / (y[1:] - y[:-1])

    tau_w: float = 1 / Re * du_dy[0]
    u_tau: float = np.sqrt(tau_w)

    y_plus: np.ndarray = Re * y * u_tau
    U_plus: np.ndarray = U_mean / u_tau

    result = (y_plus, U_plus, upup, u_tau, tau_w)

    if save_output:
        metadata = {
            "min_index": min_index,
            "max_index": max_index,
            "num_files_processed": count,
            "function": "get_numerical_data_concurrent",
        }
        save_results_to_h5(result, f"{output_dir}/numerical_data.h5", metadata)

    return result


def get_numerical_data_singlethreaded(
    min_index: Optional[int] = None, max_index: Optional[int] = None, save_output: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    sum_U_mean: Optional[np.ndarray] = None
    sum_upup: Optional[np.ndarray] = None
    count: int = 0

    y: Optional[np.ndarray] = None

    files: List[str] = sorted(glob.glob(f"./{data_dir}/Data_*.h5"))
    if min_index is not None or max_index is not None:
        filtered_files: List[str] = []
        for file in files:
            try:
                file_index: int = int(file.split("_")[-1].split(".")[0])
                if (min_index is None or file_index >= min_index) and (
                    max_index is None or file_index <= max_index
                ):
                    filtered_files.append(file)
            except (ValueError, IndexError):
                continue

        files = filtered_files
    for data_file in files:
        with h5py.File(data_file, "r") as f:
            u_raw: np.ndarray = f["u"][:]  # type: ignore
            u: np.ndarray = u_raw[:-1, :-1, :-1]
            print(f'Processing file "{data_file}" ... ')

            Ny: int = u.shape[1]

            if Ny % 2 != 0:
                u = np.delete(u, Ny // 2, axis=1)
                Ny = Ny - 1

            mid: int = Ny // 2
            sum_u: np.ndarray = np.zeros((mid,), dtype=u.dtype)
            sum_uu: np.ndarray = np.zeros((mid,), dtype=u.dtype)
            u_top: np.ndarray = u[:, :mid, :]
            u_bottom: np.ndarray = np.flip(u[:, mid:, :], axis=1)
            sum_u += u_top.sum(axis=(0, 2))
            sum_uu += (u_top * u_top).sum(axis=(0, 2))
            sum_u += u_bottom.sum(axis=(0, 2))
            sum_uu += (u_bottom * u_bottom).sum(axis=(0, 2))

            if y is None:
                y_raw: np.ndarray = f["grid"]["yc"]  # type: ignore
                y = y_raw[:-1]
                Ny = u.shape[1]
                if Ny % 2 != 0:
                    y = np.delete(y, Ny // 2)
                    Ny = Ny - 1
                y = y[:mid]

            Nz: int = u.shape[0]
            Nx: int = u_top.shape[2] + u_bottom.shape[2]
            U_mean_single: np.ndarray = sum_u / (Nx * Nz)

            UU_mean_single: np.ndarray = sum_uu / (Nx * Nz)
            upup_single: np.ndarray = UU_mean_single - U_mean_single * U_mean_single

            if sum_U_mean is None:
                sum_U_mean = np.zeros_like(U_mean_single)
                sum_upup = np.zeros_like(upup_single)

            sum_U_mean += U_mean_single
            sum_upup += upup_single
            count += 1

    U_mean: np.ndarray = np.array([])
    upup: np.ndarray = np.array([])
    if sum_U_mean is None or sum_upup is None or y is None:
        return np.array([]), np.array([]), np.array([]), 0.0, 0.0

    U_mean = np.concatenate((np.array([0]), np.squeeze(sum_U_mean / count)))
    upup = np.concatenate((np.array([0]), np.squeeze(sum_upup / count)))
    y = np.concatenate((np.array([0]), np.squeeze(y)))

    du_dy: np.ndarray = (U_mean[1:] - U_mean[:-1]) / (y[1:] - y[:-1])

    tau_w: float = 1 / Re * du_dy[0]
    u_tau: float = np.sqrt(tau_w)

    y_plus: np.ndarray = Re * y * u_tau
    U_plus: np.ndarray = U_mean / u_tau

    result = (y_plus, U_plus, upup, u_tau, tau_w)

    if save_output:
        metadata = {
            "min_index": min_index,
            "max_index": max_index,
            "num_files_processed": count,
            "function": "get_numerical_data_singlethreaded",
        }
        save_results_to_h5(result, f"{output_dir}/numerical_data.h5", metadata)

    return result

def get_numerical_data_saved(file_path: str = f"./{output_dir}/numerical_data.h5") -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Saved data file not found: {file_path}")
    
    with h5py.File(file_path, 'r') as f:
        # Load all required datasets
        y_plus = f['y_plus'][:]
        U_plus = f['U_plus'][:]
        upup = f['upup'][:]
        u_tau = f['u_tau'][()]  # [()] to get scalar value
        tau_w = f['tau_w'][()]  # [()] to get scalar value
    
    return y_plus, U_plus, upup, u_tau, tau_w


def main() -> None:
    utexas_filename: str = "./data/LM_Channel_0180_mean_prof.dat"

    y_delta_utexas: np.ndarray
    y_plus_utexas: np.ndarray
    U_plus_utexas: np.ndarray
    dU_dy_utexas: np.ndarray
    W_utexas: np.ndarray
    P_utexas: np.ndarray
    y_delta_utexas, y_plus_utexas, U_plus_utexas, dU_dy_utexas, W_utexas, P_utexas = (
        load_data_with_comments(utexas_filename)
    )

    y_plus_numerical: np.ndarray
    U_plus_numerical: np.ndarray
    upup_numerical: np.ndarray
    u_tau: float
    tau_w: float
    # y_plus_numerical, U_plus_numerical, upup_numerical, u_tau, tau_w = (
    #     get_numerical_data_singlethreaded()
    # )
    y_plus_numerical, U_plus_numerical, upup_numerical, u_tau, tau_w = (
        get_nuerical_data_concurrent()
    )
    # y_plus_numerical, U_plus_numerical, upup_numerical, u_tau, tau_w = (
    #     get_numerical_data_saved()
    # )
    Re_tau: float = Re * u_tau

    print(f"u_tau: {u_tau}, tau_w: {tau_w}, Re_tau: {Re_tau}")

    kappa_utexas: float
    C_plus_utexas: float
    kappa_utexas, C_plus_utexas = fit_law_of_the_wall_parameters(
        y_plus_utexas, U_plus_utexas
    )

    y_plus_viscous_utexas: np.ndarray
    U_plus_viscous_utexas: np.ndarray
    y_plus_log_utexas: np.ndarray
    U_plus_log_utexas: np.ndarray
    (
        y_plus_viscous_utexas,
        U_plus_viscous_utexas,
        y_plus_log_utexas,
        U_plus_log_utexas,
    ) = law_of_the_wall(y_plus_utexas, kappa_utexas, C_plus_utexas)

    kappa_PARTIES: float
    C_plus_PARTIES: float
    kappa_PARTIES, C_plus_PARTIES = fit_law_of_the_wall_parameters(
        y_plus_numerical, U_plus_numerical
    )

    y_plus_viscous_PARTIES: np.ndarray
    U_plus_viscous_PARTIES: np.ndarray
    y_plus_log_PARTIES: np.ndarray
    U_plus_log_PARTIES: np.ndarray
    (
        y_plus_viscous_PARTIES,
        U_plus_viscous_PARTIES,
        y_plus_log_PARTIES,
        U_plus_log_PARTIES,
    ) = law_of_the_wall(y_plus_numerical, kappa_PARTIES, C_plus_PARTIES)

    print(
        f"Law of the wall parameters (utexas):  kappa={kappa_utexas}, C_plus={C_plus_utexas}\n"
        f"Law of the wall parameters (PARTIES): kappa={kappa_PARTIES}, C_plus={C_plus_PARTIES}"
    )

    # ---------- U_mean Plot ----------

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.semilogx(y_plus_utexas, U_plus_utexas, "-k", label="utexas data")
    ax.semilogx(y_plus_numerical, U_plus_numerical, "-.k", label="PARTIES data")
    ax.semilogx(
        y_plus_viscous_utexas,
        U_plus_viscous_utexas,
        "--k",
        linewidth=0.9,
        label="law of the wall (utexas)",
    )
    ax.semilogx(y_plus_log_utexas, U_plus_log_utexas, "--k", linewidth=0.8)
    ax.semilogx(
        y_plus_log_PARTIES,
        U_plus_log_PARTIES,
        ":k",
        linewidth=0.9,
        label="law of the wall (PARTIES)",
    )

    viscous_boundary: float = 5.0  # end of viscous sublayer
    buffer_boundary: float = 30.0

    for x in (viscous_boundary, buffer_boundary):
        ax.axvline(x=x, color="0.25", linewidth=0.8, linestyle=":", alpha=0.7, zorder=0)

    ax.set_xlim((1e0, np.max(y_plus_utexas)))
    ax.set_ylim((0e0, np.max([np.max(U_plus_utexas), np.max(U_plus_numerical)]) * 1.05))
    ax.set_xlabel(r"$y^+$", fontsize=14)
    ax.set_ylabel(r"$u^+$", fontsize=14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.tick_params(axis="both", which="both", direction="out", labelsize=12)
    legend = ax.legend(
        loc="lower right", frameon=False, fontsize=12, bbox_to_anchor=(1.0, 0.20)
    )

    x_visc_center: float = np.sqrt(1.0 * viscous_boundary)
    x_buffer_center: float = np.sqrt(viscous_boundary * buffer_boundary)
    x_right = ax.get_xlim()[1]
    x_log_center = np.sqrt(buffer_boundary * x_right)
    y_top = ax.get_ylim()[1]
    y_label_top = 0.99 * y_top

    ax.text(
        x_visc_center,
        y_label_top,
        "Viscous sublayer\n$y^+<5$",
        ha="center",
        va="top",
        fontsize=12,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.0),
    )

    ax.text(
        x_buffer_center,
        y_label_top,
        "Buffer layer\n$5<y^+<30$",
        ha="center",
        va="top",
        fontsize=12,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.0),
    )

    ax.text(
        x_log_center,
        y_label_top,
        "Log-law region\n$30<y^+$",
        ha="center",
        va="top",
        fontsize=12,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.0),
    )

    plt.tight_layout()
    plt.savefig(f"{output_dir}/Re={Re}_Re_tau={Re_tau}-y+_u+.png", dpi=300)
    plt.show()
    plt.close(fig)

    # ---------- u_rms Plot ----------


if __name__ == "__main__":
    main()
