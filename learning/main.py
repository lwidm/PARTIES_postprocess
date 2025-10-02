import h5py  # type: ignore
import numpy as np
import matplotlib

matplotlib.use("Qt5Agg")
from matplotlib import pyplot as plt


def load_data_with_comments(filename, comment_chars="#%") -> list[np.ndarray]:
    lines: list[str]
    with open(filename, "r") as f:
        lines = f.readlines()

    data_lines: list[str] = []
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    viscous_buffer: float = 10.0
    log_buffer: float = 25.0
    viscous_mask: np.ndarray = y_plus < (5.0 + viscous_buffer)
    log_mask: np.ndarray = y_plus > (30.0 - log_buffer)

    y_plus_viscous: np.ndarray = y_plus[viscous_mask]
    y_plus_log: np.ndarray = y_plus[log_mask]

    U_plus_viscous: np.ndarray = viscous_sublayer(y_plus_viscous)
    U_plus_log: np.ndarray = log_law(y_plus_log, kappa, C_plus)

    return (y_plus_viscous, U_plus_viscous, y_plus_log, U_plus_log)


def fit_law_of_the_wall_parameters( y_plus_experimental: np.ndarray, U_plus_experimental: np.ndarray) -> tuple[float, float]:
    from scipy.optimize import curve_fit # type: ignore

    log_mask: np.ndarray = y_plus_experimental > 30
    y_plus_log: np.ndarray = y_plus_experimental[log_mask]
    U_plus_log: np.ndarray = U_plus_experimental[log_mask]

    initial_guess: tuple[float, float] = (0.41, 5.0)

    try:
        popt: tuple[float, float]
        popt, _ = curve_fit(log_law, y_plus_log, U_plus_log, p0=initial_guess)
        kappa_fit: float
        C_plus_fit: float
        kappa_fit, C_plus_fit = popt
        return kappa_fit, C_plus_fit
    except Exception as e:
        print(f"Fitting of log law parameters failed: {e}")
        return 0.41, 5.0

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

    kappa: float
    C_plus: float
    kappa, C_plus = fit_law_of_the_wall_parameters(y_plus_utexas, U_plus_utexas)
    print(f"Law of the wall parameters: kappa={kappa}, C_plus={C_plus}")

    y_plus_viscous: np.ndarray
    U_plus_viscous: np.ndarray
    y_plus_log: np.ndarray
    U_plus_log: np.ndarray
    y_plus_viscous, U_plus_viscous, y_plus_log, U_plus_log = law_of_the_wall(y_plus_utexas, kappa, C_plus)

    plt.semilogx(y_plus_utexas, U_plus_utexas, '-k', label="utexas data")
    plt.semilogx(y_plus_viscous, U_plus_viscous, '--k', label="law of the wall")
    plt.semilogx(y_plus_log, U_plus_log, '--k')
    plt.xlim([1e0, np.max(y_plus_utexas)])
    plt.ylim([0e0, np.max(U_plus_utexas)])
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
