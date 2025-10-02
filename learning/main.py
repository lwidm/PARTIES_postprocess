import h5py  # type: ignore
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
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


def main() -> None:
    utexas_filename: str = "./data/LM_Channel_0180_mean_prof.dat"

    y_delta_utexas: np.ndarray
    y_plus_utexas: np.ndarray
    U_utexas: np.ndarray
    dU_dy_utexas: np.ndarray
    W_utexas: np.ndarray
    P_utexas: np.ndarray
    y_delta_utexas, y_plus_utexas, U_utexas, dU_dy_utexas, W_utexas, P_utexas = (
        load_data_with_comments(utexas_filename)
    )
    plt.plot(y_plus_utexas, U_utexas)
    plt.show()


if __name__ == "__main__":
    main()
