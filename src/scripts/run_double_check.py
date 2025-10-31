import h5py
from pathlib import Path
import numpy as np

def main():
    data_path: Path = Path("/anvil/scratch/x-lwidmer/RUN10/Data_166.h5")
    output_data_path: Path = Path("output/RUN10/fluid/parties_reynolds.h5")
    u: np.ndarray
    with h5py.File(str(data_path), "r") as f:
        u = f["u"][:-1] # type: ignore
    u_mean = np.mean(u)
    print(f"u_mean = {u_mean}")

    Re_tau: float
    with h5py.File(str(output_data_path), "r") as f:
        Re_tau = f["Re_tau"][()] # type: ignore
    print(f"Re_tau = {Re_tau}")



if __name__ == "__main__":
    main()
