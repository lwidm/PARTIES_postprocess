import h5py # type: ignore
from pathlib import Path
import numpy as np

def main():
    data_path: Path = Path("/anvil/scratch/x-lwidmer/RUN8/Data.h5")
    output_data_path: Path = Path("output/RUN8/fluid/parties_reynolds.h5")
    u: np.ndarray
    with h5py.File(data_path, "r") as f:
        u = f["u"][:-1] # type: ignore
    u_mean = np.mean(u)
    print(f"u_mean = {u_mean}")

    Re_tau: float
    with h5py.File(output_data_path, "r") as f:
        Re_tau = f["Re_tau"][()] # type: ignore
    print(f"Re_tau = {Re_tau}")



if __name__ == "__main__":
    main()
