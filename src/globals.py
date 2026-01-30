# -- src/globals.py

import os
import sys
from pathlib import Path
from typing import TypedDict

on_anvil: bool = os.getenv("MY_MACHINE", "") == "anvil"
_use_external_drive: bool = True
# _use_data: list[str] = ["phi1p5", "phi3p0", "phi5p0_new", "phi5p0_noCo"]
_use_data: list[str] = ["phi1p5", "phi3p0", "phi5p0_new"]
# _use_data: list[str] = ["phi3p0"]

plot_dir: Path = Path("./output/plots")
parent_dir: Path
data_dir: Path
output_dir: Path


class DataSet(TypedDict):
    label: str
    trn: bool
    phi: float
    coagulation_kernel_sigma: float


# fmt: off
_all_datasets: dict[str, DataSet] = {
    "phi1p5":      {"label": r"$\phi_{1.5\%}$",           "trn": False, "phi": 1.5, "coagulation_kernel_sigma": 5},
    "phi3p0":      {"label": r"$\phi_{3\%}$",             "trn": True,  "phi": 3.0, "coagulation_kernel_sigma": 5},
    "phi5p0_new":  {"label": r"$\phi_{5\%}$",             "trn": True,  "phi": 5.0, "coagulation_kernel_sigma": 5},
    "phi5p0_noCo": {"label": r"$\phi_{5\%}$ no cohesion", "trn": False, "phi": 5.0, "coagulation_kernel_sigma": 5},
    "phi5p0":      {"label": r"$\phi_{5\%}$",             "trn": True,  "phi": 5.0, "coagulation_kernel_sigma": 5},
    "test":        {"label": "test",                      "trn": True,  "phi": 3.0, "coagulation_kernel_sigma": 5},
}
# fmt: on

_datasets: dict[str, DataSet] = {k: _all_datasets[k] for k in _use_data}
data_names: list[str] = list(_datasets.keys())
labels: list[str] = [d["label"] for d in _datasets.values()]
phi_values: list[float] = [d["phi"] for d in _datasets.values()]
coagulation_kernel_sigmas: list[float] = [
    d["coagulation_kernel_sigma"] for d in _datasets.values()
]
has_trn_data: list[bool] = [d["trn"] for d in _datasets.values()]

if on_anvil:
    parent_dir = Path("/anvil/scratch/x-lwidmer")
    data_dir = parent_dir / "output"
    output_dir = parent_dir / "output"
elif _use_external_drive:
    parent_dir = Path("/media/usb/UCSB/")
    if not parent_dir.exists():
        print(f"Error: external drive not mounted (expected {parent_dir})")
        sys.exit(1)
    data_dir = parent_dir / "output"
    output_dir = parent_dir / "output"
else:
    parent_dir = Path("./")
    data_dir = parent_dir / "data"
    output_dir = parent_dir / "data"


BLAS_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
