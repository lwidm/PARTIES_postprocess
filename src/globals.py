# -- src/globals.py

import os

on_anvil = os.getenv("MY_MACHINE", "") == "anvil"


BLAS_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
