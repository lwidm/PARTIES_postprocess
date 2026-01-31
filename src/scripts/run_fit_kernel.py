import pickle
from pathlib import Path
from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit

from src import globals


# ---------- candidate models ----------

def _model_sum(xy: np.ndarray, c: float, a: float, b: float) -> np.ndarray:
    """K = c * (x^a + y^a)^b"""
    x, y = xy
    return c * np.power(np.power(x, a) + np.power(y, a), b)

def _model_product(xy: np.ndarray, c: float, a: float, b: float) -> np.ndarray:
    """K = c * (x*y)^a * (x+y)^b"""
    x, y = xy
    return c * np.power(x * y, a) * np.power(x + y, b)

def _model_product2(xy: np.ndarray, c: float, a: float, b: float) -> np.ndarray:
    """K = c * (x*y)^a * (x^2+y^2)^b"""
    x, y = xy
    return c * np.power(x * y, a) * np.power(x**2 + y**2, b)

def _model_sqsum_linsum(xy: np.ndarray, c: float, b1: float, b2: float) -> np.ndarray:
    """K = c * (x^2+y^2)^b1 * (x+y)^b2"""
    x, y = xy
    return c * np.power(x**2 + y**2, b1) * np.power(x + y, b2)

def _model_power_sum(xy: np.ndarray, c: float, a: float) -> np.ndarray:
    """K = c * (x + y)^a"""
    x, y = xy
    return c * np.power(x + y, a)

def _model_power_product(xy: np.ndarray, c: float, a: float) -> np.ndarray:
    """K = c * (x * y)^a"""
    x, y = xy
    return c * np.power(x * y, a)

def _model_sum_product(xy: np.ndarray, c: float, a: float, b: float) -> np.ndarray:
    """K = c * (x+y)^a * (x*y)^b"""
    x, y = xy
    return c * np.power(x + y, a) * np.power(x * y, b)

def _model_general_two_term(xy: np.ndarray, c: float, a1: float, b1: float, a2: float, b2: float) -> np.ndarray:
    """K = c * (x^a1 + y^a1)^b1 * (x^a2 + y^a2)^b2"""
    x, y = xy
    return c * np.power(np.power(x, a1) + np.power(y, a1), b1) * np.power(np.power(x, a2) + np.power(y, a2), b2)


@dataclass
class ModelSpec:
    name: str
    func: callable
    p0: list[float]
    bounds: tuple[list[float], list[float]]
    param_names: list[str]


ALL_MODELS: list[ModelSpec] = [
    ModelSpec(
        name="c*(x^a + y^a)^b",
        func=_model_sum,
        p0=[1.0, 2.0, 1.0],
        bounds=([0, 1, 1], [np.inf, 4, 4]),
        param_names=["c", "a", "b"],
    ),
    ModelSpec(
        name="c*(x*y)^b1 * (x+y)^b2",
        func=_model_product,
        p0=[1.0, 1.0, 1.0],
        bounds=([0, 1, 1], [np.inf, 4, 4]),
        param_names=["c", "b1", "b2"],
    ),
    ModelSpec(
        name="c*(x*y)^b1 * (x^2+y^2)^b2",
        func=_model_product2,
        p0=[1.0, 1.0, 1.0],
        bounds=([0, 1, 1], [np.inf, 4, 4]),
        param_names=["c", "b2", "b2"],
    ),
    ModelSpec(
        name="c*(x^2+y^2)^b1 * (x+y)^b2",
        func=_model_sqsum_linsum,
        p0=[1.0, 1.0, 1.0],
        bounds=([0, 1, 1], [np.inf, 4, 4]),
        param_names=["c", "b1", "b2"],
    ),
    ModelSpec(
        name="c*(x+y)^b",
        func=_model_power_sum,
        p0=[1.0, 1.0],
        bounds=([0, 1], [np.inf, 4]),
        param_names=["c", "b"],
    ),
    ModelSpec(
        name="c*(x*y)^b",
        func=_model_power_product,
        p0=[1.0, 1.0],
        bounds=([0, 1], [np.inf, 4]),
        param_names=["c", "b"],
    ),
    ModelSpec(
        name="c*(x+y)^b1 * (x*y)^b2",
        func=_model_sum_product,
        p0=[1.0, 1.0, 1.0],
        bounds=([0, 1.0, 1.0], [np.inf, 4, 4]),
        param_names=["c", "b1", "b2"],
    ),
    ModelSpec(
        name="c*(x^a1+y^a1)^b1 * (x^a2+y^a2)^b2",
        func=_model_general_two_term,
        p0=[1.0, 2.0, 1.0, 1.0, 1.0],
        bounds=([0, 1, 1, 1, 1], [np.inf, 4, 4, 4, 4]),
        param_names=["c", "a1", "b1", "a2", "b2"],
    ),
]


def _fit_kernel(
    pickle_path: Path,
) -> None:
    with open(pickle_path, "rb") as file:
        results: dict[str, dict] = pickle.load(file)

    K: dict[tuple[int, int], float] = results["K"]
    x_list: list[float] = results["bin_info"]["center_sizes"]
    x_idx_list: list[int] = results["bin_info"]["center_idxs"]

    x_arr: np.ndarray = np.asarray(x_list, dtype=float)
    x_arr = np.pow(x_arr, 1 / 3)

    # Build flat arrays of (x, y, K_val)
    xs: list[float] = []
    ys: list[float] = []
    k_vals: list[float] = []
    for i, x1_idx in enumerate(x_idx_list):
        for j, x2_idx in enumerate(x_idx_list):
            val = K[(x1_idx, x2_idx)]
            if np.isfinite(val) and val > 0:
                xs.append(x_arr[i])
                ys.append(x_arr[j])
                k_vals.append(val)

    xs_arr = np.asarray(xs)
    ys_arr = np.asarray(ys)
    k_arr = np.asarray(k_vals)
    xy_data = np.vstack([xs_arr, ys_arr])

    n_data = len(k_arr)

    for model in ALL_MODELS:
        try:
            popt, pcov = curve_fit(
                model.func,
                xy_data,
                k_arr,
                p0=model.p0,
                bounds=model.bounds,
                maxfev=100000,
            )
            perr = np.sqrt(np.diag(pcov))

            k_pred = model.func(xy_data, *popt)
            ss_res = np.sum((k_arr - k_pred) ** 2)
            ss_tot = np.sum((k_arr - np.mean(k_arr)) ** 2)
            r_squared = 1.0 - ss_res / ss_tot

            n_params = len(popt)
            adj_r_squared = 1.0 - (1.0 - r_squared) * (n_data - 1) / (n_data - n_params - 1)

            param_str = ", ".join(
                f"{name}={val:.4f}+/-{err:.4f}"
                for name, val, err in zip(model.param_names, popt, perr)
            )
            print(f"  {model.name:35s}  R²={r_squared:.4f}  adj_R²={adj_r_squared:.4f}  {param_str}")
        except Exception as e:
            print(f"  {model.name:35s}  FAILED: {e}")


def main() -> None:
    data_names: list[str] = globals.data_names
    data_dir: Path = globals.data_dir

    for data_name in data_names:
        dataset_dir: Path = data_dir / data_name

        for corrected in [True, False]:
            tag = "corrected" if corrected else "uncorrected"
            pickle_file = f"number_density_evolution_params_{tag}.pkl"
            pickle_path = dataset_dir / pickle_file

            if not pickle_path.exists():
                print(f"  Skipping {data_name} ({tag}): not found")
                continue

            print(f"\n{'='*60}")
            print(f"Dataset: {data_name} ({tag})")
            print(f"{'-'*60}")
            _fit_kernel(pickle_path)


if __name__ == "__main__":
    main()
