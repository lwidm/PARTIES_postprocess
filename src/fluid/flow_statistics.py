# -- src/fluid/flow_statistics.py

import numpy as np
import h5py  # type: ignore
from pathlib import Path
from typing import Union, Dict, Literal, List, Tuple, Any
import tqdm  # type: ignore
import gc


def get_grid(fluid_file: Union[str, Path]) -> Dict[str, np.ndarray]:
    """Get domain information from the fluid file."""
    with h5py.File(fluid_file, "r") as f:
        xc: np.ndarray = f["grid/xc"][:-1]  # type: ignore
        yc: np.ndarray = f["grid/yc"][:-1]  # type: ignore
        zc: np.ndarray = f["grid/zc"][:-1]  # type: ignore
        xu: np.ndarray = f["grid/xu"][:]  # type: ignore
        yv: np.ndarray = f["grid/yv"][:]  # type: ignore
        zw: np.ndarray = f["grid/zw"][:]  # type: ignore
        return {
            "xc": xc,
            "yc": yc,
            "zc": zc,
            "xu": xu,
            "yv": yv,
            "zw": zw,
        }


def masked_mean(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.einsum("abc,abc->b", arr, mask) / np.einsum("abc->b", mask)


def interp_x(data: np.ndarray):
    return 0.5 * (data[:, :, 1:] + data[:, :, :-1])


def interp_y(data: np.ndarray):
    return 0.5 * (data[:, 1:] + data[:, :-1])


def interp_z(data: np.ndarray):
    return 0.5 * (data[1:] + data[:-1])


def E_ii(data: np.ndarray, axis: Literal[0, 1, 2]):
    return np.mean(np.abs(data) ** 2, axis=axis)


def E_ij(data_i: np.ndarray, data_j: np.ndarray, axis: Literal[0, 1, 2]):
    return np.mean(np.real(data_i * data_j.conj()), axis=axis)


def process_mean_flow(
    fluid_files: Union[List[str], List[Path]], grid: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    n_snapshots: int = len(fluid_files)
    results: Dict[str, np.ndarray] = {
        "U": np.zeros_like(grid["yc"]),
        "V": np.zeros_like(grid["yv"]),
        "W": np.zeros_like(grid["yc"]),
        "Phi": np.zeros_like(grid["yc"]),
    }

    for fluid_file in tqdm.tqdm(fluid_files, desc="Processing mean flow"):
        with h5py.File(fluid_file, "r") as f:
            vfu: np.ndarray = f["vfu"][:-1, :-1, :-1]  # type: ignore
            results["U"] += masked_mean(f["u"][:-1, :-1, :-1], 1 - vfu)  # type: ignore
            results["Phi"] += np.mean(vfu, axis=(0, 2))
            del vfu
            gc.collect()
            results["V"] += masked_mean(f["v"][:-1, :, :-1], 1 - f["vfv"][:-1, :, :-1])  # type: ignore
            gc.collect()
            results["W"] += masked_mean(f["w"][:-1, :-1, :-1], 1 - f["vfw"][:-1, :-1, :-1])  # type: ignore
            gc.collect()

    for key in results:
        results[key] /= n_snapshots

    return results


def process_fluctuations(
    fluid_files: Union[List[str], List[Path]],
    mean_results: Dict[str, np.ndarray],
    grid: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    n_snapshots = len(fluid_files)
    ny_c = len(grid["yc"])
    ny_v = len(grid["yv"])
    nx = len(grid["xc"]) // 2 + 1
    nz = len(grid["zc"]) // 2 + 1

    results: Dict[str, np.ndarray] = {
        "phip": np.zeros_like(grid["yc"]),
        "upup": np.zeros_like(grid["yc"]),
        "upvp": np.zeros_like(grid["yc"]),
        "vpvp": np.zeros_like(grid["yv"]),
        "wpwp": np.zeros_like(grid["yc"]),
        "E_uu_kx": np.zeros((ny_c, nx)),
        "E_vv_kx": np.zeros((ny_v, nx)),
        "E_ww_kx": np.zeros((ny_c, nx)),
        "E_uv_kx": np.zeros((ny_c, nx)),
        "E_uu_kz": np.zeros((nz, ny_c)),
        "E_vv_kz": np.zeros((nz, ny_v)),
        "E_ww_kz": np.zeros((nz, ny_c)),
        "E_uv_kz": np.zeros((nz, ny_c)),
    }

    for fluid_file in tqdm.tqdm(fluid_files, desc="Processing fluctuations"):
        with h5py.File(fluid_file, "r") as f:
            w: np.ndarray = f["w"][:-1, :-1, :-1] - mean_results["W"][np.newaxis, :, np.newaxis]  # type: ignore
            gc.collect()
            vfw: np.ndarray = 1 - f["vfw"][:, :-1, :-1]  # type: ignore
            gc.collect()

        results["E_ww_kx"] += E_ii(np.fft.rfft(w, axis=2), axis=0)
        results["E_ww_kz"] += E_ii(np.fft.rfft(w, axis=0), axis=2)
        results["wpwp"] += masked_mean(w * w, vfw[:-1])
        del w
        gc.collect()

        with h5py.File(fluid_file, "r") as f:
            u: np.ndarray = f["u"][:-1, :-1, :] - mean_results["U"][np.newaxis, :, np.newaxis]  # type: ignore
            v: np.ndarray = f["v"][:-1, :, :-1] - mean_results["V"][np.newaxis, :, np.newaxis]  # type: ignore

            # BUG : missing dimension here?
            vfu: np.ndarray = f["vfu"][:-1, :-1]  # type: ignore
            vfv: np.ndarray = 1 - f["vfv"][:-1, :, :-1]  # type: ignore

        phi: np.ndarray = (
            vfu[:, :, :-1] - mean_results["Phi"][np.newaxis, :, np.newaxis]
        )
        results["phip"] += np.mean(phi * phi, axis=(0, 2))
        vfu = 1 - vfu
        del phi

        results["upup"] += masked_mean(u[:, :, :-1] * u[:, :, :-1], vfu[:, :, :-1])
        results["vpvp"] += masked_mean(v * v, vfv)

        vfc: np.ndarray = (interp_x(vfu) + interp_y(vfv) + interp_z(vfw)) / 3

        del vfu, vfv, vfw
        gc.collect()

        u_k: np.ndarray = np.fft.rfft(u[:, :, :-1], axis=2)
        results["E_uu_kx"] += E_ii(u_k, axis=0)

        v_k: np.ndarray = np.fft.rfft(v, axis=2)
        results["E_vv_kx"] += E_ii(v_k, axis=0)

        v_k = interp_y(v_k)
        results["E_uv_kx"] += E_ij(u_k, v_k, axis=0)

        del u_k, v_k
        gc.collect()

        u_k = np.fft.rfft(u[:, :, :-1], axis=0)
        results["E_uu_kz"] += E_ii(u_k, axis=2)

        v_k = np.fft.rfft(v, axis=0)
        results["E_vv_kz"] += E_ii(v_k, axis=2)

        v_k = interp_y(v_k)
        results["E_uv_kz"] += E_ij(u_k, v_k, axis=2)

        del u_k, v_k
        gc.collect()

        u = interp_x(u)
        v = interp_y(v)
        results["upvp"] += masked_mean(u * v, vfc)
        del u, v
        gc.collect()

    for key in results:
        results[key] /= n_snapshots

    vv_cell_center: np.ndarray = 0.5 * (results["vpvp"][1:] + results["vpvp"][:-1])
    results["k"] = 0.5 * (results["upup"] + vv_cell_center + results["wpwp"])

    results["E_uu_kx"] *= 0.5
    results["E_vv_kx"] *= 0.5
    results["E_ww_kx"] *= 0.5
    results["E_uv_kx"] *= 0.5
    results["E_uu_kz"] = 0.5 * results["E_uu_kz"].T
    results["E_vv_kz"] = 0.5 * results["E_vv_kz"].T
    results["E_ww_kz"] = 0.5 * results["E_ww_kz"].T
    results["E_uv_kz"] = 0.5 * results["E_uv_kz"].T

    results["k_x"] = (
        2 * np.pi * np.fft.fftfreq(results["E_uu_kx"].shape[1], d=grid["xu"][1])
    )
    results["k_z"] = (
        2 * np.pi * np.fft.fftfreq(results["E_uu_kz"].shape[1], d=grid["zw"][1])
    )

    return results


def calc_friction_velocity(
    mean_results: Dict[str, Any], grid: Dict[str, np.ndarray], Re: float
) -> Tuple[float, float, float]:
    du_dy_0: float = (mean_results["U"][0] - (-mean_results["U"][0])) / (
        grid["yc"][0] * 2
    )
    tau_w: float = 1.0 / Re * du_dy_0
    u_tau: float = float(np.sqrt(tau_w))
    Re_tau: float = u_tau * Re

    return tau_w, u_tau, Re_tau


def get_wall_units(
    results: Dict[str, Any],
    grid: Dict[str, np.ndarray],
    Re: float,
    tau_w: float,
    u_tau: float,
) -> Dict[str, Union[np.ndarray, float]]:

    wall_results: Dict[str, Union[np.ndarray, float]] = {}
    yc_plus: np.ndarray = Re * grid["yc"] * u_tau
    yv_plus: np.ndarray = Re * grid["yv"] * u_tau
    wall_results["u_tau"] = u_tau
    wall_results["tau_w"] = tau_w
    wall_results["yc_plus"] = yc_plus
    wall_results["yv_plus"] = yv_plus
    wall_results["Re_tau"] = Re * u_tau

    # Convert to wall units
    wall_results["U_plus"] = results["U"] / u_tau
    wall_results["upup_plus"] = results["upup"] / (u_tau * u_tau)
    wall_results["vpvp_plus"] = results["vpvp"] / (u_tau * u_tau)
    wall_results["wpwp_plus"] = results["wpwp"] / (u_tau * u_tau)
    wall_results["k_plus"] = results["k"] / (u_tau * u_tau)

    if "upvp" in results:
        wall_results["upvp_plus"] = results["upvp"] / (u_tau * u_tau)
    if "upwp" in results:
        wall_results["upwp_plus"] = results["upwp"] / (u_tau * u_tau)
    if "vpwp" in results:
        wall_results["vpwp_plus"] = results["vpwp"] / (u_tau * u_tau)

    return wall_results


def calc_tot_vol_frac(path: Union[str, Path]) -> float:

    print(f"Calculating volume fraction using {path}")
    vfu: np.ndarray
    with h5py.File(path, "r") as h5_file:
        vfu = h5_file["vfu"][:]  # type: ignore

    vfu = vfu[:-1, :-1, :-1]

    Nx: int
    Ny: int
    Nz: int
    Nz, Ny, Nx = vfu.shape

    phi: float = np.sum(vfu, axis=None) / (Nx * Ny * Nz)

    return phi


def calc_tot_fluid_Ekin(fluid_file_path: Union[str, Path], Re: float) -> float:

    grid: Dict[str, np.ndarray] = get_grid(fluid_file_path)
    mean_u_squared: np.floating
    phi: float
    with h5py.File(fluid_file_path, "r") as h5_file:
        uc: np.ndarray = h5_file["u"][:-1, :-1, :-1]  # type: ignore
        vc: np.ndarray = interp_y(h5_file["v"][:-1, :, :-1])  # type: ignore
        wc: np.ndarray = h5_file["w"][:-1, :-1, :-1]  # type: ignore

        u_squared: np.ndarray = uc * uc + vc * vc + wc * wc
        del (uc, vc, wc)
        gc.collect()

        vfu: np.ndarray = h5_file["vfu"][:-1, :-1, :-1]  # type: ignore
        Nz, Ny, Nx = vfu.shape
        phi = np.sum(vfu, axis=None) / (Nx * Ny * Nz)

        mean_u_squared = np.average(u_squared, weights=(1 - vfu))

    V_fluid: float = grid["xu"][-1] * grid["yv"][-1] * grid["zw"][-1] * (1 - phi)
    E_kin = Re / 2 * V_fluid * float(mean_u_squared)
    return E_kin
