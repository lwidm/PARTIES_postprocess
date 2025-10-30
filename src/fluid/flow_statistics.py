# -- src/fluid/flow_statistics.py

import numpy as np
import h5py
from pathlib import Path
from typing import Union, Dict, Literal, List, Tuple, Any, Optional
import tqdm
import gc

from src.myio import myio


def get_grid(fluid_file: Path) -> Dict[str, np.ndarray]:
    """Get domain information from the fluid file."""
    with h5py.File(str(fluid_file), "r") as f:
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
    fluid_files: List[Path], grid: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:

    u_shape: Tuple[int, int, int]
    v_shape: Tuple[int, int, int]
    w_shape: Tuple[int, int, int]
    with h5py.File(str(fluid_files[0]), "r") as f:
        u_shape = f["u"][:-1, :-1, :-1].shape  # type: ignore
        v_shape = f["v"][:-1, :, :-1].shape  # type: ignore
        w_shape = f["w"][:-1, :-1, :-1].shape  # type: ignore

    u_buf: np.ndarray = np.empty(u_shape)
    v_buf: np.ndarray = np.empty(v_shape)
    w_buf: np.ndarray = np.empty(w_shape)
    vfu_buf: np.ndarray = np.empty(u_shape)
    vfv_buf: np.ndarray = np.empty(v_shape)
    vfw_buf: np.ndarray = np.empty(w_shape)
    n_snapshots = len(fluid_files)
    results = {
        "U": np.zeros_like(grid["yc"]),
        "V": np.zeros_like(grid["yv"]),
        "W": np.zeros_like(grid["yc"]),
        "Phi": np.zeros_like(grid["yc"]),
    }

    for fluid_file in tqdm.tqdm(fluid_files, desc="Processing mean flow"):
        with h5py.File(str(fluid_file), "r") as f:
            f["u"].read_direct(u_buf, np.s_[:-1, :-1, :-1])  # type: ignore
            f["v"].read_direct(v_buf, np.s_[:-1, :, :-1])  # type: ignore
            f["w"].read_direct(w_buf, np.s_[:-1, :-1, :-1])  # type: ignore
            f["vfu"].read_direct(vfu_buf, np.s_[:-1, :-1, :-1])  # type: ignore
            f["vfv"].read_direct(vfv_buf, np.s_[:-1, :, :-1])  # type: ignore
            f["vfw"].read_direct(vfw_buf, np.s_[:-1, :-1, :-1])  # type: ignore
        np.add(results["U"], masked_mean(u_buf, 1 - vfu_buf), out=results["U"])
        np.add(results["V"], masked_mean(v_buf, 1 - vfv_buf), out=results["V"])
        np.add(results["W"], masked_mean(w_buf, 1 - vfw_buf), out=results["W"])
        np.add(results["Phi"], np.mean(vfu_buf, axis=(0, 2)), out=results["Phi"])

    for key in results:
        np.divide(results[key], n_snapshots, out=results[key])

    return results


def process_fluctuations(
    fluid_files: List[Path],
    mean_results: Dict[str, np.ndarray],
    grid: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    n_snapshots: int = len(fluid_files)
    ny_c: int = len(grid["yc"])
    ny_v: int = len(grid["yv"])
    nx: int = len(grid["xc"]) // 2 + 1
    nz: int = len(grid["zc"]) // 2 + 1

    results = {
        "phi": np.zeros_like(grid["yc"]),
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
        w: np.ndarray
        vfw: np.ndarray
        with h5py.File(str(fluid_file), "r") as f:
            w = f["w"][:-1, :-1, :-1] - mean_results["W"][np.newaxis, :, np.newaxis]  # type: ignore
            # BUG : -1, :, -1
            vfw = 1 - f["vfw"][:, :-1, :-1]  # type: ignore

        np.add(
            results["E_ww_kx"],
            E_ii(np.fft.rfft(w, axis=2), axis=0),
            out=results["E_ww_kx"],
        )
        np.add(
            results["E_ww_kz"],
            E_ii(np.fft.rfft(w, axis=0), axis=2),
            out=results["E_ww_kz"],
        )
        np.add(results["wpwp"], masked_mean(w * w, vfw[:-1]), out=results["wpwp"])
        del w
        gc.collect()

        u: np.ndarray
        v: np.ndarray
        vfu: np.ndarray
        vfv: np.ndarray
        with h5py.File(str(fluid_file), "r") as f:
            u = f["u"][:-1, :-1, :] - mean_results["U"][np.newaxis, :, np.newaxis]  # type: ignore
            v = f["v"][:-1, :, :-1] - mean_results["V"][np.newaxis, :, np.newaxis]  # type: ignore

            # BUG : missing dimension here?
            vfu = f["vfu"][:-1, :-1]  # type: ignore
            vfv = 1 - f["vfv"][:-1, :, :-1]  # type: ignore

        phi: np.ndarray = (
            vfu[:, :, :-1] - mean_results["Phi"][np.newaxis, :, np.newaxis]
        )
        np.add(results["phip"], np.mean(phi * phi, axis=(0, 2)), out=results["phip"])
        np.subtract(1, vfu, out=vfu)
        del phi
        gc.collect()

        np.add(
            results["upup"],
            masked_mean(u[:, :, :-1] * u[:, :, :-1], vfu[:, :, :-1]),
            out=results["upup"],
        )
        np.add(results["vpvp"], masked_mean(v * v, vfv), out=results["vpvp"])

        vfc: np.ndarray = (interp_x(vfu) + interp_y(vfv) + interp_z(vfw)) / 3

        del vfu, vfv, vfw
        gc.collect()

        u_k: np.ndarray = np.fft.rfft(u[:, :, :-1], axis=2)
        np.add(results["E_uu_kx"], E_ii(u_k, axis=0), out=results["E_uu_kx"])

        v_k: np.ndarray = np.fft.rfft(v, axis=2)
        np.add(results["E_vv_kx"], E_ii(v_k, axis=0), out=results["E_vv_kx"])

        v_k = interp_y(v_k)
        np.add(results["E_uv_kx"], E_ij(u_k, v_k, axis=0), out=results["E_uv_kx"])

        del u_k, v_k
        gc.collect()

        u_k: np.ndarray = np.fft.rfft(u[:, :, :-1], axis=0)
        np.add(results["E_uu_kz"], E_ii(u_k, axis=2), out=results["E_uu_kz"])

        v_k: np.ndarray = np.fft.rfft(v, axis=0)
        np.add(results["E_vv_kz"], E_ii(v_k, axis=2), out=results["E_vv_kz"])

        v_k = interp_y(v_k)
        np.add(results["E_uv_kz"], E_ij(u_k, v_k, axis=2), out=results["E_uv_kz"])

        del u_k, v_k
        gc.collect()

        u = interp_x(u)
        v = interp_y(v)
        np.add(results["upvp"], masked_mean(u * v, vfc), out=results["upvp"])
        del u, v
        gc.collect()

    for key in results:
        results[key] /= n_snapshots

    vv_cell_center: np.ndarray = 0.5 * (results["vpvp"][1:] + results["vpvp"][:-1])
    results["k"] = 0.5 * (results["upup"] + vv_cell_center + results["wpwp"])

    np.multiply(results["E_uu_kx"], 0.5, out=results["E_uu_kx"])
    np.multiply(results["E_vv_kx"], 0.5, out=results["E_vv_kx"])
    np.multiply(results["E_ww_kx"], 0.5, out=results["E_ww_kx"])
    np.multiply(results["E_uv_kx"], 0.5, out=results["E_uv_kx"])
    np.multiply(results["E_uu_kz"].T, 0.5, out=results["E_uu_kz"])
    np.multiply(results["E_vv_kz"].T, 0.5, out=results["E_vv_kz"])
    np.multiply(results["E_ww_kz"].T, 0.5, out=results["E_ww_kz"])
    np.multiply(results["E_uv_kz"].T, 0.5, out=results["E_uv_kz"])

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


def calc_tot_vol_frac(path: Path) -> float:
    print(f"Calculating volume fraction using {path}")
    vfu: np.ndarray
    with h5py.File(str(path), "r") as h5_file:
        vfu = h5_file["vfu"][:]  # type: ignore

    vfu = vfu[:-1, :-1, :-1]

    Nx: int
    Ny: int
    Nz: int
    Nz, Ny, Nx = vfu.shape

    phi: float = np.sum(vfu, axis=None) / (Nx * Ny * Nz)

    return phi


def calc_tot_fluid_Ekin(fluid_file_path: Path, Re: float) -> float:

    grid: Dict[str, np.ndarray] = get_grid(fluid_file_path)
    mean_u_squared: np.floating
    phi: float
    with h5py.File(str(fluid_file_path), "r") as h5_file:
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


def process_mean_phi(
    parties_data_dir: Path,
    output_h5: Path,
    compute_err: bool,
    min_file_index: Optional[int],
    max_file_index: Optional[int],
) -> Dict[str, np.ndarray]:

    fluid_files: List[Path] = myio.list_data_files(
        parties_data_dir, "Data", min_file_index, max_file_index
    )
    n_snapshots: int = len(fluid_files)

    if n_snapshots == 0:
        raise ValueError(f"no fluid files provided (looking in {parties_data_dir} with min_file_index {min_file_index} and max_file_index {max_file_index})")

    grid: Dict[str, np.ndarray] = get_grid(fluid_files[0])


    dset: h5py.Dataset
    with h5py.File(str(fluid_files[0]), "r") as h5file_sample:
        dset = h5file_sample["vfv"]  # type: ignore
        print(dset)
        vfv_Nz, vfv_Ny, vfv_Nx = dset.shape
    Nx: int = (vfv_Nx - 1) * 2
    Ny: int = (vfv_Ny - 1) // 2
    Nz: int = vfv_Nz - 1
    vfv_dtype = dset.dtype

    results: Dict[str, np.ndarray] = {
        "yv": grid["yv"][:Ny].copy(),
        "Phi_mean": np.zeros(Ny, dtype=vfv_dtype),
        "Phi_mean_norm": np.zeros(Ny, dtype=vfv_dtype),
    }
    if compute_err:
        results.update(
            {
                "Phi_err": np.zeros(Ny, dtype=vfv_dtype),
                "Phi_err_norm": np.zeros(Ny, dtype=vfv_dtype),
            }
        )

    # preallocated working buffers
    vfv_buf: np.ndarray = np.empty(
        (vfv_Nz - 1, vfv_Ny - 1, vfv_Nx - 1), dtype=vfv_dtype
    )
    vfv_mirr_buf: np.ndarray = np.empty((Nz, Ny, Nx), dtype=vfv_dtype)
    vfv_mean_buf: np.ndarray = np.empty(Ny, dtype=vfv_dtype)
    vfv_err_buf: Optional[np.ndarray] = None
    if compute_err:
        vfv_err_buf = np.empty(Ny, dtype=vfv_dtype)

    # first pass: accumulate means
    for fluid_file in tqdm.tqdm(fluid_files, desc="Processing mean phi"):
        with h5py.File(str(fluid_file), "r") as h5_file:
            dset = h5_file["vfv"]  # type: ignore
            dset.read_direct(vfv_buf, np.s_[:-1, :-1, :-1])
        myio.mirror_and_append_along_y_inplace(vfv_buf, vfv_mirr_buf)
        np.mean(vfv_mirr_buf, axis=(0, 2), out=vfv_mean_buf)
        results["Phi_mean"] += vfv_mean_buf
        phi_tot: float = np.sum(vfv_mean_buf, axis=None) / float(Ny)
        np.divide(vfv_mean_buf, phi_tot, out=vfv_mean_buf)
        results["Phi_mean_norm"] += vfv_mean_buf

    np.divide(results["Phi_mean"], float(n_snapshots), out=results["Phi_mean"])
    np.divide(
        results["Phi_mean_norm"], float(n_snapshots), out=results["Phi_mean_norm"]
    )

    # second pass: compute variance
    if compute_err:
        assert vfv_err_buf is not None
        for fluid_file in tqdm.tqdm(fluid_files, desc="Processing mean phi"):
            with h5py.File(str(fluid_file), "r") as h5_file:
                dset = h5_file["vfv"]  # type: ignore
                dset.read_direct(vfv_buf, np.s_[:-1, :-1, :-1])
            myio.mirror_and_append_along_y_inplace(vfv_buf, vfv_mirr_buf)
            np.mean(vfv_mirr_buf, axis=(0, 2), out=vfv_mean_buf)
            np.subtract(vfv_mean_buf, results["Phi_mean"], out=vfv_err_buf)
            np.multiply(vfv_err_buf, vfv_err_buf, out=vfv_err_buf)
            results["Phi_err"] += vfv_err_buf

            phi_tot: float = np.sum(vfv_mean_buf, axis=None) / Ny
            np.divide(vfv_mean_buf, phi_tot, out=vfv_mean_buf)
            np.subtract(vfv_mean_buf, results["Phi_mean_norm"], out=vfv_err_buf)
            np.multiply(vfv_err_buf, vfv_err_buf, out=vfv_err_buf)
            results["Phi_err_norm"] += vfv_err_buf

        np.divide(results["Phi_err"], float(n_snapshots), out=results["Phi_err"])
        np.divide(
            results["Phi_err_norm"], float(n_snapshots), out=results["Phi_err_norm"]
        )
        np.sqrt(results["Phi_err"], out=results["Phi_err"])
        np.sqrt(results["Phi_err_norm"], out=results["Phi_err_norm"])

    myio.save_to_h5(output_h5, results)

    return results
