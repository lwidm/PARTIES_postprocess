# flocParties/io/tools.py
"""Write and read simulation metadata."""
import ast
import configparser
from pathlib import Path
from typing import Any

import numpy as np

def _read_inp(inp_file: Path | str) -> dict[str, np.ndarray | int | float]:
    """Return a dict of all parameters in a config file."""
    config_parser = configparser.ConfigParser(inline_comment_prefixes="#")
    config_parser.optionxform = lambda option: option  # type: ignore
    config_parser.read(inp_file)

    config = {k: v for s in config_parser.sections() for k, v in config_parser.items(s)}

    parsed: dict[str, np.ndarray | int | float] = {}
    for k, v in config.items():
        v.replace("{", "[").replace("}", "]")
        try:
            val = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            val = v
        parsed[k] = np.array(val) if isinstance(val, list) else val
    return parsed


def _get_phi(inp_path: Path) -> tuple[float, int, int]:
    """Compute the total particle volume fraction phi from `p_mobile.inp`, `p_fixed.inp` and `parties.inp` files"""

    mobile_path: Path = inp_path / "p_mobile.inp"
    fixed_path: Path = inp_path / "p_fixed.inp"
    parties_inp_path: Path = inp_path / "parties.inp"

    if not mobile_path.exists():
        raise ValueError(
            f'Failed to obtain phi: Could not find "p_mobile.inp" at "{mobile_path}"'
        )
    if not fixed_path.exists():
        raise ValueError(
            f'Failed to obtain phi: Could not find "p_fixed.inp" at "{fixed_path}"'
        )
    if not parties_inp_path.exists():
        raise ValueError(
            f'Failed to obtain phi: Could not find "parties.inp" at "{parties_inp_path}"'
        )

    r_mobile: np.ndarray = np.array([0.0])
    r_fixed: np.ndarray = np.array([0.0])
    num_p_mobile: int
    num_p_fixed: int
    with open(mobile_path, "r") as f:
        first_line_mobile: str = f.readline().strip()
        num_p_mobile = int(first_line_mobile)
    with open(fixed_path, "r") as f:
        first_line_fixed: str = f.readline().strip()
        num_p_fixed = int(first_line_fixed)
    if num_p_mobile > 0:
        r_mobile = np.loadtxt(mobile_path, skiprows=1, usecols=3)
    if num_p_fixed > 0:
        r_fixed = np.loadtxt(fixed_path, skiprows=1, usecols=3)
    vol_particle = 4 / 3 * np.pi * np.sum(r_mobile**3)
    vol_particle += 4 / 3 * np.pi * np.sum(r_fixed**3)

    config: configparser.ConfigParser = configparser.ConfigParser(
        inline_comment_prefixes="#"
    )
    config.optionxform = lambda option: option  # type: ignore
    config.read(parties_inp_path)
    if not config.has_section("geometry"):
        raise ValueError(
            f'Failed to obtain phi: Could not find [geometry] field in "parties.inp" at "{parties_inp_path}"'
        )
    geom: dict[str, Any] = dict(config["geometry"])
    for key in geom:
        geom[key] = float(geom[key])

    vol_channel: float = (
        (geom["xmax"] - geom["xmin"])
        * (geom["ymax"] - geom["ymin"])
        * (geom["zmax"] - geom["zmin"])
    )

    phi: float = vol_particle / vol_channel

    return phi, num_p_mobile, num_p_fixed


def write_metadata(sim_name: str, inp_dir: Path, output_dir: Path) -> None:
    """Create metadata file with basic simulation info

    This function creates a `metadata.ini` file (in non append mode) and writes the most basic simulation info (obtained from an `parties.inp` file) to it. The total particle volume fraction also get calculated and added to this metadata file. This should be the first function called before postprocessing.

    Args:
        sim_name: The name of the current simulation (E.g. "lukwidmer_phi5p0")
        inp_dir: The location of the parties input files: `parties.inp`, `p_mobile.inp`, `p_fixed.inp`.
        output_dir: The directory in which to save the generated `metadata.ini` file.
    """

    output_dir.mkdir(exist_ok=True)
    config: dict[str, np.ndarray | int | float] = _read_inp(inp_dir / "parties.inp")

    phi: float
    n_p_mobile: int
    n_p_fixed: int
    phi, n_p_mobile, n_p_fixed = _get_phi(inp_dir)
    Co: float = float(config["Co"])
    Re: float = float(config["Re"])
    rho_s: float = float(config["rho_s"])
    xmin: float = float(config["xmin"])
    xmax: float = float(config["xmax"])
    ymin: float = float(config["ymin"])
    ymax: float = float(config["ymax"])
    zmin: float = float(config["zmin"])
    zmax: float = float(config["zmax"])
    Nx: int = int(config["NXM"])
    Ny: int = int(config["NYM"])
    Nz: int = int(config["NZM"])
    h: float = (ymax - ymin) / Ny
    coh_range: float = float(config["coh_range"]) * h


    metadata: configparser.ConfigParser = configparser.ConfigParser(
        inline_comment_prefixes="#"
    )
    metadata.optionxform = lambda option: option  # type: ignore

    n_digits: int = 5

    def fl_fmt(number: float) -> str:
        return f"{number:.{n_digits}f}".rstrip("0").rstrip(".")

    metadata["General"] = {
        "Sim_name": sim_name,
        "Re": fl_fmt(Re),
        "Co": fl_fmt(Co),
        "coh_range": fl_fmt(coh_range),
        "rho_s": fl_fmt(rho_s),
    }
    metadata["Domain"] = {
        "xmin": fl_fmt(xmin),
        "xmax": fl_fmt(xmax),
        "ymin": fl_fmt(ymin),
        "ymax": fl_fmt(ymax),
        "zmin": fl_fmt(zmin),
        "zmax": fl_fmt(zmax),
        "Nx": str(Nx),
        "Ny": str(Ny),
        "Nz": str(Nz),
        "x_periodic": "1",  # Currently hardcoded, need to read from Particle_XXX.h5 files or Boundary.h (a pain)
        "y_periodic": "0",  # Currently hardcoded, ...
        "z_periodic": "1",  # Currently hardcoded, ...
    }
    metadata["Particles"] = {
        "Phi": fl_fmt(phi),
        "n_p_mobile": str(n_p_mobile),
        "n_p_fixed": str(n_p_fixed),
    }

    with open(output_dir / "metadata.ini", "w") as metadata_file:
        metadata.write(metadata_file)


def read_metadata(metadata_file: Path) -> dict[str, dict[str, float | int | str]]:
    """
    Read a simulation metadata file in INI format and return a nested dictionary.

    Args:
        metadata_file: Path to the INI file.

    Returns:
        A dictionary of sections, each containing key-value pairs with automatic type conversion.
    """
    config = configparser.ConfigParser()
    config.optionxform = lambda option: option  # type: ignore[assignment]
    config.read(metadata_file)
    metadata: dict[str, dict[str, float | int | str]] = {}

    for section in config.sections():
        section_data: dict[str, float | int | str] = {}
        for key, value in config.items(section):
            value = value.strip()
            # try to cast to int, then float, else keep string
            try:
                val: float | int | str
                if "." in value or "e" in value.lower():
                    val = float(value)
                else:
                    val = int(value)
            except ValueError:
                val = value
            section_data[key] = val
        metadata[section] = section_data

    return metadata