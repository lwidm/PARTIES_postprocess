# theory/law_of_the_wall.py
import numpy as np
from typing import Tuple
from scipy.optimize import curve_fit  # type: ignore
import warnings

def _viscous_sublayer_velocity(y_plus: np.ndarray) -> np.ndarray:
    return y_plus


def _log_law_velocity(
    y_plus: np.ndarray,
    von_karman_constant: float = 0.41,
    log_law_constant: float = 5.0,
) -> np.ndarray:
    return 1.0 / von_karman_constant * np.log(y_plus) + log_law_constant


def generate_profile(
    y_plus: np.ndarray,
    von_karman_constant: float = 0.41,
    log_law_constant: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    viscous_buffer: float = 10.0
    log_buffer: float = 24.0

    viscous_region_mask: np.ndarray = y_plus < (5.0 + viscous_buffer)
    log_region_mask: np.ndarray = y_plus > (30.0 - log_buffer)

    viscous_y_plus: np.ndarray = y_plus[viscous_region_mask]
    log_y_plus: np.ndarray = y_plus[log_region_mask]

    viscous_u_plus: np.ndarray = _viscous_sublayer_velocity(viscous_y_plus)
    log_u_plus: np.ndarray = _log_law_velocity(
        log_y_plus, von_karman_constant, log_law_constant
    )

    return viscous_y_plus, viscous_u_plus, log_y_plus, log_u_plus


def fit_parameters(
    experimental_yc_plus: np.ndarray, experimental_plus_velocity: np.ndarray
) -> Tuple[float, float]:

    log_region_mask: np.ndarray = (experimental_yc_plus > 30) & (experimental_yc_plus < 1e2)
    log_region_yc_plus: np.ndarray = experimental_yc_plus[log_region_mask]
    log_region_velocity: np.ndarray = experimental_plus_velocity[log_region_mask]

    initial_guess: Tuple[float, float] = (0.41, 5.0)

    fitted_parameters: Tuple[float, float]
    try:
        fitted_parameters, _ = curve_fit(
            _log_law_velocity, log_region_yc_plus, log_region_velocity, p0=initial_guess
        )
        fitted_kappa, fitted_constant = fitted_parameters
        return fitted_kappa, fitted_constant
    except Exception as error:
        warnings.warn(f"Fitting of log law parameters failed: {error}", category=UserWarning)
        return 0.41, 5.0
