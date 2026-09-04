"""Convert aligned MAVEN STATIC H+, O+, and O2+ spectra into v9 inputs."""

from __future__ import annotations

import numpy as np


INPUT_CHANNEL_NAMES = (
    "o2_log_def",
    "h_log_def",
    "o_log_def",
    "h_to_o2_log_ratio",
    "o_to_o2_log_ratio",
    "o2_validity",
    "h_validity",
    "o_validity",
    "log_energy",
)
ENERGY_MIN_EV = 0.2
ENERGY_MAX_EV = 3.0e4


def _validate_spectrum(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape [time, energy], got {array.shape}")
    return array


def scale_log_def(
    values: np.ndarray, minimum_log: float = 4.0, maximum_log: float = 8.0
) -> tuple[np.ndarray, np.ndarray]:
    """Scale log10 differential energy flux from [4, 8] to [0, 1]."""
    valid = np.isfinite(values) & (values >= 10.0**minimum_log)
    scaled = np.zeros_like(values, dtype=np.float32)
    scaled[valid] = (
        np.clip(np.log10(values[valid]), minimum_log, maximum_log) - minimum_log
    ) / (maximum_log - minimum_log)
    return scaled, valid.astype(np.float32)


def _scaled_log_ratio(
    numerator: np.ndarray,
    denominator: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """Map a clipped log10 ratio from [-3, 3] to [0, 1]."""
    output = np.zeros_like(denominator, dtype=np.float32)
    ratio = np.divide(
        numerator,
        denominator,
        out=np.ones_like(denominator, dtype=np.float64),
        where=valid,
    )
    output[valid] = (
        np.clip(np.log10(np.maximum(ratio[valid], 1.0e-6)), -3.0, 3.0) / 6.0
        + 0.5
    )
    return output


def build_input_channels(
    h_def: np.ndarray,
    o_def: np.ndarray,
    o2_def: np.ndarray,
    energy_ev: np.ndarray,
    h_validity: np.ndarray | None = None,
    o_validity: np.ndarray | None = None,
    o2_validity: np.ndarray | None = None,
) -> np.ndarray:
    """Build the nine normalized input channels expected by U-Net v9.

    All three DEF arrays must be aligned and have shape ``[time, energy]``.
    ``energy_ev`` is a one-dimensional array containing the energy-bin centers.
    Optional validity arrays use one for visible/valid pixels and zero otherwise.
    Pixels below 1e4 DEF or containing non-finite values are always marked invalid.
    """
    h_def = _validate_spectrum("h_def", h_def)
    o_def = _validate_spectrum("o_def", o_def)
    o2_def = _validate_spectrum("o2_def", o2_def)
    if h_def.shape != o_def.shape or h_def.shape != o2_def.shape:
        raise ValueError("H+, O+, and O2+ DEF arrays must have identical shapes")

    energy_ev = np.asarray(energy_ev, dtype=np.float64)
    if energy_ev.ndim != 1 or energy_ev.size != h_def.shape[1]:
        raise ValueError("energy_ev must contain one value per energy bin")
    if np.any(~np.isfinite(energy_ev)) or np.any(energy_ev <= 0):
        raise ValueError("energy_ev must contain finite positive values")

    h_scaled, h_from_def = scale_log_def(h_def)
    o_scaled, o_from_def = scale_log_def(o_def)
    o2_scaled, o2_from_def = scale_log_def(o2_def)

    def combine_validity(provided: np.ndarray | None, inferred: np.ndarray) -> np.ndarray:
        if provided is None:
            return inferred
        supplied = np.asarray(provided, dtype=np.float32)
        if supplied.shape != inferred.shape:
            raise ValueError("A validity array does not match the spectral shape")
        return np.minimum(supplied, inferred)

    h_valid = combine_validity(h_validity, h_from_def)
    o_valid = combine_validity(o_validity, o_from_def)
    o2_valid = combine_validity(o2_validity, o2_from_def)
    h_o2_valid = (h_valid > 0) & (o2_valid > 0)
    o_o2_valid = (o_valid > 0) & (o2_valid > 0)

    log_energy = np.log10(energy_ev)
    energy_coordinate = np.clip(
        (log_energy - np.log10(ENERGY_MIN_EV))
        / (np.log10(ENERGY_MAX_EV) - np.log10(ENERGY_MIN_EV)),
        0.0,
        1.0,
    )
    energy_channel = np.broadcast_to(energy_coordinate, o2_def.shape).astype(np.float32)

    return np.stack(
        (
            o2_scaled,
            h_scaled,
            o_scaled,
            _scaled_log_ratio(h_def, o2_def, h_o2_valid),
            _scaled_log_ratio(o_def, o2_def, o_o2_valid),
            o2_valid,
            h_valid,
            o_valid,
            energy_channel,
        ),
        axis=0,
    ).astype(np.float32)
