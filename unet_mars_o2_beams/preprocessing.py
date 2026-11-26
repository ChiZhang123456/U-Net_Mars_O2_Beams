"""Convert aligned MAVEN STATIC background-corrected O2+ spectra into v12 count2 inputs."""

from __future__ import annotations

import numpy as np


INPUT_CHANNEL_NAMES = (
    "o2_log_def",
    "o2_validity",
    "log_energy",
)
ENERGY_MIN_EV = 0.2
ENERGY_MAX_EV = 3.0e4


def _validate_spectrum(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
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


def build_input_channels(
    o2_def: np.ndarray,
    energy_ev: np.ndarray,
    o2_validity: np.ndarray | None = None,
    *,
    two_count_def: np.ndarray | None = None,
) -> np.ndarray:
    """Build the three normalized input channels expected by U-Net v12 count2.

    The background-corrected O2+ DEF array must be aligned and have shape ``[time, energy]``.
    ``energy_ev`` is a one-dimensional array containing the energy-bin centers.
    Supply an aligned two-count DEF response, or a precomputed binary count mask.
    Validity requires finite corrected DEF strictly greater than the positive,
    finite two-count response. Equality is invalid. The mask is independent of
    log-DEF clipping; values below 1e4 may still have validity one.
    """
    o2_def = _validate_spectrum("o2_def", o2_def)

    energy_ev = np.asarray(energy_ev, dtype=np.float32)
    if energy_ev.ndim != 1 or energy_ev.size != o2_def.shape[1]:
        raise ValueError("energy_ev must contain one value per energy bin")
    if np.any(~np.isfinite(energy_ev)) or np.any(energy_ev <= 0):
        raise ValueError("energy_ev must contain finite positive values")

    if o2_def.shape[1] != 32:
        raise ValueError("The released v12 count2 model requires 32 energy bins")

    o2_scaled, _ = scale_log_def(o2_def)

    if two_count_def is None and o2_validity is None:
        raise ValueError("Supply two_count_def or an explicitly computed count-based o2_validity mask")
    o2_valid = np.isfinite(o2_def)
    if two_count_def is not None:
        threshold = np.asarray(two_count_def, dtype=np.float32)
        if threshold.shape != o2_def.shape:
            raise ValueError("two_count_def must match the spectral shape")
        o2_valid &= np.isfinite(threshold) & (threshold > 0) & (o2_def > threshold)
    if o2_validity is not None:
        supplied = np.asarray(o2_validity)
        if supplied.shape != o2_def.shape:
            raise ValueError("A validity array does not match the spectral shape")
        if not np.all(np.isfinite(supplied)) or not np.all((supplied == 0) | (supplied == 1)):
            raise ValueError("Validity masks must contain only finite zeros and ones")
        o2_valid &= supplied.astype(bool)
    o2_valid = o2_valid.astype(np.float32)

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
            o2_valid,
            energy_channel,
        ),
        axis=0,
    ).astype(np.float32)
