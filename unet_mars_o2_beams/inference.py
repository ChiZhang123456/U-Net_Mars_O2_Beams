"""Checkpoint loading and overlap-averaged inference for U-Net v9."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .model import StaticUNet
from .preprocessing import INPUT_CHANNEL_NAMES


CLASS_NAMES = ("H+ contamination", "Beam", "cold ions", "other/uncertain")
DEFAULT_WINDOW_SIZE = 512
DEFAULT_STRIDE = 51


def load_pretrained_model(
    checkpoint_path: str | Path,
    device: str | torch.device | None = None,
) -> tuple[StaticUNet, dict, torch.device]:
    """Load the released v9 checkpoint and verify its public metadata."""
    selected_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    checkpoint = torch.load(
        Path(checkpoint_path), map_location=selected_device, weights_only=False
    )
    stored_channels = tuple(checkpoint.get("input_channel_names", ()))
    if stored_channels != INPUT_CHANNEL_NAMES:
        raise ValueError(
            "Checkpoint input channels do not match the released v9 preprocessing"
        )
    model = StaticUNet(in_channels=len(stored_channels), classes=len(CLASS_NAMES))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(selected_device).eval()
    return model, checkpoint, selected_device


def _window_starts(length: int, window_size: int, stride: int) -> list[int]:
    if length <= window_size:
        return [0]
    starts = list(range(0, length - window_size + 1, stride))
    final_start = length - window_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


@torch.no_grad()
def predict_spectrogram(
    model: StaticUNet,
    inputs: np.ndarray,
    device: str | torch.device,
    window_size: int = DEFAULT_WINDOW_SIZE,
    stride: int = DEFAULT_STRIDE,
) -> tuple[np.ndarray, np.ndarray]:
    """Classify a complete spectrogram with overlap-averaged time windows.

    Parameters
    ----------
    inputs
        Normalized array with shape ``[9, time, energy]``.

    Returns
    -------
    probabilities
        Array with shape ``[4, time, energy]``.
    labels
        Integer class map with shape ``[time, energy]``. Values 0 to 3 follow
        ``CLASS_NAMES``.
    """
    values = np.asarray(inputs, dtype=np.float32)
    if values.ndim != 3 or values.shape[0] != len(INPUT_CHANNEL_NAMES):
        raise ValueError(f"Expected [9, time, energy], got {values.shape}")
    if window_size <= 0 or stride <= 0 or stride > window_size:
        raise ValueError("Require 0 < stride <= window_size")

    original_length = values.shape[1]
    if original_length == 0:
        raise ValueError("The time axis is empty")
    padded_length = max(original_length, window_size)
    padded = np.pad(values, ((0, 0), (0, padded_length - original_length), (0, 0)))
    probability_sum = np.zeros(
        (len(CLASS_NAMES), padded_length, values.shape[2]), dtype=np.float32
    )
    coverage = np.zeros(padded_length, dtype=np.float32)

    for start in _window_starts(padded_length, window_size, stride):
        patch = torch.from_numpy(padded[:, start : start + window_size])[None].to(device)
        probabilities = torch.softmax(model(patch)[0], dim=0).cpu().numpy()
        probability_sum[:, start : start + window_size] += probabilities
        coverage[start : start + window_size] += 1.0

    probability_sum /= coverage[None, :, None]
    probability_sum = probability_sum[:, :original_length]
    return probability_sum, np.argmax(probability_sum, axis=0).astype(np.uint8)
