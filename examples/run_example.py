"""Run U-Net v9 on an NPZ file or on a deterministic synthetic example."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from unet_mars_o2_plume import (
    build_input_channels,
    load_pretrained_model,
    predict_spectrogram,
)
from unet_mars_o2_plume.inference import CLASS_NAMES


DEFAULT_CHECKPOINT = ROOT / "weights" / "unet_v9_best_selection.pt"


def synthetic_spectra() -> dict[str, np.ndarray]:
    """Create correctly shaped inputs for a dependency-free smoke test."""
    rng = np.random.default_rng(42)
    time_samples, energy_bins = 700, 32
    energy_ev = np.geomspace(0.2, 3.0e4, energy_bins)
    shape = (time_samples, energy_bins)
    return {
        "h_def": 10.0 ** rng.uniform(4.0, 7.0, shape),
        "o_def": 10.0 ** rng.uniform(4.0, 7.0, shape),
        "o2_def": 10.0 ** rng.uniform(4.0, 7.0, shape),
        "energy_ev": energy_ev,
    }


def load_spectra(path: Path | None) -> dict[str, np.ndarray]:
    if path is None:
        return synthetic_spectra()
    with np.load(path, allow_pickle=False) as source:
        required = ("h_def", "o_def", "o2_def", "energy_ev")
        missing = [name for name in required if name not in source]
        if missing:
            raise KeyError(f"Input NPZ is missing: {missing}")
        return {name: source[name] for name in required}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, help="Optional aligned STATIC spectra NPZ")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=Path("prediction.npz"))
    parser.add_argument("--device", help="For example, cpu, cuda, or cuda:0")
    args = parser.parse_args()

    spectra = load_spectra(args.input)
    inputs = build_input_channels(**spectra)
    model, checkpoint, device = load_pretrained_model(args.checkpoint, args.device)
    probabilities, labels = predict_spectrogram(model, inputs, device)
    np.savez_compressed(
        args.output,
        probabilities=probabilities,
        labels=labels,
        class_names=np.asarray(CLASS_NAMES),
        energy_ev=spectra["energy_ev"],
    )
    print(f"Model: {checkpoint['version']}")
    print(f"Device: {device}")
    print(f"Input shape: {inputs.shape}")
    print(f"Probability shape: {probabilities.shape}")
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
