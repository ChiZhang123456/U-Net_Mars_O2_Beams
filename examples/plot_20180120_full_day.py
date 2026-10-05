"""Run or display the U-Net v11 example for 20 January 2018.

The input NPZ file contains aligned MAVEN STATIC background-corrected O+ and O2+
differential energy flux (DEF) arrays for 00:00 to 24:00 UTC. By default,
this script plots the v11 predictions stored in that file. Pass
``--rerun-inference`` to reconstruct the three model inputs and run the
released v11 checkpoint before plotting.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm
from matplotlib.patches import Patch
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from unet_mars_o2_beams import (  # noqa: E402
    build_input_channels,
    load_pretrained_model,
    predict_spectrogram,
)


DEFAULT_CHECKPOINT = ROOT / "weights" / "unet_v11_best_validation.pt"
DEFAULT_OUTPUT = ROOT / "docs" / "static_unet_v11_20180120_abc.png"

# Production labels use zero for noise or rejected pixels, followed by the
# two accepted physical classes. These colors match the manuscript figure.
LABEL_NAMES = ("Noise", "Beam", "Low-energy ions")
LABEL_COLORS = ("#D9D9D9", "#7B2CBF", "#1B9E77")
REGION_NAMES = {
    "SW": "Solar Wind",
    "MSH": "Magnetosheath",
    "MSP": "Induced Magnetosphere",
}
REGION_COLORS = {"SW": "#D8645A", "MSH": "#4C78A8", "MSP": "#5F6368"}


def load_example(path: Path) -> dict[str, np.ndarray]:
    """Load and validate the aligned full-day STATIC arrays."""
    with np.load(path, allow_pickle=False) as source:
        required = ("time", "energy_ev", "o_def", "o2_def", "labels")
        missing = [name for name in required if name not in source]
        if missing:
            raise KeyError(f"Example NPZ is missing keys: {missing}")
        data = {name: source[name] for name in required}

    shape = data["o_def"].shape
    if shape != data["o2_def"].shape:
        raise ValueError("The two corrected DEF arrays are not aligned")
    if shape != data["labels"].shape:
        raise ValueError("The label map does not match the spectra")
    if shape != (data["time"].size, data["energy_ev"].size):
        raise ValueError("Time or energy coordinates do not match the spectra")
    if not np.all(np.isin(data["labels"], [0, 1, 2])):
        raise ValueError("Expected v11 labels: 0 Noise, 1 Beam, 2 Low-energy ions")
    return data


def infer_labels(
    data: dict[str, np.ndarray], checkpoint: Path, device: str | None
) -> np.ndarray:
    """Run v11 and return native class IDs."""
    inputs = build_input_channels(
        o2_def=data["o2_def"],
        energy_ev=data["energy_ev"],
    )
    model, _, selected_device = load_pretrained_model(checkpoint, device)
    _, raw_labels = predict_spectrogram(model, inputs, selected_device)

    labels = raw_labels
    return labels


def load_regions(path: Path | None) -> list[tuple[str, np.datetime64, np.datetime64]]:
    """Read the preselected plasma-region intervals used above panel a."""
    if path is None:
        return []
    regions = []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            regions.append(
                (
                    row["region"],
                    np.datetime64(row["start_utc"], "ns"),
                    np.datetime64(row["end_utc"], "ns"),
                )
            )
    return regions


def add_spectrum(ax, time, energy, values, ylabel):
    """Plot one STATIC DEF spectrogram on a logarithmic energy axis."""
    finite = np.isfinite(values)
    plot_values = np.where(finite & (values > 0), values, 1.0).astype(float)
    plot_values[~finite] = np.nan
    plotted = np.ma.masked_invalid(plot_values)
    cmap = mpl.colormaps["turbo"].copy()
    cmap.set_under("black")
    mesh = ax.pcolormesh(
        time,
        energy,
        plotted.T,
        shading="nearest",
        cmap=cmap,
        norm=LogNorm(vmin=1.0e4, vmax=1.0e8),
        rasterized=True,
    )
    ax.set_yscale("log")
    ax.set_ylim(0.3, 3.0e4)
    ax.set_ylabel(ylabel)
    return mesh


def add_label_panel(ax, time, energy, labels):
    """Plot the three-class U-Net result using discrete colors."""
    cmap = ListedColormap(LABEL_COLORS)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    ax.pcolormesh(
        time, energy, labels.T, shading="nearest", cmap=cmap, norm=norm,
        rasterized=True,
    )
    ax.set_yscale("log")
    ax.set_ylim(0.3, 3.0e4)
    ax.set_ylabel(r"STATIC O$_2^+$" "\n" r"$E_i/q$ (eV/q)")


def add_region_strip(fig, reference_ax, regions):
    """Add the solar-wind, magnetosheath, and magnetosphere strip."""
    position = reference_ax.get_position()
    region_ax = fig.add_axes(
        [position.x0, position.y1 + 0.004, position.width, 0.037],
        sharex=reference_ax,
    )
    region_ax.set_ylim(0, 1.4)
    for region, start, end in regions:
        region_ax.axvspan(
            start, end, ymin=0.0, ymax=0.36,
            color=REGION_COLORS[region], linewidth=0,
        )
    for region, name in REGION_NAMES.items():
        candidates = [(start, end) for key, start, end in regions if key == region]
        if candidates:
            start, end = max(candidates, key=lambda item: item[1] - item[0])
            region_ax.text(
                start + (end - start) / 2,
                0.98,
                name,
                ha="center",
                va="center",
                color=REGION_COLORS[region],
                fontsize=11,
                fontweight="bold",
                clip_on=False,
            )
    region_ax.set_axis_off()


def make_figure(data, labels, regions, output: Path) -> None:
    """Create manuscript-style panels a to c and save PNG and PDF files."""
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "mathtext.fontset": "dejavusans",
            "font.size": 13,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
        }
    )
    time = data["time"].astype("datetime64[ns]")
    energy = data["energy_ev"].astype(float)
    spectra = (
        (data["o_def"], r"STATIC O$^+$" "\n" r"$E_i/q$ (eV/q)"),
        (data["o2_def"], r"STATIC O$_2^+$" "\n" r"$E_i/q$ (eV/q)"),
    )

    # The wider canvas leaves enough room for the shared colorbar and the
    # complete three-class legend without compressing the spectrogram panels.
    fig, axes = plt.subplots(3, 1, figsize=(13.5, 7.8), sharex=True)
    fig.subplots_adjust(left=0.105, right=0.82, top=0.925, bottom=0.10, hspace=0.035)
    spectrum_mesh = None
    for ax, (values, ylabel) in zip(axes[:2], spectra):
        spectrum_mesh = add_spectrum(ax, time, energy, values, ylabel)
    add_label_panel(axes[2], time, energy, labels)
    add_region_strip(fig, axes[0], regions)

    start = np.datetime64("2018-01-20T00:00:00")
    end = np.datetime64("2018-01-21T00:00:00")
    for panel, ax in zip("abc", axes):
        ax.set_xlim(start, end)
        ax.text(
            -0.12, 1.02, panel, transform=ax.transAxes,
            fontweight="bold", fontsize=15, va="bottom",
        )
        ax.tick_params(which="both", top=True, right=True, labelsize=12)

    axes[-1].xaxis.set_major_locator(mdates.HourLocator(interval=2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes[-1].set_xlabel("UTC on 20 January 2018", fontsize=15)

    first = axes[0].get_position()
    third = axes[1].get_position()
    colorbar_ax = fig.add_axes([first.x1 + 0.008, third.y0, 0.016, first.y1 - third.y0])
    colorbar = fig.colorbar(spectrum_mesh, cax=colorbar_ax)
    colorbar.set_label(
        r"DEF (keV cm$^{-2}$ s$^{-1}$ sr$^{-1}$ keV$^{-1}$)", fontsize=13
    )
    colorbar.ax.tick_params(labelsize=12)

    handles = [
        Patch(facecolor=color, edgecolor="none", label=name)
        for color, name in zip(LABEL_COLORS, LABEL_NAMES)
    ]
    fourth = axes[2].get_position()
    fig.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(0.84, 0.5 * (fourth.y0 + fourth.y1)),
        frameon=False,
        fontsize=13,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=350)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=ROOT / "examples/data/static_20180120_v11.npz",
        help="Aligned 2018-01-20 STATIC spectra and label map in NPZ format",
    )
    parser.add_argument(
        "--regions", type=Path,
        help="Optional CSV with region, start_utc, and end_utc columns",
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", help="For example, cpu, cuda, or cuda:0")
    parser.add_argument(
        "--rerun-inference",
        action="store_true",
        help="Run the released checkpoint instead of using the saved v11 predictions",
    )
    args = parser.parse_args()

    torch.set_num_threads(4)
    data = load_example(args.data)
    labels = (
        infer_labels(data, args.checkpoint, args.device)
        if args.rerun_inference
        else data["labels"].astype(np.uint8)
    )
    make_figure(data, labels, load_regions(args.regions), args.output)
    print(f"Time interval: {data['time'][0]} to {data['time'][-1]}")
    print(f"Spectral shape: {data['o_def'].shape}")
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
