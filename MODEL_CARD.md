# Model card: MAVEN STATIC O2+ U-Net v10

## Intended use

Pixel-wise segmentation of background-corrected STATIC O+ and O2+ DEF into Noise, Beam, and Low-energy ions. This is a research model; inspect representative predictions and instrument coverage before physical interpretation.

## Architecture and input contract

Three encoder levels, a bottleneck, three transpose-convolution decoder levels, skip concatenations, and a 1×1 output convolution. Base width 12, subsequent widths 24/48/96. Each block contains two 3×3 convolutions, each followed by GroupNorm and ReLU. Parameter count: 272,787. The supplied architecture diagram omits GroupNorm and arranges input images for display rather than tensor order.

Input tensor `[batch,6,time,32]`, in this exact order:

1. O2+ log10 DEF, clipped to [4,8] and scaled to [0,1].
2. O+ log10 DEF, with the same scaling.
3. log10(O+/O2+) where both species are valid, clipped to [-3,3] and scaled to [0,1]; zero elsewhere.
4. O2+ validity, finite DEF >= 1e4.
5. O+ validity, finite DEF >= 1e4.
6. Absolute log10 energy coordinate, scaled between 0.2 eV and 30 keV and clipped to [0,1].

Background correction is applied once to native STATIC C6 with the iv4 product, before summing O+ mass 14–20 and O2+ mass 24–40 and interpolating DEF against log10 energy onto the common 32-bin grid. No H+ input is used. D1 is not the input to this classifier.

Output tensor `[batch,3,time,32]`; labels 0 Noise, 1 Beam, 2 Low-energy ions. The raw API performs no postclassification screening.

## Inference and reproducibility

512-time × 32-energy windows, stride 256, 50% overlap. Short observations are zero-padded in normalized input space. Softmax probabilities are averaged across overlapping windows and cropped to original length before argmax. No confidence, connected-component, energy-range, or event-level filtering is included.

The inference checkpoint contains weights and public metadata. Its source and release SHA-256 hashes are recorded in `weights/model_metadata.json`. The included full-day 2018-01-20 example illustrates unedited v10 predictions.

## Limitations

Performance depends on correct background calibration, species extraction, alignment, and energy scaling. Out-of-distribution plasma populations, instrumental contamination, incomplete field of view, and weak reference labels can affect predictions.
