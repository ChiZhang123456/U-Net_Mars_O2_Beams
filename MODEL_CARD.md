# MAVEN STATIC O2+ U-Net v11

Three encoder levels, a bottleneck, three transpose-convolution decoder levels, skip concatenations, and a 1x1 output convolution. Base widths 12/24/48/96. Each block has two 3x3 convolutions, each followed by GroupNorm and ReLU. Parameter count: 272,463. The supplied diagram omits GroupNorm.

Input `[batch,3,time,32]`:

1. O2+ log10 DEF clipped to [4,8] and scaled to [0,1].
2. O2+ validity: finite DEF >= 1e4.
3. Log10 energy normalized between 0.2 eV and 30 keV, clipped to [0,1].

Apply C6 iv4 background correction once before summing O2+ mass bins 24 to 40 and interpolating onto the common energy grid. No O+ or H+ input is required. D1 is not classifier input.

Output `[batch,3,time,32]`: 0 Noise, 1 Beam, 2 Low-energy ions. 512-time windows, stride 256; short input zero-padding; overlapping probability averaging before argmax. Raw API does not apply production screening. Split at outages; production uses a 60-second maximum gap and separate UTC days.

Checkpoint hashes are in `weights/model_metadata.json`. The full-day example reproduces raw labels; O+ context is optional to scientific interpretation and not passed to the model.

The 14,484 released intervals represent 2,811,946 v9-subset samples after final v11 screening, MSE validity and corrected D1 moment validation. Gaps <=600 seconds are merged. Endpoints are observed samples; detection is not continuous across internal gaps.

Calibration, alignment, contamination, incomplete FOV and inherited labels limit interpretation. Inspect representative spectra and instrument coverage.
