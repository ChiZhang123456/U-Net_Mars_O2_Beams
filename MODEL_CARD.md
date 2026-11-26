# MAVEN STATIC O2+ U-Net v12 count2

Three encoder levels, a bottleneck, three transpose-convolution decoder levels, skip concatenations, and a 1x1 output convolution. Base widths 12/24/48/96. Each block has two 3x3 convolutions, each followed by GroupNorm and ReLU. Parameter count: 272,463.

Input `[batch,3,time,32]`:

1. O2+ log10 DEF clipped to [4,8] and scaled to [0,1].
2. O2+ validity: finite corrected DEF strictly greater than the simultaneous positive, finite central-mass-channel 2-count DEF response. Equality is invalid. This channel is independent of the 1e4 DEF scaling floor.
3. Log10 energy normalized between 0.2 eV and 30 keV, clipped to [0,1].

Apply C6 iv4 background correction once before summing O2+ mass bins 24 to 40 and interpolating onto the common energy grid. No O+ or H+ input is required.

Output `[batch,3,time,32]`: 0 Noise, 1 Beam, 2 Low-energy ions. 512-time windows, inference stride 51; short input zero-padding; overlapping probability averaging before argmax. Raw API does not apply production screening. Split at outages; production uses a 60-second maximum gap and separate UTC days.

Checkpoint hashes are in `weights/model_metadata.json`. The source checkpoint records training stride 256; default inference uses stride 51.

Calibration, alignment, contamination, incomplete FOV and inherited labels limit interpretation. Inspect representative spectra and instrument coverage.

Count response: `2 * dead / (time_integ * geom_factor * gf * eff)`, interpolated linearly in log10 energy with NaN outside native coverage. Supply an aligned `two_count_def` matrix or explicit binary count mask. The v12 checkpoint was selected by validation physical macro F1; test data were not used for model selection. Reference labels outside the count mask are Noise. See README for calibration requirements.

Reference labels include expert-rule and model-assisted labels and are not independent human ground truth. Reproducibility is limited to one training seed (42) and five epochs.
