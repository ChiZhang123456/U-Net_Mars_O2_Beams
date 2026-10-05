# Corrected STATIC example

`static_20180120_v11.npz` contains 21,600 time samples and 32 energy bins for 20 January 2018. O+ and O2+ DEF were extracted from iv4-background-corrected STATIC C6 and interpolated onto the shared grid. `labels` uses native v11 IDs (0 Noise, 1 Beam, 2 Low-energy ions), with no manual prediction overrides. The example is for reproducibility and illustration.

Run `python examples/plot_20180120_full_day.py --rerun-inference` from the repository root.
