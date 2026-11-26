# Current model figure data

`static_20180805_count2.npz`: UTC time, 32-bin energy centers in eV, background-corrected O2+ DEF and aligned two-count-equivalent DEF response, covering 614 samples starting 2018-08-05 10:30:02.664857 UTC. DEF is in eV cm^-2 s^-1 sr^-1 eV^-1 (numerically identical to keV cm^-2 s^-1 sr^-1 keV^-1).

The response is 2 * dead / (time_integ * geom_factor * gf * eff) for the central O2 mass channel, interpolated linearly against log10 energy. It is derived from a verified five-count response multiplied by 2/5. Corrected species DEF sums positive background-corrected contributions from mass bins 24 to 40. Calibration uses same-day STATIC C6 L2 and iv4 products. No extrapolation is applied.

`native_1050_spectrum_count2.csv` contains the native-energy calibration snapshot at 2018-08-05 10:49:58.66 UTC. The plot omits zero/negative DEF on logarithmic axes. Validity is strict DEF > finite positive two-count response. These observations are an illustrative model input, not independent evaluation labels.

## Full-day corrected C6 dataset

See [C6_EXAMPLE.md](../C6_EXAMPLE.md) for `static_c6_20180120_corrected.npz`, count definitions, calibration, units and reproduction commands. The region CSV is observational context only.
