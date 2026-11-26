# Full-day STATIC C6 identification example

This example uses MAVEN STATIC C6 observations on **20 January 2018 (UTC)**. It supplies the corrected DEF and count arrays needed to inspect the data and reproduce identification with **v12_count2**, a 512 × 32 window and inference stride **51**.

## Files and reproduction

- [Corrected C6 data and counts](data/static_c6_20180120_corrected.npz), approximately 23.6 MB.
- [Figure PNG](../docs/static_c6_20180120_example.png) and [editable-text PDF](../docs/static_c6_20180120_example.pdf).
- [Predictions](../docs/static_c6_20180120_predictions.npz) and [model/method metadata](../docs/static_c6_20180120_method.json).
- [Plot and inference script](plot_c6_full_day.py).
- [Optional raw-CDF export script](export_c6_example.py).

## Installation from a fresh download

Use **Python 3.10 or later** with a PyTorch wheel available for your platform. Python 3.11 is the tested version. This example runs on CPU; CUDA, Conda, raw CDF files and personal helper libraries are not required. Internet access is needed to download the repository and install dependencies; inference runs offline afterward.

### 1. Download the complete repository

```bash
git clone https://github.com/ChiZhang123456/U-Net_Mars_O2_Beams.git
cd U-Net_Mars_O2_Beams
```

Alternatively, choose **Code → Download ZIP** on GitHub, extract the ZIP, and open a terminal inside the extracted repository folder (the folder containing `requirements.txt`). Do not download only the example script: the included data, weights, CSV and package folder must remain together. Git LFS is not needed.

### 2. Create and activate an isolated environment

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell does not permit activation, use `.\.venv\Scripts\python.exe` instead of `python` in every command below. Activation is optional; no execution-policy change is needed.

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Linux, install your distribution's Python venv support if `python3 -m venv` reports that it is missing.

### 3. Install the required packages

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` includes NumPy, PyTorch and Matplotlib. The C6 example uses Python's standard-library CSV reader and does not require pandas, SpacePy or `py_space_zc`. No editable package installation or `PYTHONPATH` setting is needed: the example resolves the bundled package, data and checkpoint relative to its own file.

### 4. Run the full-day example

```bash
python examples/plot_c6_full_day.py
```

The script runs the released model on CPU and writes these files under the downloaded repository's `docs/` folder:

- `static_c6_20180120_example.png`
- `static_c6_20180120_example.pdf`
- `static_c6_20180120_predictions.npz`
- `static_c6_20180120_method.json`

Existing files with these names are overwritten when the example is rerun. Completion prints the model version (`v12_count2`), inference stride (`51`), checkpoint hash and class counts. The output probabilities have shape `[3,21600,32]`. No GUI window is required, because plotting uses the noninteractive Agg backend. Runtime depends on CPU speed. The supplied NPZ contains the observations and counts; inference only loads the small arrays it needs, not the full native cubes.

### Troubleshooting

- `ModuleNotFoundError`: install requirements with the same environment's Python that runs the script. `python -m pip --version` shows which environment pip is using.
- Missing data or weights: extract/download the complete repository and keep its directory structure intact.
- Font warnings: the figure requests Arial. Install Arial if you need identical typography; on systems without Arial, Matplotlib uses an available fallback font. This does not affect inference.
- PyTorch installation failure: use a Python version supported by PyTorch on your platform (3.11 was tested), then repeat the installation step.

The included NPZ is sufficient to run the example; raw CDF access and SpacePy are only needed for the optional export described below.

## Input data dictionary

All arrays use increasing UTC timestamps. There are **21,600 time samples**, **64 native mass channels**, and **32 native energy bins**. Native energy depends on mass and sweep. The model uses a separate fixed 32-bin grid from 0.2 eV to 30 keV.

| Key | Shape | Meaning / unit |
|---|---|---|
| `time` | `[21600]` | UTC `datetime64[ns]` timestamps |
| `energy_ev` | `[32]` | Fixed model energy grid, eV |
| `h_def` | `[21600,32]` | Corrected H+ species DEF on fixed grid, context only |
| `o2_def` | `[21600,32]` | Corrected O2+ species DEF on fixed grid, classifier input |
| `two_count_def` | `[21600,32]` | Calibrated central O2 mass-channel two-count DEF response |
| `c6_raw_counts` | `[21600,64,32]` | Observed C6 counts before background subtraction |
| `c6_net_counts` | `[21600,64,32]` | Observed counts minus time-interpolated iv4 background; negative values retained |
| `c6_def_corrected` | `[21600,64,32]` | Background-corrected native DEF, negative/nonfinite responses set to zero |
| `c6_mass_amu` | `[64]` | Mean mass-channel centers, amu |
| `c6_energy_table_ev` | `[64,32,27]` | Native energy lookup table, mass × energy × sweep, eV |
| `c6_sweep_index` | `[21600]` | Zero-based sweep index for each record |
| `c6_deadtime_factor` | `[21600,64,32]` | Dead-time correction factor |
| `central_o2_mass_index` | scalar | Zero-based mass channel nearest 32 amu |

DEF units are keV cm^-2 s^-1 sr^-1 keV^-1, numerically identical to eV cm^-2 s^-1 sr^-1 eV^-1. `h_def`, `o2_def` and `two_count_def` use these same units. Counts are counts per accumulation, not count rates, DEF values, or integer class labels.

Native energy for record `i` and mass channel `m` is `c6_energy_table_ev[m, :, c6_sweep_index[i]]`. The species spectra use the central O2 channel's energy coordinate, matching the trained model's preprocessing convention. Arrays are ordered `[time,mass,energy]` in this export.

## Background correction and validity

The source products are `mvn_sta_l2_c6-32e64m_20180120_v02.cdf` and `mvn_sta_l2_c6-32e64m_20180120_iv4.cdf`. The export implements the C6/iv4 background correction explicitly:

```text
net_counts = raw_counts - interpolated_iv4_background
corrected_DEF = net_counts * dead / (time_integ * geom_factor * gf * eff)
```

Calibration factors follow attenuator, sweep and efficiency indices and are interpolated to observation time. Nonpositive/nonfinite corrected DEF is set to zero per native mass/energy channel before summing species contributions. O2+ uses mean mass centers 24 to 40 amu; H+ uses 0 to 1.55 amu. DEF is linearly interpolated against log10 energy onto the fixed grid; outside coverage is NaN. Do not apply background correction again to the supplied arrays.

The count mask is based on the response for two counts in the **single central O2 mass channel**, compared with the **summed O2+ species DEF**:

```text
two_count_DEF_native = 2 * dead_center / (time_integ * geom_factor * gf * eff_center)
valid = finite(o2_def) & finite(two_count_def) & (two_count_def > 0) & (o2_def > two_count_def)
```

Equality is invalid. This is not a threshold of two on the summed raw/net counts. Validity is independent of the log-DEF clipping floor. Native net counts retain negative values so users can inspect the subtraction directly.

## Model outputs and figure

Panel a shows corrected C6 H+ DEF, panel b corrected C6 O2+ DEF, and panel c the current model classification. H+ and the plasma-region strip provide context and are not classifier channels. The supplied region CSV records preselected observational context; no region labels are inferred by U-Net.

`probabilities` has shape `[3,21600,32]` and contains mean overlapping softmax probabilities. `raw_labels` is their argmax with shape `[21600,32]`. Class IDs are 0 Noise, 1 Beam, 2 Low-energy ions. The displayed `labels` sets invalid count-mask pixels to Noise; `validity` stores that mask. This display step is explicit and does not modify `raw_labels`. There are no manual class edits or interval-merging steps in this example. As the inference stride is 51, the figure can differ from earlier renders made with other strides.

## Optional export from raw CDFs

Install SpacePy separately if re-exporting raw observations:

```bash
python -m pip install spacepy
python examples/export_c6_example.py --l2 /path/to/c6_l2.cdf --iv4 /path/to/c6_iv4.cdf --output examples/data/static_c6_20180120_corrected.npz
```

The export script is C6-specific. It does not download files. The included full day has increasing timestamps and approximately four-second sampling. For other data, split separate UTC days and instrument outages before inference.

## Validation and limits

The export was checked against the existing calibrated-response implementation at 51 records distributed over the day, covering both corrected O2+ DEF and two-count thresholds. Native cube shapes, timestamp ordering, output probability sums and count-mask display consistency were checked. The example is an illustration, not an independent human-labeled test set. Calibration, species overlap, incomplete instrument coverage and inherited training labels limit scientific interpretation. Original observational-data rights and attribution requirements remain applicable.


### Clean-environment execution check

On 26 November 2026, the complete tracked repository was exported to a separate directory and run in a new Python 3.11.14 virtual environment with `include-system-site-packages = false`. Only `requirements.txt` was installed. The resolved main packages were NumPy 2.4.6, PyTorch 2.15.0+cpu and Matplotlib 3.11.2. `py_space_zc`, SpacePy and pandas were absent.

The C6 script completed from a working directory outside the repository using Python isolated mode (`-I`), with no editable install or custom `PYTHONPATH`. It regenerated all four output files. Probability normalization, argmax labels and validity-mask application passed validation; displayed labels exactly matched the bundled example, with class counts `[660688, 8116, 22396]`. `python -m pip check` reported no broken requirements. This execution check was performed on Windows; macOS/Linux instructions above have not been execution-tested in this check.
