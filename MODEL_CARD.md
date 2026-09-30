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

Output tensor `[batch,3,time,32]`; labels 0 Noise, 1 Beam, 2 Low-energy ions. The raw API performs no postclassification screening. Training targets set former H+ contamination and invalid/subthreshold corrected O2+ pixels to Noise.

## Training and selection

234 inherited labeled events from 2015–2025, rebuilt with corrected C6. Date-grouped disjoint splits: 167 training, 32 validation, 35 test events. Seed 42. Five CPU epochs, AdamW learning rate 0.001, weight decay 0.0001, batch size 8. Weighted cross-entropy plus 0.5 times the mean physical-class soft Dice loss. Class weights are proportional to inverse square-root training pixel counts. Epoch 3 was selected by the validation macro F1 over Beam and Low-energy ions, and the held-out test set was evaluated after selection.

Labels derive from earlier expert rules and model-assisted annotations, with one reviewed beam correction, rather than a wholly new independent manual labeling campaign. Date grouping differs from the earlier v9 event split. Metrics must not be compared directly across versions.

## Held-out test metrics

All pixels, before production postprocessing:

| Class | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Noise | 0.9994 | 0.9966 | 0.9980 |
| Beam | 0.8378 | 0.9757 | 0.9015 |
| Low-energy ions | 0.9654 | 0.9911 | 0.9781 |

All-class macro F1: 0.959172; physical-class macro F1 (Beam and Low-energy ions): 0.939768; pixel accuracy: 0.996160. Accuracy restricted to visible O2+ pixels: 0.952043.

The large Noise fraction makes all-pixel accuracy insufficient on its own. Beam precision is about 83.8% relative to inherited labels, so false-positive signatures require inspection. These results are label-agreement estimates, not archive-wide uncertainty estimates.

## Inference and reproducibility

512-time × 32-energy windows, stride 256, 50% overlap. Short observations are zero-padded in normalized input space. Softmax probabilities are averaged across overlapping windows and cropped to original length before argmax. No confidence, connected-component, energy-range, or event-level filtering is included.

The inference checkpoint contains weights plus public metadata, without optimizer states, training-case paths, or training history. Its source and release SHA-256 hashes are recorded in `weights/model_metadata.json`. The included full-day 2018-01-20 example is a training-date illustration with unedited v10 predictions, not an independent validation sample.

## Limitations

Performance depends on correct background calibration, species extraction, alignment, and energy scaling. Out-of-distribution plasma populations, instrumental contamination, incomplete field of view, and weak reference labels can affect predictions. Training loss continued to fall after epoch 3 while validation loss increased. This release uses the selected checkpoint, not the final epoch. Existing v9 learning-curve figures do not describe this model.
