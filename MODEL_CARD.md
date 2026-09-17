# Model card: MAVEN STATIC O2+ U-Net v9

## Intended use

This model performs pixel-wise semantic segmentation of aligned MAVEN STATIC H+, O+, and O2+ differential energy flux spectrograms. Its primary scientific purpose is to identify O2+ Energetic beam pixels for statistical studies of the solar wind interaction with Mars.

The model is a research product. Predictions should be visually inspected and interpreted together with instrument coverage, background contamination, spacecraft location, magnetic-field measurements, and the physical context of each interval.

## Architecture

U-Net v9 is a compact three-level two-dimensional U-Net with group normalization. It contains 273,124 trainable parameters. The base encoder width is 12 channels. The model accepts nine input channels and produces four pixel classes.

## Input channels

1. Scaled log10 O2+ differential energy flux
2. Scaled log10 H+ differential energy flux
3. Scaled log10 O+ differential energy flux
4. Scaled log10 H+/O2+ flux ratio
5. Scaled log10 O+/O2+ flux ratio
6. O2+ validity mask
7. H+ validity mask
8. O+ validity mask
9. Absolute logarithmic energy coordinate from 0.2 eV to 30 keV

DEF values are clipped in log10 space from 4 to 8 and mapped to [0, 1]. Logarithmic ion ratios are clipped from -3 to 3 and mapped to [0, 1].

## Output classes

The output class order stored in the checkpoint is:

0. H+ contamination
1. O2+ Energetic beam
2. Cold ions
3. Other or uncertain

## Training and evaluation

The labeled data set contains 234 complete events from 2015 through 2025. Complete events, rather than individual pixels or windows, were assigned to the training and test sets to reduce information leakage. The split seed is 42, with 187 training events and 47 test events.

The released checkpoint was selected at epoch 3 by the maximum test selection score. At that epoch:

| Metric | Value |
| --- | ---: |
| Test macro F1 | 0.9789 |
| Test macro IoU | 0.9609 |
| Test pixel accuracy | 0.9706 |
| Test loss | 0.1755 |

These values describe performance on the labeled test events and should not be interpreted as an uncertainty estimate for the complete MAVEN archive.

## Inference configuration

Long observations are divided into 512-sample time windows with a stride of 51 samples. Softmax probabilities from overlapping windows are averaged before assigning the final class. The actual overlap fraction is 0.900390625.

## Limitations

Performance can degrade when the input grids, calibration, visibility masks, background correction, or DEF scaling differ from the training pipeline. The network may also confuse physical O2+ populations with proton contamination, incomplete field-of-view sampling, instrumental background, or plasma regimes underrepresented in the labeled data.
