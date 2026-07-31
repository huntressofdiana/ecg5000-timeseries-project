# Dataset Description: Thermocouple Grid Sensor Reconstruction Data

## Dataset Title

Thermocouple Grid Dataset for Data-Driven Sensor Signal Reconstruction in Brake Tribometer Experiments

## Location

- **Dataset / download link:** https://doi.org/10.5281/zenodo.21043250
- Generated from experimental pin-on-disc brake-tribometer measurements.

## Background & Motivation

Temperature measurements are essential for analyzing the thermo-mechanical behavior of braking systems. However, sensor signals can be unreliable because of incorrect sensor placement, systematic measurement bias, sensor failure, or missing observations.

The dataset supports the development of data-driven **soft-sensing** methods that reconstruct unreliable temperature signals from correlated measurements recorded by neighboring sensors.

## Data Description

- The measurements originate from a thermocouple grid embedded in the pin of a pin-on-disc brake tribometer.
- The grid contains 12 pin thermocouples arranged spatially across the pin.
- Nine thermocouples provide correctly positioned reference measurements.
- Three thermocouples were positioned at an unintended depth and consequently exhibit systematically biased temperature signals.
- Each sample represents one braking event followed by a cooling phase.
- The signals are sampled at 10 Hz.
- The dataset contains 211 retained braking events.
- Sequence lengths vary slightly between approximately 510 and 532 time steps.
- Sequences can be padded to a common maximum length and accompanied by a padding mask.
- The data are distributed in NumPy-compatible format.

- **Input:** nine correctly positioned thermocouple signals.
- **Target:** three systematically biased thermocouple signals.
- Typical sample shapes are:

  - Input: $$\(X \in \mathbb{R}^{T \times 9}\)$$
  - Target: $$\(Y \in \mathbb{R}^{T \times 3}\)$$

## Typical Modeling Tasks

- **Input:** multivariate temperature signals from correctly positioned sensors.
- **Target:** complete temperature signals of the biased sensors.
- **Task type:** multivariate sequence-to-sequence regression.
- **Possible models:** MLP, LSTM, GRU, Transformer, spatial interpolation, or neural operators.

## Evaluation Metrics

Suitable evaluation metrics include:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Coefficient of determination (\(R^2\))
- Maximum Absolute Error (MaxAE)
