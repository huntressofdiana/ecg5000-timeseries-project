# Projects

This folder contains the modeling projects and datasets for the summer school.

## Overview

| Folder | Domain | Task |
|--------|--------|------|
| `project_template/` | Template | Copy to start a new project |
| `CMAPSS/` | Aerospace / Predictive Maintenance | RUL regression on turbofan engine sensor data |
| `dora/` | Nonlinear Dynamics | Autoregressive state prediction of a forced Duffing oscillator |
| `ecg5000/` | Medical / Cardiology | 5-class heartbeat classification from ECG signals |
| `ETTh1/` | Energy / Forecasting | Univariate and multivariate transformer oil-temperature forecasting |
| `fordA/` | Automotive / Diagnostics | Binary engine-noise classification (clean train & test) |
| `fordB/` | Automotive / Diagnostics | Binary engine-noise classification (noisy test set; distribution shift) |
| `sensor_reconstruction/` | Mechanical Engineering / Braking | Seq2seq regression for soft-sensing of biased thermocouple signals |

Each project folder contains a `dataset_description.md` with background information, data format, typical modeling tasks, and evaluation metrics.

## Working on a Project

1. Navigate into the relevant project folder.
2. Read `dataset_description.md` (and `instructions.md` if present) carefully before starting.
3. Work in your own copy of the folder (or a personal branch/fork).
4. Submit your results as specified in the project instructions.

## Adding a New Project

Copy `project_template/`, rename it to `project_<name>/`, and fill in `instructions.md`.
