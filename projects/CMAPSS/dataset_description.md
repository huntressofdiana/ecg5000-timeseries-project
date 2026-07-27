# Dataset Description: C-MAPSS Jet Engine Simulated Data

## Dataset Title

NASA C-MAPSS Jet Engine Simulated Data (Turbofan Engine Degradation Simulation)

## Location

- **Source / download link:** https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data
- Generated using NASA's Commercial Modular Aero-Propulsion System Simulation (C-MAPSS); maintained by NASA (Chris Teubert) as part of the NASA Prognostics Data Repository.
- Distributed as a ZIP-compressed archive of plain-text files.

## Background & Motivation

The dataset simulates run-to-failure degradation trajectories of a fleet of turbofan jet engines. It is a standard benchmark for **prognostics and health management (PHM)**, specifically for estimating the **Remaining Useful Life (RUL)** of a machine from multivariate sensor time series, which is a task directly relevant to predictive maintenance.

## Data Description

- Each engine unit starts in a healthy state and develops a fault at some point during its run; the engine continues operating until failure.
- Each row corresponds to one operational cycle of one engine and contains 26 space-separated columns:
  - Unit (engine) number
  - Cycle (time) index
  - 3 operational settings (e.g., altitude, Mach number, throttle resolver angle)
  - 21 sensor measurements (e.g., temperatures, pressures, fan/core speeds)
- Sensor readings include realistic sensor noise and normal manufacturing variation between units.
- The data is split into **four sub-datasets (FD001–FD004)**, which vary in operating conditions and fault modes:

| Sub-dataset | # Train / # Test units | Operating conditions | Fault mode(s) |
|---|---|---|---|
| FD001 | 100 / 100 | 1 (sea level) | HPC degradation |
| FD002 | 260 / 259 | 6 | HPC degradation |
| FD003 | 100 / 100 | 1 (sea level) | HPC + Fan degradation |
| FD004 | 248 / 249 | 6 | HPC + Fan degradation |

- **Training sets:** full run-to-failure trajectories (last cycle = failure).
- **Test sets:** trajectories truncated at some point before failure; a separate ground-truth file gives the true RUL (in cycles) remaining at the point of truncation, used for evaluation.

## Typical Modeling Task

- **Input:** windows of the multivariate sensor/operational-setting time series for one engine unit.
- **Target:** Remaining Useful Life (RUL), in cycles, at the end of the input window.
- **Model type:** sequence models such as LSTM/GRU, 1D-CNN, or Transformer-based architectures are commonly used.
