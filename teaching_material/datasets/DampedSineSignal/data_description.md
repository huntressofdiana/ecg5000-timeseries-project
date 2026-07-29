# Dataset Description: Exponentially Decaying Sine Signal

## Dataset Title

Synthetic Exponentially Decaying Sine-Wave Forecasting Dataset

## Location

- **Dataset type:** synthetically generated time series
- **Storage format:** NumPy array, for example `.npy` or `.npz`
- The signal is generated directly in Python using NumPy.
- The dataset contains one continuous univariate time series.

## Background & Motivation

The exponentially decaying sine signal is a simple benchmark for introducing time-series forecasting with recurrent neural networks such as LSTMs.

The signal combines two basic temporal characteristics:

- Periodic oscillation
- Gradually decreasing amplitude

The periodic component requires the model to learn the oscillation frequency and phase, while the exponential term requires it to learn how the amplitude changes over time.

## Signal Definition

The signal is defined as:

$$
x(t)
=
A e^{-\lambda t}
\sin\left(2\pi f t\right)
$$

where:

- $x(t)$ is the signal value at time $t$
- $A$ is the initial amplitude
- $\lambda$ is the exponential decay rate
- $f$ is the oscillation frequency
- The phase shift is set to zero

The exponential envelope is:

$$
E(t)=A e^{-\lambda t}
$$

The signal therefore remains within the bounds:

$$
-Ae^{-\lambda t}
\leq
x(t)
\leq
Ae^{-\lambda t}
$$

## Signal Parameters

The example dataset uses the following parameters:

| Parameter | Symbol | Value |
|---|---:|---:|
| Initial amplitude | $A$ | 1.0 |
| Decay rate | $\lambda$ | 0.05 |
| Frequency | $f$ | 0.5 Hz |
| Phase shift | $\phi$ | 0 |
| Start time | $t_0$ | 0 s |
| End time | $t_{\mathrm{end}}$ | 20 s |
| Number of time steps | $N$ | 1,000 |

The corresponding Python implementation is:

```python
time = np.linspace(
    0.0,
    20.0,
    1000,
    dtype=np.float32,
)

signal = (
    1.0
    * np.exp(-0.05 * time)
    * np.sin(2.0 * np.pi * 0.5 * time)
)