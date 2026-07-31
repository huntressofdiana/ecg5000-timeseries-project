# Dataset Description: DORA

## Dataset Title

Duffing Oscillator Response Analysis (DORA)

## Location

- **Dataset and download:** https://doi.org/10.5281/zenodo.14851014
- The data are provided as CSV files.

## Background & Motivation

DORA is a benchmark task based on an externally forced Duffing oscillator. The Duffing oscillator is a nonlinear dynamical system that can exhibit complex behavior such as:

- Multiple stable states
- Nonlinear resonance
- Bifurcations
- Period-doubling
- Periodic limit cycles
- Chaotic motion

The qualitative response of the oscillator changes when the amplitude of the external periodic forcing is varied.

The dataset can therefore be used to investigate whether a machine-learning model trained on only a small number of trajectories can generalize to unseen forcing parameters and previously unseen dynamical regimes.

## Governing Equation

The forced Duffing oscillator is described by:

$$ \frac{\mathrm{d}^2 q_1}{\mathrm{d}t^2} + c\frac{\mathrm{d}q_1}{\mathrm{d}t} + kq_1 + \beta q_1^3 = f\cos(\omega t+\phi) $$

where:

- $q_1(t)$ is the oscillator displacement
- $c$ is the damping coefficient
- $k$ is the linear stiffness coefficient
- $\beta$ is the nonlinear stiffness coefficient
- $f$ is the forcing amplitude
- $\omega$ is the forcing frequency
- $\phi$ is the phase shift

Introducing the velocity

$$
q_2(t)=\frac{\mathrm{d}q_1}{\mathrm{d}t}
$$

gives the first-order state-space representation:

$$ \frac{\mathrm{d}q_1}{\mathrm{d}t} = q_2 $$

$$ \frac{\mathrm{d}q_2}{\mathrm{d}t} = -cq_2-kq_1-\beta q_1^3 + f\cos(\omega t+\phi) $$

## Data Description

The dataset contains two primary files:

- `DORA_Train.csv`
- `DORA_Test.csv`

Each file contains five columns:

| Column | Description |
|---|---|
| `time` | Time $t$ |
| `qa(t)` | Oscillator displacement $q_1(t)$ |
| `qb(t)` | Oscillator velocity $q_2(t)$ |
| `f(t)` | Time-dependent external forcing |
| `f_amplitude` | Constant forcing amplitude for the trajectory |

The external forcing is given by:

$$
F(t)=f\cos(\omega t+\phi)
$$

The supplied data generator uses:

- Sampling interval: $\Delta t=0.1$
- Simulation duration: $T=250$
- Time steps per trajectory: 2,500
- Forcing frequency: $\omega=1.5$
- Phase shift: $\phi=0$
- Initial state:

$$
\mathbf{q}_0=
\begin{bmatrix}
0.05 \\
0.05
\end{bmatrix}
$$

## Training Data

The training dataset contains two trajectories generated using the forcing amplitudes:

$$
f \in \{0.46,\,0.49\}
$$

Each trajectory contains 2,500 time steps.

The complete training dataset therefore contains:

$$
2 \times 2500 = 5000
$$

time steps.

The training trajectories represent regular periodic dynamics and provide only a minimal representation of the complete nonlinear system behavior.

## Test Data

The test dataset contains trajectories generated using five forcing amplitudes:

$$
f \in \{0.20,\,0.35,\,0.48,\,0.58,\,0.75\}
$$

Each test trajectory contains 2,500 time steps.

The complete test dataset therefore contains:

$$
5 \times 2500 = 12500
$$

time steps.

The different forcing amplitudes produce qualitatively different system responses, including:

- Period-1 dynamics
- Period-2 dynamics
- Higher-order periodic dynamics, such as period-4
- Chaotic dynamics

## Typical Modeling Task

The main task is to learn the state evolution of the forced nonlinear oscillator from the two available training trajectories.

A one-step prediction problem can be formulated as:

$$
\begin{bmatrix}
q_1(t) \\
q_2(t) \\
F(t) \\
f
\end{bmatrix}
\longrightarrow
\begin{bmatrix}
\hat{q}_1(t+\Delta t) \\
\hat{q}_2(t+\Delta t)
\end{bmatrix}
$$

The input contains:

- The current displacement
- The current velocity
- The current external forcing
- The forcing amplitude

The target contains the oscillator state at the next time step.

## Autoregressive Prediction

During autoregressive inference, the predicted state is used as the input state for the following time step:

$$
\begin{bmatrix}
\hat{q}_1(t+\Delta t) \\
\hat{q}_2(t+\Delta t)
\end{bmatrix}
\longrightarrow
\begin{bmatrix}
\hat{q}_1(t+2\Delta t) \\
\hat{q}_2(t+2\Delta t)
\end{bmatrix}
$$

The known external forcing is supplied as an exogenous input at every prediction step.

The model can therefore generate a complete trajectory by repeatedly feeding its previous prediction back into the model.

## Goal

The goal is to train a model using only the two training trajectories and predict the response of the system for unseen forcing amplitudes.

A successful model should reproduce:

- Period-1 dynamics
- Period-2 dynamics
- Period-4 dynamics
- Chaotic motion
- Transitions between dynamical regimes
- The qualitative bifurcation behavior of the oscillator

The challenge is not only to minimize the pointwise prediction error but also to preserve the long-term qualitative dynamics of the nonlinear system.

## Physics-Informed Extension

As a project extension, a Physics-Informed Neural Network (PINN) can combine the available trajectory data with the governing Duffing oscillator equations.

A combined objective can be written as:

$$ \mathcal{L} = \mathcal{L}_{\mathrm{data}} + \lambda_{\mathrm{physics}} \mathcal{L}_{\mathrm{physics}} $$

where:

- $\mathcal{L}_{\mathrm{data}}$ measures the difference between predicted and reference trajectories.
- $\mathcal{L}_{\mathrm{physics}}$ penalizes violations of the Duffing oscillator equations.
- $\lambda_{\mathrm{physics}}$ controls the contribution of the physical constraint.

For example, the physics residuals can be defined as:

$$ r_1(t) = \frac{\mathrm{d}\hat{q}_1}{\mathrm{d}t} - \hat{q}_2 $$

and

$$ r_2(t) = \frac{\mathrm{d}\hat{q}_2}{\mathrm{d}t} + c\hat{q}_2 + k\hat{q}_1 + \beta\hat{q}_1^3 - f\cos(\omega t+\phi) $$

The physics loss can then be calculated as:

$$ \mathcal{L}_{\mathrm{physics}} = \frac{1}{N} \sum_{i=1}^{N} \left(r_1(t_i)^2+r_2(t_i)^2 \right) $$

The project can investigate whether the physical constraint improves generalization to unseen forcing amplitudes and unseen dynamical regimes.

## Evaluation Metrics

Standard pointwise trajectory metrics include:

- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Relative error

However, a low pointwise error does not necessarily mean that the predicted system exhibits the correct long-term dynamics.

Additional dynamical evaluation measures can include:

- Maximum steady-state vibration amplitude
- Mean steady-state vibration amplitude
- Oscillation periodicity
- Phase-space trajectory
- Frequency spectrum
- Entropy
- Location of bifurcation points
- Qualitative identification of periodic and chaotic regimes

The benchmark characterizes the system response using the maximum and mean squared displacement after an initial transient period.

The maximum squared displacement is:

$$ A_{\max} = \max_{t>t^*} q_1^2(t) $$

The mean squared displacement is:

$$ A_{\mathrm{mean}} = \operatorname{mean}_{t>t^*} q_1^2(t) $$

where the transient boundary is set to:

$$
t^*=20
$$

These quantities can be compared between the predicted and reference trajectories to determine whether the model reproduces the correct long-term response.
