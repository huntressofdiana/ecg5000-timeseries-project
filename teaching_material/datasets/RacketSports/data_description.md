# Dataset Description: RacketSports

## Dataset Title

RacketSports Multivariate Human-Activity Recognition Dataset

## Location

- **Dataset description and download:** https://www.timeseriesclassification.com/description.php?Dataset=RacketSports
- **Archive:** https://www.timeseriesclassification.com/
- The dataset is distributed through the UEA Multivariate Time Series Classification Archive.
- The data are available in formats such as ARFF and TS files.

## Background & Motivation

RacketSports is a multivariate human-activity recognition dataset created from motion-sensor measurements recorded while participants played badminton or squash.

The objective is to identify both:

- The racket sport being performed
- The type of stroke performed by the player

The dataset is suitable for demonstrating multivariate time-series classification using wearable inertial sensors.

## Data Collection

University students performed different badminton and squash strokes while wearing a smartwatch.

The smartwatch recorded:

- Three-axis acceleration
- Three-axis angular velocity

The measurements were transmitted to an Android smartphone and stored as time-series data.

Each movement was recorded for three seconds at a sampling frequency of:

$$
f_s = 10\ \mathrm{Hz}
$$

Each sample therefore contains:

$$
3\ \mathrm{s} \times 10\ \mathrm{Hz}
=
30
$$

time steps.

## Data Description

RacketSports contains:

| Property | Value |
|---|---:|
| Training samples | 151 |
| Test samples | 152 |
| Time steps per sample | 30 |
| Sensor channels | 6 |
| Classes | 4 |
| Sampling frequency | 10 Hz |
| Recording duration | 3 seconds |

Each sample is a multivariate time series with the shape:

$$
X^{(i)} \in \mathbb{R}^{30 \times 6}
$$

The complete training data have the shape:

$$
X_{\mathrm{train}}
\in
\mathbb{R}^{151 \times 30 \times 6}
$$

The complete test data have the shape:

$$
X_{\mathrm{test}}
\in
\mathbb{R}^{152 \times 30 \times 6}
$$

The dimensions correspond to:

$$
\text{samples}
\times
\text{time steps}
\times
\text{sensor channels}
$$

## Sensor Channels

The six sensor channels are ordered as follows:

| Channel | Measurement |
|---:|---|
| 1 | Accelerometer x-axis |
| 2 | Accelerometer y-axis |
| 3 | Accelerometer z-axis |
| 4 | Gyroscope x-axis |
| 5 | Gyroscope y-axis |
| 6 | Gyroscope z-axis |

The first three channels describe linear acceleration, while the final three channels describe angular velocity.

## Classes

The dataset contains four combinations of sport and stroke:

1. Badminton clear
2. Badminton smash
3. Squash forehand
4. Squash backhand

Each sample belongs to exactly one of these four classes.

The target can therefore be represented as:

$$
y^{(i)}
\in
\{1,2,3,4\}
$$

Alternatively, the labels can be one-hot encoded:

$$
\mathbf{y}^{(i)}
\in
\{0,1\}^{4}
$$

For example:

$$
\mathbf{y}^{(i)}
=
\begin{bmatrix}
0 & 1 & 0 & 0
\end{bmatrix}
$$

indicates that the sample belongs to the second class.

## Typical Modeling Task

The task is multivariate time-series classification.

### Input

A sequence of accelerometer and gyroscope measurements:

$$
X^{(i)}
=
\begin{bmatrix}
\mathbf{x}_1 \\
\mathbf{x}_2 \\
\vdots \\
\mathbf{x}_{30}
\end{bmatrix}
\in
\mathbb{R}^{30 \times 6}
$$

where each time step contains:

$$
\mathbf{x}_t
=
\begin{bmatrix}
a_x(t) &
a_y(t) &
a_z(t) &
g_x(t) &
g_y(t) &
g_z(t)
\end{bmatrix}
$$

### Target

One of four sport-and-stroke classes:

$$
X^{(i)}
\longrightarrow
\hat{\mathbf{y}}^{(i)}
$$

where:

$$
\hat{\mathbf{y}}^{(i)}
\in
\mathbb{R}^{4}
$$

contains the model logits or predicted class scores.

The predicted class is:

$$
\hat{c}^{(i)}
=
\operatorname*{arg\,max}_{k}
\hat{y}^{(i)}_k
$$

## Preprocessing

Possible preprocessing steps include:

- Standardization of each sensor channel
- Conversion of class names to integer or one-hot labels
- Preservation of the original training and test split
- Optional data augmentation for the small training dataset



