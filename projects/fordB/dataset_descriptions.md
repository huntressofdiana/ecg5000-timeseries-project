# Dataset Description: FordB

## Dataset Title

FordB Noisy Engine-Noise Time-Series Classification Dataset

## Location

- **Dataset description and download:** https://www.timeseriesclassification.com/description.php?Dataset=FordB
- **Archive:** https://www.timeseriesclassification.com/

The dataset was originally used in a time-series classification competition associated with the 2008 IEEE World Congress on Computational Intelligence.

## Background & Motivation

FordB addresses the same general automotive diagnosis problem as FordA: determining whether a particular symptom is present in an automotive subsystem from engine-noise measurements.

The distinguishing characteristic of FordB is the difference between the training and test conditions. The training samples were collected under typical operating conditions, whereas the test samples contain considerably more noise.

FordB can therefore be used to investigate model robustness under distribution shift and measurement noise.

## Data Description

- Each sample contains 500 consecutive engine-noise measurements.
- Every sample is a univariate time series.
- The dataset contains two classes representing the presence or absence of an anonymized automotive symptom.
- The predefined split contains:

  - 3,636 training samples
  - 810 test samples

- Training samples were recorded under typical operating conditions.
- Test samples were recorded under noisier operating conditions.
- The predefined training and test sets should not normally be mixed because their different acquisition conditions are part of the benchmark.

Typical data shapes are:

- Training input: $$\(X_{\mathrm{train}} \in \mathbb{R}^{3636 \times 500 \times 1}\)$$
- Test input: $$\(X_{\mathrm{test}} \in \mathbb{R}^{810 \times 500 \times 1}\)$$
- Labels: $$\(y \in \{0,1\}^{N}\)$$

Depending on the downloaded format, the original labels may be encoded as \(-1\) and \(1\).

## Typical Modeling Task

- **Input:** 500 engine-noise measurements.
- **Target:** binary symptom classification.
- **Task type:** univariate binary time-series classification under noisy test conditions.
- **Possible models:** 1D-CNN, LSTM, GRU, Transformer, ROCKET, or noise-robust classifiers.

## Evaluation Metrics

Suitable evaluation metrics include:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix

Comparing FordA and FordB test performance can help demonstrate how measurement noise and distribution shift affect model generalization.
