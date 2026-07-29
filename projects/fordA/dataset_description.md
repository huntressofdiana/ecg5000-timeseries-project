# Dataset Description: FordA

## Dataset Title

FordA Engine-Noise Time-Series Classification Dataset

## Location

- **Dataset description and download:** https://www.timeseriesclassification.com/description.php?Dataset=FordA
- **Archive:** https://www.timeseriesclassification.com/

The dataset was originally used in a time-series classification competition associated with the 2008 IEEE World Congress on Computational Intelligence.

## Background & Motivation

FordA is an automotive time-series classification benchmark. The task is to determine whether a particular symptom is present in an automotive subsystem using a sequence of engine-noise measurements.

The detailed physical meaning of the symptom and the exact subsystem are anonymized.

## Data Description

- Each sample contains 500 consecutive engine-noise measurements.
- Every sample is a univariate time series.
- The dataset contains two classes representing the presence or absence of the investigated symptom.
- The predefined split contains:

  - 3,601 training samples
  - 1,320 test samples

- Training and test measurements were recorded under typical operating conditions with relatively little noise contamination.
- The samples are provided with predefined class labels.

Typical data shapes are:

- Training input: \(X_{\mathrm{train}} \in \mathbb{R}^{3601 \times 500 \times 1}\)
- Test input: \(X_{\mathrm{test}} \in \mathbb{R}^{1320 \times 500 \times 1}\)
- Labels: \(y \in \{0,1\}^{N}\)

Depending on the downloaded format, the original labels may be encoded as \(-1\) and \(1\).

## Typical Modeling Task

- **Input:** 500 engine-noise measurements.
- **Target:** binary symptom classification.
- **Task type:** univariate binary time-series classification.
- **Possible models:** 1D-CNN, LSTM, GRU, Transformer, ROCKET, or MiniROCKET.

## Evaluation Metrics

Suitable evaluation metrics include:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix