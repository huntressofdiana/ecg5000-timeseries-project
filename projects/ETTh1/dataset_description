# Dataset Description: ETTh1

## Dataset Title

ETTh1 Electricity Transformer Temperature Dataset

## Location

- **Source repository:** https://github.com/zhouhaoyi/ETDataset
- **Direct dataset file:** https://github.com/zhouhaoyi/ETDataset/blob/main/ETT-small/ETTh1.csv

The dataset was introduced for research on long-sequence time-series forecasting and was used in the Informer forecasting benchmark.

## Background & Motivation

ETTh1 contains measurements from an electricity transformer station. The dataset supports forecasting of transformer oil temperature and electrical load variables.

Accurate oil-temperature forecasting can support transformer monitoring, load management, predictive maintenance, and the prevention of overheating or unnecessary equipment degradation.

## Data Description

- ETTh1 is one continuous multivariate time series.
- The data were recorded at an hourly sampling interval.
- The dataset covers approximately two years, from 2016 to 2018.
- It contains 17,420 hourly observations.
- The data are stored in a CSV file.
- Each row contains a timestamp and seven numerical variables.

The columns are:

| Column | Description |
|---|---|
| `date` | Measurement timestamp |
| `HUFL` | High Useful Load |
| `HULL` | High Useless Load |
| `MUFL` | Middle Useful Load |
| `MULL` | Middle Useless Load |
| `LUFL` | Low Useful Load |
| `LULL` | Low Useless Load |
| `OT` | Oil Temperature |

The `OT` column is commonly treated as the primary forecasting target.

Typical data shapes are:

- Complete numerical data: \(X \in \mathbb{R}^{17420 \times 7}\)
- Lookback window: \(X_{t-L+1:t} \in \mathbb{R}^{L \times 7}\)
- Univariate target: \(\hat{y}_{t+1:t+H} \in \mathbb{R}^{H}\)
- Multivariate target: \(\hat{Y}_{t+1:t+H} \in \mathbb{R}^{H \times 7}\)

## Typical Dataset Split

A commonly used chronological benchmark split is:

- First 12 months: training
- Following 4 months: validation
- Following 4 months: testing

The normalization parameters should be calculated only from the training region and subsequently applied to the validation and test regions.

## Typical Modeling Tasks

### Univariate Forecasting

- **Input:** previous values of the oil-temperature signal.
- **Target:** future values of `OT`.

### Multivariate-to-Univariate Forecasting

- **Input:** previous values of all seven variables.
- **Target:** future values of `OT`.

### Multivariate Forecasting

- **Input:** previous values of all seven variables.
- **Target:** future values of all seven variables.

Common benchmark forecast horizons include:

- 96 hours
- 192 hours
- 336 hours
- 720 hours

## Evaluation Metrics

Suitable forecasting metrics include:

- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Coefficient of determination (\(R^2\))