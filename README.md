# Double-Moving-Average-Crossover-ML

## Overview
This project aims to predict the future price movement of the SPDR S&P 500 ETF Trust (SPY) using both a traditional Double Moving Average trading strategy and an XGBoost machine learning model. The goal is to compare the performance of these two approaches against a simple buy-and-hold benchmark.

## Data Acquisition
Historical daily stock data for SPY is fetched using the `yfinance` library, covering a period from 2005-01-01 to the present.

```python
data = yf.download("SPY", start="2005-01-01", auto_adjust=False)
```

## Double Moving Average Strategy
A classic technical analysis strategy is implemented, based on the crossover of two Simple Moving Averages (SMAs):
- **Fast SMA**: 20-day Simple Moving Average
- **Slow SMA**: 50-day Simple Moving Average

A `signal` is generated: `1` (long position) when the fast SMA crosses above the slow SMA, and `0` (flat position) otherwise. The `position` column is a shifted version of the signal, accounting for trade execution on the next day. Transaction costs are also factored into the strategy returns.

## Feature Engineering for Machine Learning
Several technical indicators and price-based features are engineered from the historical data to be used as inputs for the machine learning model:
- **`sma_fast`**: 20-day Simple Moving Average
- **`sma_slow`**: 50-day Simple Moving Average
- **`volatility_20d`**: 20-day rolling standard deviation of returns
- **`rsi`**: 14-period Relative Strength Index
- **`sma_ratio`**: Ratio of `sma_fast` to `sma_slow`
- **`price_sma20`**: Price relative to `sma_fast`
- **`price_sma50`**: Price relative to `sma_slow`
- **`ret_5d`**: 5-day percentage change in 'Adj Close'
- **`ret_20d`**: 20-day percentage change in 'Adj Close'
- **`volume_ratio`**: Current volume relative to its 20-day rolling mean

## Target Variable
The project explores two types of target variables:
1. **Regression Target (`target`)**: The 5-day forward percentage change in 'Adj Close' (`data['Adj Close'].shift(-5) / data['Adj Close'] - 1`). This is used for predicting the magnitude of future price movement.
2. **Classification Target (`target_class`)**: A binary variable indicating whether the next day's return is positive or not (`(data['ret'].shift(-1) > 0).astype(int)`). This is used for predicting the direction of future price movement.

## XGBoost Machine Learning Model
An XGBoost Regressor and Classifier are implemented to predict the target variables. Time Series Split cross-validation (`TimeSeriesSplit`) is used to ensure that the model is evaluated on future data, mimicking real-world trading conditions. Early stopping is employed to prevent overfitting.

**Model Parameters:**
- `objective`: `reg:pseudohubererror` for regression, `binary:logistic` for classification
- `n_estimators`: 2000
- `learning_rate`: 0.02
- `max_depth`: 4
- `subsample`: 0.8
- `colsample_bytree`: 0.8
- `early_stopping_rounds`: 100

## Performance Metrics
The following metrics are calculated to evaluate the performance of the strategies and models:
- **Equity Curve**: Visualization of capital growth over time for both the Double MA Strategy and a Buy & Hold benchmark.
- **Sharpe Ratio**: Return per unit of risk (volatility).
- **Max Drawdown**: The largest percentage drop from a peak to a trough.
- **CAGR (Compound Annual Growth Rate)**: Annualized rate of return.
- **Information Coefficient (IC)**: Spearman's rank correlation between predictions and actual returns (for regression model).
- **Accuracy (`acc`)**: Percentage of correctly predicted directions (for regression model, by comparing signs of `pred` and `y_val`).
- **Logloss**: Evaluation metric for the classification model.

## How to Run
1. Ensure you have Google Colab environment set up.
2. Run all cells sequentially. The notebook will automatically download data, calculate indicators, implement the trading strategy, train the machine learning models, and display the results and plots.
