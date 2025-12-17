# Machine Learning–Based Stock Selection and Trading Strategy

## Research Objective

This project analyzes whether simple machine-learning models can predict short-term equity returns in a liquid U.S. equity market. The focus is on assessing whether probability forecasts derived from technical indicators can be transformed into economically relevant portfolio strategies, and how these strategies compare to simple benchmarks such as momentum and equal-weight portfolios.

## Data and Predictors

The analysis uses daily price and volume data for current S&P 500 constituents, covering five years. Data are then aggregated to a monthly frequency for the analysis.

For each stock and month, the following features are constructed:

- Momentum (1-month, 3-month, 6-month)
- Realized volatility (3-month)
- Trend indicator (50-day moving average)
- RSI (14-day, sampled monthly)

The target variable is a binary indicator equal to 1 if the next-month return is positive.

## Modeling Strategy

The prediction task is framed as a cross-sectional binary classification problem. The following models are implemented and compared:

- Logistic Regression (with and without regularization)
- Random Forest
- Gradient Boosting
- XGBoost

Models are trained using an expanding walk-forward scheme, ensuring out-of-sample predictions at each month.

## Portfolio Construction and Evaluation

Predicted probabilities are used to rank stocks monthly and build equal-weighted portfolios consisting of the top-ranked fraction of stocks. Portfolios are rebalanced monthly and evaluated both in-sample and out-of-sample.

Performance is assessed using:

- ROC–AUC and Brier score (forecast quality)
- CAGR, volatility, Sharpe ratio, and maximum drawdown (economic performance)

Results are compared against:

- An equal-weight benchmark
- A momentum strategy

Transaction costs and survivorship bias are explicitly considered.

## Expected Contribution

The project aims to clarify the extent to which machine learning adds value beyond simple rules in equity markets, highlighting the gap between statistical predictability and economically meaningful performance.
