# Column Definitions - Processed Finance Dataset

This file documents the columns available in `processed_finance_data.csv` used for time-series forecasting of TSLA stock prices and related financial indicators. It is part of the DSS5104-ForecastDL project.

---

## Common Columns

### `Date` (datetime64[ns])
The timestamp for the record.

### `Close_TSLA` (float64)
Adjusted closing price of Tesla stock. This is the main target variable.

### `Volume_TSLA` (int64)
Trading volume for TSLA stock on that day.

---

## Time Features (Engineered from `Date`)
Help capture seasonal patterns and temporal cycles:

- `TSLA_day`: Day of the month (1–31)
- `TSLA_month`: Month (1–12)
- `TSLA_weekday`: Day of week (Monday=0, Sunday=6)
- `day_sin`, `day_cos`: Sine and cosine transformation for day of month
- `month_sin`, `month_cos`: Sine and cosine transformation for month
- `weekday_sin`, `weekday_cos`: Sine and cosine transformation for weekday

---

## Lag Features (Past Close Prices)
Lagged values of `Close_TSLA` to capture temporal dependencies:

- `Close_TSLA_Lag_1`, `Lag_2`, `Lag_3`, `Lag_4`, `Lag_5`
- `Lag_10`, `Lag_20`, `Lag_30`, `Lag_60`, `Lag_90`, `Lag_120`

---

## Return Features

### Daily and Rolling Returns:
- `TSLA_Returns`: Log return between two consecutive days
- `TSLA_Daily_Return`: Daily % change in closing price
- `TSLA_Total_Return_5d`, `10d`, `20d`, `60d`, `90d`, `120d`, `252d`: Total cumulative returns over N trading days

---

## Technical Indicators

### Momentum Indicators:
- `TSLA_RSI`: Relative Strength Index
- `TSLA_MACD`: Moving Average Convergence Divergence
- `TSLA_MACD_Signal`: MACD signal line

### Trend Indicators:
- `TSLA_EMA_20`, `TSLA_EMA_40`: Exponential Moving Averages
- `TSLA_SMA_50`, `100`, `150`, `200`: Simple Moving Averages

### Volatility Indicators:
- `TSLA_BB_Mid`: Bollinger Band midline (typically 20-day SMA)
- `TSLA_BB_High`: Upper Bollinger Band
- `TSLA_BB_Low`: Lower Bollinger Band

### Ichimoku Components:
- `TSLA_Ichimoku_Conversion_Line`, `Base_Line`
- `TSLA_Ichimoku_Lagging_Span`
- `TSLA_Ichimoku_Leading_Span_A`, `Leading_Span_B`

---

## Price Gap Features

- `TSLA_Gap_Up`: Indicator (1/0) if open price > previous close
- `TSLA_Gap_Down`: Indicator if open price < previous close
- `TSLA_Gap_Size`: Magnitude of the gap

---

## Open Price Features

- `TSLA_Open_Return`: Return from previous close to open price
- `TSLA_Open_SMA_10`: 10-day simple moving average of opening prices
- `TSLA_Open_EMA_10`: 10-day exponential moving average of opening prices
- `TSLA_VWAP_Open`: Volume-weighted average price using open prices

---

## External Market Signals

### Cryptocurrency:
- `Close_BTC-USD`, `Volume_BTC-USD`, `BTC-USD_Returns`

### Commodities:
- `Close_BZ=F`, `Volume_BZ=F`, `BZ=F_Returns` (Brent Crude Oil)

### Currency Index:
- `Close_DX-Y.NYB`, `Volume_DX-Y.NYB`, `DX-Y.NYB_Returns` (US Dollar Index)

### Volatility Index:
- `Close_%5EVIX`, `Volume_%5EVIX`, `%5EVIX_Returns` (CBOE VIX)

### Market Index:
- `Close_%5EGSPC`, `Volume_%5EGSPC`, `%5EGSPC_Returns` (S&P 500)

### Sector ETFs:
- `Close_LIT`, `Volume_LIT`, `LIT_Returns` (Lithium & Battery ETF)
- `Close_SOXX`, `Volume_SOXX`, `SOXX_Returns` (Semiconductor ETF)

---

## Macroeconomic Indicators

- `PCE_Price_Index`: Personal Consumption Expenditures Price Index
- `Core_PCE_Price_Index`: Excluding food and energy
- `10-Year_Treasury_Yield`: US government bond yield
- `Federal_Funds_Rate`: Interest rate target by the Fed
- `University_of_Michigan-Consumer_Sentiment`: Consumer sentiment index
- `Consumer_Price_Index-All_Items-Total_for_United_States`: Official CPI
- `Total_Vehicle_Sales`: Indicator of economic activity

---

## Notes

- This dataset is scaled and cleaned for use in deep learning models (LSTM, TCN, Transformer).
- Missing values are imputed using forward fill or statistical estimates.
- Ideal for multivariate time-series forecasting of stock price trends, volatility, and macroeconomic influence.

---

_Last updated: April 2025_
