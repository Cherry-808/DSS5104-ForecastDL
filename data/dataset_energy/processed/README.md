# Column Definitions - Processed Energy Dataset

This file documents the columns available in `df_with_lag_indicators.csv.gz` used for time-series forecasting of energy prices in New South Wales, Australia. It is part of the DSS5104-ForecastDL project.
/
The data covers the period from 1 January 2022 to 31 December 2024, and includes both local and global features.
---

## Common Columns

| **Data**                     | **Type**    | **Boundary** | **Frequency** | **Source** |
|-----------------------------|-------------|--------------|----------------|------------|
| Timestamp                   | Timeindex   | NSW          | 5-min          | NA
| Energy Price                | Output      | NSW          | 5-min          | [Australian Energy Market Operator (AEMO)] |
| Energy Demand               | Predictor   | NSW          | 5-min          | [Australian Energy Market Operator (AEMO)]|
| Global Crude Oil Price      | Predictor   | Global       | Daily          | [U.S. Energy Information Administration (EIA)] |
| Global Natural Gas Price    | Predictor   | Global       | Daily          | [U.S. Energy Information Administration (EIA)] |
| Rain                        | Predictor   | NSW          | Daily          | [Australia Bureau of Meteorology (BoM)] |
| Solar Exposure              | Predictor   | NSW          | 5-min          | [Solcast]|
| Temperature                 | Predictor   | NSW          | 5-min          | [Solcast]|
| Renewable Mix               | Predictor   | NSW          | Hourly         | [Electricity Maps]|


### `timestamp` (datetime64[ns])
The timeindex for the record. It is based on NSW timezone and is in 5-min intervals.

### `Energy_Price` (float64)
The log-adjusted energy price. This is the main target variable. It has been treated to remove outliers (>3 standard deviation), as energy prices in Australia tend to surge during peak period. There are also instances where energy prices become negative when the supply exceeds the demand. All such outliers have been removed.

### `Other Data Processing` (float64)
Where predictors have a interval frequency larger than the target variable, e.g. daily data or hourly data, they have been upsampled to match the 5-min frequency of the target variable.

---

## Time Features (Engineered from `Timestamp`)
In addition to the original predictors, additional time features were added, since some models require lag indicators for time-series forecasting.

- `hour`: Hour of the day
- `dayofweek`: Day of the week (Monday through Sunday)
- `month`: Month of the year (January through December)
- `lag_1': Lagged energy price by 5-min
- `lag_12': Lagged energy price by 1-hour
- `lag_288': Lagged energy price by 24-hours
- `lag_576': Lagged energy price by 48-hours 
- `rolling_mean_24': The rolling mean energy price for the past 2 hours
- `rolling_std_24': The rolling standard deviation of energy price for the past 2 hours

---

## Notes

- This dataset is scaled and cleaned for use in deep learning models (LSTM, TCN, Transformer).
- Missing values are removed.
- Ideal for multivariate time-series forecasting of energy price trends, based on macroeconomic and geospatial indicators.

---

_Last updated: April 2025_
