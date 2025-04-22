# Column Definitions - Temperature Data of California Dataset

This file documents the columns available in `processed_finance_data.csv` used for time-series forecasting of TSLA stock prices and related financial indicators. It is part of the DSS5104-ForecastDL project.

---

## Common Columns

### `TIME` (string)
The timestamp for the record.

### `Mean_Temperature` (float64)
Adjusted closing price of Tesla stock. This is the main target variable.

---

## Lag Features (Past Temperature)
Lagged values of `Mean_Temperature` to capture temporal dependencies:

- `Lag_1`, `Lag_12`

---

## Technical Indicators
- `Mean_Temperature`: Temperature of California

---
