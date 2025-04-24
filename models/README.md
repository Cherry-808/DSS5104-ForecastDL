
# 📘 Model Hyperparameter Reference

This document lists the full set of hyperparameters used for all forecasting models in the project **"Deep Learning vs Classical Methods for Time Series Forecasting"**. These configurations were applied consistently across datasets unless otherwise specified.

---

## ⚙️ Classical Models

### 🔢 ARIMA
- **Order (p, d, q)**: Auto-selected using AIC/BIC via `pmdarima.auto_arima` or `statsmodels.ARIMA`
- **Seasonality**: Not used (non-seasonal modeling)
- **Differencing Check**: Stationarity tested using ADF (Augmented Dickey-Fuller)

---

### 📈 Prophet
- **`changepoint_prior_scale`**: `0.05`
- **`seasonality_mode`**: `'additive'`
- **`yearly_seasonality`**: Auto (depends on dataset frequency)
- **`weekly_seasonality`**: Auto
- **`daily_seasonality`**: Auto
- **`interval_width`**: `0.80`

---

### 🌲 XGBoost
- **`n_estimators`**: `100`
- **`learning_rate`**: `0.1`
- **`max_depth`**: `5`
- **`subsample`**: `0.8`
- **`objective`**: `'reg:squarederror'`
- **`booster`**: `'gbtree'`
- **Feature Engineering**: Lag features, rolling means, and calendar encodings generated per dataset

---

## 🤖 Deep Learning Models

### 🧠 LSTM
- **`sequence_length`**: `30`
- **`n_features`**: Dataset-dependent (multivariate or univariate)
- **`units`**: `50`
- **`activation`**: `'relu'`
- **`dropout`**: `0.2`
- **`optimizer`**: `'adam'`
- **`loss`**: `'mse'`
- **`batch_size`**: `32`
- **`epochs`**: `10`
- **`validation_split`**: `0.2`
- **`early_stopping`**: Enabled (patience=3)
- **Scaling**: `MinMaxScaler`

---

### 🔀 Transformer
- **`n_heads`**: `4`
- **`n_layers`**: `2`
- **`ff_dim`**: `64`
- **`dropout`**: `0.1`
- **`embedding`**: Sinusoidal
- **`sequence_length (look-back)`**: `30`
- **`batch_size`**: `32`
- **`epochs`**: `10`
- **`optimizer`**: `'adam'`
- **`loss`**: `'mse'`

---

### 🧩 TCN (Temporal Convolutional Network)
- **`num_filters`**: `64`
- **`kernel_size`**: `3`
- **`dilation_rates`**: `[1, 2, 4, 8]`
- **`residual_blocks`**: `3`
- **`dropout`**: `0.2`
- **`activation`**: `'relu'`
- **`optimizer`**: `'adam'`
- **`loss`**: `'mse'`
- **`batch_size`**: `32`
- **`epochs`**: `10`
- **Receptive Field**: ~30 time steps

---

## 📝 Notes
- Hyperparameters were selected to **balance fairness and feasibility** across domains.
- No grid search or Bayesian tuning was performed—values were chosen based on literature and reproducibility.
- ARIMA's `order` is dataset-specific and may vary slightly.
- Deep models used fixed architectures to ensure consistent comparison unless specified otherwise.

---

## 📎 Reference
For implementation details and reproducibility, see the corresponding scripts in:
- `src/models/`
