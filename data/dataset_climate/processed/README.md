# Column Definitions - Temperature Data of California Dataset

This file contains monthly mean temperature data of California derived from the PRISM Climate Group.

---

## Common Columns

| **Data**           | **Type**     | **Boundary** | **Frequency** | **Source**                 |
|--------------------|--------------|--------------|---------------|----------------------------|
| TIME               | Timedex      | California   | Monthly       | PRISM Climate Group (OSU)  |
| Mean_Temperature   | Target       | California   | Monthly       | PRISM Climate Group (OSU)  |

---

### `TIME` (datetime64[M])
The time index for each observation, provided in `YYYY-MM` format. This defines the **monthly resolution** of the dataset and is used as the time axis for modeling and forecasting.

### `Mean_Temperature` (float64)
This is the **target variable** for forecasting. It represents the monthly mean surface air temperature, derived from PRISM’s high-resolution gridded climate model. Values reflect climatological conditions over time, with both seasonal patterns and long-term trends.

---

## Additional Info

### Time Range
The dataset begins in **January 1981** and ends in **September 2024**, covering over four decades of monthly climate observations.

### Temperature Values
The `Mean_Temperature` values range approximately from **-3.68°C** to **29.09°C**, capturing seasonal and interannual temperature variability across California.

---

### Notes on Source
PRISM (Parameter-elevation Regressions on Independent Slopes Model) is a high-quality climate interpolation system developed by Oregon State University. It combines point-based station data with geographic and topographic information to produce reliable gridded climate datasets across the U.S.
