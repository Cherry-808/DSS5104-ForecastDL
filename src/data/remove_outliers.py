from scipy.stats import zscore

def remove_outliers_zscore(df, threshold):
    # Select only numeric columns
    numeric_cols = df.select_dtypes(include='number').columns
    clean_df = df.copy()

    # Calculate Z-scores for numeric columns
    z_scores = clean_df[numeric_cols].apply(zscore)

    # Create a boolean mask for values with abs(Z-score) <= threshold
    mask = (z_scores.abs() <= threshold).all(axis=1)

    # Apply mask to filter out rows with outliers
    clean_df = clean_df[mask]

    return clean_df


def remove_outliers_iqr(df, factor):
    numeric_cols = df.select_dtypes(include='number').columns
    clean_df = df.copy()

    for col in numeric_cols:
        Q1 = clean_df[col].quantile(0.25)
        Q3 = clean_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR

        # Mask out outliers
        clean_df = clean_df[(clean_df[col] >= lower_bound) & (clean_df[col] <= upper_bound)]

    return clean_df

