
import numpy as np 
import pandas as pd
from scipy.stats import zscore

# load file 'combined_df.csv'
df = pd.read_csv('combined_df.csv')

#set timestamp as index
df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d/%m/%y %H:%M')
df.set_index('timestamp', inplace=True)

df_target = 'Energy_Price'
df_predictors = [col for col in df.columns if col != df_target]

#drop all rows with NaN values
df = df.dropna()

#drop all rows with negative value for target variable
df = df[df[df_target] >= 0]

#remove outliers using z-score
z_scores = zscore(df[df_target])
df = df[(abs(z_scores) < 3)]

#log transform "df_target"
df[df_target] = np.log1p(df[df_target])