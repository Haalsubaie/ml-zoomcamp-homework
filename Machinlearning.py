import pandas as pd
import numpy as np

url = "https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/car_fuel_efficiency_2026.csv"

df = pd.read_csv(url)

print(df.head())
print("Pandas version:", pd.__version__)
print("Records count:", len(df))

print("Fuel types:", df['fuel_type'].nunique())

print("Missing values by column:")
print(df.isnull().sum())

print("Columns with missing values:", (df.isnull().sum() > 0).sum())

max_efficiency_asia = df[df['origin'] == 'Asia']['fuel_efficiency_mpg'].max()
print("Max fuel efficiency in Asia:", max_efficiency_asia)

median_before = df['horsepower'].median()
mode_hp = df['horsepower'].mode()[0]

df_q6 = df.copy()
df_q6['horsepower'] = df_q6['horsepower'].fillna(mode_hp)

median_after = df_q6['horsepower'].median()

print("Median before:", median_before)
print("Most frequent horsepower:", mode_hp)
print("Median after:", median_after)

X = df[df['origin'] == 'Asia'][['vehicle_weight', 'model_year']].head(7).values

XTX = X.T @ X
XTX_inv = np.linalg.inv(XTX)

y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200])

w = XTX_inv @ X.T @ y

print("w:", w)
print("Sum of w:", w.sum())