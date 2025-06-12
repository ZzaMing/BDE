import numpy as np
import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/Datamanim/pandas/main/timeTest.csv')
result = 0

df['Yr_Mo_Dy'] = pd.to_datetime(df['Yr_Mo_Dy'])
# result = df['Yr_Mo_Dy']

# result = df['Yr_Mo_Dy'].dt.year.unique()
# print(dir(df['Yr_Mo_Dy']))

df.loc[df['Yr_Mo_Dy'].dt.year >= 2061, 'Yr_Mo_Dy'] = df.loc[df['Yr_Mo_Dy'].dt.year >= 2061, 'Yr_Mo_Dy'] - pd.DateOffset(years=100)
# result = df.head()

# result = df.groupby(df['Yr_Mo_Dy'].dt.year).mean()

df['weeekday'] = df['Yr_Mo_Dy'].dt.day_of_week
# result = df['weeekday'].head(10)

df['week_check'] = df['weeekday'].isin([5,6]).astype(int)
# result = df['week_check'].head(10)

# result = df.groupby(df['Yr_Mo_Dy'].dt.month).mean()

# df = df.fillna(method = 'ffill').fillna('bfill')
# result = df.isnull().sum()

# result = df.groupby(df['Yr_Mo_Dy'].dt.to_period('M')).mean()

#-----------------------73
print(result)