import numpy as np
import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com//YoungjinBD/data/main/exam/3_1.csv')

# print(df['lotSizeSqFt'].head(10))
df_sorted = df.sort_values('lotSizeSqFt',ascending = False).reset_index(drop = False)
df_sorted.loc[:9, 'lotSizeSqFt'] = df_sorted['lotSizeSqFt'].min()
result = df_sorted.loc[df['yearBuilt'] >= 2000, 'lotSizeSqFt'].mean().round()
# print(result)

df = pd.read_csv('https://raw.githubusercontent.com//YoungjinBD/data/main/exam/3_1.csv')
# print(df.isna().sum())
df_bf = df['numOfBathrooms'].std()
df = df.fillna(df['numOfBathrooms'].median())
df_af = df['numOfBathrooms'].std()
# print(abs(df_af - df_bf).round(3))
# print(df)

df = pd.read_csv('https://raw.githubusercontent.com//YoungjinBD/data/main/exam/3_1.csv')
q3_std = df['MedianStudentsPerTeacher'].std() * 1.5
q3_mean = df['MedianStudentsPerTeacher'].mean()
q3_up = q3_mean + q3_std
q3_down = q3_mean - q3_std
result = df.loc[(df['MedianStudentsPerTeacher'] > q3_up) | (df['MedianStudentsPerTeacher'] < q3_down)]
print(len(result))
print(q3_up,q3_down)