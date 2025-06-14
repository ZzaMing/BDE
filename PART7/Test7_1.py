import pandas as pd
import numpy as np

df = pd.read_csv('https://raw.githubusercontent.com//YoungjinBD/data/main/exam/7_1_2.csv')

# print(df.info())
# print(df.head())

df_mp = df.loc[df['STATION_ADDR1'].str.contains('마포구'), 'dist']
df_sd = df.loc[df['STATION_ADDR1'].str.contains('성동구'), 'dist']

# print(df_mp.mean())
# print(df_sd.mean())

df = pd.read_csv('https://raw.githubusercontent.com//YoungjinBD/data/main/exam/7_1_3.csv')

# print(df.info())
# print(df.head())

df['총판매량'] = df['제품A'] + df['제품B'] + df['제품C'] + df['제품D'] + df['제품E']

df[['연도', '월']] = df['기간'].str.split('_', expand= True)
df['월'] = df['월'].str.replace('월', '').astype(int)
df['분기'] = pd.cut(df['월'], bins = [0,3,6,9,12], labels=[ 1,2,3,4 ], right=True)

df_grouped = df.groupby(['연도','분기'])['총판매량'].mean().reset_index()
print(df_grouped.loc[df_grouped['총판매량'].idxmax()])
