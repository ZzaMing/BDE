import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
# df = pd.read_csv('https://raw.githubusercontent.com//YoungjinBD/data/main/exam/6_1_1.csv')

# # print(df.info())
# # print(df.head())

# df = df.loc[(df['ProductA가격'] != 0) & (df['ProductB가격'] != 0)]
# df['가격차이'] = abs(df['ProductA가격'] - df['ProductB가격'])
# result = df.groupby('도시명')['가격차이'].mean().max()
# print(round(result, 1))

# df = pd.read_csv('https://raw.githubusercontent.com//YoungjinBD/data/main/exam/6_1_2.csv')

# print(df.head())
# # print(df.info())

# Height_M = df['Height_cm'] / 100
# df['BMI'] = round(df['Weight_kg'] / (Height_M **2), 1)
# df['구분'] = 'tmp'
# df.loc[df['BMI'] < 18.5, '구분'] = '저체중'
# df.loc[(df['BMI'] >= 18.5) & (df['BMI'] < 23), '구분'] = '정상'
# df.loc[(df['BMI'] >= 23) & (df['BMI'] < 25), '구분'] = '과체중'
# df.loc[df['BMI'] >=25, '구분'] = '비만'
# print(df.groupby('구분').size().loc[['저체중', '비만']].sum())

# df = pd.read_csv('https://raw.githubusercontent.com//YoungjinBD/data/main/exam/6_1_3.csv')

# print(df.info())
# print(df.head())

# df['생성수량'] = df['products_made_domestic'] + df['products_made_international']
# df['판매수량'] = df['products_sold_domestic'] + df['products_sold_international']
# df['순생산량'] = df['생성수량'] - df['판매수량']

# # print(df.groupby(['year','factory'])['순생산량'].sum())
# result = df.loc[df.groupby('year')['순생산량'].idxmax(), ['year', 'factory', '순생산량']]
# print(result)
# print(result['순생산량'].sum())
