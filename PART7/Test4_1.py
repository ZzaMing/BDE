import numpy as np
import pandas as pd

# df = pd.read_csv('https://raw.githubusercontent.com//YoungjinBD/data/main/exam/4_1_1.csv')
# print(df.info())
# print(df.head())
# df = df.dropna()
# df = df.loc[:int(len(df) * 0.7)]
# print(round(df['PTRATIO'].quantile(0.25), 2))

# df = pd.read_csv('https://raw.githubusercontent.com//YoungjinBD/data/main/exam/4_1_2.csv')

# df_year1 = df.loc[  (df['yearBuilt'] >= 1991)  &  (df['yearBuilt'] <= 2000)  ]
# result1 = df_year1.loc[df_year1['avgSchoolRating'] <= df['avgSchoolRating'].mean(),'uid'].shape[0]
# df_year2 = df.loc[ (df['yearBuilt'] >= 2001)  &  (df['yearBuilt'] <= 2010)   ]
# result2 = df_year2.loc[df_year2['avgSchoolRating'] >= df['avgSchoolRating'].mean(),'uid'].shape[0]

# print(result1, result2)

# print(df.isna().sum().idxmax())
# print(df.isna().sum().sort_values(ascending = False).index[0])