import numpy as np
import pandas as pd
import re
pd.set_option('display.max_columns', None)

# df = pd.read_csv('https://raw.githubusercontent.com//YoungjinBD/data/main/exam/5_1_1.csv')

# print(df.info())
# print(df.head())

# result = df.loc[((df['minority'] / df['poverty']) > 2) & (df['city'] == 'state'), 'crime'].mean()

# print(round(result, 0))

# df = pd.read_csv('https://raw.githubusercontent.com//YoungjinBD/data/main/exam/5_1_2.csv')

# print(df.info())
# print(df.head())

# df['date'] = pd.to_datetime(dict(year = df['year'], month = df['month'], day=df['day']))
# df.set_index('date', inplace=True)
# print(df)
# print(df.loc[:'2016-09-01', 'actual'].mean())
