import numpy as np
import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/Datamanim/pandas/main/mergeTEst.csv',index_col= 0)
result = 0

df3 = df.iloc[:2,:4]
df4 = df.iloc[5:,3:]

# result = pd.concat([df3,df4],join='inner')

# result = pd.concat([df3,df4], join='outer').fillna(0)

df5 = df.T.iloc[:7,:3]
df6 = df.T.iloc[6:,2:5]

# result = pd.merge(df5,df6, on='Algeria', how='inner')

result = pd.merge(df5,df6, on='Algeria',how='outer')







print(result)