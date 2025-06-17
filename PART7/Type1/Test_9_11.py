import numpy as np
import pandas as pd



df = pd.read_csv('https://raw.githubusercontent.com//YoungjinBD/data/main/exam/9_1_3.csv')

# print(df.isna().sum())

mean = df['평균만족도'].mean()
df['평균만족도'] = df['평균만족도'].fillna(mean)
# print(df)
df['근속연수'] = df['근속연수'].fillna(np.floor(df.groupby(['부서', '등급'])['근속연수'].transform('mean')))
# print(df)

result1 = df.loc[(df['부서'] == 'HR') & (df['등급'] == 'A'), '근속연수'].mean()
print(result1)