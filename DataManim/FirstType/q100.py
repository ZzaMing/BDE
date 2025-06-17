import numpy as np
import pandas as pd

df= pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/worldcup/worldcupgoals.csv')
# print(df.head())

result = df.groupby('Country')['Goals'].sum()
result = pd.DataFrame(result.sort_values(ascending=False))

# result = df.groupby('Country').size().sort_values(ascending=False)[:5]

df['yearList'] = df.Years.str.split('-')

def checkFour(x):
    for value in x:
        if len(str(value)) != 4:
            return False
        
    return True
    
df['check'] = df['yearList'].apply(checkFour)

# result = len(df[df.check ==False])

df2 = df[df.check ==True].reset_index(drop=True)

df2['LenCup'] = df2['yearList'].str.len()
# result = df2['LenCup']
result = df2.loc[df2['LenCup'] == 4, 'LenCup'].value_counts()

result = len(df2.loc[(df2['Country'] == 'Yugoslavia') & (df2['LenCup'] == 2), 'Player'])

result = len(df2.loc[df2['Years'].str.contains('2002'), 'Player'])

result = len(df2.loc[df2['Player'].str.lower().str.contains('carlos')])

result = df2.loc[df2['LenCup'] == 1,'Country'].value_counts().index[0]

# --------------------------------30 번 문제까지 품 ----------------------------
print(result)