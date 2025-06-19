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

df =pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/bicycle/seoul_bi.csv')

result = df.groupby('대여일자').size().sort_values(ascending=False).index[0]

df['대여일자'] = pd.to_datetime(df['대여일자'])
df['day_name'] = df['대여일자'].dt.day_name()
result = df['day_name'].value_counts()

result = df.groupby('연령대코드')['이동거리'].mean()

df_20 = df.loc[df['연령대코드'] == '20대']
df_20_mean = df.loc[df['연령대코드'] == '20대', '이동거리'].mean()
df_res = df_20.loc[df_20['이동거리'] >= df_20_mean, ['대여일자', '대여소번호','탄소량']]
result = df_res.sort_values(['대여일자', '대여소번호'], ascending=False)[0:200]
result = result['탄소량'].astype(float).mean().round(3)

result = df.loc[(df['대여일자'] >= '2021-06-07') & (df['대여일자'] <= '2021-06-10'), '이용건수'].median()

weekend = ['Saturday', 'Sunday']

df_weekday = df.loc[(~df['day_name'].isin(weekend)) & (df['대여시간'].isin([6,7,8]))]

result = (df_weekday
          .groupby(['대여시간','대여소번호'])
          .size().to_frame('이용 횟수')
          .reset_index()
          .sort_values(['대여시간','이용 횟수'],ascending=False)
          .groupby('대여시간').head(3)
)

result = df.loc[df['이동거리'] >= df['이동거리'].mean(), '이동거리'].std()

df['성별'] = df['성별'].str.lower()
result = df.groupby('성별')['이동거리'].mean()


df =pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/happy2/happiness.csv',encoding='utf-8')

result = df.loc[df['행복랭킹'] <= 50,  ['년도','점수']].groupby('년도').mean()

# result = df.loc[df['년도'] == 2018,['점수', '부패에 대한인식'] ].corr().iloc[0,1]

# df_2018 = df.loc[df['년도'] == 2018, '행복랭킹']
# result = df.loc[df['년도'] == 2019, '행복랭킹']

# df.loc[df['년도'] == 2019].corr()
#---------------50번 문제----------------

df= pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/consum/Tetuan%20City%20power%20consumption.csv')

print(df.info())

df['DateTime'] = pd.to_datetime(df['DateTime'])
result = df['DateTime'].dt.month.value_counts().sort_index().to_frame()

result = df.loc[df['DateTime'].dt.month == 3, ['DateTime', 'Temperature']].groupby(df['DateTime'].dt.hour)['Temperature'].mean().max()

result = df.loc[df['Zone 1 Power Consumption'] > df['Zone 2  Power Consumption'], 'Humidity'].mean()

result = df[['Zone 1 Power Consumption', 'Zone 2  Power Consumption', 'Zone 3  Power Consumption' ]].corr()

df['rank'] = 'D'
df.loc[df['Temperature'] < 10, 'rank'] = 'A'
df.loc[(df['Temperature'] >= 10) & (df['Temperature'] < 20), 'rank'] = 'B'
df.loc[(df['Temperature'] >= 20) & (df['Temperature'] < 30), 'rank'] = 'C'

result = df['rank'].value_counts()
 
result = df.loc[(df['DateTime'].dt.month == 6) & (df['DateTime'].dt.hour == 12), 'Temperature'].std()
result = df.loc[(df['DateTime'].dt.month == 6) & (df['DateTime'].dt.hour == 12), 'Temperature'].var()
 
result = df.loc[df['Temperature'] >= df['Temperature'].mean()].sort_values('Temperature')['Humidity'].values[3]
 
print(result)

