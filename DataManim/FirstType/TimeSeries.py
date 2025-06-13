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

# result = df['RPT'].diff()

# result = df[['RPT', 'VAL']].rolling(7).mean()

df =pd.read_csv('https://raw.githubusercontent.com/Datamanim/pandas/main/seoul_pm.csv')

def change_date(x):
    import datetime
    hour = x.split(':')[1]
    date = x.split(":")[0]
    
    if hour =='24':
        hour ='00:00:00'
        
        FinalDate = pd.to_datetime(date +" "+hour) +datetime.timedelta(days=1)
        
    else:
        hour = hour +':00:00'
        FinalDate = pd.to_datetime(date +" "+hour)
    
    return FinalDate

df['(년-월-일:시)'] = df['(년-월-일:시)'].apply(change_date)

df['day_name'] = df['(년-월-일:시)'].dt.day_name()
# result = df['day_name']

# result = df.groupby(['day_name','PM10등급']).size().reset_index()

# result = df.groupby(df['(년-월-일:시)'].dt.hour)['PM10'].mean().loc[[10,22]]

df.set_index('(년-월-일:시)', inplace=True, drop = True)

print(result)