import pandas as pd
import numpy as np
result = 0

df = pd.read_csv('https://raw.githubusercontent.com//YoungjinBD/data/main/exam/8_1_1.csv')
# print(df.head(10))
# result = df.groupby('대륙')['맥주소비량'].mean().idxmax()
# SA

#------------------------------------------------------------

# df_sa = df.loc[df['대륙'] == 'SA']
# result = df_sa.groupby('국가')['맥주소비량'].sum().sort_values(ascending=False)
# result = df_sa.sort_values(by ='맥주소비량', ascending=False)

# result = df.groupby(['대륙', '국가'])['맥주소비량'].sum().reset_index()
# result = result.loc[result['대륙'] == 'SA'].sort_values('맥주소비량', ascending=False).iloc[4,1]

#-----------------------------------------------------------

# result = df.loc[df['국가'] == 'Venezuela']['맥주소비량'].mean()
# result = round(result)

#=============================================================

df = pd.read_csv('https://raw.githubusercontent.com//YoungjinBD/data/main/exam/8_1_2.csv')

# print(df.head(10))
# print(df.info())

# df['합계'] = df['관광'] + df['사무'] + df['공무'] + df['유학'] + df['기타']
# df['비율'] = df['관광'] / df['합계']
# result = df.sort_values('비율', ascending=False)
# result = round(result ,4).iloc[1]['관광']

# df = df.groupby('국가').sum()
# df['합계'] = df['관광'] + df['사무'] + df['공무'] + df['유학'] + df['기타']
# df['비율'] = df['관광'] / df['합계']
# result = df.sort_values('비율', ascending=False)

# result = df.head(50)
#------------------------------------
# result = df.sort_values('관광',ascending=False).head(2) #이스라엘
# df_loc = df.loc[df['국가'] == '이스라엘']
# result = df_loc.groupby('국가')['공무'].mean().reset_index()
# result = round(result)

#===============================================
df = pd.read_csv('https://raw.githubusercontent.com//YoungjinBD/data/main/exam/8_1_3.csv')
# result =df.head(10)

# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()

# df = df.loc[:, ['CO(GT)', 'NMHC(GT)']]
# scaled_df = scaler.fit_transform(df)
# print(scaled_df[:, 0].std().round(2))
# print(scaled_df[:, 1].std().round(2))

# print(result)

#1-1 SA
#1-2 Venezuela 
#1-3 253
#
#2-1 7831 /9039
#2-2  494
#2-3 8325
#
#3-1

