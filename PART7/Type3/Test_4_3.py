import numpy as np
import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com//YoungjinBD/data/main/exam/4_3_1.csv')
# print(df.head())

result = df.groupby('department')['hours_worked'].agg(['mean','std']).round(2)
# print(result)

# 2표본(독립) t-검정
from scipy import stats
t_statistic, p_value = stats.ttest_ind(
    df.loc[df['department'] == 'A', 'hours_worked'],
    df.loc[df['department'] == 'B', 'hours_worked'])
# print('검정통계랑: ', t_statistic.round(2))
# print('p_value: ', p_value.round(2))
if p_value < 0.05: #작대 큰귀
    result = '귀무가설 기각'
else:
    result = '귀무가설 채택'
# print(result)

#================================================================

df = pd.read_csv('https://raw.githubusercontent.com//YoungjinBD/data/main/exam/4_3_2.csv')
# print(df.head())

result = df.groupby('factory')['quality_score'].agg(['mean','std']).round(2)
# print(result)

# 크루스칼-왈리스 검정 
from scipy import stats
t_statistic, p_value = stats.kruskal(
    df.loc[df['factory'] == 'A', 'quality_score'],
    df.loc[df['factory'] == 'B', 'quality_score'],
    df.loc[df['factory'] == 'C', 'quality_score']
)
print('검정통계랑: ', t_statistic.round(2))
print('p_value: ', p_value.round(2))
if p_value < 0.05: #작대 큰귀
    result = '귀무가설 기각'
else:
    result = '귀무가설 채택'
print(result)