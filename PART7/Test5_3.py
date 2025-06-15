import numpy as np
import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com//YoungjinBD/data/main/exam/5_3_1.csv')
# print(df.head())

# print(df.agg(['mean','std']))

# 피어슨 상관계수 -> pearsonr
from scipy import stats
corr, p_value = stats.pearsonr(df['study_hours'], df['exam_scores'])
# print('상관계수: ', corr.round(3))
# print('p_value: ', p_value.round(3))
if p_value < 0.05: # 작대 큰귀
    result = '귀무가설 기각'
else:
    result = '귀무가설 채택'
# print(result)
    

# ===========================================================

df = pd.read_csv('https://raw.githubusercontent.com//YoungjinBD/data/main/exam/5_3_2.csv')
print(df.head())

print(df.groupby('campaign')['satisfaction_score'].agg(['mean', 'std']).round(2))

# ANOVA 검정
from scipy import stats
# print(dir(stats))
f_statistic, p_value = stats.f_oneway(
    df.loc[df['campaign'] == 'A', 'satisfaction_score'],
    df.loc[df['campaign'] == 'B', 'satisfaction_score'],
    df.loc[df['campaign'] == 'C', 'satisfaction_score']
)

print(f_statistic.round(3))
print(p_value.round(3))

























