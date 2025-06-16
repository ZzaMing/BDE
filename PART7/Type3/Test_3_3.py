import numpy as np
import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com//YoungjinBD/data/main/exam/3_3_1.csv')

mean_score = round(df['score'].mean(),2)
std_dev_score = round(df['score'].std(),2)
# print(mean_score)
# print(std_dev_score)

from scipy import stats
# 모평균 평균 가설값 설정
population_mean = 75

# 단일 표본 t-검정 수행
t_statistic, p_value = stats.ttest_1samp(df['score'], population_mean)
# print(t_statistic.round(2))
# print(p_value.round(2))
if p_value < 0.05: # 작대 큰귀
    result = '귀무가설 기각'
else:
    result = '귀무가설 채택'
# print(result)

#################################################################################

df = pd.read_csv('https://raw.githubusercontent.com//YoungjinBD/data/main/exam/3_3_2.csv')

cross_tab = pd.crosstab(df['gender'], df['club_membership'])
print(cross_tab)

#카이제곱 독립
from scipy import stats 
chi2, p_value, dof, expected = stats.chi2_contingency(cross_tab, correction=False)
print(chi2.round(2))

if p_value < 0.05: # 작대 큰귀
    result = '귀무가설 기각'
else:
    result = '귀무가설 채택'
print(result)