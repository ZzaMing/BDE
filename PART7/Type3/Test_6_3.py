import numpy as np
import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com//YoungjinBD/data/main/exam/6_3_1.csv')

# print(df.head())
# print(df.agg(['mean', 'std']).round(3))
mean_wh = df['work_hours'].mean().round(3)
std_wh = df['work_hours'].std().round(3)

# 정규분포 따르는지 K-S 검정 실시
from scipy import stats
from scipy.stats import norm
statistic, p_value = stats.kstest(df, 'norm', args=(mean_wh, std_wh))
# print(statistic)
# print(p_value)

#======================================================


df = pd.read_csv('https://raw.githubusercontent.com//YoungjinBD/data/main/exam/6_3_2.csv')

import statsmodels.api as sm

# 독립변수와 종속변수 설정
X = df[['area', 'rooms', 'age']] # 독립변수
y = df['price'] # 종속변수


# 상수항 추가
X = sm.add_constant(X)

# 다중회귀 모델 분석
model = sm.OLS(y, X).fit()

# 회귀변수가 가장높은 변수를 확인
coefficients = model.params[1:]
print('회귀변수가 가장 큰 변수: ', coefficients.idxmax())

p_values = model.pvalues[1:]
print('유의미한 변수 개수: ', np.sum(p_values < 0.05))

print(p_values)











