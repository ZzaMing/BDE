import numpy as np
import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com//YoungjinBD/data/main/exam/8_3_1.csv')

# print(df)

# print(df.agg(['mean', 'std']))

# 부호 순위 검정
from scipy import stats
statistic, p_value = stats.wilcoxon(df['before'], df['after'])
# print(statistic.round(2))
# print(p_value < 0.05) # 작대 큰귀 

#===============================================================

train = pd.read_csv('https://raw.githubusercontent.com//YoungjinBD/data/main/exam/8_3_2_train.csv')
test = pd.read_csv('https://raw.githubusercontent.com//YoungjinBD/data/main/exam/8_3_2_test.csv')

# print(train.head())
X_train = train.drop('productivity', axis = 1)
y_train = train['productivity']

import statsmodels.api as sm
X_train = sm.add_constant(X_train)

model = sm.OLS(y_train, X_train).fit() # 다중회귀분석 모델 적합

# 회귀계수 추출
coefficients = model.params[1:]
# print(coefficients.idxmax())

#p-value 확인 및 변수 개수 # 작대 큰귀 # 유의미한가 ? -> 대립가설이 유의마한가 ?
p_values = model.pvalues[1:].round(4)
p_sum = (p_values < 0.05).sum()
# print(p_sum)

X_test = test.drop('productivity', axis = 1)
y_test = test['productivity']

X_test = sm.add_constant(X_test)
y_pred = model.predict(X_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)

# print(round(r2,3))