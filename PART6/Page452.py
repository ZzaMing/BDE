import numpy as np
import pandas as pd

np.random.seed(42)
n_sample = 210
X = np.random.randn(n_sample, 4)
y = (X[:,0] + X[:,1] * 0.5 + np.random.randn(n_sample) *0.5 >0).astype(int)
df = pd.DataFrame(X, columns=['weight', 'height', 'age', 'income'])
df['gender'] = y
# print(df.head())

# 1. 성별 변수를 사용해여 몸무게 변수에 대한 로지스틱 회귀모델을 적합하고, 해당하는 오즈비를 계산하시오.
# 오즈비 = 로지스틱 회귀계수의 지수값
import statsmodels.api as sm

X_weight = df[['weight']]
X_weight = sm.add_constant(X_weight)
y = df['gender']
logit_model_weight = sm.Logit(y, X_weight).fit()
# print(logit_model_weight.summary())
odds_ratio_weight = np.exp(logit_model_weight.params['weight'])
# print(f"weight의 오즈비: {odds_ratio_weight}")

# 2. 성별 변수를 주어진 4개 변수를 사용하여 로지스틱 회귀모델을 적합했을 때,  residual deviance를 계산하시오.
# residual deviance = -2 * LLF
X_all = df[['weight', 'height', 'age', 'income']]
X_all = sm.add_constant(X_all)
logit_model_all = sm.Logit(y, X_all).fit()
residual_deviance = -2 * logit_model_all.llf
# print(f"resudual deviance: {residual_deviance.round(3)}")

# 3. 1번 문제의 모델 데이터를 학습 데이터와 평가 데이터로 분류 한 후 , 오분류율을 계산하시오.
# 오분류율 = 1 - accuracy_score()
from sklearn.model_selection import train_test_split
df_train, df_test = train_test_split(df, test_size=90, random_state=42)
X_train = sm.add_constant(df_train['weight'])
y_train = df_train['gender']
X_test = sm.add_constant(df_test['weight'])
y_test = df_test['gender']
# print(X_train.shape)
# print(X_test.shape)

logit_model_train = sm.Logit(y_train, X_train).fit()

from sklearn.metrics import accuracy_score

y_pred = logit_model_train.predict(X_test) > 0.5

error_rate = 1- accuracy_score(y_pred, y_pred)
print(f"오분류율: {round(error_rate, 4)}")