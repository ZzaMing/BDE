import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes

diabetes = load_diabetes(as_frame=True)
df = diabetes.frame
# print(df.head())

# 4. target 변수를 중앙값을 기준으로 낮으면 0, 높으면 1 로 이진화 후,
#    로지스틱 회귀모델을 적합시키고, 유의하지 않은 변수의 개수를 구하시오.
import statsmodels.api as sm

X = df.iloc[:, 0:4] # 모든 행, 0~3까지 인덱스에 해당하는 열
X = sm.add_constant(X)

y = (df['target'] > df['target'].median()).astype(int)

logit_model = sm.Logit(y,X).fit()
print(logit_model.summary())

p_values = logit_model.pvalues
non_significant_vars = p_values[p_values >= 0.05] # 대립가설이 유의미하며, 귀무가설은 무의미... 로 생각해야함.. # 작대,큰귀
num_non_significant_vars = len(non_significant_vars)

print(f"유의 하지 않은 변수의 수: {num_non_significant_vars}")

# 5. 4번 문제에서 유의한 변수들만 사용하여 다시 로지스틱 회귀적합하고, 유의한 변수들의 회귀계수 평균을 구하시오.
significant_vars = p_values[p_values < 0.05]
significant_vars_names = significant_vars.index.drop('const', errors = 'ignore') #상수 제외.

X_significant = X[significant_vars_names]
X_significant = sm.add_constant(X_significant)

logit_model_significant = sm.Logit(y, X_significant).fit()

significant_coef_mean = logit_model_significant.params.mean()
print(f"유의미한 변수들만 사용시 회귀계수들의 평균: {significant_coef_mean}")

# 6. 4번 문제에서 나이가 1 단위 증가할 대 오즈비를 계산하시오.
coef_bmi = logit_model.params['age']
delta_x = 1
odds_ratio = np.exp(coef_bmi * delta_x)
print(f"age 변수가 1단위 증가할 때 오즈비: {odds_ratio}")