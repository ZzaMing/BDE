import pandas as pd
import numpy as np
from sklearn.datasets import make_regression
import statsmodels.api as sm

X, y = make_regression(n_samples= 100, n_features=3, noise=0.1, random_state=42)
df = pd.DataFrame(X, columns=[f'var{i}' for i in range(3)])
df['target'] = y
# print(df.head())

# 4. 유의확률(p-value)이 가장 작은 변수의 회귀계수를 구하시오.
X = df.drop(columns= 'target')
y = df['target']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
# print(model.summary())
p_values = model.pvalues
min_p_var = p_values.idxmin()
min_p_coef = model.params[min_p_var]
print(f"p값이 제일 작은 변수의 회귀계수: {min_p_coef}")

# 5. 적합된 회귀모델의 결정계수를 구하시오.
r_squared = model.rsquared
print(f"결정계수 : {r_squared}")

# 6. 적합된 회귀모델을 사용하여 var() 변수가 0.5, var1은 1.2 그리고 var2는 0.3일때 예측값을 계산하시오.
new_data = pd.DataFrame({'const': [1.0], 'var0': [0.5],'var1': [1.2],'var2': [0.3]})
print(new_data)
pred_value = model.predict(new_data)
print(f"예측값: {pred_value[0]}")