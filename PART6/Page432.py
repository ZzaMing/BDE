import pandas as pd
import numpy as np

np.random.seed(42)
n_samples = 100
X = np.random.randn(n_samples, 5)
y = 3 * X[:, 0] + 2 * X[:, 1] + X[:, 2] + np.random.randn(n_samples)
df = pd.DataFrame(X, columns=['var1', 'var2', 'var3', 'var4', 'var5'])
df['target'] = y

# print(df.head())

# 1. target변수와 가장 큰 상관 관계를 갖는 변수의 상관계수를 구하시오.
correlation_matrix = df.corr()
target_corr = correlation_matrix['target'].drop('target')
max_corr_var = target_corr.abs().idxmax()
max_corr_value = target_corr.abs().max()
print(f"가장 큰 상관관계를 갖는 변수: {max_corr_var}, 상관계수: {max_corr_value}")

# 2. 다중 선형회귀 모형으로 target 변수를 예측할 때, 모델의 결정계수를 계산하시오.
import statsmodels.api as sm
X = df.drop(columns= 'target')
y = df['target']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()

r_squared = model.rsquared
print(f"결정계수: {r_squared}")

# 3. 앞에서 사용된 모델의 계수 검정에서 대응하는 유의확률이 가장 큰 변수와 그때의 p-value를 구하시오.
p_values = model.pvalues.drop('const')
max_p_value_var = p_values.idxmax()
max_p_value = p_values.max()
print(f"가장 큰 p-value를 갖는 변수: {max_p_value_var}, p-value: {max_p_value}")