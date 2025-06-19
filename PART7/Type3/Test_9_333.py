import numpy as np
import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com//YoungjinBD/data/main/exam/9_3_1.csv')

train = df.loc[df['id'] <= 140]
test = df.loc[df['id'] >=141]

from statsmodels.formula.api import ols
model = ols('design~tenure+f2+f3+f4+f5', train).fit()

p_values = model.pvalues[1:]

result = (p_values >= 0.05).sum() #작대 큰귀 대립 = 유의미= 작음 / 귀무 = 무의미 = 큼
print(result)

from scipy import stats
pred = model.predict(train)
st, p_values = stats.pearsonr(train['design'], pred)
print(st)

from sklearn.metrics import root_mean_squared_error
pred = model.predict(test)
print(root_mean_squared_error(test['design'], pred))

df = pd.read_csv('https://raw.githubusercontent.com//YoungjinBD/data/main/exam/9_3_2.csv')

from statsmodels.formula.api import logit
model = logit('churn ~ col1 + col2 + Phone_Service + Tech_Insurance', df).fit()

print(model.pvalues['col1'])
print(np.exp(model.params['Phone_Service']),3)
print((model.predict(df) > 0.3).sum())