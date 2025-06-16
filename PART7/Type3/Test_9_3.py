import numpy as np
import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com//YoungjinBD/data/main/exam/9_3_1.csv')

train = df.loc[df['id'] <= 140]
test = df.loc[df['id'] > 140]

from statsmodels.formula.api import ols # 범주형 잇을 경우 C()를 사용
#볌주형 예시 => ols('design ~ tenure+f2 +f3 +f4+f5+C(예시컬럼)'
model = ols('design ~ tenure+f2 +f3 +f4+f5', train).fit()
p_values = model.pvalues[1:]
print((p_values >= 0.05).sum())

y_pred = model.predict(train)
y_real = train['design']
from scipy import stats
correlation, _ = stats.pearsonr(y_real, y_pred)
print(correlation)


y_pred_test = model.predict(test)
from sklearn.metrics import root_mean_squared_error
print(root_mean_squared_error(test['design'],y_pred_test))

#==============================================================================================

df = pd.read_csv('https://raw.githubusercontent.com//YoungjinBD/data/main/exam/9_3_2.csv')

from statsmodels.formula.api import logit

# print(df.head())
model = logit('churn ~ col1 + col2 + Phone_Service + Tech_Insurance', df).fit()

print(round(model.pvalues['col1'],3))

odds_radio_phnoe_service = np.exp(model.params['Phone_Service'])
print(round(odds_radio_phnoe_service,3))

y_pred = model.predict(df)
print((y_pred >= 0.3).sum())
