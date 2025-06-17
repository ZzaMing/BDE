import numpy as np
import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com//YoungjinBD/data/main/exam/9_3_1.csv')

train = df.loc[df['id'] <= 140]
test = df.loc[df['id'] >=141]

from statsmodels.formula.api import ols
model = ols('design ~ f2 + f3+ f4+ f5+ tenure', train).fit()

p_values = model.pvalues[1:]

print((p_values >= 0.05).sum())  #작대(기각, 유의미) 큰귀(채택, 무의미)

train_pred = model.predict(train)
train_y = train['design']

from scipy import stats
corr, _ = stats.pearsonr(train_y, train_pred)
print(corr)

from sklearn.metrics import root_mean_squared_error
test_pred = model.predict(test)
print(root_mean_squared_error(test['design'], test_pred))



# ===========================================================

df = pd.read_csv('https://raw.githubusercontent.com//YoungjinBD/data/main/exam/9_3_2.csv')

from statsmodels.formula.api import logit
model = logit('churn ~ col1 + col2 + Phone_Service + Tech_Insurance', df).fit()

p_value_col1 = model.pvalues['col1']

print(round(p_value_col1,3))

odds_ratio = round(np.exp(model.params['Phone_Service']), 3)
print(odds_ratio)

pred = model.predict(df)
print(sum(pred> 0.3))