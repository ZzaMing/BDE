import  numpy as np
import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com//YoungjinBD/data/main/exam/8_3_1.csv')
print(df.agg(['mean','std']).round(2))

from scipy import stats
statistic, p_value = stats.wilcoxon(df['before'], df['after'])
# print(statistic.round(2))
# print('기각' if p_value < 0.05 else '채택')

train = pd.read_csv('https://raw.githubusercontent.com//YoungjinBD/data/main/exam/8_3_2_train.csv')
test = pd.read_csv('https://raw.githubusercontent.com//YoungjinBD/data/main/exam/8_3_2_test.csv')


print(train.head())

from statsmodels.formula.api import ols
model = ols('productivity ~ hours + age + experience', train).fit()

corr = model.params[1:]
print(corr.idxmax())

print((model.pvalues[1:] < 0.05).sum())

from sklearn.metrics import r2_score
test_pred = model.predict(test)
print(r2_score(test['productivity'], test_pred))