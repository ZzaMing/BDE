import numpy as np
import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com//YoungjinBD/data/main/exam/7_3_1.csv')
# print(df.head())

# print(df.agg(['mean', 'std']).round(3))

# 대응 표본 t-검정
from scipy import stats
statistic, p_values = stats.ttest_rel(df['before'], df['after'])

# print(statistic.round(2))
# print('기각' if p_values < 0.05 else '채택')


#============================================================




train = pd.read_csv('https://raw.githubusercontent.com//YoungjinBD/data/main/exam/7_3_2_train.csv')
test = pd.read_csv('https://raw.githubusercontent.com//YoungjinBD/data/main/exam/7_3_2_test.csv')

# 독립변수 종속변수 설정
X_train = train[['age', 'income', 'family_members']]
y_train = train['purchase']

# 로지스틱 회귀 분석 & 소득 변수 오즈비
import statsmodels.api as sm
X_train = sm.add_constant(X_train)              # 상수항 절편 추가
logit_model = sm.Logit(y_train, X_train).fit()  # 로지스틱 회귀 분석 모델 적합
print(logit_model.summary())                    # 모델 요약 출력

odds_ratios = np.exp(logit_model.params)        
print('소득 오즈비: ', odds_ratios['income'].round(3))

#잔차 이탈도 = -2 * llf
residual_deviance = -2 * logit_model.llf
print(residual_deviance)

# test 데이터로 오분류율 예측
X_test = test[['age', 'income', 'family_members']]
y_test = test['purchase']

import statsmodels.api as sm
X_test  = sm.add_constant(X_test)

y_pred_prob = logit_model.predict(X_test)
y_pred = (y_pred_prob >= 0.5).astype(int) # 임계값 0.5 (문제에 주어짐.)





