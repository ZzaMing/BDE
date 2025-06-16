import numpy as np
import pandas as pd
train = pd.read_csv('https://raw.githubusercontent.com//YoungjinBD/data/main/exam/8_2_train.csv')
test = pd.read_csv('https://raw.githubusercontent.com//YoungjinBD/data/main/exam/8_2_test.csv')

# 회귀. 값 예측
#1. 결측치 확인
print(train.info())
print(test.info())
# print(train.head())
# print(test.head())

# 2. 데이터 분할
test_id = test['ID']

train_X = train.drop(['count'], axis=1)
train_y = train['count']

test_X = test.drop(['count'], axis=1)
test_y = test['count']

from sklearn.model_selection import train_test_split
train_X, valid_X, train_y, valid_y = train_test_split(train_X, train_y, test_size=0.3, random_state=1) # 막힌부분

# 3. 데이터 전처리
from sklearn.preprocessing import OneHotEncoder
cat_columns = train_X.select_dtypes('object').columns.to_list() # 막힌ㅂ부분
onehotencoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore') # 막힌부분

train_X_preprocessed = onehotencoder.fit_transform(train_X[cat_columns])
valid_X_preprocessed = onehotencoder.transform(valid_X[cat_columns])
test_X_preprocessed = onehotencoder.transform(test_X[cat_columns])

# 4. 모델 적합
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(random_state=1)
rf = rf.fit(train_X_preprocessed, train_y)

from sklearn.metrics import mean_absolute_error
valid_pred = rf.predict(valid_X_preprocessed)
print('MAE: ', mean_absolute_error(valid_y, valid_pred))

test_pred = rf.predict(test_X_preprocessed)
test_pred = pd.DataFrame(test_pred, columns=['pred'])
result = pd.concat([test_id, test_pred], axis=1)
result.to_csv('result_8_2.csv', index=False)