import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)

train = pd.read_csv('https://raw.githubusercontent.com//YoungjinBD/data/main/exam/6_2_train.csv')
test = pd.read_csv('https://raw.githubusercontent.com//YoungjinBD/data/main/exam/6_2_test.csv')

# 1. 결측치 확인 & 데이터 탐색
# print(train.info()) #gender결측치
# print(test.info()) #gender결측치
# print(train.head())
# print(test.head())

# 2. 데이터 분할
test_id = test['ID']

train = train.drop(['level_0', 'ID'], axis=1)
test = test.drop(['level_0', 'ID'], axis=1)

train_X = train.drop('DBP', axis=1)
train_y = train['DBP']

test_X = test.drop('DBP', axis=1)
test_y = test['DBP']

from sklearn.model_selection import train_test_split
train_X, valid_X, train_y, valid_y = train_test_split(train_X, train_y, test_size = 0.3, random_state= 1)

# 3. 데이터 전처리
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
onehotencoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
imputer = SimpleImputer(strategy='most_frequent')
cat_columns = train_X.select_dtypes('object').columns.to_list()
num_columns = train_X.select_dtypes('number').columns.to_list()

train_X_encoded = onehotencoder.fit_transform(train_X[cat_columns])
valid_X_encoded = onehotencoder.transform(valid_X[cat_columns])
test_X_encoded = onehotencoder.transform(test_X[cat_columns])

train_X_imputed = imputer.fit_transform(train_X[num_columns])
valid_X_imputed = imputer.transform(valid_X[num_columns])
test_X_imputed = imputer.transform(test_X[num_columns])

train_X_preprocessed = np.concatenate([train_X_encoded, train_X_imputed], axis=1)
valid_X_preprocessed = np.concatenate([valid_X_encoded, valid_X_imputed], axis=1)
test_X_preprocessed = np.concatenate([test_X_encoded, test_X_imputed], axis=1)

# 4. 모델 학습
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(random_state = 1)


# from sklearn.metrics import get_scorer_names # scorer 이름 찾기
# print(get_scorer_names())
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import root_mean_squared_error
rf = RandomForestRegressor(random_state=1)
param_grid = {"max_depth": [10,20,30],
               "min_samples_split": [2,5,10]}
rf_search = GridSearchCV(estimator=rf,
                         param_grid=param_grid,
                         scoring= 'neg_root_mean_squared_error',
                         cv = 3)
train_X_full = np.concatenate([train_X_preprocessed, valid_X_preprocessed],axis=0)
train_y_full = np.concatenate([train_y, valid_y], axis=0)
rf_search.fit(train_X_full, train_y_full)
print("rf_search_best_score: ", -rf_search.best_score_)
valid_pred2 = rf_search.predict(valid_X_preprocessed)
print("rf_search_RMSE: ", root_mean_squared_error(valid_y, valid_pred2))

rf = rf.fit(train_X_preprocessed, train_y)
valid_pred = rf.predict(valid_X_preprocessed)
print("rf_RMSE: ", root_mean_squared_error(valid_y, valid_pred))

# 5. 테스트 데이터 예측
# test_pred = rf.predict(test_X_encoded)
# test_pred = pd.DataFrame(test_pred, columns=['pred'])
# result = pd.concat([test_id, test_pred], axis= 1)
# tmp = pd.concat([result, test_y], axis=1)
test_pred = rf_search.predict(test_X_preprocessed)
test_pred = pd.DataFrame(test_pred, columns=['pred'])
result = pd.concat([test_id, test_pred], axis=1)
result.to_csv('result_6_2.csv', index=False)

# 13.085002924372565 // 결측치 처리 못한거
# 13.074669905805962 // 결측치 처리 못하고 하이퍼파라미터 적용
# 12.537756340985977 // 결측치 처리.
# 11.098293072705692 // 결측치 처리하고 하이퍼파라미터 적용
# rf_search_best_score:  124.04110237275903
# rf_search_RMSE:  4.854267299797955