import numpy as np
import pandas as pd

train = pd.read_csv('https://raw.githubusercontent.com//YoungjinBD/data/main/exam/5_2_train.csv')
test = pd.read_csv('https://raw.githubusercontent.com//YoungjinBD/data/main/exam/5_2_test.csv')
# AUC , 1일 확률
#1. 결측치 / fractal_dimension_error
# print(train.info())
print(train['fractal_dimension_error'].head())
# print(test.info())
# print(test.head())

#2. 데이터 분할
test_id = test['ID']

train = train.drop(['level_0','ID'], axis=1)
test = test.drop(['level_0','ID'], axis=1)

train_X = train.drop('target', axis=1)
train_y = train['target']

test_X = test.drop('target', axis=1)
test_y = test['target']

from sklearn.model_selection import train_test_split
train_X, valid_X, train_y, valid_y = train_test_split(train_X, train_y, test_size=0.3, random_state=1)

#3. 데이터 전처리
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = 'mean')
num_columns = train_X.select_dtypes('number').columns.to_list()
# cat_columns = train_X.select_dtypes('object').columns.to_list()

train_X_preprocessed = imputer.fit_transform(train_X[num_columns])
valid_X_preprocessed = imputer.transform(valid_X[num_columns])
test_X_preprocessed = imputer.transform(test_X[num_columns])

#4. 모델 적합
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=1)
from sklearn.model_selection import GridSearchCV
param_grid = {'max_depth': [10,20,30],
              'min_samples_split': [2,5,10]}
rf_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=3,
    scoring= 'roc_auc')
train_X_full = np.concatenate([train_X_preprocessed, valid_X_preprocessed], axis = 0)
train_y_full = np.concatenate([train_y, valid_y], axis = 0 )
rf_search.fit(train_X_full, train_y_full)
print(rf_search.best_score_)

test_pred = rf_search.predict_proba(test_X_preprocessed)[:,1]
test_pred = pd.DataFrame(test_pred, columns=['prob'])
result = pd.concat([test_id, test_pred], axis=1)
result.to_csv('result_5_2.csv', index=False)

# rf.fit(train_X_preprocessed, train_y)

# from sklearn.metrics import roc_auc_score
# # valid_pred = rf.predict_proba(valid_X_preprocessed)[:,1]
# print(valid_pred)
# print("AUC: ", roc_auc_score(valid_y, valid_pred))

#5. 테스트값 입력
# test_pred = rf.predict_proba(test_X_preprocessed)
# test_pred = pd.DataFrame(test_pred, columns=['pred'])
# result = pd.concat([test_id, test_pred], axis=1)
#0.9915433403805497
#0.9942283419391852 하이퍼 파라미터