import pandas as pd
import numpy as np

train = pd.read_csv('https://raw.githubusercontent.com//YoungjinBD/data/main/exam/9_2_train.csv')
test = pd.read_csv('https://raw.githubusercontent.com//YoungjinBD/data/main/exam/9_2_test.csv')

# print(train.head())
# print(test.head())

# 1. 결측치 확인
# print(test.info())
# print(train.info())


# 2. 데이터 분할
test_id = test['ID']

train_X = train.drop('라벨', axis = 1)
train_y = train['라벨']

test_X = test.drop('라벨', axis =1)
test_y = test['라벨']

# from sklearn.model_selection import train_test_split
# train_X, valid_X, train_y, valid_y = train_test_split(train_X, train_y, test_size=0.3, random_state=1)

# 3. 데이터 전처리
from sklearn.preprocessing import OneHotEncoder
cat_columns = train_X.select_dtypes('object').columns.to_list()
num_columns = train_X.select_dtypes('number').columns.to_list()
onehotencoder = OneHotEncoder(sparse_output=False, handle_unknown = 'ignore').set_output(transform='pandas')

train_X_encoded = onehotencoder.fit_transform(train_X[cat_columns])
# valid_X_encoded= onehotencoder.transform(valid_X[cat_columns])
test_X_encoded = onehotencoder.transform(test_X[cat_columns])

train_X_preprocessed = pd.concat([train_X_encoded, train_X[num_columns]], axis=1)
# valid_X_preprocessed = pd.concat([valid_X_encoded, valid_X[num_columns]], axis=1)
test_X_preprocessed = pd.concat([test_X_encoded, test_X[num_columns]], axis=1)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
rf = RandomForestClassifier(random_state=1)
param_grid = {'max_depth': [10,20,30],
              'min_samples_split': [2,5,10]}
rf_search = GridSearchCV(
    estimator = rf,
    param_grid= param_grid,
    cv = 3,
    scoring='f1_macro' 
)
# train_X_full = pd.concat([train_X_preprocessed, valid_X_preprocessed], axis=0)
# train_y_full = pd.concat([train_y, valid_y], axis=0)
rf_search = rf_search.fit(train_X_preprocessed,train_y)
print(rf_search.best_score_)