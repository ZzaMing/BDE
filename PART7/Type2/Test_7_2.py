import numpy as np
import pandas as pd

train = pd.read_csv('https://raw.githubusercontent.com//YoungjinBD/data/main/exam/7_2_train.csv')
test = pd.read_csv('https://raw.githubusercontent.com//YoungjinBD/data/main/exam/7_2_test.csv')


print(train['Target'].head(10))

test_id = test['ID']

train = train.drop(['level_0','ID'], axis=1)
test = test.drop(['level_0','ID'], axis=1)

train_X = train.drop('Target', axis=1)
train_y = train['Target']

test_X = test.drop('Target', axis=1)
test_y = test['Target']

from sklearn.model_selection import train_test_split
train_X, valid_X, train_y, valid_y = train_test_split(train_X,train_y, test_size=0.3, random_state=1) # 막힌부분

# from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder # 범주형이 많으니까 라벨인코딩
from sklearn.preprocessing import StandardScaler
cat_columns = train_X.select_dtypes('object').columns.to_list()
num_columns = train_X.select_dtypes('number').columns.to_list()
# onehotencoder = OneHotEncoder(sparse_output = False, handle_unknown='ignore')
ordinalencoder = OrdinalEncoder(unknown_value= -1,handle_unknown= 'use_encoded_value')
stdscaler = StandardScaler()

# train_X_preprocessed = onehotencoder.fit_transform(train_X[cat_columns])
# valid_X_preprocessed = onehotencoder.transform(valid_X[cat_columns])
# test_X_preprocessed = onehotencoder.transform(test_X[cat_columns])

train_X_encoded = ordinalencoder.fit_transform(train_X[cat_columns])
valid_X_encoded = ordinalencoder.transform(valid_X[cat_columns])
test_X_encoded = ordinalencoder.transform(test_X[cat_columns])

train_X_scaled = stdscaler.fit_transform(train_X[num_columns])
valid_X_scaled = stdscaler.transform(valid_X[num_columns])
test_X_scaled = stdscaler.transform(test_X[num_columns])

train_X_preprocessed = np.concatenate([train_X_encoded, train_X_scaled], axis = 1)
valid_X_preprocessed = np.concatenate([valid_X_encoded, valid_X_scaled], axis = 1)
test_X_preprocessed = np.concatenate([test_X_encoded, test_X_scaled], axis = 1)

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=1) # 막힌부분
rf = rf.fit(train_X_preprocessed, train_y)

from sklearn.metrics import f1_score
valid_pred = rf.predict(valid_X_preprocessed) # 막힌부분
print("f1-score: ", f1_score(valid_y, valid_pred, average='macro')) 


# 하이퍼파라미터
train_X_full = np.concatenate([train_X_preprocessed, valid_X_preprocessed], axis=0)
train_y_full = np.concatenate([train_y, valid_y], axis=0)

from sklearn.model_selection import GridSearchCV
# print(RandomForestClassifier().get_params())
from sklearn.metrics import get_scorer_names
# print(get_scorer_names())
rf = RandomForestClassifier(random_state=1)
param_grid = {'max_depth': [10,20,30],
              'min_samples_split': [2,5,10]}
rf_search = GridSearchCV(estimator=rf,
                         param_grid=param_grid,
                         scoring = 'f1_macro',
                         cv=3)
rf_search.fit(train_X_full, train_y_full)
print(rf_search.best_score_) # 0.4198295241182763

test_pred = rf_search.predict(test_X_preprocessed)
test_pred = pd.DataFrame(test_pred, columns=['pred'])
result = pd.concat([test_id, test_pred], axis=1)
tmp = pd.concat([result, test_y], axis=1)
print(len(tmp.loc[tmp['pred'] == tmp['Target']]))


# test_pred = rf.predict(test_X_preprocessed)
# test_pred = pd.DataFrame(test_pred, columns=['pred'])
# result = pd.concat([test_id, test_pred], axis=1)
# result.to_csv('result_7_2.csv', index=False)

# tmp = pd.concat([result, test_y], axis=1)
# print(len(tmp.loc[tmp['pred'] == tmp['Target']])) # 72/133 , 0.3973271300857508
# 72/133 , 0.3973271300857508 // 내가 한거
# 77/133 , 0.4198295241182763 // 내가 전처리 &하이퍼파라미터
# 86/133 , 0.5602603402553307 // 전처리 바꾼거
# 89/133 , 0.5786882239001523 // 바꾼 전처리 + 하이퍼파라미터