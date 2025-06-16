import numpy as np
import pandas as pd

train = pd.read_csv('https://raw.githubusercontent.com//YoungjinBD/data/main/exam/4_2_train.csv')
test = pd.read_csv('https://raw.githubusercontent.com//YoungjinBD/data/main/exam/4_2_test.csv')

# 1. 결측치 확인 / 결측치 없음.
# print(train.info()) 
# print(test.info())
# print(train.head(10))

# 2. 데이터 분할
from sklearn.model_selection import train_test_split

test_id = test['ID']

train = train.drop('ID', axis = 1)
test = test.drop('ID', axis = 1)

train_X = train.drop('Attrition_Flag', axis=1)
train_y = train['Attrition_Flag']

test_X = test.drop('Attrition_Flag', axis=1)
test_y = test['Attrition_Flag']

train_X, valid_X, train_y, valid_y =train_test_split(train_X, train_y, test_size= 0.3, random_state=1)

# 3. 데이터 전처리
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
stdscaler = StandardScaler()
onehotencoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
cat_columns = train_X.select_dtypes('object').columns.to_list()
num_columns = train_X.select_dtypes('number').columns.to_list()

train_X_encoded = onehotencoder.fit_transform(train_X[cat_columns])
valid_X_encoded = onehotencoder.transform(valid_X[cat_columns])
test_X_encoded = onehotencoder.transform(test_X[cat_columns])

train_X_scaled = stdscaler.fit_transform(train_X[num_columns])
valid_X_scaled = stdscaler.transform(valid_X[num_columns])
test_X_scaled = stdscaler.transform(test_X[num_columns])

train_X_preprocessed = np.concatenate([train_X_encoded, train_X_scaled], axis=1)
valid_X_preprocessed = np.concatenate([valid_X_encoded, valid_X_scaled], axis=1)
test_X_preprocessed = np.concatenate([test_X_encoded, test_X_scaled], axis=1)

# 4. 모델 적합 / AUC:  0.3760302197802198 / AUC:  0.773695054945055
# from sklearn.ensemble import RandomForestClassifier
# rf = RandomForestClassifier(random_state=1)
# rf = rf.fit(train_X_preprocessed, train_y)

# from sklearn.metrics import roc_auc_score
# valid_pred = rf.predict_proba(valid_X_preprocessed)[:,1]
# print('AUC: ', roc_auc_score(valid_y, valid_pred))

# 4+. 하이퍼파라미터 / AUC:  0.7451995685005394
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import get_scorer_names
# print(get_scorer_names())
train_X_full = np.concatenate([train_X_preprocessed, valid_X_preprocessed], axis= 0)
train_y_full = np.concatenate([train_y, valid_y])
rf = RandomForestClassifier(random_state=1)
param_gird  = {'max_depth': [10,20,30],
               'min_samples_split': [2,5,10]}
rf_search = GridSearchCV(
    estimator =  rf,
    param_grid = param_gird ,
    cv= 3,
    scoring = 'roc_auc'
)
rf_search.fit(train_X_full, train_y_full)
print('AUC: ', rf_search.best_score_)

# 5. 테스트 데이터 예측
# test_pred = rf.predict_proba(test_X_preprocessed)[:,1]
# test_pred = pd.DataFrame(test_pred, columns=['prob'])
# result = pd.concat([test_id, test_pred], axis=1)
# result.to_csv('result_4_2.csv', index=False)
# print(result)

# 5+. 데스트 데이터 예측
test_pred = rf_search.predict_proba(test_X_preprocessed)[:,1]
test_pred = pd.DataFrame(test_pred, columns=['prob'])
result = pd.concat([test_id, test_pred], axis = 1)
result.to_csv('result_4_2.csv', index=False)
print(result)