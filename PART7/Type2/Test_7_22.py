import numpy as np
import pandas as pd

train = pd.read_csv('https://raw.githubusercontent.com//YoungjinBD/data/main/exam/7_2_train.csv')
test = pd.read_csv('https://raw.githubusercontent.com//YoungjinBD/data/main/exam/7_2_test.csv')

print(train.info())
print(test.info())


test_id = test['ID']
train = train.drop(['ID', 'level_0'], axis=1)
test = test.drop(['ID', 'level_0'], axis=1)

train_X = train.drop('Target', axis=1)
train_y = train['Target']
test_X = test.drop('Target', axis=1)
test_y = test['Target']

from sklearn.model_selection import train_test_split
train_X, valid_X, train_y, valid_y = train_test_split(train_X, train_y, test_size=0.3, random_state=1)

from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore').set_output(transform='pandas')

num_columns = train_X.select_dtypes('number').columns.to_list()
cat_columns = train_X.select_dtypes('object').columns.to_list()

train_X_encoded = onehotencoder.fit_transform(train_X[cat_columns])
valid_X_encoded = onehotencoder.transform(valid_X[cat_columns])
test_X_encoded = onehotencoder.transform(test_X[cat_columns])

train_X_preprocessed = pd.concat([train_X_encoded, train_X[num_columns]], axis=1)
valid_X_preprocessed = pd.concat([valid_X_encoded, valid_X[num_columns]], axis=1)
test_X_preprocessed = pd.concat([test_X_encoded, test_X[num_columns]], axis=1)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
rf = RandomForestClassifier(random_state=1)
param_grid = {'max_depth': [10,20,30],
              'min_samples_split': [2,5,10]}
rf_search = GridSearchCV(
    estimator = rf,
    param_grid= param_grid,
    cv = 5,
    scoring='f1_macro' 
)
train_X_full = pd.concat([train_X_preprocessed, valid_X_preprocessed], axis=0)
train_y_full = pd.concat([train_y, valid_y], axis=0)
rf_search = rf_search.fit(train_X_full,train_y_full)
print(rf_search.best_score_)

test_pred2 = rf_search.predict(test_X_preprocessed)
test_pred2 = pd.DataFrame(test_pred2,columns=['pred'])
tmp = pd.concat([test_pred2['pred'], test_y], axis=1)
# print(len(tmp.loc[tmp['pred'] == tmp['Target']]))

#87 0.5739267666876169

# 

# from sklearn.ensemble import RandomForestClassifier
# rf = RandomForestClassifier(random_state=1)
# rf = rf.fit(train_X_preprocessed, train_y)

# from sklearn.metrics import f1_score
# valid_pred = rf.predict(valid_X_preprocessed)
# print(f1_score(valid_y, valid_pred, average='macro'))

# test_pred = rf.predict(test_X_preprocessed)
# test_pred = pd.DataFrame(test_pred,columns=['pred'])
# result = pd.concat([test_id, test_pred], axis=1)
# result.to_csv('result_7_2.csv', index=False)

# tmp = pd.concat([result['pred'], test_y], axis=1)
# print(len(tmp.loc[tmp['pred'] == tmp['Target']]))


#84 0.6128970106561143


