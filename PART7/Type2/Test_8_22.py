import pandas as pd
import numpy as np

train = pd.read_csv('https://raw.githubusercontent.com//YoungjinBD/data/main/exam/8_2_train.csv')
test = pd.read_csv('https://raw.githubusercontent.com//YoungjinBD/data/main/exam/8_2_test.csv')


print(train['count'].head())
print(test.info())

train = train.drop('ID', axis = 1)
test = test.drop('ID', axis = 1)

train_X = train.drop('count', axis=1)
train_y = train['count']

test_X = test.drop('count', axis=1)
test_y = test['count']

from sklearn.model_selection import train_test_split
train_X, valid_X, train_y, valid_y = train_test_split(train_X, train_y, test_size = 0.3, random_state = 1)

from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(sparse_output = False, handle_unknown = 'ignore').set_output(transform = 'pandas')

numc_columns = train_X.select_dtypes('number').columns.to_list()
cat_columns = train_X.select_dtypes('object').columns.to_list()

train_X_encoded = onehotencoder.fit_transform(train_X[cat_columns])
valid_X_encoded = onehotencoder.transform(valid_X[cat_columns])
test_X_encoded = onehotencoder.transform(test_X[cat_columns])

train_X_preprocessed = pd.concat([train_X_encoded, train_X[numc_columns]], axis= 1)
valid_X_preprocessed = pd.concat([valid_X_encoded, valid_X[numc_columns]], axis= 1)
test_X_preprocessed = pd.concat([test_X_encoded, test_X[numc_columns]], axis= 1)

trian_X_full = pd.concat([train_X_preprocessed, valid_X_preprocessed], axis = 0)
train_y_full = pd.concat([train_y, valid_y], axis = 0)
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

param_grid = {'max_depth': [10,20,30],
              'min_samples_split': [2,5,10]}
rf = RandomForestRegressor(random_state = 1)
rf_search = GridSearchCV(
    estimator= rf,
    param_grid= param_grid,
    cv = 3,
    scoring = 'neg_mean_absolute_error'
)
rf_search = rf_search.fit(trian_X_full, train_y_full)
print(-rf_search.best_score_)
print(rf_search.best_params_)



# rf = rf.fit(train_X_preprocessed, train_y)
# from sklearn.metrics import mean_absolute_error
# valid_pred = rf.predict(valid_X_preprocessed)
# print(mean_absolute_error(valid_y, valid_pred))
# test_pred = rf.predict(test_X_preprocessed)
# result = pd.DataFrame(test_pred, columns=['pred'])
# # result.to_csv('result_8_22.csv', index=False)
# print(result)