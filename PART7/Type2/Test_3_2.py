import numpy as np
import pandas as pd

train_X = pd.read_csv('https://raw.githubusercontent.com//YoungjinBD/data/main/exam/3_2_trainX.csv')
train_y = pd.read_csv('https://raw.githubusercontent.com//YoungjinBD/data/main/exam/3_2_trainy.csv')
test_X = pd.read_csv('https://raw.githubusercontent.com//YoungjinBD/data/main/exam/3_2_testX.csv')

train_y = train_y['Outcome'] # 오류 방지....?

# print(train_X.info())
# print(train_y.info())
# print(test_X.info())

from sklearn.model_selection import train_test_split
train_X, valid_X, train_y, valid_y = train_test_split(train_X, train_y, test_size=0.3, random_state=1)

from sklearn.preprocessing import StandardScaler
stdscaler = StandardScaler().set_output(transform='pandas')

num_columns = train_X.select_dtypes('number').columns.to_list()
cat_columns = train_X.select_dtypes('object').columns.to_list()

train_X_scaled = stdscaler.fit_transform(train_X[num_columns])
valid_X_scaled = stdscaler.transform(valid_X[num_columns])
# train_X_scaled = stdscaler.transform(train_X[num_columns])

train_X_preprocessed = pd.concat([train_X_scaled, train_X[cat_columns]], axis = 1)
valid_X_preprocessed = pd.concat([valid_X_scaled, valid_X[cat_columns]], axis = 1)
# print(train_X_preprocessed)

from sklearn.ensemble import RandomForestClassifier #0.7543800539083558
from sklearn.model_selection import GridSearchCV
rf = RandomForestClassifier(random_state=1)
param_grid = {'max_depth': [10,20,30],
              'min_samples_split': [2,5,10]}
rf_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=3,
    scoring='f1_macro'
)
train_X_full = pd.concat([train_X_preprocessed,valid_X_preprocessed], axis=0)
train_y_full = pd.concat([train_y, valid_y], axis = 0)
rf_search = rf_search.fit(train_X_full, train_y_full)
print(rf_search.best_score_)

test_pred = rf_search.predict(test_X)
test_pred = pd.DataFrame(test_pred, columns=['pred'])
# result = pd.concat()

# rf = rf.fit(train_X_preprocessed, train_y)
# from sklearn.metrics import f1_score
# valid_pred = rf.predict(valid_X_preprocessed)
# print(f1_score(valid_y, valid_pred, average='macro'))




