import pandas as pd
import numpy as np

train = pd.read_csv(
    "https://raw.githubusercontent.com/YoungjinBD/data/main/st_train.csv"
)
test = pd.read_csv("https://raw.githubusercontent.com/YoungjinBD/data/main/st_test.csv")

# 1. 데이터 탐색 / 결측치 확인 -> goout 컬럼에 결측치 존재.
print(train.info())
print(test.info())

# 2. 데이터 분할 / 훈련데이터의 일부를 검증데이터로 분할 -> 과적합 방지를 위한것.
train_X = train.drop(["grade"], axis=1)
train_y = train["grade"]

test_X = test.drop(["grade"], axis=1)
test_y = test["grade"]

from sklearn.model_selection import train_test_split

train_X, valid_X, train_y, valid_y = train_test_split(
    train_X, train_y, test_size=0.3, random_state=1
)

print(train_X.shape, train_y.shape, valid_X.shape, valid_y.shape)

# 3. 데이터 전처리 / 범주형은 원핫인코딩, 수치형은 임퓨터, 결측치는 대치
num_columns = train_X.select_dtypes("number").columns.to_list()
cat_columns = train_X.select_dtypes("object").columns.to_list()

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

imputer = SimpleImputer(strategy="mean")
onehotencoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

train_X_numeric_imputed = imputer.fit_transform(train_X[num_columns])
valid_X_numeric_imputed = imputer.transform(valid_X[num_columns])
test_X_numeric_imputed = imputer.transform(test_X[num_columns])

train_X_categorical_encoded = onehotencoder.fit_transform(train_X[cat_columns])
valid_X_categorical_encoded = onehotencoder.transform(valid_X[cat_columns])
test_X_categorical_encoded = onehotencoder.transform(test_X[cat_columns])

train_X_preprocessed = np.concatenate(
    [train_X_numeric_imputed, train_X_categorical_encoded], axis=1
)
valid_X_preprocessed = np.concatenate(
    [valid_X_numeric_imputed, valid_X_categorical_encoded], axis=1
)
test_X_preprocessed = np.concatenate(
    [test_X_numeric_imputed, test_X_categorical_encoded], axis=1
)

# 4. 모델 적합 / 기본 성능 보장되는 랜덤포레스트 사용
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(random_state=1)
rf.fit(train_X_preprocessed, train_y)
RandomForestRegressor(random_state=1)

from sklearn.metrics import root_mean_squared_error

pred_val = rf.predict(valid_X_preprocessed)
# print('valid RMSE: ', mean_squared_error(valid_y, pred_val))
print("valid RMSE: ", root_mean_squared_error(valid_y, pred_val))


# 5. 테스트 데이터로 예측
test_pred = rf.predict(test_X_preprocessed)
test_pred = pd.DataFrame(test_pred, columns=["pred"])
# test_pred.to_csv('result.csv', index = False)

# 6. 하이퍼파라미터 튜닝
train_X_full = np.concatenate([train_X_preprocessed, valid_X_preprocessed], axis=0)
train_y_full = np.concatenate([train_y, valid_y], axis=0)

from sklearn.model_selection import GridSearchCV

param_grid = {"max_depth": [10, 20, 30], "min_samples_split": [2, 5, 10]}
rf = RandomForestRegressor(random_state=1)
rf_search = GridSearchCV(estimator= rf,
                         param_grid= param_grid,
                         cv = 3,
                         scoring= 'neg_root_mean_squared_error')
rf_search.fit(train_X_full, train_y_full)
print('교차검증 RMSE: ', -rf_search.best_score_)

test_pred2 = rf_search.predict(test_X_preprocessed)
test_pred2 = pd.DataFrame(test_pred2, columns= ['pred'])
# test_pred2.to_csv('result.csv', index=False)