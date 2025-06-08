import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()

X = data.data
y = data.target

df = pd.DataFrame(X, columns=data.feature_names)
df["target"] = y

# print(df.head())

# 0. 훈련데이터와 테스트 데이터 분할
from sklearn.model_selection import train_test_split

train_X, test_X, train_y, text_y = train_test_split(
    df.drop(columns="target"), df["target"], test_size=0.3, random_state=42
)

# 1. 데이터 탐색 / 결측치 또는 특이치 확인
# print(train_X.info())
# print(test_X.info())

# 2. 데이터 분할 / 훈련데이터의 일부를 검증데이터로 분할
from sklearn.model_selection import train_test_split

train_X, valid_X, train_y, valid_y = train_test_split(
    train_X, train_y, test_size=0.3, random_state=1
)
print(train_X.shape, train_y.shape, valid_X.shape, valid_y.shape)

# 3. 데이터 전처리 / 범주형은 원핫인코딩, 결측치는 결측치 대치방법 수행
cat_columns = train_X.select_dtypes("object").columns.to_list()
num_columns = train_X.select_dtypes("number").columns.to_list()

from sklearn.preprocessing import OneHotEncoder

onehotencoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

train_X_categorical_encoded = onehotencoder.fit_transform(train_X[cat_columns])
valid_X_categorical_encoded = onehotencoder.transform(valid_X[cat_columns])
test_X_categorical_encoded = onehotencoder.transform(test_X[cat_columns])

train_X_preprocessed = np.concatenate(
    [train_X[num_columns], train_X_categorical_encoded], axis=1
)
valid_X_preprocessed = np.concatenate(
    [valid_X[num_columns], valid_X_categorical_encoded], axis=1
)
test_X_preprocessed = np.concatenate(
    [test_X[num_columns], test_X_categorical_encoded], axis=1
)

# 4. 모델 적합 / 기본 성능 보장되는 랜덤포레스트 사용
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=1)
rf.fit(train_X_preprocessed, train_y)
RandomForestClassifier(random_state=1)

from sklearn.metrics import f1_score

pred_val = rf.predict(valid_X_preprocessed)
print(f1_score(valid_y, pred_val, average="macro"))

# 5. 테스트 데이터로 예측
test_pred = rf.predict(test_X_preprocessed)
test_pred = pd.DataFrame(test_pred, columns=['pred'])
# test_pred.to_csv('result.csv', index=False)

# 6. 하이퍼파라미터 튜닝 / 가능하면 데이터 전처리과정에서 실수가 있었느지 먼저 확인
train_X_full = np.concatenate([train_X_preprocessed, valid_X_preprocessed], axis=0)
train_y_full = np.concatenate([train_y, valid_y], axis = 0)

from sklearn.model_selection import GridSearchCV
param_grid = {'max_depth': [10,20,30],
              'min_samples_split': [2,5,10]}

rf = RandomForestClassifier(random_state= 1)
rf_search = GridSearchCV(estimator= rf,
                         param_grid= param_grid,
                         cv = 3,
                         scoring='f1_macro')

rf_search.fit(train_X_full, train_y_full)

print('f1_score: ', rf_search.best_score_)

test_pred2 = rf_search.predict(test_X_preprocessed)
test_pred2 = pd.DataFrame(test_pred2, columns=['pred'])

# test_pred2.to_csv('result.csv', index = False)