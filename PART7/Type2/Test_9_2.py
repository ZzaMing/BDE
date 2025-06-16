# # Target: 농업유형, 평가지표: Macro F1 Score
# import numpy as np
# import pandas as pd

# train = pd.read_csv('https://raw.githubusercontent.com//YoungjinBD/data/main/exam/9_2_train.csv')
# test = pd.read_csv('https://raw.githubusercontent.com//YoungjinBD/data/main/exam/9_2_test.csv')

# # 1. 결측치 확인
# # print(train.info())
# # print(test.info())
# test_id = test['ID']

# train_X = train.drop(['라벨'], axis= 1)
# train_y = train['라벨'] #종속변수

# test_X = test.drop(['라벨'], axis=1)
# test_y = test['라벨']

# # 2. 데이터 분할 / valid스플릿
# from sklearn.model_selection import train_test_split
# train_X, valid_X, train_y, valid_y =  train_test_split(train_X, train_y, test_size=0.3, random_state=1)


# # 3. 데이터 전처리 / 원핫인코더
# cat_columns = train_X.select_dtypes('object').columns.to_list()
# num_columns = train_X.select_dtypes('number').columns.to_list()

# from sklearn.preprocessing import OneHotEncoder
# onehotencoder = OneHotEncoder(sparse_output = False, handle_unknown='ignore')

# train_X_preprocessed = onehotencoder.fit_transform(train_X[cat_columns])
# valid_X_preprocessed = onehotencoder.transform(valid_X[cat_columns])
# test_X_preprocessed = onehotencoder.transform(test_X[cat_columns])

# # 4. 모델 적합
# from sklearn.ensemble import RandomForestClassifier
# rf = RandomForestClassifier(random_state = 1)
# # rf.fit(입력데이터, 정답데이터) // 인코딩(입력데이터) 된거 알려주고 , 분류되어있는거(정답데이터)로 학습시키기.
# rf.fit(train_X_preprocessed, train_y)

# from sklearn.metrics import f1_score
# # 평가 지표가 ROC, AUC, 확률 이라는 표현이 있으면 predict_proba 사용 !
# pred_val = rf.predict(valid_X_preprocessed)
# print('valid f1-macro: ', f1_score(valid_y, pred_val, average='macro')) #f1_score(정답데이터, 입력데이터예측한거)

# test_pred = rf.predict(test_X_preprocessed) # 인코딩 된거 
# test_pred = pd.DataFrame(test_pred, columns=['pred'])

# result = pd.concat([test_id, test_pred], axis = 1)
# # print(result)
# result.to_csv('9_2_result.csv', index= False)




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

from sklearn.model_selection import train_test_split
train_X, valid_X, train_y, valid_y = train_test_split(train_X, train_y, test_size=0.3, random_state=1)

# 3. 데이터 전처리
from sklearn.preprocessing import OneHotEncoder
cat_columns = train_X.select_dtypes('object').columns.to_list()
onehotencoder = OneHotEncoder(sparse_output=False, handle_unknown = 'ignore')

train_X_preprocessed = onehotencoder.fit_transform(train_X[cat_columns])
valid_X_preprocessed = onehotencoder.transform(valid_X[cat_columns])
test_X_preprocessed = onehotencoder.transform(test_X[cat_columns])


# 4. 모델 적합
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=1)
rf = rf.fit(train_X_preprocessed, train_y)

from sklearn.metrics import f1_score
val_pred = rf.predict(valid_X_preprocessed)
print(f1_score(valid_y, val_pred, average='macro'))

test_pred = rf.predict(test_X_preprocessed)
test_pred = pd.DataFrame(test_pred, columns=['pred'])
result = pd.concat([test_id, test_pred], axis=1)
result.to_csv('result.csv', index=False)






