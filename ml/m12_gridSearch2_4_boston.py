# 모델 : RandomForestClasstifier

import numpy as np
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score, r2_score

dataset = load_boston()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=44)

kfold = KFold(n_splits=5, shuffle=True) # 5등분으로 나눈다 

# parameters = [
#     {'n_estimators' : [100,200]},
#     {'max_depth' : [6, 8, 10 ,12]},
#     {'min_samples_leaf' : [3,5,7,10]},
#     {'min_samples_split' : [2, 3, 5, 10]},
#     {'n_jobs' : [-1, 2, 4]}
# ]

parameters = [
    {'n_estimators' : [100,10], 'n_jobs' : [-1]},
    {'max_depth' : [6,8,10,12], 'min_samples_split' : [2,3,5,10]},
    {'min_samples_split' : [2,3,5,10], 'max_depth' : [6,8,10,12]},
    {'n_jobs' : [-1], 'min_samples_leaf' : [3,5,7,10]},
    {'max_depth' : [6,8,10]}
]

#2. 모델
# model = RandomForestClassifier(SVC(), parameters, cv=kfold )
model = GridSearchCV(RandomForestRegressor(), parameters, cv = kfold)

scores = cross_val_score(model,x_train, y_train, cv=kfold) # cross_validation

#3.훈련 
model.fit(x_train, y_train)
# fit , train 이 다 포함되어있다.

#4. 평가, 예측
print("최적의 매개변수 : ", model.best_estimator_)

y_pred = model.predict(x_test) # gird 서치에서 가장 좋은놈을 빼준다 
print("최종정답률 : ", r2_score(y_pred, y_test))

# aaa = model.score(x_test, y_test)

# print("aaa : " , aaa)