# 실습
# 1.상단모델에 그리드서치 또는 랜덤서치로 튜닝한 모델 구성 
# 최적의 R2값과 피쳐임포턴스 구할것

#2. 위 쓰레드 값으로 SelectFromModel을 구해서 
# 최적의 피처 갯수를 구할것

#3. 위 피쳐 갯수로 뎅이터(피처)를 수정(삭제)해서
# 그리드서치 또는 랜덤서치 적용해서
# 최적의 R2 구할것

# 1번값과 2번값을 비교 

from xgboost import XGBClassifier, XGBRFRegressor, XGBRegressor
from sklearn.datasets import load_boston, load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score, accuracy_score
from sklearn.svm import LinearSVC, SVC

dataset = load_boston()
x = dataset.data
y = dataset.target
# x, y = load_boston(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

kfold = KFold(n_splits=5, shuffle=True) # 5등분으로 나눈다

parameters = [
    {'n_estimators' : [100,10], 'n_jobs' : [-1]},
    {'max_depth' : [6,8,10,12], 'min_samples_split' : [2,3,5,10]},
    {'min_samples_split' : [2,3,5,10], 'max_depth' : [6,8,10,12]},
    {'n_jobs' : [-1], 'min_samples_leaf' : [3,5,7,10]},
    {'max_depth' : [6,8,10]}
]

model = GridSearchCV(SVC(), parameters, cv=kfold) # n_jobs 는 학습을 수행하기 위해 CPU 코어 8개를 병렬적으로 사용한다는 의미 
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
scores = r2_score(y_pred, y_test) # cross_validation



# score = model.score(x_test, y_test)

print('r2 : ', scores)
'''
thresholds = np.sort(model.feature_importances_) # 디폴트 오름차순 
print(thresholds)
# [0.00134153 0.00363372 0.01203115 0.01220458 0.01447935 0.01479119
#  0.0175432  0.03041655 0.04246345 0.0518254  0.06949984 0.30128643
#  0.42848358] 13개 

# for thresh in thresholds :
#     selection = SelectFromModel(model, threshold=thresh, prefit=True) # prefit = 접두사 
#     select_x_train = selection.transform(x_train)   
#     print(select_x_train.shape)

#     selection_model = XGBRegressor(n_jobs=8)
#     selection_model.fit(select_x_train, y_train)

#     select_x_test = selection.transform(x_test)
#     y_predict = selection_model.predict(select_x_test)
#     score = r2_score(y_test, y_predict)

#     print("Thresh=%.3f, n=%d, R2: %.2f%%" %(thresh, select_x_train.shape[1],
#           score*100))

     #column 4개를 빼고 작업하는게 더 낫다 

print("최적의 매개변수 : ", model.best_estimator_)

y_pred = model.predict(x_test) # gird 서치에서 가장 좋은놈을 빼준다 
print("최종정답률 : ", accuracy_score(y_test, y_pred))

aaa = model.score(x_test, y_test)

print("aaa : " , aaa)
'''