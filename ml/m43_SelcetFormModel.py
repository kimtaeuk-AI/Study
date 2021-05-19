from xgboost import XGBClassifier, XGBRFRegressor, XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score, accuracy_score

print('____')

x, y = load_boston(return_X_y=True) #x 랑 y로 바로 간다 x는 특성 y는 타겟
# dataset = load_boston()
# x = dataset.data
# y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

# model = XGBRegressor(n_jobs=8,n_estimators=1000, learning_rate=0.05) # n_jobs 는 학습을 수행하기 위해 CPU 코어 8개를 병렬적으로 사용한다는 의미 

model = XGBRegressor(n_jobs=8)

model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print('r2 : ', score)

thresholds = np.sort(model.feature_importances_) # 디폴트 오름차순 
print(thresholds)
# [0.00134153 0.00363372 0.01203115 0.01220458 0.01447935 0.01479119
#  0.0175432  0.03041655 0.04246345 0.0518254  0.06949984 0.30128643
#  0.42848358] 13개 

for thresh in thresholds :
    selection = SelectFromModel(model, threshold=thresh, prefit=True) # prefit = 접두사 
    select_x_train = selection.transform(x_train)   
    print(select_x_train.shape)

    selection_model = XGBRegressor(n_jobs=8)
    selection_model.fit(select_x_train, y_train)

    select_x_test = selection.transform(x_test)
    y_predict = selection_model.predict(select_x_test)
    score = r2_score(y_test, y_predict)

    print("Thresh=%.3f, n=%d, R2: %.2f%%" %(thresh, select_x_train.shape[1],
          score*100))

     #column 4개를 빼고 작업하는게 더 낫다 

# print(model.coef_)
# print(model.intercept_)
# Coefficients are not defined for Booster type None
    