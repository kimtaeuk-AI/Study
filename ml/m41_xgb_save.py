from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score,accuracy_score
#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ#

# x, y = load_boston(return_X_y=True)
datasets = load_boston()
x=datasets.data
y=datasets['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=77)

#2. 모델 
model = XGBRegressor(n_estimators=1000, learning_rate=0.01, n_jobs=8)

#3. 훈련

model.fit(x_train, y_train, verbose=1, eval_metric=['rmse'],
         eval_set=[(x_train,y_train), (x_test, y_test)],
         early_stopping_rounds=20)

aaa = model.score(x_test, y_test)
print('score : ' , aaa)

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print('r2 : ', r2)

#validation 0 - train
#validation 1 = test

# print('==================================')
results = model.evals_result()
# print(results)
#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ#


#============================pikcle===============================#
#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ#

# 저장
import pickle

pickle.dump(model, open('./data/xgb_save/m39_pickle.data','wb')) #wb : 쓰겠다 ?
print('저장완료')
print('================== pickle 불러오기 ================')

#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ#

# 불러오기
model2 = pickle.load(open('./data/xgb_save/m39_pickle.data','rb')) #rb : 불러오겠다?
print('불러왔다!')
r22 = model2.score(x_test, y_test)
print('r22 : ', r22)

#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ#
#============================pikcle===============================#



#============================joblib===============================#
#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ#

# import joblib
# joblib.dump(model, './data/xgb_save/m39_joblib.data')
# print('저장완료')

# model2 = joblib.load('./data/xgb_save/m39_joblib.data')
# print('불러왔다!')
# r22 = model2.score(x_test, y_test)
# print('r22 : ', r22)
#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ#
#============================joblib===============================#



#============================xgb_save===============================#
#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ#

# model.save_model('./data/xgb_save/m39_xgb.model')
# print('저장완료')

# model2 = XGBRegressor() # 헷갈릴수 있어서 ?
# model2.load_model('./data/xgb_save/m39_xgb.model')
# r22 = model2.score(x_test, y_test)
# print('불러왔다!')
# print('r22 : ', r22)
#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ#
#============================xgb_save===============================#
