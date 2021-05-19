from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score,accuracy_score

# x, y = load_boston(return_X_y=True)
datasets = load_boston()
x=datasets.data
y=datasets['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=77)

#2. 모델 
model = XGBRegressor(n_estimators=1000, learning_rate=0.01, n_jobs=8)

#3. 훈련

model.fit(x_train, y_train, verbose=1, eval_metric=['rmse','logloss'],
         eval_set=[(x_train,y_train), (x_test, y_test)],
         early_stopping_rounds=20)

aaa = model.score(x_test, y_test)
print('score : ' , aaa)

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print('r2 : ', r2)

#validation 0 - train
#validation 1 = teest

# print('==================================')
results = model.evals_result()
print(results)

import matplotlib.pyplot as plt

epochs = len(results['validation_0']['logloss']) 
x_axis = range(0, epochs)

fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
ax.plot(x_axis, results['validation_1']['logloss'], label='Test')
ax.legend()
plt.ylabel('Log Loss')
plt.title('XGBoost Log Loss')

fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['rmse'], label='Train')
ax.plot(x_axis, results['validation_0']['rmse'], label='Test')
ax.legend()
plt.ylabel('Rmse')
plt.title('XGBoost RMSE')
plt.show()
