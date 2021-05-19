import numpy as np
import os
import pandas as pd

from sklearn.datasets import load_wine
#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ#
# datasets = load_wine()
# print(datasets.feature_names)
# print(datasets.target_names)

# x= datasets.data
# y= datasets.target

# print(x.shape, y.shape)
# print(np.unique(y))

#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ#

wine = pd.read_csv('C:/Study/winequality-white.csv', index_col=None, header=0, sep=';')  # index 없음 

# print(wine.head()) 
# print(wine.shape) #(4898, 12)
# print(wine.describe()) # -pandas라 describe 먹힘. 25 % 50 % 75% 어제배운거 검색해보기 

wine_npy = wine.values

x = wine_npy[:, :11]
y = wine_npy[:, 11]
# print(x)
# print(y)
print(wine)
# print(x.shape) #(4898, 11)
# print(y.shape) #(4898,)
'''
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=77)

scale = StandardScaler()
scale.fit(x_train)
x_train = scale.transform(x_train)
x_test = scale.transform(x_test)
print(x_train.shape, x_test.shape) #(3918, 11) (980, 11)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# model = KNeighborsClassifier() #score :  0.5642857142857143
model = RandomForestClassifier() #score :  0.6693877551020408 , score :  0.6846938775510204
# model = XGBClassifier() #score :  0.6755102040816326

model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print("score : " , score)

# 원래 RandomForestClassifier 이게 더좋아야하는데..?





'''