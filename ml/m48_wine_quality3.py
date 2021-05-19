import pandas as pd
import numpy as np


wine = pd.read_csv('C:/Study/winequality-white.csv', index_col=None, header=0, sep=';')

count_data = wine.groupby('quality')['quality'].count() # 데이터 분포 
print(count_data)

# print(pd.Series.unique(count_data('quality'))) #다시알아보기 
# print(np.unique(count_data['quality'])) #다시알아보기 
print(np.unique(wine['quality'])) # [3 4 5 6 7 8 9]

import matplotlib.pyplot as plt
count_data.plot()
# plt.show()


#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ#

wine_npy = wine.values

# x = wine_npy[:, :11]
# y = wine_npy[:, 11]

y = wine['quality']
x = wine.drop('quality', axis=1)

print(y[100:110]) # 5,6 이 들어있다

newlist = []
for i in list(y):             # 4 보다 작으면 0 등급, 7보다 작으면 1등급 나머지는 2등급 
    if i <=4:  
        newlist += [0] 
    elif i <=7:
        newlist += [1]
    else:
        newlist += [2]
y = newlist


# print(x.shape) #(4898, 11)
# print(y.shape) #(4898,)

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

# model = KNeighborsClassifier() #score :  0.9275510204081633 상중하로 나눴다. 카테고리를 줄인다.(무조건줄이는건 좋지않음)
model = RandomForestClassifier() #score :  0.9377551020408164 
# model = XGBClassifier() #score :  0.9357142857142857

model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print("score : " , score)
