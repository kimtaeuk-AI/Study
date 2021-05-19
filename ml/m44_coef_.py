x = [-3, 31, -11, 4, 0, 22, -2, -5, -25, -14]

# y = [-2, 32, -10, 5, 1, 23, -1, -4, -24, -13]

# y = [-5, 63, -21, 9, 1, 45, -3, -9, -49 ,-27]

y = [-3, 65, -19, 11, 3, 47, -1, -7, -47, -25]

print(x, '\n', y)

import matplotlib.pyplot as plt

plt.plot(x,y)
# plt.show()
import pandas as pd 

df = pd.DataFrame({'X':x, 'Y':y})

print(df)
print(df.shape)

x_train = df.loc[:, 'X']
y_train = df.loc[:, 'Y'] #(10,) 라 시리즈
print(x_train.shape, y_train.shape) #(10) (10,)
print(type(x_train))

# 스칼라 -> 벡터 -> 행렬 -> 텐서 

x_train = x_train.values.reshape(len(x_train),1)
print(x_train.shape, y_train.shape) #(10,1) (10,)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train, y_train)

score = model.score(x_train, y_train)
print('score : ', score)

print('기울기 : ' ,model.coef_) # 가중치 weight
print('절편 : ' , model.intercept_) # bias 
