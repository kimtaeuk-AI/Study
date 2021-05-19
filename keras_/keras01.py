import numpy as np
import tensorflow as tf

#1. 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

#2. 모델구성
from tensorflow.keras.models import Sequential #폴더구조 , 마지막만 쓸꺼니까 import 씀
from tensorflow.keras.layers import Dense #기초적

model = Sequential()
model.add(Dense(5, input_dim=1, activation='linear')) # input dimention 한계??
model.add(Dense(3, activation='linear'))
model.add(Dense(4))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam')
model.fit(x, y, epochs = 1000, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x, y, batch_size=1)
print('loss : ', loss)

x_pred = np.array([4])

#result = model.predict([4])
#result = model.predict(x)

print('result : ', result)