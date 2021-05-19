# (n, 28, 28) = (n, 28*28) = (n, 764)
# 주말 과제
#  dense  모델로 구성  input_shape=(28*28,)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tensorflow.keras.datasets import mnist
#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ#

# 데이터

(x_train, _), (x_test, _) =mnist.load_data() # _ -> 안하겠다.
x = np.append(x_train, x_test, axis=0)
x = x.reshape(70000, 784)

(_, y_train), (_, y_test) = mnist.load_data()
y_train = y_train.reshape(y_train.shape[0],1)
y_test = y_test.reshape(y_test.shape[0],1)


x_train = x_train.reshape(60000,784)
x_test = x_test.reshape(10000,784)

#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ#

# MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# PCA
pca = PCA(n_components=156)
x_train = pca.fit_transform(x_train)
x_test = pca.fit_transform(x_test)

# OneHotencoding
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ#

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

# 모델
model = Sequential()

model.add(Dense(200, input_dim=156, activation='relu'))
model.add(Dense(70))
model.add(Dense(50))
model.add(Dense(50))
model.add(Dense(50,activation='relu'))
model.add(Dense(10))

#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ#

# 컴파일, 훈련 

from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
early_stopping = EarlyStopping(monitor='loss', patience=5, mode='auto')

reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, verbose=1)



model.compile(loss='mse', optimizer='adam', metrics='acc')
model.fit(x_train, y_train, epochs=30, verbose=1, batch_size=10, validation_split=0.2, callbacks=[early_stopping,reduce_lr])
#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ#

# 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print('loss, acc : ', loss, acc)

# loss, acc :  0.00907091423869133 0.9747999906539917

# PCA (0.95)
# loss, acc :  0.1311441957950592 0.15209999680519104

