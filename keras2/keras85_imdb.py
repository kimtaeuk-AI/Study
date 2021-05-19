from tensorflow.keras.datasets import reuters, imdb
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words=10000
)


print(x_train.shape, x_test.shape) #(25000,) (25000,)
print(y_train.shape, y_test.shape) #(25000,) (25000,)
# 1개짜리 이진분류
#[실습 / 과제] embedding 으로 모델 만들것!

from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical


x_train = pad_sequences(x_train, maxlen=100, padding='pre')
x_test = pad_sequences(x_test, maxlen=100, padding='pre')
print(x_train.shape, x_test.shape) #(8982, 100) (2246, 100)

# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)
print(y_train.shape, y_test.shape) #(8982, 46) (2246,46)

# 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM

model = Sequential()
# model.add(Embedding(input_dim=10000, output_dim=64, input_length=100)) #print(x_train.shape, x_test.shape) #(8982, 100) (2246, 100)
model.add(Embedding(10000,64))
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
# sparse_categorical_crossentropy - 원핫인코딩 하기 귀찮을때 sparse 을 쓴다
model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=1, validation_split=0.2)

results = model.evaluate(x_test, y_test)

print('loss : ', results[0])
print('acc : ', results[1])

# loss :  0.7341302633285522
# acc :  0.8242800235748291