# 인공지능계의 hellow world라 불리는 40_mnist2!!

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000,28,28,1).astype('float32')/255. #데이터는 0~255여서 /255를 해주면 0~1로 좁혀진다 =>전처리 float-실수형  
x_test = x_test.reshape(10000,28,28,1)/255. #이것도 가능 

y_train = y_train.reshape(y_train.shape[0],1)
y_test = y_test.reshape(y_test.shape[0],1)
#OneHotencoding
#직접하기

from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder()
onehotencoder.fit(y_train)
y_train = onehotencoder.transform(y_train).toarray()
y_test = onehotencoder.transform(y_test).toarray()


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='auto')

#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ분산처리ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ#
import tensorflow as tf

strategy = tf.distribute.MirroredStrategy(cross_device_ops=\
    tf.distribute.HierarchicalCopyAllReduce()
    )
#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ분산처리ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ#
with strategy.scope():
    model = Sequential()
    model.add(Conv2D(filters=10, kernel_size=(2,2), padding='same', strides=1, input_shape=(28,28,1)))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dense(10))
    model.add(Dropout(0.2))
    model.add(Conv2D(10, (2,2)))
    # model.add(Conv2D(10, (2,2)))
    model.add(Flatten()) #Flatten = 평평하게해주다. Dense 4차원을 2차원으로 바꾸기 위해.
    model.add(Dense(8, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    #실습 완성하기
    #지표는 acc /// 0.985 이상


    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
hist = model.fit(x_train, y_train, epochs=10, verbose=1,validation_split=0.2, batch_size=10, callbacks=[early_stopping])

result = model.evaluate(x_test, y_test, batch_size=10)
print('loss : ', result[0])
print('acc : ', result[1])


#과제 1.matplotlib 한글깨짐 처리할것
