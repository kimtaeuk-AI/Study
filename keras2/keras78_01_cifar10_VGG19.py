# 실습
# cifar10 으로 vgg16 넣어서 만들것

# 결과치에 대한 기존값과 비교

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import VGG19
from tensorflow.keras.datasets import cifar10


(x_train, y_train), (x_test, y_test) = cifar10.load_data()


x_train = x_train.astype('float32')/255. #데이터는 0~255여서 /255를 해주면 0~1로 좁혀진다 =>전처리 float-실수형  
x_test = x_test.astype('float32')/255. #이것도 가능 


# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)
# (50000, 32, 32, 3)
# (10000, 32, 32, 3)
# (50000, 1)
# (10000, 1)

from sklearn.preprocessing import OneHotEncoder
onehotencder = OneHotEncoder()
onehotencder.fit(y_train)
y_train = onehotencder.transform(y_train).toarray()
y_test = onehotencder.transform(y_test).toarray()

vgg19 = VGG19(weights='imagenet', include_top=False, input_shape=(32,32,3))
# print(vgg16.weights)
vgg19.trainable = True
vgg19.summary()
print(len(vgg19.weights))
print(len(vgg19.trainable_weights))

model = Sequential()
model.add(vgg19)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(80))
model.add(Dense(50))
model.add(Dense(10, activation='softmax'))
model.summary()
print("그냥 가중치의 수 : ",len(model.weights))
print("동결하기 전 훈련되는 가중치의 수 : ", len(model.trainable_weights))

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# modelpath = './modelCheckpoint/k45_mnist_{epoch:02d}-{val_loss:.4f}.hdf5' #. <-지금 study 폴더, 02d -정수형 , f= float 

early_stopping = EarlyStopping(monitor='loss', patience=5, mode='min')

# model_checkpoint = ModelCheckpoint(filepath=modelpath, monitor = 'val_loss', save_best_only=True, mode='auto')

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')
model.fit(x_train, y_train, epochs=100, verbose=1, batch_size=20, callbacks=[early_stopping], validation_split=0.2)

loss, acc = model.evaluate(x_test, y_test, batch_size=20)
print('loss, acc : ', loss, acc)

# vgg16 Ture 일때가 더 좋다 
# 500/500 [==============================] - 3s 5ms/step - loss: 0.9384 - acc: 0.7009
# loss, acc :  0.9384399652481079 0.7009000182151794
# conv1d 
# loss, acc :  1.3426063060760498 0.5562999844551086

#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ요기 하단때문에 파일 분리 했다.ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ#

import pandas as pd

pd.set_option('max_colwidth', -1)
layers = [(layer, layer.name, layer.trainable) for layer in model.layers]
aaa = pd.DataFrame(layers, columns = ['Layer Type', 'Layer Name', 'Layer Trainable'])

print(aaa)
