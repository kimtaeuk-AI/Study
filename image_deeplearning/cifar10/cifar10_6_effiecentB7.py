# batch normalization 이랑 dropout이랑 같은효과


# resize를 안해주면 결과값이 엄청 안좋다. 
# 해상도 32 x 32 라.. 255 까지 올려줘야 한다 .
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from tensorflow.python.keras import activations
from tensorflow.python.keras.layers.pooling import MaxPooling2D
from tensorflow.keras.applications import EfficientNetB7
import cv2

(x_train, y_train), (x_test, y_test)  = cifar10.load_data()

print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)

x_train = tf.image.resize(x_train,(254,254))
# img = img.astype(np.float32)/255.
# x_train = cv2.imread(x_train)
# x_train = cv2.resize(x_train,(255,255),interpolation=cv2.INTER_CUBIC)
print(x_train.shape,y_train.shape)

'''
# (50000, 32, 32, 3) (50000, 1)
# (10000, 32, 32, 3) (10000, 1)
# plt.figure(figsize=(15,2))
# plt.imshow(x_train[0])
# plt.show()
print(y_train.shape)
# y_train = y_train.reshape(-1,) # 쭉 길게 늘어뜨려진다. 1차원 배열 
print(y_train.shape)
# y_test = y_test.reshape(-1,)

classes = ['airplane','automobile','bird','cat','deer','dog','forg','horse','ship','truck']

x_train = x_train/255.
x_test = x_test/255.


y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


from tensorflow.keras.layers import Dense, Conv1D, Conv2D,Flatten,BatchNormalization, Dropout, DepthwiseConv2D
from tensorflow.keras.models import Sequential

vgg19 = EfficientNetB7(weights='imagenet', include_top=False, input_shape=(32,32,3))
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
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['acc'])
model.fit(x_train, y_train, validation_split=0.2,epochs=10)

loss = model.evaluate(x_test,y_test)
print(loss)

# [1.2406651973724365, 0.6080999970436096]
# [0.9790632724761963, 0.6723999977111816]
# [1.117068886756897, 0.6829000115394592] #relu를 층마다 써주면 효과가 더 좋았다 loss 는 늘어났지만..
# [0.6480008959770203, 0.7811999917030334] # batch nomalization, dropout  사용
# [0.7740620374679565, 0.7724000215530396] # batch nomalization 사용 (Dense만 dropout 사용) 더안좋아졌다 
# [3.2146787643432617, 0.60589998960495] 흠.. depth_wise 가져온건 안좋다. 에폭이 많으면 더좋을수도?
# [0.8409419655799866, 0.7382000088691711] vgg19.. 
'''