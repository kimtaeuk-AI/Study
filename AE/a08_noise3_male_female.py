# keras67_1 남자 여자에 잡음넣어서 기미 주근깨 여드름을 제거하시오 

import numpy as np
from tensorflow.keras.datasets import mnist
import pandas as pd
import os
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator


train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True, #수평방향뒤집기 
    vertical_flip=True, #수직방향뒤집기 
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest', # 빈자리를 채워준다 
    validation_split=0.3
)

test_datagen = ImageDataGenerator(rescale=1./255)

xy_train = train_datagen.flow_from_directory( 
    './image/human',
    target_size=(150,150),
    batch_size=10, 
    class_mode='binary', 
    #(14, 150, 150, 3)
    subset='training'
    
)

# test_generater
xy_val = train_datagen.flow_from_directory( 
    './image/human',
    target_size=(150,150),
    batch_size=1, 
    class_mode='binary', # 앞에있는놈이 0 뒤에는 1 
    subset='validation'
)

# (x_train, _), (x_test, _) = mnist.load_data()
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# _ 빼버림 - 자릿수는 맞춰줌

x_train = x_train.reshape(60000, 784).astype('float32')/255  
x_test = x_test.reshape(10000, 784)/255. #위에꺼랑 같음

#  노이즈 임의로 만든다 
x_train_noised = x_train + np.random.normal(0, 0.1, size=x_train.shape)
x_test_noised = x_test + np.random.normal(0, 0.1, size=x_test.shape)
x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1) #고정시킨다. 1이상이 되면 1로 고정 시킨다 . (a_min은 상관없음)
x_test_noised = np.clip(x_test_noised, a_min=0, a_max=1)

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense

def autoencoder (hidden_layer_size):
    model = Sequential()
    model.add(Conv2D(units=hidden_layer_size,kernel_size=3, input_shape=(150,150,3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(units=784, activation='sigmoid'))
    return model

model = autoencoder(hidden_layer_size=154) #가장 안정적?? 

model.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['acc'])
model.fit(x_train_noised, x_train, epochs=10)
# 노이즈 없는놈과 노이즈 있는놈 비교하며 훈련 

output = model.predict(x_test_noised)

from matplotlib import pyplot as plt
import random
fig, ((ax1,ax2,ax3,ax4,ax5), (ax6,ax7,ax8,ax9,ax10), (ax11,ax12,ax13,ax14,ax15)) = plt.subplots(3, 5, figsize=(20,7))

# 이미지 다섯개를 무작위로 고른다
random_images = random.sample(range(output.shape[0]), 5)

# 원본(입력) 이미지를 맨 위에 그린다.
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28,28), cmap='gray')
    if i==0:
        ax.set_ylabel("INPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])


# 잡음을 넣은 이미지
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(x_test_noised[random_images[i]].reshape(28,28), cmap='gray')
    if i==0:
        ax.set_ylabel("NOISE", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 오토인코더가 출력한 이미지를 아랭 그린다.
for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(output[random_images[i]].reshape(28,28), cmap='gray')
    if i==0:
        ax.set_ylabel("OUTPUTT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()