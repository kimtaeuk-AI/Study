import numpy as np
from tensorflow.keras.datasets import mnist

(x_train, _), (x_test, _) = mnist.load_data()
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# _ 빼버림 - 자릿수는 맞춰줌
print(x_train.shape) #(60000, 28, 28)
x_train = x_train.reshape(60000, 784).astype('float32')/255  
x_test = x_test.reshape(10000, 784)/255. #위에꺼랑 같음

# print(x_train[0])
# print(x_test[0])


from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

input_img = Input(shape=(784,))
encoded = Dense(64, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(64, activation='relu')(encoded)
# decoded = Dense(784, activation='sigmoid')(encoded)
decoded = Dense(784, activation='relu')(encoded)

autoencoder = Model(input_img, decoded)

autoencoder.summary()

# autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
autoencoder.compile(optimizer='adam', loss='mse', metrics=['acc'])

# acc 엉망인이유 ?? 오픈인코더는 loss 기준으로 잡아야한다 

autoencoder.fit(x_train, x_train, epochs=30, batch_size=256,
                validation_split=0.2) # x 랑 y 가 동일하다!

decoded_imgs = autoencoder.predict(x_test)

import matplotlib.pyplot as plt
n=10
plt.figure(figsize=(20,4))
for i in range(n):
    ax = plt.subplot(2, n, i+1) 
    plt.imshow(x_test[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(decoded_imgs[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()
