# 인공지능계의 hellow world라 불리는 40_mnist2!!

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist

import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus :
    try: 
        tf.config.experimental.set_visible_devices(gpus[0],'GPU')
    except RuntimeError as e:
        print(e)

#1. 데이터
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
import tensorflow as tf
import numpy as np
# tf.set_random_seed(66)


tf.compat.v1.disable_eager_execution()
# print(tf.execution_eagerly())
print(tf.__version__)


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus :
    try: 
        tf.config.experimental.set_visible_devices(gpus[1],'GPU')
    except RuntimeError as e:
        print(e)

#1. 데이터
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255

learing_rate = 0.001
training_epochs = 15
batch_size = 100
total_batch = int(len(x_train)/batch_size) # 60000 / 100

x = tf.compat.v1.placeholder(tf.float32, [None, 28, 28, 1])
y = tf.compat.v1.placeholder(tf.float32, [None, 10])

#2. 모델 구성 

#L1.

w1 = tf.compat.v1.get_variable('w1', shape=[3,3,1,32])   #  3,3 = > kernel_size , 1 - 채널(흑백), 32 출력
L1 = tf.nn.conv2d(x, w1, strides=[1,1,1,1], padding='SAME') # strides = shape 맞춰줌 
print(L1)
# conv2D(filter, kernel_size, input_shape, input_shape)  서머리???
# conv2D(10, (3,2), input_shape=(7,7,1)) 파라미터의 갯수???   70 => 10 x 1 x 
# conv2D(32, (3,3), input_shape=(28,28,1)) 파라미터의 갯수???   70 => 10 x 1 x 
L1 = tf.nn.selu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
# Tensor("MaxPool:0", shape=(?, 14, 14, 32), dtype=float32)
print(L1)

#L2.
w2 = tf.compat.v1.get_variable('w2', shape=[3,3,32,64])   #  3,3 = > kernel_size , 32 - 위에 입력 , 64 출력 
L2 = tf.nn.conv2d(L1, w2, strides=[1,1,1,1], padding='SAME') # strides = shape 맞춰줌 
print(L2)
# Tensor("Conv2D_1:0", shape=(?, 14, 14, 64), dtype=float32)
L2 = tf.nn.elu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
print(L2)
# Tensor("MaxPool_1:0", shape=(?, 7, 7, 64), dtype=float32)

#L3.
w3 = tf.compat.v1.get_variable('w3', shape=[3,3,64,128])   #  3,3 = > kernel_size , 64 - 위에 입력, 128 출력
L3 = tf.nn.conv2d(L2, w3, strides=[1,1,1,1], padding='SAME') # strides = shape 맞춰줌 
print(L3)
# Tensor("Conv2D_2:0", shape=(?, 7, 7, 128), dtype=float32)
L3 = tf.nn.relu(L3)
L3 = tf.nn.max_pool(L3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
print(L3)
# Tensor("MaxPool_2:0", shape=(?, 4, 4, 128), dtype=float32)

#L4.
w4 = tf.compat.v1.get_variable('w4', shape=[3,3,128,64])   #  3,3 = > kernel_size , 64 - 위에 입력, 128 출력
L4 = tf.nn.conv2d(L3, w4, strides=[1,1,1,1], padding='SAME') # strides = shape 맞춰줌 
print(L4)
# Tensor("Conv2D_3:0", shape=(?, 4, 4, 64), dtype=float32)
L4 = tf.nn.selu(L4)
L4 = tf.nn.max_pool(L4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
print(L4)
# Tensor("MaxPool_3:0", shape=(?, 2, 2, 64), dtype=float32)

# Flatten
L_flat = tf.reshape(L4, [-1, 2*2*64])
print('플래튼 : ', L_flat)

#L5.
w5 = tf.compat.v1.get_variable('w5', shape=[2*2*64, 64],#, 
                    #  initializer=tf.compat.v1.contrib.layers.xavier_initializer()) # 64 내가주고싶은만큼
                   initializer =  tf.keras.initializers.HeNormal()) 
b5 = tf.Variable(tf.compat.v1.random_normal([64]), name='b5')
L5 = tf.nn.selu(tf.matmul(L_flat, w5)+ b5)
# L5 = tf.nn.dropout(L5, keep_prob=0.2)
print(L5)
# Tensor("dropout/mul_1:0", shape=(?, 64), dtype=float32)


#L6.
w6 = tf.compat.v1.get_variable('w6', shape=[64, 32],#, 
                    #  initializer=tf.compat.v1.contrib.layers.xavier_initializer()) # 32 내가주고싶은만큼 
                    initializer = tf.keras.initializers.HeNormal()) 
b6 = tf.Variable(tf.compat.v1.random_normal([32]), name='b6')
L6 = tf.nn.selu(tf.matmul(L5, w6)+ b6)
# L6 = tf.nn.dropout(L6, keep_prob=0.2)
print(L6)
# Tensor("dropout_1/mul_1:0", shape=(?, 32), dtype=float32)

#L7.
w7 = tf.compat.v1.get_variable('w7', shape=[32, 10],#, 
                    #  initializer=tf.contrib.layers.xavier_initializer()) # 32 내가주고싶은만큼 
                    initializer = tf.keras.initializers.HeNormal()) 
b7 = tf.Variable(tf.compat.v1.random_normal([10]), name='b6')
hypothesis = tf.nn.softmax(tf.matmul(L6, w7)+ b7)
print("최종 출력 : ", hypothesis)
# Tensor("Softmax:0", shape=(?, 10), dtype=float32)

#8. 컴파일, 훈련
loss = tf.reduce_mean(-tf.reduce_sum(y*tf.math.log(hypothesis), axis=1)) # categorical_crossentropy
optimizer = tf.compat.v1.train.AdamOptimizer(0.01).minimize(loss)

#9. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for epoch in range(training_epochs):
    avg_cost = 0

    for i in range(total_batch):
        start = i * batch_size
        end = start + batch_size

        batch_x, batch_y = x_train[start:end], y_train[start:end]
        feed_dict = {x:batch_x, y:batch_y}
        c, _ = sess.run([loss, optimizer], feed_dict=feed_dict)
        avg_cost += c/total_batch

    print('Epoch : ', '%04d' %(epoch +1), 'cost = {:.9f}'.format(avg_cost))
print("훈련 끗!")

prediction = tf.equal(tf.compat.v1.arg_max(hypothesis, 1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
print('Acc : ', sess.run(accuracy, feed_dict={x:x_test, y:y_test}))

