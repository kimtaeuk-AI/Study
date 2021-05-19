import numpy as np
import os
import pandas as pd


#  1. 데이터

dataset_train = pd.read_csv('C:/Study/winequality-white.csv',sep=";", header=0 )
# dataset_train = dataset_train.drop(['volatile acidity','citric acid'], axis=1)
# dataset_trian = dataset_train.iloc[:,10]
# dataset_test = pd.read_csv('C:/Study/winequality-white.csv',sep=';', header=0)

# print(dataset_train)

x = dataset_train.values/255
y = dataset_train['quality'].values

print(x)
print(y)

# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y)

from sklearn.preprocessing import OneHotEncoder



from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=0.8, random_state=66)

print(x_train.shape) #(3918, 12)
print(x_val.shape) #(980, 1)
print(y_train.shape) #(3918, 12)
print(y_val.shape) #(980, 1)

print(dataset_train.head())

#"volatile acidity";"citric acid";"residual sugar";"chlorides";"free sulfur dioxide";"total sulfur dioxide";"density";"pH";"sulphates";"alcohol";"quality"


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import * 

model = Sequential()
model.add(Dense(100, input_dim=12, activation='relu'))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(10, activation='softmax'))


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train,y_train,epochs=2000, batch_size=10, validation_split=0.2)

loss = model.evaluate(x_val, y_val)

print(loss)