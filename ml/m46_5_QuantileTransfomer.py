import numpy as np

from sklearn.datasets import load_boston

dataset = load_boston()
x = dataset.data
y = dataset.target

print(x.shape)
print(y.shape)

print("===============================")
print(x[:5]) #0~4
print(y[:10]) 

print(np.max(x), np.min(x)) # 711.0 0.0
print(dataset.feature_names)
# print(dataset.DESCR) #describe 

#data (MinMax)

# x = x /711.  #0 ~ 1 
# x = (x - 최소) (최대 -최소 )
# x = (x - np.min(x)) / (np.max(x)- np.min(x))
print(np.max(x[0]))


from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer
# scaler = MinMaxScaler()
scaler = RobustScaler() # 중위값 기준으로 한다 , 중위값이 1이다 
#장점 : 이상치제거하지않은상태에서 이상치에 대해 효과가 좋다. 100%신뢰 x 
# 분위수로 계산한다   
scaler = QuantileTransformer() # 디폴트: 균등분포 
scaler = QuantileTransformer(output_distribution='normal') #정규분포
scaler.fit(x)
x = scaler.transform(x)
#MinMaxScaler()
print(np.max(x), np.min(x)) # 1.0, 0.0
print(np.max(x[0])) # 0.9999999999999999

#StandardScaler
print(np.max(x), np.min(x)) # 9.933930601860268 , -3.9071933049810337
print(np.max(x[0])) #0.44105193260704206




from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(x, y, train_size=0.8, random_state=66, shuffle=True)

print(x1_train.shape)
print(x1_test.shape)
print(y1_train.shape)
print(y1_test.shape)


from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

model = Sequential()
# model.add(Dense(128, activation='relu', input_shape=(13,))
model.add(Dense(128, activation='relu', input_dim=13))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))



# input1 = Input(shape=(13,))
# dense1 = Dense(100, activation='relu')(input1)
# dense1 = Dense(100)(dense1)
# dense1 = Dense(10)(dense1)
# output1 = Dense(1)(dense1)



# model = Model(input1, output1)

model.compile(loss = 'mse', optimizer='adam', metrics='mae')
model.fit(x1_train, y1_train, epochs=100, validation_split=0.2, batch_size=8)

loss, mae = model.evaluate(x1_test, y1_test, batch_size=8)
print('loss, mae = ', loss, mae)

y1_predict = model.predict(x1_test)

from sklearn.metrics import r2_score
r2 = r2_score(y1_predict, y1_test)


print('r2 = ', r2)

# loss, mae =  16.858667373657227 2.9712021350860596
# r2 =  0.7938000536056695 전처리 전

# loss, mae =  11.793758392333984 2.6175220012664795
# r2 =  0.847403366879222 전처리 후 [X/711.]

# loss, mae =  9.03050422668457 2.062678337097168
# r2 =  0.8975892517708454 전처리 후 MinMaxScalar

#StanardScaler
# loss, mae =  14.151742935180664 2.342193365097046
# r2 =  0.8201837155849974
