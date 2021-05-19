import numpy as np
#1.data
x = np.array([range(100), range(301, 401), range(1, 101)])
y = np.array([range(711, 811), range(1,101), range(201, 301)])

#z = x.reshape(10,2)
print(x.shape) #(3,100)
print(y.shape) #(3,100)# shape 권장함. (10,) = 스칼라가 10개  -> (2, 10) 2행 10열  행 무시 열 우선  (11, 3 이면 input_dim= 3)
#print(z.shape)

x = np.transpose(x)
y = np.transpose(y)
print(x)
print(x.shape) # (100,3)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, train_size=0.8, random_state=66) # 순서 기억, 랜덤 스테이트:랜덤 단수 고정 

print(x_train.shape) #(80, 3)
print(y_train.shape) #(80, 3)

#2. model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# from keras.layers import Dense 2.0 때부터 위에 것보다 안좋음  
model = Sequential()
model.add(Dense(10, input_dim=3))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(3))

#3 compile

model.compile(loss = 'mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs = 100, batch_size=1, validation_split=0.2)


 #20% 검증데이터. 엑스:[1,2], [11,12] 그리고 와이:[1,2] 씀

#4 평가 예측

loss, mae = model.evaluate(x_test, y_test)
print('loss = ', loss)
print('mae = ', mae)

y_predict = model.predict(x_test)
#print(y_predict)

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):                              #mse만있고 rmse가 없어서 루트를 씌워줘야 한다.
    return np.sqrt(mean_squared_error(y_test, y_predict)) #sqrt -  루트
print("RMSE : ", RMSE(y_test, y_predict))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)
