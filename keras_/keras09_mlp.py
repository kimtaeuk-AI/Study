import numpy as np
#1.data
x = np.array([[1,2,3,4,5,6,7,8,9,10],
             [11,12,13,14,15,16,17,18,19,20]])

y = np.array([1,2,3,4,5,6,7,8,9,10])
#z = x.reshape(10,2)
print(x.shape) # shape 권장함. (10,) = 스칼라가 10개  -> (2, 10) 2행 10열  행 무시 열 우선  (11, 3 이면 input_dim= 3)
#print(z.shape)

x = np.transpose(x)
print(x)
print(x.shape) # (10,2)

#2. model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# from keras.layers import Dense 2.0 때부터 위에 것보다 안좋음  
model = Sequential()
model.add(Dense(10, input_dim=2))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1))

#3 compile

model.compile(loss = 'mse', optimizer='adam', metrics=['mae'])
model.fit(x, y, epochs=100, batch_size=1, validation_split=0.2) #20% 검증데이터. 엑스:[1,2], [11,12] 그리고 와이:[1,2] 씀

#4 평가 예측

loss, mae = model.evaluate(x, y)
print('loss = ', loss)
print('mae = ', mae)

y_predict = model.predict(x)
#print(y_predict)

'''from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):                              #mse만있고 rmse가 없어서 루트를 씌워줘야 한다.
    return np.sqrt(mean_squared_error(y_test, y_predict)) #sqrt -  루트
print("RMSE : ", RMSE(y_test, y_predict))
print("mse : ", mean_squared_error(y_predict, y_test) )'''