from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.datasets import reuters
import matplotlib.pyplot as plt
import numpy as np


(x_train, y_train), (x_test, y_test) = reuters.load_data(
    num_words=10000, test_split=0.2
)

from tensorflow.keras.preprocessing.sequence import pad_sequences
x_train = pad_sequences(x_train, maxlen=200)
x_test = pad_sequences(x_test, maxlen=200)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# unique_elemets, counts_elements = np.unique(y_train, return_counts=True)


# pad_x = pad_sequences(x_train, padding='post', maxlen=200)

#문장의 길이를 maxlen 인자로 맞춰줍니다. 예를 들어 200으로 지정했다면 
# 200보다 짧은 문장은 0으로 채워서 200단어로 맞춰주고 
# 200보다 긴 문장은 200단어까지만 잘라냅니다.

# print(pad_x)
# print(pad_x.shape)   #(8982, 200)
# print(unique_elemets.shape) #(46,)
# print(counts_elements.shape)
# print(np.unique(pad_x)) 
# print(len(np.unique(pad_x)))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM, Flatten, Conv1D

#https://tykimos.tistory.com/24 참고

model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=11, input_length=500)) #word_size = input_dim, input_length= 위에- 안넣어도됨
# model.add(Embedding(28,11))
#lstm은 앞서 설명했듯이 rnn에서 기억 값에 대한 가중치를 제어하며 lstm의 활성화 함수로는 tanh를 사용합니다
model.add(LSTM(32,activation='tanh'))
# model.add(Dense(80))
model.add(Dense(60, activation='tanh'))
# model.add(Dense(50))
model.add(Dense(46, activation='softmax'))

# model.add(Flatten())
# model.add(Dense(1))

model.summary()
model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics=['acc'] )
model.fit(x_train, y_train, epochs=100, batch_size=20, validation_split=0.2)

acc = model.evaluate(x_test, y_test)

print('acc : ', acc)
# 튜닝 안한것 
# acc :  [3.579822301864624, 0.6086375713348389]