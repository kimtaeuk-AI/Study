from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

docs = ['너무 재밌어요', '참 최고에요', '참 잘 만든 영화에요',
        '추천하고 싶은 영화입니다.', '한 번 더 보고 싶네요', '글쎄요',
        '별로에요', '생각보다 지루해요', '연기가 어색해요',
        '재미없어요', '너무 재미없다', '참 재밌네요', '규현이가 잘 생기긴 했어요'
]

# 긍정 1, 부정 0 
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,1])

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)

x = token.texts_to_sequences(docs)
print(x)

from tensorflow.keras.preprocessing.sequence import pad_sequences
pad_x = pad_sequences(x, padding='post', maxlen=5)
#pre - 앞 자리가 0으로 채워짐
#post - 뒷 자리가 0으로 채워짐
#maxlen = 잘려짐
print(pad_x)
print(pad_x.shape)   #(13,5) 

print(np.unique(pad_x)) #11 이없다 
print(len(np.unique(pad_x)))

print(pad_x.shape)
print(labels.shape)
pad_x = pad_x.reshape(13,5,1)
# (13, 5)
# (13,)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM, Flatten, Conv1D

#[실습]임베딩 레이어를 빼고 모델 구성 
model = Sequential()
# model.add(Embedding(input_dim=28, output_dim=11, input_length=5)) #word_size = input_dim, input_length= 위에- 안넣어도됨
# model.add(Embedding(28,11))
model.add(LSTM(100, input_shape=(5,1), activation='relu'))
model.add(Dense(50))
model.add(Dense(20))
# model.add(LSTM(32))
# model.add(Conv1D(32,2))
# model.add(Flatten())    #안해줘도 먹히긴 하나 안해주면 ㅎㄷㄷ
model.add(Dense(1, activation='sigmoid'))

# model.add(Flatten())
# model.add(Dense(1))

model.summary()
model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['acc'] )
model.fit(pad_x, labels, epochs=100)

acc = model.evaluate(pad_x, labels)[1]

print('acc : ', acc)
