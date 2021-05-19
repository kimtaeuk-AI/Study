from tensorflow.keras.preprocessing.text import Tokenizer

text = '나는 진짜 진짜 맛있는 밥을 진짜 마구 마구 먹었다.'

token = Tokenizer()
token.fit_on_texts([text])

print(token.word_index)
# 빈도수가 많으면 앞쪽에 배치 , 같으면 순서대로 

x = token.texts_to_sequences([text])
print(x)

from tensorflow.keras.utils import to_categorical
word_size = len(token.word_index)
print(word_size)
x = to_categorical(x)
print(x)
print(x.shape)
#문제점 : 인덱스 0 부터 시작 
