# 이미지는

# data/image/vgg 에 4개를 넣으시오
# 개 고양이 라이언 슈트 
# 욜케 넣을것
# 파일명 : 
# dog1.jpg cat1.jpg, lion.jpg, suit1.jpg 

# E:\gemanas\AIA2021\data\image\vgg

from tensorflow.keras.applications import VGG16,EfficientNetB0
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np


#1. 데이터

img_dog = load_img('./data/image/vgg/dog1.jpg', target_size=(224,224))
img_cat = load_img('./data/image/vgg/cat1.jpg', target_size=(224,224))
img_lion = load_img('./data/image/vgg/lion1.jpg', target_size=(224,224))
img_suit = load_img('./data/image/vgg/suit1.jpg', target_size=(224,224))
#VGG16 은 224사이즈 ?

arr_dog = img_to_array(img_dog)
#형식 자체가 달라서 array로 바꾼다 => 수치화시킨다 
arr_cat = img_to_array(img_cat)
arr_lion = img_to_array(img_lion)
arr_suit = img_to_array(img_suit)
print(arr_dog)
print(type(arr_dog)) #class 'numpy.ndarray'
print(arr_dog.shape) #(224, 224, 3)
#RGB -> BGR 바꿔줘야한다
from tensorflow.keras.applications.vgg16 import preprocess_input
#array 한다음 preprocess 해주면 알아서 vgg16에 맞춰서 변환 
arr_dog = preprocess_input(arr_dog)
arr_cat = preprocess_input(arr_cat)
arr_lion = preprocess_input(arr_lion)
arr_suit = preprocess_input(arr_suit)
print(arr_dog) #위에랑 달라짐 
print(arr_dog.shape) #(224, 224, 3)

arr_input = np.stack([arr_dog, arr_cat, arr_lion, arr_suit])
#stack - 순서대로 (224,244,3)이 합쳐짐
print(arr_input.shape) #(4, 224, 224, 3)


#2. 모델구성
model = EfficientNetB0()
results = model.predict(arr_input)

print(results)
print(' results.shape : ' , results.shape) #(4,1000)

# 이미지 결과 확인

from tensorflow.keras.applications.vgg16 import decode_predictions

decode_results = decode_predictions(results)
print('=====================================')
print("results[0] : ", decode_results[0])
#[('n02085936', 'Maltese_dog', 0.9767469) - 말티즈 0.9%
print('=====================================')
print("results[1] : ", decode_results[1])
print('=====================================')
print("results[2] : ", decode_results[2])
print('=====================================')
print("results[3] : ", decode_results[3])
#'n04350905', 'suit', 0.75699496
print('=====================================')
