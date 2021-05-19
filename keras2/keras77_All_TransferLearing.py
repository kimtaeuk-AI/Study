from tensorflow.keras.applications import VGG16, VGG19, Xception
from tensorflow.keras.applications import ResNet101, ResNet101V2, ResNet152, ResNet152V2
from tensorflow.keras.applications import ResNet50, ResNet50V2
from tensorflow.keras.applications import InceptionV3, InceptionResNetV2
from tensorflow.keras.applications import DenseNet121, DenseNet169, DenseNet201
from tensorflow.keras.applications import NASNetLarge, NASNetMobile
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1


model = VGG16()
# 32
# 0
model = VGG19()
# 38
# 0
model = Xception()
# 236
# 0
model = ResNet101()
# 626
# 0
model = ResNet101V2()
# 544
# 0
model = ResNet152()
# 932
# 0
model = ResNet152V2()
# 816
# 0
model = ResNet50()
# 320
# 0
model = ResNet50V2()
# 272
# 0
model = InceptionV3()
# 378
# 0
model = InceptionResNetV2()
# 898
# 0
model = DenseNet121()
# 606
# 0
model = DenseNet169()
# 846
# 0
model = DenseNet201()
# 1006
# 0
model = NASNetLarge()
# 1546
# 0
model = NASNetMobile()
# 1126
# 0
model = EfficientNetB0()
# 314
# 0
model = EfficientNetB1()
# 442
# 0
model.trainable = False



# 모델별로 파라미터와 웨이트 수들 정리할 것

model.summary()
print(len(model.weights))
print(len(model.trainable_weights))


