import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications import VGG16

model = VGG16(weights='imagenet', include_top=False, input_shape=(32,32,3))
# print(vgg16.weights)
model.trainable = False
model.summary()
print(len(model.weights))
print(len(model.trainable_weights))

# Total params: 14,714,688
# Trainable params: 0
# Non-trainable params: 14,714,688
# _________________________________________________________________
# 26
# 0

model.trainable = True
model.summary()
print(len(model.weights))
print(len(model.trainable_weights))

# Total params: 14,714,688
# Trainable params: 14,714,688
# Non-trainable params: 0
# _________________________________________________________________
# 26
# 26