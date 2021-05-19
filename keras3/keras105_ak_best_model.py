import numpy as np
import tensorflow as tf
import autokeras as ak
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255.

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = ak.ImageClassifier(
    overwrite=True,
    max_trials=1,
    loss = 'mse',
    metrics=['acc']

)

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau

es = EarlyStopping(monitor='val_loss', mode='min', patience=6)
lr = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, verbose=2)
ck = ModelCheckpoint('./temp/',save_wieght_only=True, save_best_only=True, monitor='val_loss', verbose=1)
model.fit(x_train, y_train, epochs=1, validation_split=0.2,
          callbacks=[es,lr,ck]
) # default validation_split = 0.2로 되어있다.

results = model.evaluate(x_test, y_test)

print(results)

model2 = model.export_model() # 모델을 저장시킨다 
model2.save('./test_auto/save/aaa.h5')

best_model = model.tuner.get_best_model()
best_model.save('./test_auto/save/best_aaa.h5')

# [0.06797606498003006, 0.9771000146865845] loss, accuracy