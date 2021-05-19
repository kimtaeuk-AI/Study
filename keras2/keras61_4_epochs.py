#epochs 100
#validation_split callback 적용
# early_stopping 5 
#reduce lr 3

#modelCheckpoint 폴더에 hdf5 파일 저장 



import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

#1. 데이터 / 전처리 

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28*28).astype('float32')/255.
x_test = x_test.reshape(10000, 28*28).astype('float32')/255.

#2. 모델

def build_model(drop=0.5, optimizer='adam'):
    inputs = Input(shape=(28*28,), name='input')
    x = Dense(512, activation='relu', name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256, activation='relu', name='hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(128, activation='relu', name='hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(10, activation='softmax', name='outputs')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, metrics=['acc'], loss='categorical_crossentropy')
    return model

def create_hyperparameters():
    batchs = [10,20,30,40,50]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    dropout = [0.1, 0.2, 0.3]
    return{'batch_size' : batchs, 'optimizer': optimizers, 'drop':dropout}

hyperparameters = create_hyperparameters()
model2 = build_model()

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
early_stopping = EarlyStopping(monitor='loss', patience=5, mode='auto')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, verbose=1)
modelpath = './modelCheckpoint/k45_mnist_{epoch:02d}-{val_loss:.4f}.hdf5' #. <-지금 study 폴더, 02d -정수형 , f= float
model_checkpoint = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')


from tensorflow.keras.wrappers.scikit_learn import KerasClassifier  #머신러닝이 케라스보다 더 먼저 나와서 랩핑을 해줘야 한다
model2 = KerasClassifier(build_fn=build_model, verbose=1) # epochs, validation_split를 밑에랑 여기에 넣을수 있다. *callbacks는 안들어가진다 


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
search = RandomizedSearchCV(model2, hyperparameters, cv=3)  #cv cross validation
# search = GridSearchCV(model2, hyperparameters, cv=3)  #cv cross validation

# search.fit(x_train, y_train, verbose=1, ephocs=2, validation_split=0.2) 가능
search.fit(x_train, y_train, verbose=1, epochs=100, validation_split=0.2, callbacks=[early_stopping,reduce_lr,model_checkpoint])

print(search.best_params_) #{'optimizer': 'adam', 'drop': 0.3, 'batch_size': 20}
print(search.best_estimator_)
print(search.best_score_) #0.9815499981244405
acc = search.score(x_test, y_test)
print('최종 스코어 : ', acc ) #최종 스코어 :  0.9857000112533569

#{'optimizer': 'rmsprop', 'drop': 0.2, 'batch_size': 30} 고쳐도 된다 
#좋아졌다!