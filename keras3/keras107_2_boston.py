import numpy as np
import tensorflow as tf
import autokeras as ak
from tensorflow.keras.datasets import mnist,boston_housing
from tensorflow.keras.models import load_model
from sklearn.metrics import r2_score

(x_train, y_train),(x_test, y_test) = boston_housing.load_data()

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
# (404, 13) (404,)
# (102, 13) (102,)

# from sklearn.preprocessing import MinMaxScaler, StandardScaler 
# scaler = StandardScaler()
# scaler.fit(x_train)
# x = scaler.transform(x_train) 
# x = scaler.transform(x_test) 

# ---------------------------------------------

model = ak.StructuredDataRegressor(
    overwrite=True,
    max_trials=3
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
stop = EarlyStopping(monitor='val_loss', mode='min', patience=6)
lr = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, verbose=1)
filepath = 'C:/data/h5/autokeras/'
mc = ModelCheckpoint(filepath, save_best_only=True, verbose=1)
model.fit(x_train, y_train, epochs=10, validation_split=0.2, callbacks=[stop, lr, mc])
results = model.evaluate(x_test, y_test)
print(results)
# [82.98064422607422, 82.98064422607422]
# ---------------------------------------------

model2 = model.export_model()

model2.save('./test_auto/save/boston.h5')

model3 = load_model('./test_auto/save/boston.h5')
result_boston = model3.evaluate(x_test, y_test)

y_pred = model3.predict(x_test)
r2 = r2_score(y_test, y_pred)

print("load_result :", result_boston, r2)


# load_result : [52.26278305053711, 52.26278305053711] 0.3721724725126627