import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


#1. 데이터
dataset = load_iris()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=77)

#2. 모델
model = DecisionTreeClassifier(max_depth=4)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
acc = model.score(x_test, y_test)

print(model.feature_importances_) #[0.0244404  0.01669101 0.00766884 0.95119975] 다 합치면 1
print('acc : ', acc)

import matplotlib.pyplot as plt

def plot_feature_importances_dataset(model):
    n_features = dataset.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), dataset.feature_names)
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1,n_features)

plot_feature_importances_dataset(model)
plt.show()

# acc :  0.9/ width