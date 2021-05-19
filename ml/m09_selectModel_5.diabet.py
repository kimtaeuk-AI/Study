from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , r2_score
# from sklearn.utils.testing import all_estimators
from sklearn.utils._testing import all_estimators
import warnings


from sklearn.datasets import load_diabetes
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings('ignore') # warning 무시하겠다! 

dataset = load_diabetes()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=44)

allAlgorithms = all_estimators(type_filter = 'regressor')

for (name, algorithm) in allAlgorithms:  # allAlgorithms 에서 인자가 2개 나온다 .
    try:                                 
        model = algorithm()

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        print(name, '의 정답률 : ', r2_score(y_test, y_pred))
    except: #예외가 발생하면 
        print(name, '은 없는 놈!')
        

import sklearn
# print(sklearn.__version__)