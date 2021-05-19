import pandas as pd
import numpy as np


wine = pd.read_csv('C:/Study/winequality-white.csv', index_col=None, header=0, sep=';')

count_data = wine.groupby('quality')['quality'].count() # 데이터 분포 
print(count_data)

# print(np.unique(count_data['quality'])) #다시알아보기 

import matplotlib.pyplot as plt
count_data.plot()
plt.show()