import numpy as np

aaa = np.array([[1,2,3,4,10000,6,7,5000,90,100],
               [1000,2000,3,4000,5000,6000,-7000,8,9000,10000]])
aaa = aaa.transpose()
print(aaa.shape) # (10, 2)


def outliers(data_out):
    quartile_1, q2, quartile_3 = np.percentile(data_out, [25, 50, 75])
    print("1사분위 :",quartile_1)
    print("q2 :",q2)
    print("3사분위 :", quartile_3)
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr*1.5)
    upper_bound = quartile_3 + (iqr*1.5)
    return np.where((data_out>upper_bound) | (data_out<lower_bound))

outlier_loc1 = outliers(aaa[:,0])
outlier_loc2 = outliers(aaa[:,1])
print("이상치의 위치 :", outlier_loc1)
print("이상치의 위치 :", outlier_loc2)


import matplotlib.pyplot as plt

plt.boxplot(aaa[:,1])
plt.show()