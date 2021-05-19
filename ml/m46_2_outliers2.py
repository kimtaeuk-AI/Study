#실습
# outliers1 을 행렬형태로 적용할수 있도록 수정 
# 이상치 데이터 -> 이상한 데이터라 제거

import numpy as np


# aaa = np.array([1,2,3,4,10000,6,7,5000,90,100])

aaa = np.array([[1,2,3,4,10000,6,7,5000,90,100],
                [1000,2000,3,4000,5000,6000,7000,8,9000,10000]])

aaa = aaa.transpose()
print(aaa.shape) #(10, 2)

def outliers(data_out):
    quartile_1, quartile_2, quarilte_3 = np.percentile(data_out, [25, 50, 75]) # 25%, 50% ,75%
    print("1사분위 : ", quartile_1)
    print("q2 : ", quartile_2)
    print("3사분위 : ", quarilte_3)
    iqr = quarilte_3 - quartile_1
    print('iqr  : ', iqr)
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quarilte_3 + (iqr * 1.5)
    return np.where((data_out>upper_bound) | (data_out<lower_bound))

# outlier_loc = outliers(aaa)
# print('이상치의 위치 : ', outlier_loc)

# import matplotlib.pyplot as plt

# plt.boxplot([aaa[0], aaa[1]])
# plt.show()

outlier_loc1 = outliers(aaa[0])
outlier_loc2 = outliers(aaa[1])
print('이상치의 위치 : ', outlier_loc1)
print('이상치의 위치 : ', outlier_loc2)

import matplotlib.pyplot as plt

plt.boxplot([outlier_loc1, outlier_loc2])
plt.show()



