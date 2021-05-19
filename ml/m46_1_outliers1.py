# 이상치 처리
# 1. 0 처리
# 2. Nan 처리 후 보간
# 3.4.5.... 알아서 해

import numpy as np

aaa = np.array([1,2,3,4,6,7,90,100,5000,10000])

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

outlier_loc = outliers(aaa)
print('이상치의 위치 : ', outlier_loc)

# 실습
# 위 outlier 데이터를 boxplot으로 그리시오
import matplotlib.pyplot as plt

plt.boxplot(outlier_loc)
plt.show()
