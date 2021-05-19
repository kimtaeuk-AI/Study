import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)

x = np.arange(-5, 5, 0.1)

y = relu(x)

print(x)
print(y)

plt.plot(x,y)
plt.grid()
plt.show()


# 음수값들은 0으로 표시

#ㅡㅡㅡㅡ 과제 ㅡㅡㅡㅡ#
# elu, selu, reaky relu
# 72_2,3,4번으로 파일을 만들것

