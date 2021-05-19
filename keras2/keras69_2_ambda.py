import numpy as np
import matplotlib.pyplot as plt

f = lambda x : x**2 -4*x + 6 # 이차함수
x = np.linspace(-1, 6, 100)
y = f(x)

#  그림

plt.plot(x, y, 'k-')
plt.plot(2,2, 'sk') #2,2에 점 하나 찍겟다 
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.show()
