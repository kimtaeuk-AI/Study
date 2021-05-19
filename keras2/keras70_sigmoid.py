import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x)) #exp - exponential 지수 

x = np.arange(-5, 5, 0.1)
y = sigmoid(x)

print(x)
print(y)

plt.plot(x,y)
plt.grid()
plt.show()