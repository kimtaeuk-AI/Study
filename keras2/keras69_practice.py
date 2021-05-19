import numpy as np
import matplotlib.pyplot as plt

f = lambda x : x**2 -4*x + 6

gradient = lambda x : 2*x -4 

x0 = 5.0 #(랜덤)
# epoch = 30
epoch = 100
# learning_rate = 0.1
learning_rate = 0.1

print("step\tx\tf(x)") # \ : 한탭간격 건너 뜀


for i in range(epoch):
    temp = x0 - learning_rate * gradient(x0)
    x0 = temp

    print("{:02d}\t{:6.5f}\t{:6.5f}".format(i+1, x0, f(x0))) 
