import numpy as np

f = lambda x : x**2 -4*x + 6

gradient = lambda x : 2*x -4 

x0 = 5.0 #(랜덤)
# epoch = 30
epoch = 100
# learning_rate = 0.1
learning_rate = 0.1

print("step\tx\tf(x)") # \ : 한탭간격 건너 뜀

# print("{:02d}\t{:6.5f}\t{:6.5f}".format(0, x0, f(x0)))
# print("{:02d}\t{:6.5f}\t{:6.5f}".format(0, x0, f(x0)))

#step    x       f(x)
#00      10.00000        66.00000

for i in range(epoch):
    temp = x0 - learning_rate * gradient(x0)
    x0 = temp

    print("{:02d}\t{:6.5f}\t{:6.5f}".format(i+1, x0, f(x0))) 
    #learning rate가 줄어들면서 2차함수에서 미분값으로 수렴 
    #미분의결과치 2로 수렴한다.
    #keras69_2_에서의 matplotlib의 2에 수렴하는걸 보여준다.
