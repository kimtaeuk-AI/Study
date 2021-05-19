x_train = 0.5
y_train = 0.8
# 이 소스의 목적은 y_predict 0.8 을 찾아내는것! 


weight = 0.5
lr = 0.01
epoch =1000

for iteration in range(epoch):
    y_predict = x_train * weight  # 1epoch = 0.25
    error = (y_predict - y_train) **2 

    print("Error : " + str(error) + "\ty_predict : " + str(y_predict))
    #                   0.55 제곱 
    up_y_predict = x_train * (weight + lr)
    #                0.5 * 0.51
    up_error = (y_train - up_y_predict) ** 2
    #               
    down_y_predict = x_train * (weight - lr)
    down_error = (y_train - down_y_predict) **2

    if(down_error <= up_error):
        weigght = weight - lr
    if(down_error > up_error):
        weight = weight + lr 