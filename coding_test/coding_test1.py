# 그리디 알고리즘
# 탐욕적으로 문제를 푸는 알고리즘
# '가장 큰 단위부터 계산'


# 거스름돈 계산문제

'''
나동빈 코딩테스트 p87
'''

n = 1260

coin_types=[500,100,50,10]
count = 0
for coin in coin_types:
    count += n //coin
    n %= coin
    
    print(count)

    

