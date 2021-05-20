# 큰 수의 법칙
# 다양한 수로 이루어진 배열이 있을 때 주어진 수들을 M번 더하여 가장 큰수를 만드는 법칙
# 배열의 특정한 인덱스(번호) 에 해당하는 수가 연속해서 K번을 초과하여 더해질 수 없다
'''
나동빈 코딩테스트 p92
''' 

n,m,k = map(int, input().split())
data = list(map(int, input().split())) # list로 받아야 sort 가능 
# 위에 n m k 는 3개를 불러와서 list형식인가보다
# data는 한번에 여러개 데이터를 불러온다. 

data.sort()
print(n)
print(m)
print(k)

first = data[n-1] # 가장 큰수
second = data[n-2] # 두번째로 큰수 
print(first)
print(second)
'''
result = 0 

while True :
    for i in range(k): # 가장 큰 수를 K번 더하기
        if m == 0 : # m이 0이라면 반복문 탈출
            break
        result += first 
        m -= 1 # 더할 때마다 1씩 빼기
    if m == 0:
        break
    result += second # 두번째로 큰 수를 한 번 더하기
    m -= 1 # 더할 때마다 1씩 빼기

print(result)
'''