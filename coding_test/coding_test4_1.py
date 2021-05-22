# 어떠한 수 N이 1이 될 때까지 다음의 두 과정 중 하나를 반복적으로 선택하여 수행
# 단 두 번째 연산은 N이 K로 나누어 떨어질때만 선택

'''
나동빈 코딩테스트 p99
'''

n, k = map(int,input().split())
result = 0
while n >= k :
    while n % k != 0:
        n -= 1
        result += 1
    n //= k
    result += 1

while n > 1:
    n -= 1
    result += 1

print(result)