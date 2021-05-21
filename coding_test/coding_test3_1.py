# 숫자 카드 게임
# 숫자 카드 게임은 여러 개의 숫자 카드 중에서 가장 높은 숫자가 쓰인 카드 한장을 뽑는 게임

'''
나동빈 코딩테스트 p96
'''

n,m = map(int, input().split())
result = 0
for i in range(n):
    data = list(map(int, input().split()))
    min_value = min(data)
    result = max(result,min_value)

print(result)

# 프로그램은 가장작은 수를 뽑아내는게 결과인거 같다. 