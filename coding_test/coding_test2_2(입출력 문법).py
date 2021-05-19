# 입력을 받을때는 input()
# 만약 입력을 받을 데이터가 정수형이라면 int() 함수를 사용

# 여러개의 데이터를 입력 받을 때는 데이터가 공백으로 구분되는 경우가 많다.
# 띄어쓰기로 구분하기위해 각각 정수 자료형의 데이터로 저장
# 이때 list(map(int, input().split()))를 이용 

# list(map(int, input().split()))의 동작과정
# input()으로 입력받은 문자열을 split()을 이용해 공백으로 나눈 리스트로 바꾼 뒤, 
# map을 이용해 해당 리스트의 모든 원소에 int() 함수를 적용
# 최종적으로 그 결과를 list()로 다시 바꿈으로써 입력받은 문자열을 띄어쓰기로 구분하여 각각 숫자 자료형으로 저장

# 정말 많이 사용된다. 공백, 줄바꿈이 사용하기위해. 

# 데이터 개수 입력
n = int(input())
# 각 데이터를 공백으로 구분하여 입력
data = list(map(int, input().split()))

data.sort(reverse=True)
print(data)

# 5
# 65 90 75 34 99
# [99, 90, 75, 65, 34]

