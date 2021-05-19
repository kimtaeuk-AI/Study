# 입력의 개수가 많은 경우에는 단순히 input() 함수를 그대로 사용하지는 않는다.
# 파이썬의 기본 input()함수는 동작 속도가 느려서 시간 초과로 오답 판정을 받을 수 있기 때문
# 이 경우 sys 라이브러리에 정의되어 있는 sys.stdin.readline() 함수를 사용
'''
import sys
sys.stdin.readline().rstrip()
'''
# readline() 으로 입력하면 입력후 엔터가 줄바꿈 기호로 입력, 이 공백 문자를 제거하려면 rstrip() 함수를 사용 

import sys
data = sys.stdin.readline().rstrip()
print(data)

# hellow world
# hellow world