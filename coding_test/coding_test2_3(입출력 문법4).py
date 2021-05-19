a = 1 
b = 2 

print(a , b)
print(a+b)

answer = 7

# print('정답은'+ answer+'입니다.') ->TypeError: can only concatenate str (not "int") to str

print('정답은' + str(answer)+'입니다.')
print('정답은', str(answer),'입니다.')
print(f'정답은{answer}입니다.')

