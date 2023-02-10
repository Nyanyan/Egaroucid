arr = [
    8, 8, 8, 5, 
    6, 7, 8, 9, 
    10, 10, 10, 10, 
    10, 10, 10, 10
]
res = [2]
for elem in arr:
    res.append(res[-1] + 3 ** elem)
print(res)

res2 = []
for elem in res:
    res2.append(elem - res[4])
print(res2[4:8])