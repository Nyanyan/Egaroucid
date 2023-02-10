arr = [10, 10, 10, 9, 8, 8, 8, 5, 6, 7, 8, 10, 10, 10, 10, 10]
res = [1]
for elem in arr:
    res.append(res[-1] + 3 ** elem)
print(res)