def cant_create():
    print('cannot create number')
    exit()

def digit_fill(n, r):
    n = str(n)
    l = len(n)
    for i in range(r - l):
        n = '0' + n
    return n

bits = []
while True:
    digit = input('bit: ')
    if digit == '':
        break
    bits.append(int(digit))

bits.sort(reverse=True)

upper_bit = 63
res_each_bit = [0 for _ in range(64)]
affect_bit = [0 for _ in range(64)]
for bit in bits:
    shift = upper_bit - bit
    if res_each_bit[63 - shift]:
        cant_create()
    res_each_bit[63 - shift] = 1
    for b in bits:
        if 63 - (b + shift) >= 0:
            if affect_bit[63 - (b + shift)]:
                cant_create()
            affect_bit[63 - (b + shift)] = 1
    upper_bit -= 1

bin_num = ''.join([str(elem) for elem in res_each_bit])
print(bin_num)
num = int(bin_num, base=2)
print(num)
hex_num = hex(num)[2:]
hex_num = digit_fill(hex_num, 16)
res = '0x' + hex_num + 'ULL'
print(res)