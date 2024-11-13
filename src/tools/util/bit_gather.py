def cant_create(s):
    print('[ERR]', s)

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

print('\nans')

for all_shift in range(min(bits) + 1):
    upper_bit = 63
    res_each_bit = [0 for _ in range(64)]
    affect_bit = [0 for _ in range(64)]

    def plus_one(bit):
        global affect_bit
        if bit < 0:
            return
        if affect_bit[bit] == 0:
            affect_bit[bit] = 1
        else:
            plus_one(bit - 1)

    err = False
    for bit in bits:
        shift = upper_bit - bit
        if res_each_bit[63 - shift]:
            cant_create('bit conflict')
            err = True
            break
        res_each_bit[63 - shift] = 1
        for b in bits:
            if b == bit:
                continue
            plus_one(63 - (b + shift))
        for i in range(len(bits)):
            if affect_bit[i]:
                cant_create('affect conflict')
                err = True
                break
        upper_bit -= 1
        if err:
            break
    
    if not err:

        bin_num = ''.join([str(elem) for elem in res_each_bit])
        num = int(bin_num, base=2)
        hex_num = hex(num)[2:]
        hex_num = digit_fill(hex_num, 16)
        res = '0x' + hex_num + 'ULL'
        print(all_shift, res)