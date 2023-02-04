N_BIT = 64
UNKNOWN = '-'
'''
problem = [
     '',  '',  '',  '',  '',  '',  '',  '',
     '',  '',  '',  '',  '',  '',  '',  '',
     '',  '',  '',  '',  '',  '',  '',  '',
     '',  '',  '',  '',  '',  '',  '',  '',
     '',  '',  '',  '',  '',  '',  '',  '',
     '',  '',  '',  '',  '',  '',  '',  '',
     '',  '',  '',  '',  '',  '',  '',  '',
     '',  '',  '',  '',  '',  '',  '',  ''
]
'''
'''
problem = [
     '',  '',  '',  '',  '', '2', '1', '0',
     '',  '',  '',  '',  '',  '', '4', '3',
     '',  '',  '',  '',  '',  '',  '', '5',
     '',  '',  '',  '',  '',  '',  '',  '',
     '',  '',  '',  '',  '',  '',  '',  '',
    'f',  '',  '',  '',  '',  '',  '',  '',
    'd', 'e',  '',  '',  '',  '',  '',  '',
    'a', 'b', 'c',  '',  '',  '',  '',  ''
]
'''
problem = [
     '',  '',  '',  '',  '', '2', '1', '0',
     '',  '',  '',  '',  '',  '', '4', '3',
     '',  '',  '',  '',  '',  '',  '', '5',
     '',  '',  '',  '',  '',  '',  '',  '',
     '',  '',  '',  '',  '',  '',  '',  '',
     '',  '',  '',  '',  '',  '',  '', 'f',
     '',  '',  '',  '',  '',  '', 'e', 'd',
     '',  '',  '',  '',  '', 'c', 'b', 'a'
]

def ext_bit(src, bit):
    return src[N_BIT - 1 - bit]

def set_bit(src, bit, val):
    if bit >= N_BIT or bit < 0:
        return
    src[N_BIT - 1 - bit] = val

def add_1bit(src, bit, val):
    if bit >= N_BIT or bit < 0:
        return
    if ext_bit(src, bit):
        set_bit(src, bit, UNKNOWN)
        add_1bit(src, bit + 1, UNKNOWN)
    else:
        set_bit(src, bit, val)

def mul_1bits(src, mul):
    dst = ['' for _ in range(N_BIT)]
    for bit in range(N_BIT):
        src_bit = ext_bit(src, bit)
        if src_bit:
            for m in mul:
                add_1bit(dst, bit + m, src_bit)
    return dst

def rshift(src, s):
    dst = ['' for _ in range(N_BIT)]
    for bit in range(N_BIT):
        set_bit(dst, bit + s, ext_bit(src, bit))
    return dst

def and_1bits(src, bit_lst):
    dst = ['' for _ in range(N_BIT)]
    for bit in bit_lst:
        set_bit(dst, bit, ext_bit(src, bit))
    return dst

def print_bit_line_head():
    print('|', end='')
    for i in range(8):
        print('       ', end='')
        print(7 - i, end='')
    print('|')

def print_bit_line(src, comment):
    print('|', end='')
    for rbit in range(N_BIT):
        if src[rbit]:
            print(src[rbit], end='')
        else:
            print(' ', end='')
    print('|', end='')
    print('', comment)

print_bit_line_head()
print_bit_line(problem, 'problem')

res = mul_1bits(problem, [0, 6, 11, 21])
print_bit_line(res, 'res mul [0, 6, 11, 21]')

res1 = and_1bits(res, [11, 12, 13, 14, 15, 16, 56, 57, 58, 59, 60, 61])
print_bit_line(res1, 'res1 mask')

res2 = mul_1bits(res1, [0, 37])
print_bit_line(res2, 'res2 mul [0, 37]')
