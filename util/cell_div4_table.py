
def digit(n, r):
    n = str(n)
    l = len(n)
    for i in range(r - l):
        n = '0' + n
    return n

for parity in range(16):
    mask = 0
    if parity & 1:
        mask |= 0x000000000F0F0F0F
    if parity & 2:
        mask |= 0x00000000F0F0F0F0
    if parity & 4:
        mask |= 0x0F0F0F0F00000000
    if parity & 8:
        mask |= 0xF0F0F0F000000000
    hex_s = hex(mask)[2:]
    hex_s_fill0 = digit(hex_s, 16).upper()
    res = '0x' + hex_s_fill0 + 'ULL'
    print(res + ', ', end='')