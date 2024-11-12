def digit(n, r):
    n = str(n)
    l = len(n)
    for i in range(r - l):
        n = '0' + n
    return n

for coord in range(64):
    place = 1 << coord
    res = (place & 0x7F7F7F7F7F7F7F7F) << 1
    res |= (place & 0xFEFEFEFEFEFEFEFE) >> 1
    res |= (place & 0x00FFFFFFFFFFFFFF) << 8
    res |= (place & 0xFFFFFFFFFFFFFF00) >> 8
    res |= (place & 0x00FEFEFEFEFEFEFE) << 7
    res |= (place & 0x7F7F7F7F7F7F7F00) >> 7
    res |= (place & 0x007F7F7F7F7F7F7F) << 9
    res |= (place & 0xFEFEFEFEFEFEFE00) >> 9
    res_hex = hex(res)[2:]
    res_hex_filled = digit(res_hex, 16).upper()
    res_str = '0x' + res_hex_filled + 'ULL'
    print(res_str + ', ', end='')
    if coord % 8 == 7:
        print('')