#pragma once
#include <iostream>
#ifdef _MSC_VER
    #include <intrin.h>
#else
    #include <x86intrin.h>
#endif
#include "bit.hpp"
#include "setting.hpp"

inline uint64_t calc_some_mobility(uint64_t p, uint64_t o){
    uint64_t p1 = (p & 0x7F7F7F7F7F7F7F7FULL) << 1;
    uint64_t res = ~(p1 | o) & (p1 + (o & 0x7F7F7F7F7F7F7F7FULL));
    o = horizontal_mirror(o);
    p1 = (horizontal_mirror(p) & 0x7F7F7F7F7F7F7F7FULL) << 1;
    return res | horizontal_mirror(~(p1 | o) & (p1 + (o & 0x7F7F7F7F7F7F7F7FULL)));
}

inline uint64_t calc_some_mobility_diag9(uint64_t p, uint64_t o){
    uint64_t p1 = (p & 0x5F6F777B7D7E7F3FULL) << 1;
    uint64_t res = ~(p1 | o) & (p1 + (o & 0x5F6F777B7D7E7F3FULL));
    o = horizontal_mirror(o);
    p1 = (horizontal_mirror(p) & 0x7D7B776F5F3F7F7EULL) << 1;
    return res | horizontal_mirror(~(p1 | o) & (p1 + (o & 0x7D7B776F5F3F7F7EULL)));
}

inline uint64_t calc_some_mobility_diag7(uint64_t p, uint64_t o){
    uint64_t p1 = (p & 0x7D7B776F5F3F7F7EULL) << 1;
    uint64_t res = ~(p1 | o) & (p1 + (o & 0x7D7B776F5F3F7F7EULL));
    o = horizontal_mirror(o);
    p1 = (horizontal_mirror(p) & 0x5F6F777B7D7E7F3FULL) << 1;
    return res | horizontal_mirror(~(p1 | o) & (p1 + (o & 0x5F6F777B7D7E7F3FULL)));
}

inline uint64_t calc_mobility(uint64_t p, uint64_t o){
    uint64_t res = 
        calc_some_mobility(p, o) | 
        black_line_mirror(calc_some_mobility(black_line_mirror(p), black_line_mirror(o))) | 
        unrotate_45(calc_some_mobility_diag9(rotate_45(p), rotate_45(o))) | 
        unrotate_135(calc_some_mobility_diag7(rotate_135(p), rotate_135(o)));
    return res & ~(p | o);
}
