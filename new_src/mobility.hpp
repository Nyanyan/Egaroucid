#pragma once
#include <iostream>
#include "bit.hpp"
#include "setting.hpp"

inline uint64_t calc_some_mobility(uint64_t p, uint64_t o){
    uint64_t p1 = (p & 0x7F7F7F7F7F7F7F7FULL) << 1;
    uint64_t res = ~(p1 | o) & (p1 + (o & 0x7F7F7F7F7F7F7F7FULL));
    horizontal_mirror_double(&p, &o);
    p1 = (p & 0x7F7F7F7F7F7F7F7FULL) << 1;
    return res | horizontal_mirror(~(p1 | o) & (p1 + (o & 0x7F7F7F7F7F7F7F7FULL)));
}

inline uint64_t calc_some_mobility_hv(uint64_t ph, uint64_t oh){
    uint64_t pv, ov;
    black_line_mirror_double(ph, oh, &pv, &ov);
    __m128i p1, o, res1, res2, mask;
    mask = _mm_set1_epi64x(0x7F7F7F7F7F7F7F7FULL);
    p1 = _mm_set_epi64x(ph, pv) & mask;
    p1 = _mm_slli_epi64(p1, 1);
    o = _mm_set_epi64x(oh, ov);
    res1 = ~(p1 | o) & (_mm_add_epi64(p1, o & mask));
    
    horizontal_mirror_double(&ph, &pv);
    p1 = _mm_set_epi64x(ph, pv) & mask;
    p1 = _mm_slli_epi64(p1, 1);
    horizontal_mirror_m128(&o);
    res2 = ~(p1 | o) & (_mm_add_epi64(p1, o & mask));
    horizontal_mirror_m128(&res2);

    res1 |= res2;
    return black_line_mirror(_mm_cvtsi128_si64(res1)) | _mm_cvtsi128_si64(_mm_unpackhi_epi64(res1, res1));
}
/*
inline uint64_t calc_some_mobility_hv(uint64_t ph, uint64_t oh){
    uint64_t pv, ov, phr, ohr, pvr, ovr;
    black_line_mirror_double(ph, oh, &pv, &ov);
    horizontal_mirror_double(ph, oh, &phr, &ohr);
    horizontal_mirror_double(pv, ov, &pvr, &ovr);
    __m256i p1, o, res, mask;
    mask = _mm256_set1_epi64x(0x7F7F7F7F7F7F7F7FULL);
    p1 = _mm256_set_epi64x(ph, pv, phr, pvr) & mask;
    p1 = _mm256_slli_epi64(p1, 1);
    o = _mm256_set_epi64x(oh, ov, ohr, ovr);
    res = ~(p1 | o) & (_mm256_add_epi64(p1, o & mask));
    
    uint64_t a, b;
    a = _mm256_extract_epi64(res, 2);
    b = _mm256_extract_epi64(res, 3);
    horizontal_mirror_double(&a, &b);
    a |= _mm256_extract_epi64(res, 0);
    b |= _mm256_extract_epi64(res, 1);
    return a | black_line_mirror(b);
}
*/

inline uint64_t calc_some_mobility_diag9(uint64_t p, uint64_t o){
    uint64_t p1 = (p & 0x5F6F777B7D7E7F3FULL) << 1;
    uint64_t res = ~(p1 | o) & (p1 + (o & 0x5F6F777B7D7E7F3FULL));
    horizontal_mirror_double(&p, &o);
    p1 = (p & 0x7D7B776F5F3F7F7EULL) << 1;
    return res | horizontal_mirror(~(p1 | o) & (p1 + (o & 0x7D7B776F5F3F7F7EULL)));
}

inline uint64_t calc_some_mobility_diag7(uint64_t p, uint64_t o){
    uint64_t p1 = (p & 0x7D7B776F5F3F7F7EULL) << 1;
    uint64_t res = ~(p1 | o) & (p1 + (o & 0x7D7B776F5F3F7F7EULL));
    horizontal_mirror_double(&p, &o);
    p1 = (p & 0x5F6F777B7D7E7F3FULL) << 1;
    return res | horizontal_mirror(~(p1 | o) & (p1 + (o & 0x5F6F777B7D7E7F3FULL)));
}

inline uint64_t calc_mobility(uint64_t p, uint64_t o){
    uint64_t res = 
        calc_some_mobility_hv(p, o) | 
        unrotate_45(calc_some_mobility_diag9(rotate_45(p), rotate_45(o))) | 
        unrotate_135(calc_some_mobility_diag7(rotate_135(p), rotate_135(o)));
    return res & ~(p | o);
}
