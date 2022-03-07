#pragma once
#include <iostream>
#include "bit.hpp"
#include "setting.hpp"
/*
inline uint64_t calc_some_mobility(uint64_t p, uint64_t o){
    uint64_t p1 = (p & 0x7F7F7F7F7F7F7F7FULL) << 1;
    uint64_t res = ~(p1 | o) & (p1 + (o & 0x7F7F7F7F7F7F7F7FULL));
    horizontal_mirror_double(&p, &o);
    p1 = (p & 0x7F7F7F7F7F7F7F7FULL) << 1;
    return res | horizontal_mirror(~(p1 | o) & (p1 + (o & 0x7F7F7F7F7F7F7F7FULL)));
}
*/

#if LEGAL_CALCULATION_MODE < 2
    #if LEGAL_CALCULATION_MODE == 0

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

    #else

        inline uint64_t calc_some_mobility_hv(uint64_t ph, uint64_t oh){
            uint64_t pv, ov, phr, ohr, pvr, ovr;
            black_line_mirror_double(ph, oh, &pv, &ov);
            horizontal_mirror_quad(ph, oh, pv, ov, &phr, &ohr, &pvr, &ovr);
            __m256i p1, o, res, mask;
            mask = _mm256_set1_epi64x(0x7F7F7F7F7F7F7F7FULL);
            p1 = _mm256_and_si256(_mm256_set_epi64x(ph, pv, phr, pvr), mask);
            p1 = _mm256_slli_epi64(p1, 1);
            o = _mm256_set_epi64x(oh, ov, ohr, ovr);
            res = _mm256_and_si256(~_mm256_or_si256(p1, o), _mm256_add_epi64(p1, o & mask));
            
            uint64_t a, b;
            a = _mm256_extract_epi64(res, 1);
            b = _mm256_extract_epi64(res, 0);
            horizontal_mirror_double(&a, &b);
            a |= _mm256_extract_epi64(res, 3);
            b |= _mm256_extract_epi64(res, 2);
            return a | black_line_mirror(b);
        }

    #endif

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

    inline uint64_t calc_legal(uint64_t p, uint64_t o){
        uint64_t res = 
            calc_some_mobility_hv(p, o) | 
            unrotate_45(calc_some_mobility_diag9(rotate_45(p), rotate_45(o))) | 
            unrotate_135(calc_some_mobility_diag7(rotate_135(p), rotate_135(o)));
        return res & ~(p | o);
    }

#else

    inline uint64_t calc_legal(const uint64_t P, const uint64_t O){
        uint64_t moves, mO, flip1, pre1, flip8, pre8;
        __m128i    PP, mOO, MM, flip, pre;
        mO = O & 0x7e7e7e7e7e7e7e7eULL;
        PP  = _mm_set_epi64x(vertical_mirror(P), P);
        mOO = _mm_set_epi64x(vertical_mirror(mO), mO);
        flip = _mm_and_si128(mOO, _mm_slli_epi64(PP, 7));                           flip1  = mO & (P << 1);             flip8  = O & (P << 8);
        flip = _mm_or_si128(flip, _mm_and_si128(mOO, _mm_slli_epi64(flip, 7)));     flip1 |= mO & (flip1 << 1);         flip8 |= O & (flip8 << 8);
        pre  = _mm_and_si128(mOO, _mm_slli_epi64(mOO, 7));                          pre1   = mO & (mO << 1);            pre8   = O & (O << 8);
        flip = _mm_or_si128(flip, _mm_and_si128(pre, _mm_slli_epi64(flip, 14)));    flip1 |= pre1 & (flip1 << 2);       flip8 |= pre8 & (flip8 << 16);
        flip = _mm_or_si128(flip, _mm_and_si128(pre, _mm_slli_epi64(flip, 14)));    flip1 |= pre1 & (flip1 << 2);       flip8 |= pre8 & (flip8 << 16);
        MM = _mm_slli_epi64(flip, 7);                                               moves = flip1 << 1;                 moves |= flip8 << 8;
        flip = _mm_and_si128(mOO, _mm_slli_epi64(PP, 9));                           flip1  = mO & (P >> 1);             flip8  = O & (P >> 8);
        flip = _mm_or_si128(flip, _mm_and_si128(mOO, _mm_slli_epi64(flip, 9)));     flip1 |= mO & (flip1 >> 1);         flip8 |= O & (flip8 >> 8);
        pre = _mm_and_si128(mOO, _mm_slli_epi64(mOO, 9));                           pre1 >>= 1;                         pre8 >>= 8;
        flip = _mm_or_si128(flip, _mm_and_si128(pre, _mm_slli_epi64(flip, 18)));    flip1 |= pre1 & (flip1 >> 2);       flip8 |= pre8 & (flip8 >> 16);
        flip = _mm_or_si128(flip, _mm_and_si128(pre, _mm_slli_epi64(flip, 18)));    flip1 |= pre1 & (flip1 >> 2);       flip8 |= pre8 & (flip8 >> 16);
        MM = _mm_or_si128(MM, _mm_slli_epi64(flip, 9));                             moves |= flip1 >> 1;                moves |= flip8 >> 8;
        moves |= _mm_cvtsi128_si64(MM) | vertical_mirror(_mm_cvtsi128_si64(_mm_unpackhi_epi64(MM, MM)));
        return moves & ~(P|O);
    }

#endif