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
            res = _mm256_andnot_si256(_mm256_or_si256(p1, o), _mm256_add_epi64(p1, o & mask));
            
            uint64_t a, b;
            a = _mm256_extract_epi64(res, 1);
            b = _mm256_extract_epi64(res, 0);
            horizontal_mirror_double(&a, &b);
            a |= _mm256_extract_epi64(res, 3);
            b |= _mm256_extract_epi64(res, 2);
            return a | black_line_mirror(b);
        }

    #endif
    /*
    inline uint64_t calc_some_mobility_diag9(uint64_t p, uint64_t o){
        rotate_45_double(&p, &o);
        uint64_t p_r, o_r;
        horizontal_mirror_double(p, o, &p_r, &o_r);
        __m128i mask = _mm_set_epi64x(0x5F6F777B7D7E7F3FULL, 0x7D7B776F5F3F7F7EULL);
        __m128i p1 = _mm_and_si128(_mm_set_epi64x(p, p_r), mask);
        p1 = _mm_slli_epi64(p1, 1);
        __m128i oo = _mm_set_epi64x(o, o_r);
        __m128i res = _mm_and_si128(~_mm_or_si128(p1, oo), _mm_add_epi64(p1, oo & mask));
        return unrotate_45(horizontal_mirror(_mm_cvtsi128_si64(res)) | _mm_cvtsi128_si64(_mm_unpackhi_epi64(res, res)));
    }

    inline uint64_t calc_some_mobility_diag7(uint64_t p, uint64_t o){
        rotate_135_double(&p, &o);
        uint64_t p_r, o_r;
        horizontal_mirror_double(p, o, &p_r, &o_r);
        __m128i mask = _mm_set_epi64x(0x7D7B776F5F3F7F7EULL, 0x5F6F777B7D7E7F3FULL);
        __m128i p1 = _mm_and_si128(_mm_set_epi64x(p, p_r), mask);
        p1 = _mm_slli_epi64(p1, 1);
        __m128i oo = _mm_set_epi64x(o, o_r);
        __m128i res = _mm_and_si128(~_mm_or_si128(p1, oo), _mm_add_epi64(p1, oo & mask));
        return unrotate_135(horizontal_mirror(_mm_cvtsi128_si64(res)) | _mm_cvtsi128_si64(_mm_unpackhi_epi64(res, res)));
    }
    */

    inline uint64_t calc_some_mobility_diag(uint64_t p, uint64_t o){
        uint64_t p45, o45, p135, o135, p45r, o45r, p135r, o135r;
        rotate_45_double_135_double(p, o, &p45, &o45, &p135, &o135);
        horizontal_mirror_quad(p45, o45, p135, o135, &p45r, &o45r, &p135r, &o135r);
        __m256i mask = _mm256_set_epi64x(0x5F6F777B7D7E7F3FULL, 0x7D7B776F5F3F7F7EULL, 0x7D7B776F5F3F7F7EULL, 0x5F6F777B7D7E7F3FULL);
        __m256i p1 = _mm256_and_si256(_mm256_set_epi64x(p45, p135, p45r, p135r), mask);
        p1 = _mm256_slli_epi64(p1, 1);
        __m256i oo = _mm256_set_epi64x(o45, o135, o45r, o135r);
        __m256i res = _mm256_andnot_si256(_mm256_or_si256(p1, oo), _mm256_add_epi64(p1, _mm256_and_si256(oo, mask)));
        uint64_t a, b;
        a = _mm256_extract_epi64(res, 1);
        b = _mm256_extract_epi64(res, 0);
        horizontal_mirror_double(&a, &b);
        a |= _mm256_extract_epi64(res, 3);
        b |= _mm256_extract_epi64(res, 2);
        return unrotate_45_135(a, b);
    }

    inline uint64_t calc_legal(uint64_t p, uint64_t o){
        /*
        uint64_t res = 
            calc_some_mobility_hv(p, o) | 
            calc_some_mobility_diag9(p, o) | 
            calc_some_mobility_diag7(p, o);
        */
        uint64_t res = 
            calc_some_mobility_hv(p, o) | 
            calc_some_mobility_diag(p, o);
        return res & ~(p | o);
    }

#elif LEGAL_CALCULATION_MODE == 2

    inline uint64_t calc_legal(const uint64_t P, const uint64_t O){
        uint64_t moves, mO, flip1, pre1, flip8, pre8;
        __m128i    PP, mOO, MM, flip, pre;
        mO = O & 0x7E7E7E7E7E7E7E7EULL;
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

#elif LEGAL_CALCULATION_MODE == 3

    inline uint64_t calc_mobility_left(uint64_t p, uint64_t o){
        uint64_t p1 = (p & 0x7F7F7F7F7F7F7F7FULL) << 1;
        return ~(p1 | o) & (p1 + (o & 0x7F7F7F7F7F7F7F7FULL));
    }

    inline uint64_t calc_legal(const uint64_t P, const uint64_t O){
        uint64_t moves, mO, flip1, pre1, flip8, pre8;
        __m128i    PP, mOO, MM, flip, pre;
        mO = O & 0x7E7E7E7E7E7E7E7EULL;
        PP  = _mm_set_epi64x(vertical_mirror(P), P);
        mOO = _mm_set_epi64x(vertical_mirror(mO), mO);
        flip = _mm_and_si128(mOO, _mm_slli_epi64(PP, 7));                                                               flip8  = O & (P << 8);
        flip = _mm_or_si128(flip, _mm_and_si128(mOO, _mm_slli_epi64(flip, 7)));                                         flip8 |= O & (flip8 << 8);
        pre  = _mm_and_si128(mOO, _mm_slli_epi64(mOO, 7));                                                              pre8   = O & (O << 8);
        flip = _mm_or_si128(flip, _mm_and_si128(pre, _mm_slli_epi64(flip, 14)));                                        flip8 |= pre8 & (flip8 << 16);
        flip = _mm_or_si128(flip, _mm_and_si128(pre, _mm_slli_epi64(flip, 14)));                                        flip8 |= pre8 & (flip8 << 16);
        MM = _mm_slli_epi64(flip, 7);                                               moves = calc_mobility_left(P, O);   moves |= flip8 << 8;
        flip = _mm_and_si128(mOO, _mm_slli_epi64(PP, 9));                           flip1  = mO & (P >> 1);             flip8  = O & (P >> 8);
        flip = _mm_or_si128(flip, _mm_and_si128(mOO, _mm_slli_epi64(flip, 9)));     flip1 |= mO & (flip1 >> 1);         flip8 |= O & (flip8 >> 8);
        pre = _mm_and_si128(mOO, _mm_slli_epi64(mOO, 9));                           pre1   = mO & (mO >> 1);            pre8 >>= 8;
        flip = _mm_or_si128(flip, _mm_and_si128(pre, _mm_slli_epi64(flip, 18)));    flip1 |= pre1 & (flip1 >> 2);       flip8 |= pre8 & (flip8 >> 16);
        flip = _mm_or_si128(flip, _mm_and_si128(pre, _mm_slli_epi64(flip, 18)));    flip1 |= pre1 & (flip1 >> 2);       flip8 |= pre8 & (flip8 >> 16);
        MM = _mm_or_si128(MM, _mm_slli_epi64(flip, 9));                             moves |= flip1 >> 1;                moves |= flip8 >> 8;
        moves |= _mm_cvtsi128_si64(MM) | vertical_mirror(_mm_cvtsi128_si64(_mm_unpackhi_epi64(MM, MM)));
        return moves & ~(P | O);
    }

#elif LEGAL_CALCULATION_MODE == 4

    inline uint64_t calc_some_mobility_hv(uint64_t ph, uint64_t oh){
        uint64_t pv, ov, phr, ohr, pvr, ovr;
        black_line_mirror_double(ph, oh, &pv, &ov);
        horizontal_mirror_quad(ph, oh, pv, ov, &phr, &ohr, &pvr, &ovr);
        __m256i p1, o, res, mask;
        mask = _mm256_set1_epi64x(0x7F7F7F7F7F7F7F7FULL);
        p1 = _mm256_and_si256(_mm256_set_epi64x(ph, pv, phr, pvr), mask);
        p1 = _mm256_slli_epi64(p1, 1);
        o = _mm256_set_epi64x(oh, ov, ohr, ovr);
        res = _mm256_andnot_si256(_mm256_or_si256(p1, o), _mm256_add_epi64(p1, o & mask));
        
        uint64_t a, b;
        a = _mm256_extract_epi64(res, 1);
        b = _mm256_extract_epi64(res, 0);
        horizontal_mirror_double(&a, &b);
        a |= _mm256_extract_epi64(res, 3);
        b |= _mm256_extract_epi64(res, 2);
        return a | black_line_mirror(b);
    }

    inline uint64_t calc_legal(const uint64_t P, const uint64_t O){
        uint64_t moves, mO;
        __m128i    PP, mOO, MM, flip, pre;
        mO = O & 0x7E7E7E7E7E7E7E7EULL;
        PP  = _mm_set_epi64x(vertical_mirror(P), P);
        mOO = _mm_set_epi64x(vertical_mirror(mO), mO);
        flip = _mm_and_si128(mOO, _mm_slli_epi64(PP, 7));
        flip = _mm_or_si128(flip, _mm_and_si128(mOO, _mm_slli_epi64(flip, 7)));
        pre  = _mm_and_si128(mOO, _mm_slli_epi64(mOO, 7));
        flip = _mm_or_si128(flip, _mm_and_si128(pre, _mm_slli_epi64(flip, 14)));
        flip = _mm_or_si128(flip, _mm_and_si128(pre, _mm_slli_epi64(flip, 14)));
        MM = _mm_slli_epi64(flip, 7);
        flip = _mm_and_si128(mOO, _mm_slli_epi64(PP, 9));
        flip = _mm_or_si128(flip, _mm_and_si128(mOO, _mm_slli_epi64(flip, 9)));
        pre = _mm_and_si128(mOO, _mm_slli_epi64(mOO, 9));
        flip = _mm_or_si128(flip, _mm_and_si128(pre, _mm_slli_epi64(flip, 18)));
        flip = _mm_or_si128(flip, _mm_and_si128(pre, _mm_slli_epi64(flip, 18)));
        MM = _mm_or_si128(MM, _mm_slli_epi64(flip, 9));
        moves = calc_some_mobility_hv(P, O) | _mm_cvtsi128_si64(MM) | vertical_mirror(_mm_cvtsi128_si64(_mm_unpackhi_epi64(MM, MM)));
        return moves & ~(P | O);
    }

#elif LEGAL_CALCULATION_MODE == 5

    uint64_t calc_legal(const uint64_t P, const uint64_t O){
        __m256i	PP, mOO, MM, flip_l, flip_r, pre_l, pre_r, shift2;
        __m128i	M;
        const __m256i shift1897 = _mm256_set_epi64x(7, 9, 8, 1);
        const __m256i mflipH = _mm256_set_epi64x(0x7E7E7E7E7E7E7E7E, 0x7E7E7E7E7E7E7E7E, -1, 0x7E7E7E7E7E7E7E7E);
        PP = _mm256_broadcastq_epi64(_mm_cvtsi64_si128(P));
        mOO = _mm256_and_si256(_mm256_broadcastq_epi64(_mm_cvtsi64_si128(O)), mflipH);
        flip_l = _mm256_and_si256(mOO, _mm256_sllv_epi64(PP, shift1897));
        flip_r = _mm256_and_si256(mOO, _mm256_srlv_epi64(PP, shift1897));
        flip_l = _mm256_or_si256(flip_l, _mm256_and_si256(mOO, _mm256_sllv_epi64(flip_l, shift1897)));
        flip_r = _mm256_or_si256(flip_r, _mm256_and_si256(mOO, _mm256_srlv_epi64(flip_r, shift1897)));
        pre_l = _mm256_and_si256(mOO, _mm256_sllv_epi64(mOO, shift1897));
        pre_r = _mm256_srlv_epi64(pre_l, shift1897);
        shift2 = _mm256_add_epi64(shift1897, shift1897);
        flip_l = _mm256_or_si256(flip_l, _mm256_and_si256(pre_l, _mm256_sllv_epi64(flip_l, shift2)));
        flip_r = _mm256_or_si256(flip_r, _mm256_and_si256(pre_r, _mm256_srlv_epi64(flip_r, shift2)));
        flip_l = _mm256_or_si256(flip_l, _mm256_and_si256(pre_l, _mm256_sllv_epi64(flip_l, shift2)));
        flip_r = _mm256_or_si256(flip_r, _mm256_and_si256(pre_r, _mm256_srlv_epi64(flip_r, shift2)));
        MM = _mm256_sllv_epi64(flip_l, shift1897);
        MM = _mm256_or_si256(MM, _mm256_srlv_epi64(flip_r, shift1897));
        M = _mm_or_si128(_mm256_castsi256_si128(MM), _mm256_extracti128_si256(MM, 1));
        M = _mm_or_si128(M, _mm_unpackhi_epi64(M, M));
        return _mm_cvtsi128_si64(M) & ~(P | O);
    }

#elif LEGAL_CALCULATION_MODE == 6
    inline uint64_t calc_mobility_hv(uint64_t p, uint64_t o){
        u64_4 p_o_pb_ob(p, o, p, o);
        p_o_pb_ob = black_line_mirror_3_4(p_o_pb_ob);
        u64_4 pr_or_pbr_obr = horizontal_mirror(p_o_pb_ob);
        u64_4 p1, oo;
        p1.data = _mm256_unpackhi_epi64(pr_or_pbr_obr, p_o_pb_ob); // p, pr, pb, pbr
        oo.data = _mm256_unpacklo_epi64(pr_or_pbr_obr, p_o_pb_ob); //o, or, ob, obr

        p1 = (p1 & 0x7F7F7F7F7F7F7F7FULL) << 1;
        oo = oo & 0x7F7F7F7F7F7F7F7FULL;
        u64_4 legal = (~(p1 | oo)) & (p1 + oo);

        legal = horizontal_mirror_1_3(legal); // p, p, pb, pb
        legal = black_line_mirror_3_4(legal); // p, p, p, p

        return _mm256_extract_epi64(legal.data, 3) | _mm256_extract_epi64(legal.data, 2) | _mm256_extract_epi64(legal.data, 1) | _mm256_extract_epi64(legal.data, 0);
    }

    inline uint64_t calc_legal(uint64_t p, uint64_t o){
        return
            (calc_mobility_hv(p, o)// | 
            /*calc_some_mobility_diag(p, o)*/) & ~(p | o);
    }
#endif