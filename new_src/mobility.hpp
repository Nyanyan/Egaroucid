#pragma once
#include <iostream>
#include "bit.hpp"
#include "setting.hpp"

#if MOBILITY_MODE == 0

inline uint8_t calc_mobility_line(uint8_t p, uint8_t o){
    uint8_t p1 = p << 1;
    return ~(p1 | o) & (p1 + o);
}

inline uint64_t calc_mobility_left(uint64_t p, uint64_t o){
    return (uint64_t)calc_mobility_line(p & 0xFF, o & 0xFF) | 
        ((uint64_t)calc_mobility_line((p >> 8) & 0xFF, (o >> 8) & 0xFF) << 8) | 
        ((uint64_t)calc_mobility_line((p >> 16) & 0xFF, (o >> 16) & 0xFF) << 16) | 
        ((uint64_t)calc_mobility_line((p >> 24) & 0xFF, (o >> 24) & 0xFF) << 24) | 
        ((uint64_t)calc_mobility_line((p >> 32) & 0xFF, (o >> 32) & 0xFF) << 32) | 
        ((uint64_t)calc_mobility_line((p >> 40) & 0xFF, (o >> 40) & 0xFF) << 40) | 
        ((uint64_t)calc_mobility_line((p >> 48) & 0xFF, (o >> 48) & 0xFF) << 48) | 
        ((uint64_t)calc_mobility_line((p >> 56) & 0xFF, (o >> 56) & 0xFF) << 56);
}

inline uint64_t calc_mobility(uint64_t p, uint64_t o){
    uint64_t res = calc_mobility_left(p, o) | 
        horizontal_mirror(calc_mobility_left(horizontal_mirror(p), horizontal_mirror(o))) | 
        black_line_mirror(calc_mobility_left(black_line_mirror(p), black_line_mirror(o))) | 
        white_line_mirror(calc_mobility_left(white_line_mirror(p), white_line_mirror(o)));
    constexpr uint8_t mask[N_DIAG_LINE] = {
        0b11100000, 0b11110000, 0b11111000, 0b11111100, 0b11111110, 0b11111111,
        0b01111111, 0b00111111, 0b00011111, 0b00001111, 0b00000111
    };
    uint32_t i;
    for (i = 0; i < N_DIAG_LINE; ++i){
        res |= split_d7_lines[calc_mobility_line(join_d7_lines[i](p), join_d7_lines[i](o)) & mask[i]][i];
        res |= split_d9_lines[calc_mobility_line(join_d9_lines[i](p), join_d9_lines[i](o)) & mask[N_DIAG_LINE_M1 - i]][i];
    }
    res = white_line_mirror(res);
    p = white_line_mirror(p);
    o = white_line_mirror(o);
    for (i = 0; i < N_DIAG_LINE; ++i){
        res |= split_d7_lines[calc_mobility_line(join_d7_lines[i](p), join_d7_lines[i](o)) & mask[i]][i];
        res |= split_d9_lines[calc_mobility_line(join_d9_lines[i](p), join_d9_lines[i](o)) & mask[N_DIAG_LINE_M1 - i]][i];
    }
    res = black_line_mirror(res);
    p = black_line_mirror(p);
    o = black_line_mirror(o);
    for (i = 0; i < N_DIAG_LINE; ++i){
        res |= split_d7_lines[calc_mobility_line(join_d7_lines[i](p), join_d7_lines[i](o)) & mask[i]][i];
        res |= split_d9_lines[calc_mobility_line(join_d9_lines[i](p), join_d9_lines[i](o)) & mask[N_DIAG_LINE_M1 - i]][i];
    }
    res &= ~(p | o);
    return rotate_180(res);
}

#elif MOBILITY_MODE == 1

inline uint8_t calc_mobility_line(uint8_t p, uint8_t o){
    uint8_t p1 = p << 1;
    return ~(p1 | o) & (p1 + o);
}

inline uint64_t calc_mobility_left(uint64_t p, uint64_t o){
    uint64_t p1 = (p & 0x7F7F7F7F7F7F7F7FULL) << 1;
    return ~(p1 | o) & (p1 + (o & 0x7F7F7F7F7F7F7F7FULL));
}

inline uint64_t calc_mobility_diagleft(uint64_t p, uint64_t o){
    uint64_t p1 = (p & 0x5F6F777B7D7E7F3FULL) << 1;
    return ~(p1 | o) & (p1 + (o & 0x5F6F777B7D7E7F3FULL));
}

inline uint64_t calc_mobility(uint64_t p, uint64_t o){
    uint64_t res = calc_mobility_left(p, o) | 
        horizontal_mirror(calc_mobility_left(horizontal_mirror(p), horizontal_mirror(o))) | 
        black_line_mirror(calc_mobility_left(black_line_mirror(p), black_line_mirror(o))) | 
        white_line_mirror(calc_mobility_left(white_line_mirror(p), white_line_mirror(o))) | 
        unrotate_45(calc_mobility_diagleft(rotate_45(p), rotate_45(o))) | 
        unrotate_135(calc_mobility_diagleft(rotate_135(p), rotate_135(o))) | 
        unrotate_225(calc_mobility_diagleft(rotate_225(p), rotate_225(o))) | 
        unrotate_315(calc_mobility_diagleft(rotate_315(p), rotate_315(o)));
    return res & ~(p | o);
}

#ifdef _MSC_VER
    #include <intrin.h>
#else
    #include <x86intrin.h>
#endif

#ifdef _MSC_VER
	#define	mirror_v(x)	_byteswap_uint64(x)
#else
	#define	mirror_v(x)	__builtin_bswap64(x)
#endif

inline unsigned long long get_mobility(const unsigned long long P, const unsigned long long O){
    unsigned long long moves, mO, flip1, pre1, flip8, pre8;
    __m128i	PP, mOO, MM, flip, pre;
    mO = O & 0x7e7e7e7e7e7e7e7eULL;
    PP  = _mm_set_epi64x(mirror_v(P), P);
    mOO = _mm_set_epi64x(mirror_v(mO), mO);
    flip = _mm_and_si128(mOO, _mm_slli_epi64(PP, 7));				            flip1  = mO & (P << 1);		    flip8  = O & (P << 8);
    flip = _mm_or_si128(flip, _mm_and_si128(mOO, _mm_slli_epi64(flip, 7)));		flip1 |= mO & (flip1 << 1);	    flip8 |= O & (flip8 << 8);
    pre  = _mm_and_si128(mOO, _mm_slli_epi64(mOO, 7));				            pre1   = mO & (mO << 1);	    pre8   = O & (O << 8);
    flip = _mm_or_si128(flip, _mm_and_si128(pre, _mm_slli_epi64(flip, 14)));	flip1 |= pre1 & (flip1 << 2);	flip8 |= pre8 & (flip8 << 16);
    flip = _mm_or_si128(flip, _mm_and_si128(pre, _mm_slli_epi64(flip, 14)));	flip1 |= pre1 & (flip1 << 2);	flip8 |= pre8 & (flip8 << 16);
    MM = _mm_slli_epi64(flip, 7);							                    moves = flip1 << 1;		        moves |= flip8 << 8;
    flip = _mm_and_si128(mOO, _mm_slli_epi64(PP, 9));				            flip1  = mO & (P >> 1);		    flip8  = O & (P >> 8);
    flip = _mm_or_si128(flip, _mm_and_si128(mOO, _mm_slli_epi64(flip, 9)));		flip1 |= mO & (flip1 >> 1);	    flip8 |= O & (flip8 >> 8);
    pre = _mm_and_si128(mOO, _mm_slli_epi64(mOO, 9));				            pre1 >>= 1;			            pre8 >>= 8;
    flip = _mm_or_si128(flip, _mm_and_si128(pre, _mm_slli_epi64(flip, 18)));	flip1 |= pre1 & (flip1 >> 2);	flip8 |= pre8 & (flip8 >> 16);
    flip = _mm_or_si128(flip, _mm_and_si128(pre, _mm_slli_epi64(flip, 18)));	flip1 |= pre1 & (flip1 >> 2);	flip8 |= pre8 & (flip8 >> 16);
    MM = _mm_or_si128(MM, _mm_slli_epi64(flip, 9));					            moves |= flip1 >> 1;		    moves |= flip8 >> 8;
    moves |= _mm_cvtsi128_si64(MM) | mirror_v(_mm_cvtsi128_si64(_mm_unpackhi_epi64(MM, MM)));
    return moves & ~(P|O);
}

#endif