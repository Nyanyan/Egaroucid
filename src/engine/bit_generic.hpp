/*
    Egaroucid Project

    @file bit_generic.hpp
        Bit manipulation without AVX2
    @date 2021-2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once
#include <iostream>
#include "common.hpp"

/*
    @brief popcount algorithm

    @param x                    an integer
*/
#define pop_count_ull(x) std::popcount(x)
#define pop_count_uint(x) std::popcount(x)
#define pop_count_uchar(x) std::popcount(x)

/*
inline int pop_count_ull(uint64_t x) {
    x = x - ((x >> 1) & 0x5555555555555555ULL);
    x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL);
    x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0FULL;
    x = (x * 0x0101010101010101ULL) >> 56;
    return x;
}

inline int pop_count_uint(uint32_t x) {
    x = (x & 0x55555555) + ((x & 0xAAAAAAAA) >> 1);
    x = (x & 0x33333333) + ((x & 0xCCCCCCCC) >> 2);
    x = (x & 0x0F0F0F0F) + ((x & 0xF0F0F0F0) >> 4);
    x = (x & 0x00FF00FF) + ((x & 0xFF00FF00) >> 8);
    return (x & 0x0000FFFF) + ((x & 0xFFFF0000) >> 16);
}

inline int pop_count_uchar(uint8_t x) {
    x = (x & 0b01010101) + ((x & 0b10101010) >> 1);
    x = (x & 0b00110011) + ((x & 0b11001100) >> 2);
    return (x & 0b00001111) + ((x & 0b11110000) >> 4);
}
*/

/*
    @brief extract a digit of an integer

    @param x                    an integer
    @param place                a digit to extract
*/
inline uint32_t pop_digit(uint64_t x, int place) {
    return (uint32_t)(1ULL & (x >> place));
}

/*
    @brief mirroring a bitboard in white line

    @param x                    a bitboard
*/
inline uint64_t white_line_mirror(uint64_t x) {
    uint64_t a = (x ^ (x >> 7)) & 0x00AA00AA00AA00AAULL;
    x = x ^ a ^ (a << 7);
    a = (x ^ (x >> 14)) & 0x0000CCCC0000CCCCULL;
    x = x ^ a ^ (a << 14);
    a = (x ^ (x >> 28)) & 0x00000000F0F0F0F0ULL;
    return x ^ a ^ (a << 28);
}

/*
    @brief mirroring a bitboard in black line

    @param x                    a bitboard
*/
inline uint64_t black_line_mirror(uint64_t x) {
    uint64_t a = (x ^ (x >> 9)) & 0x0055005500550055ULL;
    x = x ^ a ^ (a << 9);
    a = (x ^ (x >> 18)) & 0x0000333300003333ULL;
    x = x ^ a ^ (a << 18);
    a = (x ^ (x >> 36)) & 0x000000000F0F0F0FULL;
    return x ^ a ^ (a << 36);
}

/*
    @brief mirroring a bitboard in vertical

    @param x                    a bitboard
*/
inline uint64_t vertical_mirror(uint64_t x) {
    x = ((x >> 8) & 0x00FF00FF00FF00FFULL) | ((x << 8) & 0xFF00FF00FF00FF00ULL);
    x = ((x >> 16) & 0x0000FFFF0000FFFFULL) | ((x << 16) & 0xFFFF0000FFFF0000ULL);
    return ((x >> 32) & 0x00000000FFFFFFFFULL) | ((x << 32) & 0xFFFFFFFF00000000ULL);
}

/*
    @brief mirroring a bitboard in horizontal

    @param x                    a bitboard
*/
inline uint64_t horizontal_mirror(uint64_t x) {
    x = ((x >> 1) & 0x5555555555555555ULL) | ((x << 1) & 0xAAAAAAAAAAAAAAAAULL);
    x = ((x >> 2) & 0x3333333333333333ULL) | ((x << 2) & 0xCCCCCCCCCCCCCCCCULL);
    return ((x >> 4) & 0x0F0F0F0F0F0F0F0FULL) | ((x << 4) & 0xF0F0F0F0F0F0F0F0ULL);
}

/*
    @brief rotate a board in 90 degrees in counter clockwise

    @param x                    a bitboard
*/
inline uint64_t rotate_90(uint64_t x) {
    return vertical_mirror(white_line_mirror(x));
}

/*
    @brief rotate a board in 270 degrees in counter clockwise

    @param x                    a bitboard
*/
inline uint64_t rotate_270(uint64_t x) {
    return vertical_mirror(black_line_mirror(x));
}

/*
    @brief rotate a board in 180 degrees

    @param x                    a bitboard
*/
inline uint64_t rotate_180(uint64_t x) {
    x = ((x & 0x5555555555555555ULL) << 1) | ((x & 0xAAAAAAAAAAAAAAAAULL) >> 1);
    x = ((x & 0x3333333333333333ULL) << 2) | ((x & 0xCCCCCCCCCCCCCCCCULL) >> 2);
    x = ((x & 0x0F0F0F0F0F0F0F0FULL) << 4) | ((x & 0xF0F0F0F0F0F0F0F0ULL) >> 4);
    x = ((x & 0x00FF00FF00FF00FFULL) << 8) | ((x & 0xFF00FF00FF00FF00ULL) >> 8);
    x = ((x & 0x0000FFFF0000FFFFULL) << 16) | ((x & 0xFFFF0000FFFF0000ULL) >> 16);
    return ((x & 0x00000000FFFFFFFFULL) << 32) | ((x & 0xFFFFFFFF00000000ULL) >> 32);
}

/*
    @brief NTZ (number of trailing zero) algorithm

    @param x                    a pointer of a bitboard
*/
#if USE_MINUS_NTZ
inline uint_fast8_t ctz(uint64_t *x) {
    return pop_count_ull((*x & (-(*x))) - 1);
}

inline uint_fast8_t ctz(uint64_t x) {
    return pop_count_ull((x & (-x)) - 1);
}

inline uint_fast8_t ctz_uint32(uint32_t x) {
    return pop_count_uint((x & (-x)) - 1);
}
#else
inline uint_fast8_t ctz(uint64_t *x) {
    return pop_count_ull((~(*x)) & ((*x) - 1));
}

inline uint_fast8_t ctz(uint64_t x) {
    return pop_count_ull((~x) & (x - 1));
}

inline uint_fast8_t ctz_uint32(uint32_t x) {
    return pop_count_uint((~x) & (x - 1));
}
#endif

/*
    @brief get the place of the first bit of a given board

    @param x                    a pointer of a bitboard
*/
inline uint_fast8_t first_bit(uint64_t *x) {
    return ctz(x);
}

/*
    @brief get the place of the next bit of a given board

    This function unsets the first bit.

    @param x                    a pointer of a bitboard
*/
inline uint_fast8_t next_bit(uint64_t *x) {
    *x &= *x - 1;
    return ctz(x);
}

inline uint_fast8_t join_h_line(uint64_t x, int t) {
    return (x >> (HW * t)) & 0b11111111U;
}

inline uint64_t split_h_line(uint_fast8_t x, int_fast8_t t) {
    return (uint64_t)x << (HW * t);
}

inline uint_fast8_t join_v_line(uint64_t x, int c) {
    return (((x >> c) & 0x0101010101010101ULL) * 0x0102040810204080ULL) >> 56;
}

inline uint64_t split_v_line(uint8_t x, int c) {
    return (((uint64_t)x * 0x0002040810204081ULL) & 0x0101010101010101ULL) << c;
}


/*
    . . . . e . . .
    . . . d . . . .
    . . c . . . . .
    . b . . . . . .
    a . . . . . . D
    . . . . . . C .
    . . . . . B . .
    . . . . A . . .
    to
    0b abcde000
    0b 0000ABCD


    t = x + y
     14 13 12 11 10  9  8  7
     13 12 11 10  9  8  7  6
     12 11 10  9  8  7  6  5
     11 10  9  8  7  6  5  4
     10  9  8  7  6  5  4  3
      9  8  7  6  5  4  3  2  
      8  7  6  5  4  3  2  1
      7  6  5  4  3  2  1  0

       ^
       |
       y
    <- x

*/

constexpr uint64_t join_d7_line_mask[15] = {
    0ULL, 0ULL, 0x0000000000010204ULL, 
    0x0000000001020408ULL, 0x0000000102040810ULL, 0x0000010204081020ULL, 
    0x0001020408102040ULL, 0x0102040810204080ULL, 0x0204081020408000ULL, 
    0x0408102040800000ULL, 0x0810204080000000ULL, 0x1020408000000000ULL, 
    0x2040800000000000ULL, 0ULL, 0ULL
};

// constexpr uint_fast8_t D7_LINE_MASK[HW2] = {
//     0x01, 0x03, 0x07, 0x0F, 0x1F, 0x3F, 0x7F, 0xFF,
//     0x03, 0x07, 0x0F, 0x1F, 0x3F, 0x7F, 0xFF, 0xFE,
//     0x07, 0x0F, 0x1F, 0x3F, 0x7F, 0xFF, 0xFE, 0xFC,
//     0x0F, 0x1F, 0x3F, 0x7F, 0xFF, 0xFE, 0xFC, 0xF8,
//     0x1F, 0x3F, 0x7F, 0xFF, 0xFE, 0xFC, 0xF8, 0xF0,
//     0x3F, 0x7F, 0xFF, 0xFE, 0xFC, 0xF8, 0xF0, 0xE0,
//     0x7F, 0xFF, 0xFE, 0xFC, 0xF8, 0xF0, 0xE0, 0xC0,
//     0xFF, 0xFE, 0xFC, 0xF8, 0xF0, 0xE0, 0xC0, 0x80
// };

inline int join_d7_line(uint64_t x, const int t) {
    return ((x & join_d7_line_mask[t]) * 0x0101010101010101ULL) >> 56;
}

inline uint64_t split_d7_line(uint8_t x, const int t) {
    return (((uint64_t)x * 0x0101010101010101ULL) & join_d7_line_mask[t]);
}


/*
    . . . . A . . .
    . . . . . B . .
    . . . . . . C .
    a . . . . . . D
    . b . . . . . .
    . . c . . . . .
    . . . d . . . .
    . . . . e . . .
    to
    0b abcde000
    0b 0000ABCD

    t = x + 7 - y
      7  6  5  4  3  2  1  0
      8  7  6  5  4  3  2  1
      9  8  7  6  5  4  3  2
     10  9  8  7  6  5  4  3
     11 10  9  8  7  6  5  4
     12 11 10  9  8  7  6  5
     13 12 11 10  9  8  7  6
     14 13 12 11 10  9  8  7

       ^
       |
       y
    <- x
*/

constexpr uint64_t join_d9_line_mask[15] = {
    0ULL, 0ULL, 0x0402010000000000ULL, 
    0x0804020100000000ULL, 0x1008040201000000ULL, 0x2010080402010000ULL, 
    0x4020100804020100ULL, 0x8040201008040201ULL, 0x0080402010080402ULL, 
    0x0000804020100804ULL, 0x0000008040201008ULL, 0x0000000080402010ULL, 
    0x0000000000804020ULL, 0ULL, 0ULL
};

// constexpr uint_fast8_t D9_LINE_MASK[HW2] = {
//     0xFF, 0xFE, 0xFC, 0xF8, 0xF0, 0xE0, 0xC0, 0x80,
//     0x7F, 0xFF, 0xFE, 0xFC, 0xF8, 0xF0, 0xE0, 0xC0,
//     0x3F, 0x7F, 0xFF, 0xFE, 0xFC, 0xF8, 0xF0, 0xE0,
//     0x1F, 0x3F, 0x7F, 0xFF, 0xFE, 0xFC, 0xF8, 0xF0,
//     0x0F, 0x1F, 0x3F, 0x7F, 0xFF, 0xFE, 0xFC, 0xF8,
//     0x07, 0x0F, 0x1F, 0x3F, 0x7F, 0xFF, 0xFE, 0xFC,
//     0x03, 0x07, 0x0F, 0x1F, 0x3F, 0x7F, 0xFF, 0xFE,
//     0x01, 0x03, 0x07, 0x0F, 0x1F, 0x3F, 0x7F, 0xFF
// };

inline int join_d9_line(uint64_t x, int t) {
    return ((x & join_d9_line_mask[t]) * 0x0101010101010101ULL)  >> 56;
}

inline uint64_t split_d9_line(uint8_t x, int t) {
    return ((uint64_t)x * 0x0101010101010101ULL) & join_d9_line_mask[t];
}

constexpr uint_fast8_t DIAGONAL_LINE_MASK_T[15] = {
    0x01, 0x03, 0x07, 
    0x0F, 0x1F, 0x3F, 
    0x7F, 0xFF, 0xFE,
    0xFC, 0xF8, 0xF0,
    0xE0, 0xC0, 0x80
};


/*
    @brief bit initialize
*/
void bit_init() {
}