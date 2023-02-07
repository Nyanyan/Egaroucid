/*
    Egaroucid Project

    @file bit_generic.hpp
        Bit manipulation without AVX2
    @date 2021-2023
    @author Takuto Yamana (a.k.a. Nyanyan)
    @license GPL-3.0 license
*/

#pragma once
#include <iostream>
#include "common.hpp"

/*
    @brief print bits in reverse

    @param x                    an integer to print
*/
inline void bit_print_reverse(uint64_t x){
    for (uint32_t i = 0; i < HW2; ++i)
        std::cerr << (1 & (x >> i));
    std::cerr << std::endl;
}

/*
    @brief print bits

    @param x                    an integer to print
*/
inline void bit_print(uint64_t x){
    for (uint32_t i = 0; i < HW2; ++i)
        std::cerr << (1 & (x >> (HW2_M1 - i)));
    std::cerr << std::endl;
}

/*
    @brief print bits of uint8_t

    @param x                    an integer to print
*/
inline void bit_print_uchar(uint8_t x){
    for (uint32_t i = 0; i < HW; ++i)
        std::cerr << (1 & (x >> (HW_M1 - i)));
    std::cerr << std::endl;
}

/*
    @brief print a board in reverse

    @param x                    an integer to print
*/
inline void bit_print_board_reverse(uint64_t x){
    for (uint32_t i = 0; i < HW2; ++i){
        std::cerr << (1 & (x >> i));
        if (i % HW == HW_M1)
            std::cerr << std::endl;
    }
    std::cerr << std::endl;
}

/*
    @brief print a board

    @param x                    an integer to print
*/
inline void bit_print_board(uint64_t x){
    for (uint32_t i = 0; i < HW2; ++i){
        std::cerr << (1 & (x >> (HW2_M1 - i)));
        if (i % HW == HW_M1)
            std::cerr << std::endl;
    }
    std::cerr << std::endl;
}

/*
    @brief print a board

    @param p                    an integer representing the player
    @param o                    an integer representing the opponent
*/
void print_board(uint64_t p, uint64_t o){
    for (int i = 0; i < HW2; ++i){
        if (1 & (p >> (HW2_M1 - i)))
            std::cerr << '0';
        else if (1 & (o >> (HW2_M1 - i)))
            std::cerr << '1';
        else
            std::cerr << '.';
        if (i % HW == HW_M1)
            std::cerr << std::endl;
    }
}

/*
    @brief popcount algorithm

    @param x                    an integer
*/
inline int pop_count_ull(uint64_t x){
    x = x - ((x >> 1) & 0x5555555555555555ULL);
    x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL);
    x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0FULL;
    x = (x * 0x0101010101010101ULL) >> 56;
    return x;
}

inline int pop_count_uint(uint32_t x){
    x = (x & 0x55555555) + ((x & 0xAAAAAAAA) >> 1);
    x = (x & 0x33333333) + ((x & 0xCCCCCCCC) >> 2);
    return (x & 0x0F0F0F0F) + ((x & 0xF0F0F0F0) >> 4);
}

inline int pop_count_uchar(uint8_t x){
    x = (x & 0b01010101) + ((x & 0b10101010) >> 1);
    x = (x & 0b00110011) + ((x & 0b11001100) >> 2);
    return (x & 0b00001111) + ((x & 0b11110000) >> 4);
}

/*
    @brief extract a digit of an integer

    @param x                    an integer
    @param place                a digit to extract
*/
inline uint32_t pop_digit(uint64_t x, int place){
    return (uint32_t)(1ULL & (x >> place));
}

/*
    @brief mirroring a bitboard in white line

    @param x                    a bitboard
*/
inline uint64_t white_line_mirror(uint64_t x){
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
inline uint64_t black_line_mirror(uint64_t x){
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
inline uint64_t vertical_mirror(uint64_t x){
    x = ((x >> 8) & 0x00FF00FF00FF00FFULL) | ((x << 8) & 0xFF00FF00FF00FF00ULL);
    x = ((x >> 16) & 0x0000FFFF0000FFFFULL) | ((x << 16) & 0xFFFF0000FFFF0000ULL);
    return ((x >> 32) & 0x00000000FFFFFFFFULL) | ((x << 32) & 0xFFFFFFFF00000000ULL);
}

/*
    @brief mirroring a bitboard in horizontal

    @param x                    a bitboard
*/
inline uint64_t horizontal_mirror(uint64_t x){
    x = ((x >> 1) & 0x5555555555555555ULL) | ((x << 1) & 0xAAAAAAAAAAAAAAAAULL);
    x = ((x >> 2) & 0x3333333333333333ULL) | ((x << 2) & 0xCCCCCCCCCCCCCCCCULL);
    return ((x >> 4) & 0x0F0F0F0F0F0F0F0FULL) | ((x << 4) & 0xF0F0F0F0F0F0F0F0ULL);
}

/*
    @brief rotate a board in 90 degrees in counter clockwise

    @param x                    a bitboard
*/
inline uint64_t rotate_90(uint64_t x){
    return vertical_mirror(white_line_mirror(x));
}

/*
    @brief rotate a board in 270 degrees in counter clockwise

    @param x                    a bitboard
*/
inline uint64_t rotate_270(uint64_t x){
    return vertical_mirror(black_line_mirror(x));
}

/*
    @brief rotate a board in 180 degrees

    @param x                    a bitboard
*/
inline uint64_t rotate_180(uint64_t x){
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
    inline uint_fast8_t ntz(uint64_t *x){
        return pop_count_ull((*x & (-(*x))) - 1);
    }

    inline uint_fast8_t ntz(uint64_t x){
        return pop_count_ull((x & (-x)) - 1);
    }
#else
    inline uint_fast8_t ntz(uint64_t *x){
        return pop_count_ull((~(*x)) & ((*x) - 1));
    }

    inline uint_fast8_t ntz(uint64_t x){
        return pop_count_ull((~x) & (x - 1));
    }
#endif

/*
    @brief get the place of the first bit of a given board

    @param x                    a pointer of a bitboard
*/
inline uint_fast8_t first_bit(uint64_t *x){
    return ntz(x);
}

/*
    @brief get the place of the next bit of a given board

    This function unsets the first bit.

    @param x                    a pointer of a bitboard
*/
inline uint_fast8_t next_bit(uint64_t *x){
    *x &= *x - 1;
    return ntz(x);
}

inline uint_fast8_t join_h_line(uint64_t x, int t){
    return (x >> (HW * t)) & 0b11111111U;
}

inline uint64_t split_h_line(uint_fast8_t x, int_fast8_t t){
    return (uint64_t)x << (HW * t);
}

inline int join_v_line(uint64_t x, int c){
    x = (x >> c) & 0x0101010101010101ULL;
    return (x * 0x0102040810204080ULL) >> 56;
}

inline uint64_t split_v_line(uint8_t x, int c){
    uint64_t res = ((uint64_t)x * 0x0002040810204081ULL) & 0x0101010101010101ULL;
    return res << c;
}

constexpr uint64_t join_d7_line_mask[15] = {
    0ULL, 0ULL, 0x0000000000010204ULL, 0x0000000001020408ULL, 
    0x0000000102040810ULL, 0x0000010204081020ULL, 0x0001020408102040ULL, 0x0102040810204080ULL, 
    0x0204081020408000ULL, 0x0408102040800000ULL, 0x0810204080000000ULL, 0x1020408000000000ULL, 
    0x2040800000000000ULL, 0ULL, 0ULL
};

constexpr uint8_t join_d7_line_leftshift[15] = {
    0, 0, 5, 4, 
    3, 2, 1, 0, 
    0, 0, 0, 0, 
    0, 0, 0
};

constexpr uint8_t join_d7_line_rightshift[15] = {
    0, 0, 0, 0, 
    0, 0, 0, 0, 
    8, 16, 24, 32, 
    40, 0, 0
};

inline int join_d7_line(uint64_t x, const int t){
    x = (x & join_d7_line_mask[t]);
    x <<= join_d7_line_leftshift[t];
    x >>= join_d7_line_rightshift[t];
    uint64_t res = ((x * 0x0002082080000000ULL) & 0x0F00000000000000ULL) | ((x * 0x0000000002082080ULL) & 0xF000000000000000ULL);
    return res >> 56;
}

inline uint64_t split_d7_line(uint8_t x, const int t){
    uint64_t res = ((uint64_t)(x & 0b00001111) * 0x0000000002082080ULL) & 0x0000000010204080ULL;
    res |= ((uint64_t)(x & 0b11110000) * 0x0002082080000000ULL) & 0x0102040800000000ULL;
    res >>= join_d7_line_leftshift[t];
    res <<= join_d7_line_rightshift[t];
    return res;
}

constexpr uint64_t join_d9_line_mask[15] = {
    0ULL, 0ULL, 0x0402010000000000ULL, 0x0804020100000000ULL, 
    0x1008040201000000ULL, 0x2010080402010000ULL, 0x4020100804020100ULL, 0x8040201008040201ULL, 
    0x0080402010080402ULL, 0x0000804020100804ULL, 0x0000008040201008ULL, 0x0000000080402010ULL, 
    0x0000000000804020ULL, 0ULL, 0ULL
};

constexpr uint8_t join_d9_line_rightshift[15] = {
    0, 0, 40, 32, 
    24, 16, 8, 0, 
    1, 2, 3, 4, 
    5, 0, 0
};

inline int join_d9_line(uint64_t x, int t){
    x = x & join_d9_line_mask[t];
    x >>= join_d9_line_rightshift[t];
    return (x * 0x0101010101010101ULL) >> 56;
}

inline uint64_t split_d9_line(uint8_t x, int t){
    uint64_t res = ((uint64_t)x * 0x0101010101010101ULL) & 0x8040201008040201ULL;
    res <<= join_d9_line_rightshift[t];
    return res;
}


/*
    @brief bit initialize
*/
void bit_init(){
}