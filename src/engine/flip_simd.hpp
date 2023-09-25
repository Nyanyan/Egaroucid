/*
    Egaroucid Project

    @file flip_simd.hpp
        Flip calculation with SIMD
    @date 2021-2023
    @author Takuto Yamana
    @author Toshihiko Okuhara
    @license GPL-3.0 license
    @notice I referred to codes written by others
*/

#pragma once
#include "setting.hpp"
#include "common.hpp"
#include "bit.hpp"

__m256i lmask_v4[HW2], rmask_v4[HW2];

/*
    @brief Flip class

    @param pos                  a cell to put disc
    @param flip                 a bitboard representing flipped discs
*/
class Flip{

    public:
        uint_fast8_t pos;
        uint64_t flip;
    
    public:
        // original code from http://www.amy.hi-ho.ne.jp/okuhara/bitboard.htm
        // by Toshihiko Okuhara
        static inline __m128i calc_flip(__m128i OP, const uint_fast8_t place) {
            __m256i  PP, OO, flip4, outflank, eraser, mask;
            __m128i  flip2;

            PP = _mm256_broadcastq_epi64(OP);
            OO = _mm256_permute4x64_epi64(_mm256_castsi128_si256(OP), 0x55);

            mask = rmask_v4[place];
              // isolate non-opponent MS1B by clearing lower bits
            eraser = _mm256_andnot_si256(OO, mask);
            outflank = _mm256_sllv_epi64(_mm256_and_si256(PP, mask), _mm256_set_epi64x(7, 9, 8, 1));
            eraser = _mm256_or_si256(eraser, _mm256_srlv_epi64(eraser, _mm256_set_epi64x(7, 9, 8, 1)));
            outflank = _mm256_andnot_si256(eraser, outflank);
            outflank = _mm256_andnot_si256(_mm256_srlv_epi64(eraser, _mm256_set_epi64x(14, 18, 16, 2)), outflank);
            outflank = _mm256_andnot_si256(_mm256_srlv_epi64(eraser, _mm256_set_epi64x(28, 36, 32, 4)), outflank);
              // set mask bits higher than outflank
            flip4 = _mm256_and_si256(mask, _mm256_sub_epi64(_mm256_setzero_si256(), outflank));

            mask = lmask_v4[place];
              // look for non-opponent LS1B
            outflank = _mm256_andnot_si256(OO, mask);
            outflank = _mm256_and_si256(outflank, _mm256_sub_epi64(_mm256_setzero_si256(), outflank));  // LS1B
            outflank = _mm256_and_si256(outflank, PP);
              // set all bits if outflank = 0, otherwise higher bits than outflank
            eraser = _mm256_sub_epi64(_mm256_cmpeq_epi64(outflank, _mm256_setzero_si256()), outflank);
            flip4 = _mm256_or_si256(flip4, _mm256_andnot_si256(eraser, mask));

            flip2 = _mm_or_si128(_mm256_castsi256_si128(flip4), _mm256_extracti128_si256(flip4, 1));
            return _mm_or_si128(flip2, _mm_shuffle_epi32(flip2, 0x4e));	// SWAP64
        }

        inline uint64_t calc_flip(const uint64_t player, const uint64_t opponent, const uint_fast8_t place) {
            pos = place;
            flip = _mm_cvtsi128_si64(calc_flip(_mm_set_epi64x(opponent, player), place));
            return flip;
        }
};

/*
    @brief Flip initialize
*/
void flip_init() {
    for (int x = 0; x < 8; ++x) {
        __m256i lmask = _mm256_set_epi64x(
            (0x0102040810204080ULL >> ((7 - x) * 8)) & 0xffffffffffffff00ULL,
            (0x8040201008040201ULL >> (x * 8)) & 0xffffffffffffff00ULL,
            (0x0101010101010101ULL << x) & 0xffffffffffffff00ULL,
            (0xfe << x) & 0xff
        );
        __m256i rmask = _mm256_set_epi64x(
            (0x0102040810204080ULL << (x * 8)) & 0x00ffffffffffffffULL,
            (0x8040201008040201ULL << ((7 - x) * 8)) & 0x00ffffffffffffffULL,
            (0x0101010101010101ULL << x) & 0x00ffffffffffffffULL,
            (uint64_t)(0x7f >> (7 - x)) << 56
        );

        for (int y = 0; y < 8; ++y) {
            lmask_v4[y * 8 + x] = lmask;
            rmask_v4[(7 - y) * 8 + x] = rmask;
            lmask = _mm256_slli_epi64(lmask, 8);
            rmask = _mm256_srli_epi64(rmask, 8);
        }
    }
}
