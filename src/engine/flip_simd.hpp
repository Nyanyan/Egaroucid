/*
    Egaroucid Project

    @file flip_simd.hpp
        Flip calculation with SIMD
    @date 2021-2024
    @author Takuto Yamana
    @author Toshihiko Okuhara
    @license GPL-3.0 license
    @notice I referred to codes written by others
*/

#pragma once
#include "setting.hpp"
#include "common.hpp"
#include "bit.hpp"

union V8DI {
    uint64_t ull[8];
    __m256i v4[2];
    #ifdef USE_AVX512
        __m512i v8;
    #endif
};
V8DI lrmask[HW2];

__m256i bb_shift[3];

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
    #if USE_AVX512

        static inline __m128i calc_flip(__m128i OP, const uint_fast8_t place) {
            __m256i PP = _mm256_broadcastq_epi64(OP);
            __m256i OO = _mm256_permute4x64_epi64(_mm256_castsi128_si256(OP), 0x55);
            __m256i mask = lrmask[place].v4[1];
              // right: look for non-opponent (or edge) bit with lzcnt
            __m256i outflank = _mm256_andnot_si256(OO, mask);
            outflank = _mm256_srlv_epi64(_mm256_set1_epi64x(0x8000000000000000), _mm256_lzcnt_epi64(outflank));
            outflank = _mm256_and_si256(outflank, PP);
              // set all bits higher than outflank
            __m256i flip4 = _mm256_andnot_si256(_mm256_or_si256(_mm256_add_epi64(outflank, _mm256_set1_epi64x(-1)), outflank), mask);

              // left: look for non-opponent LS1B
            mask = lrmask[place].v4[0];
            outflank = _mm256_andnot_si256(OO, mask);
            outflank = _mm256_xor_si256(outflank, _mm256_add_epi64(outflank, _mm256_set1_epi64x(-1)));	// BLSMSK
            outflank = _mm256_and_si256(outflank, mask);	// non-opponent LS1B and opponent inbetween
              // apply flip if P is in BLSMSK, i.e. LS1B is P
            flip4 = _mm256_mask_or_epi64(flip4, _mm256_test_epi64_mask(outflank, PP), flip4, _mm256_and_si256(outflank, OO));

            __m128i flip2 = _mm_or_si128(_mm256_castsi256_si128(flip4), _mm256_extracti128_si256(flip4, 1));
            return _mm_or_si128(flip2, _mm_shuffle_epi32(flip2, 0x4e));	// SWAP64
        }

    #else

        static inline __m128i calc_flip(__m128i OP, const uint_fast8_t place) {
            __m256i bp = _mm256_broadcastq_epi64(OP);
            __m256i bo = _mm256_permute4x64_epi64(_mm256_castsi128_si256(OP), 0x55);

            __m256i rp = _mm256_and_si256(bp, lrmask[place].v4[1]);
            __m256i lo = _mm256_andnot_si256(bo, lrmask[place].v4[0]);
            __m256i t5 = _mm256_or_si256(_mm256_srlv_epi64(rp, bb_shift[0]), rp);
                    t5 = _mm256_or_si256(t5, _mm256_srlv_epi64(t5, bb_shift[1]));
                    t5 = _mm256_or_si256(t5, _mm256_srlv_epi64(t5, bb_shift[2]));
            __m256i rf = _mm256_cmpgt_epi64(rp, _mm256_xor_si256(_mm256_andnot_si256(bo, lrmask[place].v4[1]), rp));
                    rf = _mm256_and_si256(_mm256_andnot_si256(t5, lrmask[place].v4[1]), rf);

            __m256i lp = _mm256_add_epi64(lo, _mm256_set1_epi64x(-1));
                    lp = _mm256_and_si256(_mm256_xor_si256(lp, lo), lrmask[place].v4[0]);
            __m256i lf = _mm256_andnot_si256(bp, lp);
                    lf = _mm256_andnot_si256(_mm256_cmpeq_epi64(lp, lf), lf);

            __m256i f4 = _mm256_or_si256(rf, lf);
            __m128i fl = _mm_or_si128(_mm256_castsi256_si128(f4), _mm256_extracti128_si256(f4, 1));
            return _mm_or_si128(fl, _mm_shuffle_epi32(fl, 0x4e));
        }

    #endif

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
            lrmask[y * 8 + x].v4[0] = lmask;
            lrmask[(7 - y) * 8 + x].v4[1] = rmask;
            lmask = _mm256_slli_epi64(lmask, 8);
            rmask = _mm256_srli_epi64(rmask, 8);
        }
    }
    bb_shift[0] = _mm256_set_epi64x(7, 9, 8, 1);
    bb_shift[1] = _mm256_set_epi64x(14, 18, 16, 2);
    bb_shift[2] = _mm256_set_epi64x(21, 27, 24, 3);
}
