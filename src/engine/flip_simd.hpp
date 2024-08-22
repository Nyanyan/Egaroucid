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
            __m256i PP = _mm256_broadcastq_epi64(OP);
            __m256i OO = _mm256_broadcastq_epi64(_mm_unpackhi_epi64(OP, OP));
            __m256i rM = lrmask[place].v4[1];
            __m256i lM = lrmask[place].v4[0];
            __m256i lO = _mm256_andnot_si256(OO, lM);

    #ifdef USE_AVX512
        #if !(ACEPCK & 1) // right: use prove
              // look for non-opponent (or edge) bit with lzcnt
            __m256i t0 = _mm256_lzcnt_epi64(_mm256_andnot_si256(OO, rM));
            t0 = _mm256_and_si256(_mm256_srlv_epi64(_mm256_set1_epi64x(0x8000000000000000), t0), PP);
              // clear masked OO lower than outflank
            __m256i F4 = _mm256_andnot_si256(_mm256_or_si256(_mm256_add_epi64(t0, _mm256_set1_epi64x(-1)), t0), rM);

        #else // right: use mask by acepck, step on a MSVC landmine?
              // shadow mask lower than leftmost P
            __m256i rP = _mm256_and_si256(PP, rM);
            __m256i t0 = _mm256_srlv_epi64(_mm256_set1_epi64x(-1), _mm256_lzcnt_epi64(rP));
              // apply flip if non-opponent MS1B is P
            // __m256i rE = _mm256_andnot_si256(OO, _mm256_andnot_si256(rP, rM));
            __m256i rE = _mm256_ternarylogic_epi64(OO, rM, rP, 0x04);	// masked empty
            __m256i F4 = _mm256_maskz_andnot_epi64(_mm256_cmpgt_epi64_mask(rP, rE), t0, rM);
        #endif

              // left: look for non-opponent LS1B
            // __m256i t2 = _mm256_xor_si256(_mm256_add_epi64(lO, _mm256_set1_epi64x(-1)), lO);	// BLSMSK
            // t2 = _mm256_and_si256(lM, t2);	// non-opponent LS1B and opponent inbetween
            __m256i t2 = _mm256_ternarylogic_epi64(lM, _mm256_add_epi64(lO, _mm256_set1_epi64x(-1)), lO, 0x60);
              // apply flip if P is in BLSMSK, i.e. LS1B is P
            // __m256i F4 = _mm256_mask_or_epi64(F4, _mm256_test_epi64_mask(PP, t2), F4, _mm256_andnot_si256(PP, t2));
            __m256i F4 = _mm256_mask_ternarylogic_epi64(F4, _mm256_test_epi64_mask(PP, t2), PP, t2, 0xf2);

    #else // AVX2
        #if !(ACEPCK & 1)
              // right: isolate non-opponent MS1B by clearing lower bits
            __m256i eraser = _mm256_andnot_si256(OO, rM);
            __m256i rO = _mm256_sllv_epi64(_mm256_and_si256(PP, rM), _mm256_set_epi64x(7, 9, 8, 1));
            eraser = _mm256_or_si256(eraser, _mm256_srlv_epi64(eraser, _mm256_set_epi64x(7, 9, 8, 1)));
            rO = _mm256_andnot_si256(eraser, rO);
            rO = _mm256_andnot_si256(_mm256_srlv_epi64(eraser, _mm256_set_epi64x(14, 18, 16, 2)), rO);
            rO = _mm256_andnot_si256(_mm256_srlv_epi64(eraser, _mm256_set_epi64x(28, 36, 32, 4)), rO);
              // set mask bits higher than outflank
            __m256i F4 = _mm256_and_si256(rM, _mm256_sub_epi64(_mm256_setzero_si256(), rO));

        #else
              // right: shadow mask lower than leftmost P
            __m256i rP = _mm256_and_si256(PP, rM);
            __m256i rS = _mm256_or_si256(rP, _mm256_srlv_epi64(rP, _mm256_set_epi64x(7, 9, 8, 1)));
            rS = _mm256_or_si256(rS, _mm256_srlv_epi64(rS, _mm256_set_epi64x(14, 18, 16, 2)));
            rS = _mm256_or_si256(rS, _mm256_srlv_epi64(rS, _mm256_set_epi64x(28, 36, 32, 4)));
              // erase if non-opponent MS1B is not P
            __m256i rE = _mm256_xor_si256(_mm256_andnot_si256(OO, rM), rP);	// masked Empty
            __m256i F4 = _mm256_and_si256(_mm256_andnot_si256(rS, rM), _mm256_cmpgt_epi64(rP, rE));
        #endif

        #if !(ACEPCK & 2)
              // left: look for non-opponent LS1B
            lO = _mm256_and_si256(lO, _mm256_sub_epi64(_mm256_setzero_si256(), lO));  // LS1B
            lO = _mm256_and_si256(lO, PP);
              // set all bits if outflank = 0, otherwise higher bits than outflank
            __m256i lF = _mm256_sub_epi64(_mm256_cmpeq_epi64(lO, _mm256_setzero_si256()), lO);
            F4 = _mm256_or_si256(F4, _mm256_andnot_si256(lF, lM));

        #else
              // left: non-opponent BLSMSK
            lO = _mm256_and_si256(_mm256_xor_si256(_mm256_add_epi64(lO, _mm256_set1_epi64x(-1)), lO), lM);
              // clear MSB of BLSMSK if it is P
            __m256i lF = _mm256_andnot_si256(PP, lO);
              // erase lF if lO = lF (i.e. MSB is not P)
            F4 = _mm256_or_si256(F4, _mm256_andnot_si256(_mm256_cmpeq_epi64(lF, lO), lF));
        #endif
    #endif

            __m128i F2 = _mm_or_si128(_mm256_castsi256_si128(F4), _mm256_extracti128_si256(F4, 1));
            return _mm_or_si128(F2, _mm_shuffle_epi32(F2, 0x4e));	// SWAP64
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
            lrmask[y * 8 + x].v4[0] = lmask;
            lrmask[(7 - y) * 8 + x].v4[1] = rmask;
            lmask = _mm256_slli_epi64(lmask, 8);
            rmask = _mm256_srli_epi64(rmask, 8);
        }
    }
}
