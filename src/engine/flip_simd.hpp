/*
    Egaroucid Project

    @file flip_simd.hpp
        Flip calculation with SIMD
    @date 2021-2025
    @author Takuto Yamana
    @author Toshihiko Okuhara
    @license GPL-3.0-or-later
    @notice I referred to codes written by others
*/

#pragma once
#include "setting.hpp"
#include "common.hpp"
#include "bit.hpp"

#if AUTO_FLIP_OPT_BY_COMPILER
#ifdef __clang_version__
#define ACEPCK_RIGHT true
#define ACEPCK_LEFT true
#elif defined __GNUC__
#define ACEPCK_RIGHT false
#define ACEPCK_LEFT true
#elif defined _MSC_VER
#define ACEPCK_RIGHT true
#define ACEPCK_LEFT false
#else
#define ACEPCK_RIGHT false
#define ACEPCK_LEFT false
#endif
#else
#define ACEPCK_RIGHT false
#define ACEPCK_LEFT false
#endif

union V8DI {
    uint64_t ull[8];
    __m256i v4[2];
};
V8DI lrmask[HW2];

/*
    @brief Flip class

    @param pos                  a cell to put disc
    @param flip                 a bitboard representing flipped discs
*/
class Flip {

    public:
        uint_fast8_t pos;
        uint64_t flip;
    
    public:
        // original code from http://www.amy.hi-ho.ne.jp/okuhara/bitboard.htm
        // by Toshihiko Okuhara

        static inline __m128i calc_flip(__m128i OP, const uint_fast8_t place) {
            __m256i PP = _mm256_broadcastq_epi64(OP);
#if USE_AMD
            __m256i OO = _mm256_broadcastq_epi64(_mm_unpackhi_epi64(OP, OP)); // fast with AMD
#else
            __m256i OO = _mm256_permute4x64_epi64(_mm256_castsi128_si256(OP), 0x55); // fast with Intel
#endif
            __m256i mask = lrmask[place].v4[1];
        
#if ACEPCK_RIGHT
              // right: shadow mask lower than leftmost P
            __m256i rP = _mm256_and_si256(PP, mask);
            __m256i rS = _mm256_or_si256(rP, _mm256_srlv_epi64(rP, _mm256_set_epi64x(7, 9, 8, 1)));
            rS = _mm256_or_si256(rS, _mm256_srlv_epi64(rS, _mm256_set_epi64x(14, 18, 16, 2)));
            rS = _mm256_or_si256(rS, _mm256_srlv_epi64(rS, _mm256_set_epi64x(28, 36, 32, 4)));
              // erase if non-opponent MS1B is not P
            __m256i rE = _mm256_xor_si256(_mm256_andnot_si256(OO, mask), rP);	// masked Empty
            __m256i F4 = _mm256_and_si256(_mm256_andnot_si256(rS, mask), _mm256_cmpgt_epi64(rP, rE));
#else
              // right: isolate non-opponent MS1B by clearing lower bits
            __m256i eraser = _mm256_andnot_si256(OO, mask);
            __m256i rO = _mm256_sllv_epi64(_mm256_and_si256(PP, mask), _mm256_set_epi64x(7, 9, 8, 1));
            eraser = _mm256_or_si256(eraser, _mm256_srlv_epi64(eraser, _mm256_set_epi64x(7, 9, 8, 1)));
            rO = _mm256_andnot_si256(eraser, rO);
            rO = _mm256_andnot_si256(_mm256_srlv_epi64(eraser, _mm256_set_epi64x(14, 18, 16, 2)), rO);
            rO = _mm256_andnot_si256(_mm256_srlv_epi64(eraser, _mm256_set_epi64x(28, 36, 32, 4)), rO);
              // set mask bits higher than outflank
            __m256i F4 = _mm256_and_si256(mask, _mm256_sub_epi64(_mm256_setzero_si256(), rO));
#endif

            mask = lrmask[place].v4[0];
            __m256i lO = _mm256_andnot_si256(OO, mask);
#if ACEPCK_LEFT
              // left: non-opponent BLSMSK
            lO = _mm256_and_si256(_mm256_xor_si256(_mm256_add_epi64(lO, _mm256_set1_epi64x(-1)), lO), mask);
              // clear MSB of BLSMSK if it is P
            __m256i lF = _mm256_andnot_si256(PP, lO);
              // erase lF if lO = lF (i.e. MSB is not P)
            F4 = _mm256_or_si256(F4, _mm256_andnot_si256(_mm256_cmpeq_epi64(lF, lO), lF));
#else
              // left: look for non-opponent LS1B
            lO = _mm256_and_si256(lO, _mm256_sub_epi64(_mm256_setzero_si256(), lO));  // LS1B
            lO = _mm256_and_si256(lO, PP);
              // set all bits if outflank = 0, otherwise higher bits than outflank
            __m256i lF = _mm256_sub_epi64(_mm256_cmpeq_epi64(lO, _mm256_setzero_si256()), lO);
            F4 = _mm256_or_si256(F4, _mm256_andnot_si256(lF, mask));
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