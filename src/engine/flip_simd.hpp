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

        static inline __m128i calc_flip(__m128i OP, const uint_fast8_t place) {
    #ifdef USE_AVX512
			auto bd = OP;
			auto bp = _mm256_broadcastq_epi64 (bd);
			auto bo = _mm256_permute4x64_epi64 (_mm256_castsi128_si256 (bd), 0x55);

			auto ml = lrmask[place].v4[0];
			auto mr = lrmask[place].v4[1];
			auto rr = _mm256_and_si256 (bp, mr);
			auto t0 = _mm256_srlv_epi64 (_mm256_set1_epi64x (-1), _mm256_lzcnt_epi64 (rr));
			auto ll = _mm256_andnot_si256 (bo, ml);
			auto t1 = _mm256_add_epi64 (ll, _mm256_set1_epi64x (-1));
			auto t2 = _mm256_ternarylogic_epi64 (ml, t1, ll, 0x60);
			auto k0 = _mm256_test_epi64_mask (bp, t2);
			auto fl = _mm256_maskz_andnot_epi64 (k0, bp, t2);
			auto t3 = _mm256_ternarylogic_epi64 (bo, mr, rr, 0x04);
			auto k1 = _mm256_cmp_epi64_mask (t3, rr, _MM_CMPINT_LT);
			auto f4 = _mm256_mask_ternarylogic_epi64 (fl, k1, t0, mr, 0xf2);

			auto f2 = _mm_or_si128 (_mm256_castsi256_si128 (f4), _mm256_extracti128_si256 (f4, 1));

			return _mm_or_si128 (f2, _mm_shuffle_epi32 (f2, 0x4e));
    #else
			auto bd = OP;
			auto bp = _mm256_broadcastq_epi64 (bd);
			auto bo = _mm256_permute4x64_epi64 (_mm256_castsi128_si256 (bd), 0x55);

			auto ml = lrmask[place].v4[0];
			auto mr = lrmask[place].v4[1];
			auto rp = _mm256_and_si256 (bp, mr);
			auto lo = _mm256_andnot_si256 (bo, ml);
			auto cr = _mm256_cmpgt_epi64 (rp, _mm256_xor_si256 (_mm256_andnot_si256 (bo, mr), rp));
			auto t1 = _mm256_or_si256 (_mm256_srlv_epi64 (rp, bb_shift[0]), rp);
			auto t3 = _mm256_or_si256 (t1, _mm256_srlv_epi64 (t1, bb_shift[1]));
			auto t5 = _mm256_or_si256 (t3, _mm256_srlv_epi64 (t3, bb_shift[2]));
			auto rf = _mm256_and_si256 (_mm256_andnot_si256 (t5, mr), cr);
			auto la = _mm256_add_epi64 (lo, _mm256_set1_epi64x (-1));
			auto lp = _mm256_and_si256 (_mm256_xor_si256 (la, lo), ml);
			auto ll = _mm256_andnot_si256 (bp, lp);
			auto lf = _mm256_andnot_si256 (_mm256_cmpeq_epi64 (lp, ll), ll);
			auto f4 = _mm256_or_si256 (rf, lf);

			auto fl = _mm_or_si128 (_mm256_castsi256_si128 (f4), _mm256_extracti128_si256 (f4, 1));

			return _mm_or_si128 (fl, _mm_shuffle_epi32 (fl, 0x4e));
    #endif
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
	bb_shift[0] = _mm256_set_epi64x (7, 9, 8, 1);
	bb_shift[1] = _mm256_set_epi64x (14, 18, 16, 2);
	bb_shift[2] = _mm256_set_epi64x (21, 27, 24, 3);
}
