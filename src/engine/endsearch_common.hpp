/*
    Egaroucid Project

    @file endsearch_common.hpp
        Common things for endgame search
    @date 2021-2024
    @author Takuto Yamana
    @author Toshihiko Okuhara
    @license GPL-3.0 license
*/

#pragma once
#include <iostream>
#include <vector>
#include <functional>
#include "setting.hpp"
#include "common.hpp"
#include "board.hpp"
#include "search.hpp"
#if USE_SIMD
    #ifdef _MSC_VER
        #include <intrin.h>
    #else
        #include <x86intrin.h>
    #endif
#endif

#if USE_END_PO
    const uint8_t parity_case[64] = {        /* p3p2p1p0 = */
        /*0000*/  0, /*0001*/  0, /*0010*/  1, /*0011*/  9, /*0100*/  2, /*0101*/ 10, /*0110*/ 11, /*0111*/  3,
        /*0002*/  0, /*0003*/  0, /*0012*/  0, /*0013*/  0, /*0102*/  4, /*0103*/  4, /*0112*/  5, /*0113*/  5,
        /*0020*/  1, /*0021*/  0, /*0030*/  1, /*0031*/  0, /*0120*/  6, /*0121*/  7, /*0130*/  6, /*0131*/  7,
        /*0022*/  9, /*0023*/  0, /*0032*/  0, /*0033*/  9, /*0122*/  8, /*0123*/  0, /*0132*/  0, /*0133*/  8,
        /*0200*/  2, /*0201*/  4, /*0210*/  6, /*0211*/  8, /*0300*/  2, /*0301*/  4, /*0310*/  6, /*0311*/  8,
        /*0202*/ 10, /*0203*/  4, /*0212*/  7, /*0213*/  0, /*0302*/  4, /*0303*/ 10, /*0312*/  0, /*0313*/  7,
        /*0220*/ 11, /*0221*/  5, /*0230*/  6, /*0231*/  0, /*0320*/  6, /*0321*/  0, /*0330*/ 11, /*0331*/  5,
        /*0222*/  3, /*0223*/  5, /*0232*/  7, /*0233*/  8, /*0322*/  8, /*0323*/  7, /*0332*/  5, /*0333*/  3
    };

    #if USE_SIMD
        union V4SI {
            unsigned int ui[4];
            __m128i v4;
        };
        const V4SI parity_ordering_shuffle_mask_last4[] = {      	// B15:4th, B11:3rd, B7:2nd, B3:1st, lower 3 bytes for 3 empties
            {{ 0x00010203, 0x01000203, 0x02000103, 0x03000102 }},   //  0: 1(p0) 3(p1 p2 p3), 1(p0) 1(p1) 2(p2 p3), 1 1 1 1, 4
            {{ 0x00010203, 0x01000203, 0x02010003, 0x03010002 }},   //  1: 1(p1) 3(p0 p2 p3)
            {{ 0x00020103, 0x01020003, 0x02000103, 0x03020100 }},   //  2: 1(p2) 3(p0 p1 p3)
            {{ 0x00030201, 0x01030200, 0x02030100, 0x03000102 }},   //  3: 1(p3) 3(p0 p1 p2)
            {{ 0x00020103, 0x02000103, 0x01000203, 0x03000201 }},   //  4: 1(p0) 1(p2) 2(p1 p3)     p1<->p2
            {{ 0x00030201, 0x03000201, 0x02000301, 0x01000302 }},   //  5: 1(p0) 1(p3) 2(p1 p2)     p1<->p3
            {{ 0x02010003, 0x01020003, 0x00020103, 0x03020100 }},   //  6: 1(p1) 1(p2) 2(p0 p3)     p0<->p2
            {{ 0x03010200, 0x01030200, 0x02030100, 0x00030102 }},   //  7: 1(p1) 1(p3) 2(p0 p2)     p0<->p3
            {{ 0x02030001, 0x03020001, 0x00020301, 0x01020300 }},   //  8: 1(p2) 1(p3) 2(p0 p1)     p0<->p2, p1<->p3
            {{ 0x00010203, 0x01000203, 0x02030100, 0x03020100 }},   //  9: 2(p0 p1) 2(p2 p3)
            {{ 0x00020103, 0x01030200, 0x02000103, 0x03010002 }},   // 10: 2(p0 p2) 2(p1 p3)
            {{ 0x00030201, 0x01020003, 0x02010003, 0x03000102 }}    // 11: 2(p0 p3) 2(p1 p2)
        };

    #else
        const uint16_t parity_ordering_last3[] = {
            0x0000, //  0: 1(p0) 3(p1 p2 p3), 1(p0) 1(p1) 2(p2 p3), 1 1 1 1, 4
            0x1100, //  1: 1(p1) 3(p0 p2 p3)        p0p1p2p3-p1p0p2p3-p2p1p0p3-p3p1p0p2
            0x2011, //  2: 1(p2) 3(p0 p1 p3)        p0p2p1p3-p1p2p0p3-p2p0p1p3-p3p2p1p0
            0x0222, //  3: 1(p3) 3(p0 p1 p2)        p0p3p2p1-p1p3p2p0-p2p3p1p0-p3p0p1p2
            0x0000, //  4: 1(p0) 1(p2) 2(p1 p3)     p1<->p2
            0x0000, //  5: 1(p0) 1(p3) 2(p1 p2)     p1<->p3
            0x0000, //  6: 1(p1) 1(p2) 2(p0 p3)     p0<->p2
            0x0000, //  7: 1(p1) 1(p3) 2(p0 p2)     p0<->p3
            0x0000, //  8: 1(p2) 1(p3) 2(p0 p1)     p0<->p2, p1<->p3
            0x2200, //  9: 2(p0 p1) 2(p2 p3)        p0p1p2p3-p1p0p2p3-p2p3p1p0-p3p2p1p0
            0x1021, // 10: 2(p0 p2) 2(p1 p3)        p0p2p1p3-p1p3p2p0-p2p0p1p3-p3p1p0p2
            0x0112  // 11: 2(p0 p3) 2(p1 p2)        p0p3p2p1-p1p2p0p3-p2p1p0p3-p3p0p1p2
        };
    #endif
#endif

/*
    @brief Get a final score from bitboard with last 1 empty

    Special optimization from an idea of https://github.com/abulmo/edax-reversi/blob/1ae7c9fe5322ac01975f1b3196e788b0d25c1e10/src/endgame.c#L85 ,
    which has been adapted from Zebra by Gunnar Anderson.

    @param search               search information (board ignored)
    @param player               player bitboard
    @param alpha                alpha value
    @param p0                   last empty square
    @return the final score
*/
inline int last1(Search *search, uint64_t player, int alpha, uint_fast8_t p0){
    ++search->n_nodes;
    #if USE_SEARCH_STATISTICS
        ++search->n_nodes_discs[63];
    #endif
    int n_flip = count_last_flip(player, p0);
    int score = 2 * (pop_count_ull(player) + n_flip + 1) - HW2;	// (P + n_flip + 1) - (HW2 - 1 - P - n_flip)
    if (n_flip == 0) {
        ++search->n_nodes;
        #if USE_SEARCH_STATISTICS
            ++search->n_nodes_discs[63];
        #endif
        int score2 = score - 2;	// empty for opponent
        if (score <= 0)
            score = score2;
        if (score > alpha) {
            n_flip = count_last_flip(~player, p0);
            if (n_flip)
                score = score2 - 2 * n_flip;
        }
    }
    return score;
}

#if USE_SIMD
    // vector otpimized version imported from Edax AVX, (C) 1998 - 2018 Richard Delorme, 2014 - 23 Toshihiko Okuhara

    #if defined(_MSC_VER) || defined(__clang__)
        #define	vectorcall	__vectorcall
    #else
        #define	vectorcall
    #endif

    #define	SWAP64	0x4e	// for _mm_shuffle_epi32
    #define	DUPHI	0xee

    /** coordinate to bit table converter */
    static uint64_t X_TO_BIT[64];
    /** last1 simd mask */
    union V4DI {
        uint64_t u64[4];
        __m256i v4;
        __m128i v2[2];
    };
    static V4DI mask_dvhd[64];

    static inline int vectorcall TESTZ_FLIP(__m128i X) { return _mm_testz_si128(X, X); }

    /*
        @brief evaluation function for game over

        @param b                    board.player
        @param e                    number of empty squares
        @return final score
    */
    static inline int end_evaluate(uint64_t b, int e) {
        int score = pop_count_ull(b) * 2 - HW2;	// in case of opponents win
        int diff = score + e;		// = n_discs_p - (64 - e - n_discs_p)

        if (diff == 0)
            score = diff;
        else if (diff > 0)
            score = diff + e;
        return score;
    }

    /**
     * @brief Compute a board resulting of a move played on a previous board.
     *
     * @param OP board to play the move on.
     * @param x move to play.
     * @param flipped flipped returned from mm_Flip.
     * @return resulting board.
     */
    static inline __m128i vectorcall board_flip_next(__m128i OP, int x, __m128i flipped)
    {
        OP = _mm_xor_si128(OP, _mm_or_si128(flipped, _mm_loadl_epi64((__m128i*) & X_TO_BIT[x])));
        return _mm_shuffle_epi32(OP, SWAP64);
    }
#endif

void endsearch_init(){
    #if USE_SIMD
        for (int i = 0; i < 64; ++i) {
              // X_TO_BIT
            X_TO_BIT[i] = 1ULL << i;
              // mask_dvhd
            int x = i & 7;
            int y = i >> 3;
            int s = (7 - y) - x;
            uint64_t d = (s >= 0) ? 0x0102040810204080 >> (s * 8) : 0x0102040810204080 << (-s * 8);
            mask_dvhd[i].u64[0] = d;
            mask_dvhd[i].u64[1] = 0xffULL << (i & 0x38);
            s = y - x;
            d = (s >= 0) ? 0x8040201008040201 << (s * 8) : 0x8040201008040201 >> (-s * 8);
            mask_dvhd[i].u64[2] = 0x0101010101010101 << (i & 7);
            mask_dvhd[i].u64[3] = d;
        }
    #endif
}