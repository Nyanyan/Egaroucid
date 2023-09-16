/*
    Egaroucid Project

    @file endsearch_simd.hpp
        Search near endgame, imported from Edax AVX
    @date 2021-2023
    @author Takuto Yamana
    @author Toshihiko Okuhara
    @license GPL-3.0 license
*/
#pragma once
#include "setting.hpp"
#include "search.hpp"

/*
    @brief Get a final score with last 2 empties

    No move ordering. Just search it.

    @param search               search information (board ignored)
    @param OP                   vectored board
    @param alpha                alpha value
    @param beta                 beta value
    @param empties_simd         vectored empties
    @return the final score
*/
static int vectorcall last2(Search* search, __m128i OP, int alpha, int beta, __m128i empties_simd) {
    __m128i flipped;
    int nodes, v;
    int p0 = _mm_extract_epi16(empties_simd, 1);
    int p1 = _mm_extract_epi16(empties_simd, 0);
    uint64_t opponent = _mm_extract_epi64(OP, 1);;

    ++search->n_nodes;
    #if USE_SEARCH_STATISTICS
        ++search->n_nodes_discs[search->n_discs];
    #endif
    if ((bit_around[p0] & opponent) && !TESTZ_FLIP(flipped = Flip::calc_flip(OP, p0))) {
        v = last1n(search, _mm_xor_si128(OP, flipped), beta, p1);
 
        if ((v < beta) && (bit_around[p1] & opponent) && !TESTZ_FLIP(flipped = Flip::calc_flip(OP, p1))) {
            int g = last1n(search, _mm_xor_si128(OP, flipped), beta, p0);
            if (v < g)
                v = g;
        }
    }

    else if ((bit_around[p1] & opponent) && !TESTZ_FLIP(flipped = Flip::calc_flip(OP, p1)))
        v = last1n(search, _mm_xor_si128(OP, flipped), beta, p0);
 
    else {	// pass
        beta = -alpha;
        __m128i PO = _mm_shuffle_epi32(OP, SWAP64);
        if (!TESTZ_FLIP(flipped = Flip::calc_flip(PO, p0))) {
            v = last1n(search, _mm_xor_si128(PO, flipped), beta, p1);
 
           if ((v < beta) && !TESTZ_FLIP(flipped = Flip::calc_flip(PO, p1))) {
                int g = last1n(search, _mm_xor_si128(PO, flipped), beta, p0);
                if (v > g)
                    v = g;
            }
        }

        else if (!TESTZ_FLIP(flipped = Flip::calc_flip(PO, p1)))
            v = last1n(search, _mm_xor_si128(PO, flipped), beta, p0);

        else	// gameover
            v = end_evaluate(opponent, 2);

        v = -v;
    }
    return v;
}

/*
    @brief Get a final score with last 3 empties

    Only with parity-based ordering.

    @param search               search information (board ignored)
    @param OP                   vectored board
    @param alpha                alpha value
    @param beta                 beta value
    @param empties_simd         vectored empties
    @return the final score
*/
static int vectorcall last3(Search* search, __m128i OP, int alpha, int beta, __m128i empties_simd) {
    __m128i flipped;

    // if (!global_searching || !(*searching))
    //  return SCORE_UNDEFINED;
#if USE_SEARCH_STATISTICS
        ++search->n_nodes_discs[search->n_discs];
#endif
    int pol = -1;
    do {
        ++search->n_nodes;
        int t = alpha;  alpha = -beta;  beta = -t;
        int v = SCORE_INF;	// Negative score
        uint64_t opponent = _mm_extract_epi64(OP, 1);
        int x = _mm_extract_epi16(empties_simd, 2);
        if ((bit_around[x] & opponent) && !TESTZ_FLIP(flipped = Flip::calc_flip(OP, x))) {
            v = last2(search, board_flip_next(OP, x, flipped), alpha, beta, empties_simd);
            if (alpha >= v)
                return v * pol;
            if (beta > v)
                beta = v;
        }

        int g;
        x = _mm_extract_epi16(empties_simd, 1);
        if ((bit_around[x] & opponent) && !TESTZ_FLIP(flipped = Flip::calc_flip(OP, x))) {
            g = last2(search, board_flip_next(OP, x, flipped), alpha, beta, _mm_shufflelo_epi16(empties_simd, 0xd8));
            if (alpha >= g)
                return g * pol;
            if (v > g)
                v = g;
            if (beta > g)
                beta = g;
        }

        x = _mm_extract_epi16(empties_simd, 0);
        if ((bit_around[x] & opponent) && !TESTZ_FLIP(flipped = Flip::calc_flip(OP, x))) {
            g = last2(search, board_flip_next(OP, x, flipped), alpha, beta, _mm_shufflelo_epi16(empties_simd, 0xc9));
            if (v > g)
                v = g;
            return v * pol;
        }

        if (v < SCORE_INF)
            return v * pol;

        OP = _mm_shuffle_epi32(OP, SWAP64);	// pass
    } while ((pol = -pol) >= 0);

    return end_evaluate(_mm_cvtsi128_si64(OP), 3);	// gameover	// = end_evaluate(opponent, 3)
}

/*
    @brief Get a final score with last 4 empties

    Only with parity-based ordering.

    @param search               search information
    @param alpha                alpha value
    @param beta                 beta value
    @param skipped              already passed?
    @return the final score

    This board contains only 4 empty squares, so empty squares on each part will be:
        4 - 0 - 0 - 0
        3 - 1 - 0 - 0
        2 - 2 - 0 - 0
        2 - 1 - 1 - 0 > need to sort
        1 - 1 - 1 - 1
    then the parities for squares will be:
        0 - 0 - 0 - 0
        1 - 1 - 0 - 0
        0 - 0 - 0 - 0
        1 - 1 - 0 - 0 > need to sort
        1 - 1 - 1 - 1
*/
int last4(Search* search, int alpha, int beta) {
    __m128i flipped;
    uint64_t opponent;
#if USE_END_PO
    static constexpr uint8_t parity_case[64] = {        /* p3p2p1p0 = */
        /*0000*/  0, /*0001*/  0, /*0010*/  1, /*0011*/  9, /*0100*/  2, /*0101*/ 10, /*0110*/ 11, /*0111*/  3,
        /*0002*/  0, /*0003*/  0, /*0012*/  0, /*0013*/  0, /*0102*/  4, /*0103*/  4, /*0112*/  5, /*0113*/  5,
        /*0020*/  1, /*0021*/  0, /*0030*/  1, /*0031*/  0, /*0120*/  6, /*0121*/  7, /*0130*/  6, /*0131*/  7,
        /*0022*/  9, /*0023*/  0, /*0032*/  0, /*0033*/  9, /*0122*/  8, /*0123*/  0, /*0132*/  0, /*0133*/  8,
        /*0200*/  2, /*0201*/  4, /*0210*/  6, /*0211*/  8, /*0300*/  2, /*0301*/  4, /*0310*/  6, /*0311*/  8,
        /*0202*/ 10, /*0203*/  4, /*0212*/  7, /*0213*/  0, /*0302*/  4, /*0303*/ 10, /*0312*/  0, /*0313*/  7,
        /*0220*/ 11, /*0221*/  5, /*0230*/  6, /*0231*/  0, /*0320*/  6, /*0321*/  0, /*0330*/ 11, /*0331*/  5,
        /*0222*/  3, /*0223*/  5, /*0232*/  7, /*0233*/  8, /*0322*/  8, /*0323*/  7, /*0332*/  5, /*0333*/  3
    };
    union V4SI {
        unsigned int ui[4];
        __m128i v4;
    };
    static constexpr V4SI parity_ordering_shuffle_mask_last4[] = {      // make search order identical to endsearch.hpp
        {{ 0x03020100, 0x02030100, 0x01030200, 0x00030201 }},   //  0: 1(p0) 3(p1 p2 p3), 1(p0) 1(p1) 2(p2 p3), 1 1 1 1, 4
        {{ 0x03020100, 0x02030100, 0x01020300, 0x00020301 }},   //  1: 1(p1) 3(p0 p2 p3)
        {{ 0x03010200, 0x02010300, 0x01030200, 0x00010203 }},   //  2: 1(p2) 3(p0 p1 p3)
        {{ 0x03000102, 0x02000103, 0x01000203, 0x00030201 }},   //  3: 1(p3) 3(p0 p1 p2)
        {{ 0x03010200, 0x01030200, 0x02030100, 0x00030102 }},   //  4: 1(p0) 1(p2) 2(p1 p3)     p1<->p2
        {{ 0x03000102, 0x00030102, 0x01030002, 0x02030001 }},   //  5: 1(p0) 1(p3) 2(p1 p2)     p1<->p3
        {{ 0x01020300, 0x02010300, 0x03010200, 0x00010203 }},   //  6: 1(p1) 1(p2) 2(p0 p3)     p0<->p2
        {{ 0x00020103, 0x02000103, 0x01000203, 0x03000201 }},   //  7: 1(p1) 1(p3) 2(p0 p2)     p0<->p3
        {{ 0x01000302, 0x00010302, 0x03010002, 0x02010003 }},   //  8: 1(p2) 1(p3) 2(p0 p1)     p0<->p2, p1<->p3
        {{ 0x03020100, 0x02030100, 0x01000203, 0x00010203 }},   //  9: 2(p0 p1) 2(p2 p3)
        {{ 0x03010200, 0x02000103, 0x01030200, 0x00020301 }},   // 10: 2(p0 p2) 2(p1 p3)
        {{ 0x03000102, 0x02010300, 0x01020300, 0x00030201 }}    // 11: 2(p0 p3) 2(p1 p2)
    };
#endif
    __m128i OP = _mm_loadu_si128((__m128i*) & search->board);
    uint64_t empties = ~(search->board.player | search->board.opponent);
    uint_fast8_t p0 = first_bit(&empties);
    uint_fast8_t p1 = next_bit(&empties);
    uint_fast8_t p2 = next_bit(&empties);
    uint_fast8_t p3 = next_bit(&empties);

    // if (!global_searching || !(*searching))
    //  return SCORE_UNDEFINED;
    #if USE_SEARCH_STATISTICS
        ++search->n_nodes_discs[search->n_discs];
    #endif
    #if USE_LAST4_SC
        int stab_res = stability_cut(search, &alpha, &beta);
        if (stab_res != SCORE_UNDEFINED) {
            return stab_res;
        }
    #endif
    __m128i empties_simd = _mm_cvtsi32_si128((p0 << 24) | (p1 << 16) | (p2 << 8) | p3);
    #if USE_END_PO
                // parity ordering optimization
                // I referred to http://www.amy.hi-ho.ne.jp/okuhara/edaxopt.htm
        const int paritysort = parity_case[((p2 ^ p3) & 0x24) + ((((p1 ^ p3) & 0x24) * 2 + ((p0 ^ p3) & 0x24)) >> 2)];
        empties_simd = _mm_shuffle_epi8(empties_simd, parity_ordering_shuffle_mask_last4[paritysort].v4);
    #else
        empties_simd = _mm_shuffle_epi8(empties_simd, _mm_set_epi32(0x00030201, 0x01030200, 0x02030100, 0x03020100));
    #endif

    int pol = -1;
    do {
        ++search->n_nodes;
        int t = alpha;  alpha = -beta;  beta = -t;
        int v = SCORE_INF;	// Negative score
        opponent = _mm_extract_epi64(OP, 1);
        p0 = _mm_extract_epi8(empties_simd, 3);
        if ((bit_around[p0] & opponent) && !TESTZ_FLIP(flipped = Flip::calc_flip(OP, p0))) {
            v = last3(search, board_flip_next(OP, p0, flipped), alpha, beta, _mm_cvtepu8_epi16(empties_simd));
            if (alpha >= v)
                return v * pol;
            if (beta > v)
                beta = v;
        }

        int g;
        empties_simd = _mm_shuffle_epi32(empties_simd, ROTR32);
        p1 = _mm_extract_epi8(empties_simd, 3);
        if ((bit_around[p1] & opponent) && !TESTZ_FLIP(flipped = Flip::calc_flip(OP, p1))) {
            g = last3(search, board_flip_next(OP, p1, flipped), alpha, beta, _mm_cvtepu8_epi16(empties_simd));
            if (alpha >= g)
                return g * pol;
            if (v > g)
                v = g;
            if (beta > g)
                beta = g;
        }
 
        empties_simd = _mm_shuffle_epi32(empties_simd, ROTR32);
        p2 = _mm_extract_epi8(empties_simd, 3);
        if ((bit_around[p2] & opponent) && !TESTZ_FLIP(flipped = Flip::calc_flip(OP, p2))) {
            g = last3(search, board_flip_next(OP, p2, flipped), alpha, beta, _mm_cvtepu8_epi16(empties_simd));
            if (alpha >= g)
                return g * pol;
            if (v > g)
                v = g;
            if (beta > g)
                beta = g;
        }
 
        empties_simd = _mm_shuffle_epi32(empties_simd, ROTR32);
        p3 = _mm_extract_epi8(empties_simd, 3);
        if ((bit_around[p3] & opponent) && !TESTZ_FLIP(flipped = Flip::calc_flip(OP, p3))) {
            g = last3(search, board_flip_next(OP, p3, flipped), alpha, beta, _mm_cvtepu8_epi16(empties_simd));
            if (v > g)
                v = g;
            return v * pol;
        }

        if (v < SCORE_INF)
            return v * pol;

        OP = _mm_shuffle_epi32(OP, SWAP64);	// pass
        empties_simd = _mm_shuffle_epi32(empties_simd, ROTR32);
    } while ((pol = -pol) >= 0);

    return end_evaluate(opponent, 4);	// gameover
}
