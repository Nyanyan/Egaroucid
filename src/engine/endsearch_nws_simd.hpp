/*
    Egaroucid Project

    @file endsearch_nws_simd.hpp
        Null windows search near endgame
        imported from Edax AVX, (C) 1998 - 2018 Richard Delorme, 2014 - 23 Toshihiko Okuhara
    @date 2021-2023
    @author Takuto Yamana
    @author Toshihiko Okuhara
    @license GPL-3.0 license
*/
#pragma once
#include "setting.hpp"
#include "search.hpp"

/*
    @brief Get a final max score with last 2 empties (NWS)

    No move ordering. Just search it.

    @param search               search information (board ignored)
    @param OP                   vectored board
    @param alpha                alpha value (beta value is alpha + 1)
    @param empties_simd         vectored empties (2 Words)
    @return the final max score
*/
static int vectorcall last2_nws(Search *search, __m128i OP, int alpha, __m128i empties_simd) {
    __m128i flipped;
    int p0 = _mm_extract_epi16(empties_simd, 1);
    int p1 = _mm_extract_epi16(empties_simd, 0);
    uint64_t opponent = _mm_extract_epi64(OP, 1);

    ++search->n_nodes;
    #if USE_SEARCH_STATISTICS
        ++search->n_nodes_discs[search->n_discs];
    #endif
    int beta = alpha + 1;
    int v;
    if ((bit_around[p0] & opponent) && !TESTZ_FLIP(flipped = Flip::calc_flip(OP, p0))) {
        v = last1(search, _mm_xor_si128(OP, flipped), beta, p1);
 
        if ((v < beta) && (bit_around[p1] & opponent) && !TESTZ_FLIP(flipped = Flip::calc_flip(OP, p1))) {
            int g = last1(search, _mm_xor_si128(OP, flipped), beta, p0);
            if (v < g)
                v = g;
        }
    }

    else if ((bit_around[p1] & opponent) && !TESTZ_FLIP(flipped = Flip::calc_flip(OP, p1)))
        v = last1(search, _mm_xor_si128(OP, flipped), beta, p0);
 
    else {	// pass
        ++search->n_nodes;
        beta = -(beta - 1);	// -alpha
        __m128i PO = _mm_shuffle_epi32(OP, SWAP64);
        if (!TESTZ_FLIP(flipped = Flip::calc_flip(PO, p0))) {
            v = last1(search, _mm_xor_si128(PO, flipped), beta, p1);

            if ((v < beta) && !TESTZ_FLIP(flipped = Flip::calc_flip(PO, p1))) {
                int g = last1(search, _mm_xor_si128(PO, flipped), beta, p0);
                if (v < g)
                    v = g;
            }
        }

        else if (!TESTZ_FLIP(flipped = Flip::calc_flip(PO, p1)))
            v = last1(search, _mm_xor_si128(PO, flipped), beta, p0);

        else	// gameover
            v = end_evaluate(opponent, 2);

        v = -v;
    }
    return v;
}

/*
    @brief Get a final min score with last 3 empties (NWS)

    Only with parity-based ordering.

    @param search               search information (board ignored)
    @param OP                   vectored board
    @param alpha                alpha value (beta value is alpha + 1)
    @param empties_simd         vectored empties (3 Bytes)
    @return the final min score
*/
static int vectorcall last3_nws(Search *search, __m128i OP, int alpha, __m128i empties_simd) {
    __m128i flipped;

    // if (!global_searching || !(*searching))
    //  return SCORE_UNDEFINED;
    #if USE_SEARCH_STATISTICS
        ++search->n_nodes_discs[search->n_discs];
    #endif
    empties_simd = _mm_cvtepu8_epi16(empties_simd);
    int v = SCORE_INF;	// min stage
    int pol = 1;
    do {
        ++search->n_nodes;
        uint64_t opponent = _mm_extract_epi64(OP, 1);
        int x = _mm_extract_epi16(empties_simd, 2);
        if ((bit_around[x] & opponent) && !TESTZ_FLIP(flipped = Flip::calc_flip(OP, x))) {
            v = last2_nws(search, board_flip_next(OP, x, flipped), alpha, empties_simd);
            if (alpha >= v)
                return v * pol;
        }

        int g;
        x = _mm_extract_epi16(empties_simd, 1);
        if ((bit_around[x] & opponent) && !TESTZ_FLIP(flipped = Flip::calc_flip(OP, x))) {
            g = last2_nws(search, board_flip_next(OP, x, flipped), alpha, _mm_shufflelo_epi16(empties_simd, 0xd8));	// (d3d1)d2d0
            if (alpha >= g)
                return g * pol;
            if (v > g)
                v = g;
        }

        x = _mm_extract_epi16(empties_simd, 0);
        if ((bit_around[x] & opponent) && !TESTZ_FLIP(flipped = Flip::calc_flip(OP, x))) {
            g = last2_nws(search, board_flip_next(OP, x, flipped), alpha, _mm_shufflelo_epi16(empties_simd, 0xc9));	// (d3d0)d2d1
            if (v > g)
                v = g;
            return v * pol;
        }

        if (v < SCORE_INF)
            return v * pol;

        OP = _mm_shuffle_epi32(OP, SWAP64);	// pass
        alpha = -alpha - 1;
    } while ((pol = -pol) < 0);

    return end_evaluate(_mm_extract_epi64(OP, 1), 3);	// gameover
}

/*
    @brief Get a final max score with last 4 empties (NWS)

    Only with parity-based ordering.

    @param search               search information
    @param alpha                alpha value (beta value is alpha + 1)
    @return the final max score

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
int last4_nws(Search *search, int alpha) {
    __m128i flipped;
    uint64_t opponent;
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
        int stab_res = stability_cut_nws(search, alpha);
        if (stab_res != SCORE_UNDEFINED) {
            return stab_res;
        }
    #endif
    __m128i empties_simd = _mm_cvtsi32_si128((p3 << 24) | (p2 << 16) | (p1 << 8) | p0);
    #if USE_END_PO
                // parity ordering optimization
                // I referred to http://www.amy.hi-ho.ne.jp/okuhara/edaxopt.htm
        const int paritysort = parity_case[((p2 ^ p3) & 0x24) + ((((p1 ^ p3) & 0x24) * 2 + ((p0 ^ p3) & 0x24)) >> 2)];
        empties_simd = _mm_shuffle_epi8(empties_simd, parity_ordering_shuffle_mask_last4[paritysort].v4);
    #else
        empties_simd = _mm_shuffle_epi8(empties_simd, _mm_set_epi32(0x03000102, 0x02000103, 0x01000203, 0x00010203));
    #endif

    int v = -SCORE_INF;
    int pol = 1;
    do {
        ++search->n_nodes;
        opponent = _mm_extract_epi64(OP, 1);
        p0 = _mm_extract_epi8(empties_simd, 3);
        if ((bit_around[p0] & opponent) && !TESTZ_FLIP(flipped = Flip::calc_flip(OP, p0))) {
            v = last3_nws(search, board_flip_next(OP, p0, flipped), alpha, empties_simd);
            if (alpha < v)
                return v * pol;
        }

        int g;
        p1 = _mm_extract_epi8(empties_simd, 7);
        if ((bit_around[p1] & opponent) && !TESTZ_FLIP(flipped = Flip::calc_flip(OP, p1))) {
            g = last3_nws(search, board_flip_next(OP, p1, flipped), alpha, _mm_srli_si128(empties_simd, 4));
            if (alpha < g)
                return g * pol;
            if (v < g)
                v = g;
        }
 
        p2 = _mm_extract_epi8(empties_simd, 11);
        if ((bit_around[p2] & opponent) && !TESTZ_FLIP(flipped = Flip::calc_flip(OP, p2))) {
            g = last3_nws(search, board_flip_next(OP, p2, flipped), alpha, _mm_srli_si128(empties_simd, 8));
            if (alpha < g)
                return g * pol;
            if (v < g)
                v = g;
        }
 
        p3 = _mm_extract_epi8(empties_simd, 15);
        if ((bit_around[p3] & opponent) && !TESTZ_FLIP(flipped = Flip::calc_flip(OP, p3))) {
            g = last3_nws(search, board_flip_next(OP, p3, flipped), alpha, _mm_srli_si128(empties_simd, 12));
            if (v < g)
                v = g;
            return v * pol;
        }

        if (v > -SCORE_INF)
            return v * pol;

        OP = _mm_shuffle_epi32(OP, SWAP64);	// pass
        alpha = -alpha - 1;
    } while ((pol = -pol) < 0);

    return end_evaluate(opponent, 4);	// gameover (opponent is P here)
}
