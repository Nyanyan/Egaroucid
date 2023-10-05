/*
    Egaroucid Project

    @file endsearch_simd.hpp
        Search near endgame
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
    @brief Get a final score with last 1 empty

    @param search               search information (board ignored)
    @param PO                   vectored board (O ignored)
    @param alpha                alpha value
    @param place                last empty
    @return the final opponent's score
*/
static inline int vectorcall last1(Search *search, __m128i PO, int alpha, int place) {
    __m128i M0 = mask_dvhd[place].v2[0];
    __m128i M1 = mask_dvhd[place].v2[1];
    __m128i PP = _mm_shuffle_epi32(PO, DUPHI);
    __m128i II = _mm_sad_epu8(_mm_and_si128(PP, M0), _mm_setzero_si128());
    const int x = place & 7;
    const int y = place >> 3;

    ++search->n_nodes;
    #if USE_SEARCH_STATISTICS
        ++search->n_nodes_discs[63];
    #endif
    uint_fast8_t n_flip = n_flip_pre_calc[_mm_extract_epi16(II, 4)][x];
    n_flip += n_flip_pre_calc[_mm_cvtsi128_si32(II)][x];
    int t = _mm_movemask_epi8(_mm_sub_epi8(_mm_setzero_si128(), _mm_and_si128(PP, M1)));
    n_flip += n_flip_pre_calc[t >> 8][y];
    n_flip += n_flip_pre_calc[t & 0xFF][y];

    int score = 2 * (pop_count_ull(_mm_cvtsi128_si64(PP)) + n_flip + 1) - HW2;	// (n_P + n_flip + 1) - (HW2 - 1 - n_P - n_flip)

    if (n_flip == 0) {
        ++search->n_nodes;
        #if USE_SEARCH_STATISTICS
            ++search->n_nodes_discs[63];
        #endif
        int score2 = score - 2;	// empty for player
        if (score <= 0)
            score = score2;

        if (score > alpha) {	// lazy cut-off
            II = _mm_sad_epu8(_mm_andnot_si128(PP, M0), _mm_setzero_si128());
            n_flip = n_flip_pre_calc[_mm_extract_epi16(II, 4)][x];
            n_flip += n_flip_pre_calc[_mm_cvtsi128_si32(II)][x];
            t = _mm_movemask_epi8(_mm_sub_epi8(_mm_setzero_si128(), _mm_andnot_si128(PP, M1)));
            n_flip += n_flip_pre_calc[t >> 8][y];
            n_flip += n_flip_pre_calc[t & 0xFF][y];

            if (n_flip != 0)
                score = score2 - 2 * n_flip;
        }
    }

    return score;
}

/*
    @brief Get a final min score with last 2 empties

    No move ordering. Just search it.

    @param search               search information (board ignored)
    @param OP                   vectored board
    @param alpha                alpha value
    @param beta                 beta value
    @param empties_simd         vectored empties (2 Words)
    @return the final min score
*/
static int vectorcall last2(Search *search, __m128i OP, int alpha, int beta, __m128i empties_simd) {
    __m256i flipped;
    int p0 = _mm_extract_epi16(empties_simd, 1);
    int p1 = _mm_extract_epi16(empties_simd, 0);
    uint64_t opponent = _mm_extract_epi64(OP, 1);

    ++search->n_nodes;
    #if USE_SEARCH_STATISTICS
        ++search->n_nodes_discs[62];
    #endif
    int v;
    if ((bit_around[p0] & opponent) && !TESTZ_FLIP(flipped = Flip::calc_flip(OP, p0))) {
        v = last1(search, _mm_xor_si128(OP, Flip::reduce_vflip(flipped)), alpha, p1);
 
        if ((v > alpha) && (bit_around[p1] & opponent) && !TESTZ_FLIP(flipped = Flip::calc_flip(OP, p1))) {
            int g = last1(search, _mm_xor_si128(OP, Flip::reduce_vflip(flipped)), alpha, p0);
            if (v > g)
                v = g;
        }
    }

    else if ((bit_around[p1] & opponent) && !TESTZ_FLIP(flipped = Flip::calc_flip(OP, p1)))
        v = last1(search, _mm_xor_si128(OP, Flip::reduce_vflip(flipped)), alpha, p0);
 
    else {	// pass
        ++search->n_nodes;
        #if USE_SEARCH_STATISTICS
            ++search->n_nodes_discs[62];
        #endif
        alpha = -beta;
        __m128i PO = _mm_shuffle_epi32(OP, SWAP64);
        if (!TESTZ_FLIP(flipped = Flip::calc_flip(PO, p0))) {
            v = last1(search, _mm_xor_si128(PO, Flip::reduce_vflip(flipped)), alpha, p1);
 
           if ((v > alpha) && !TESTZ_FLIP(flipped = Flip::calc_flip(PO, p1))) {
                int g = last1(search, _mm_xor_si128(PO, Flip::reduce_vflip(flipped)), alpha, p0);
                if (v > g)
                    v = g;
            }
        }

        else if (!TESTZ_FLIP(flipped = Flip::calc_flip(PO, p1)))
            v = last1(search, _mm_xor_si128(PO, Flip::reduce_vflip(flipped)), alpha, p0);

        else	// gameover
            v = end_evaluate(_mm_extract_epi64(PO, 1), 2);

        v = -v;
    }
    return v;
}

/*
    @brief Get a final max score with last 3 empties

    Only with parity-based ordering.

    @param search               search information (board ignored)
    @param OP                   vectored board
    @param alpha                alpha value
    @param beta                 beta value
    @param empties_simd         vectored empties (3 Bytes)
    @return the final max score
*/
static int vectorcall last3(Search *search, __m128i OP, int alpha, int beta, __m128i empties_simd) {
    __m256i flipped;
    uint64_t opponent;

    // if (!global_searching || !(*searching))
    //  return SCORE_UNDEFINED;
    empties_simd = _mm_cvtepu8_epi16(empties_simd);
    int v = -SCORE_INF;
    int pol = 1;
    do {
        ++search->n_nodes;
        #if USE_SEARCH_STATISTICS
            ++search->n_nodes_discs[61];
        #endif
        opponent = _mm_extract_epi64(OP, 1);
        int x = _mm_extract_epi16(empties_simd, 2);
        if ((bit_around[x] & opponent) && !TESTZ_FLIP(flipped = Flip::calc_flip(OP, x))) {
            v = last2(search, board_flip_next(OP, x, flipped), alpha, beta, empties_simd);
            if (beta <= v)
                return v * pol;
            if (alpha < v)
                alpha = v;
        }

        int g;
        x = _mm_extract_epi16(empties_simd, 1);
        if ((bit_around[x] & opponent) && !TESTZ_FLIP(flipped = Flip::calc_flip(OP, x))) {
            g = last2(search, board_flip_next(OP, x, flipped), alpha, beta, _mm_shufflelo_epi16(empties_simd, 0xd8));	// (d3d1)d2d0
            if (beta <= g)
                return g * pol;
            if (v < g)
                v = g;
            if (alpha < g)
                alpha = g;
        }

        x = _mm_extract_epi16(empties_simd, 0);
        if ((bit_around[x] & opponent) && !TESTZ_FLIP(flipped = Flip::calc_flip(OP, x))) {
            g = last2(search, board_flip_next(OP, x, flipped), alpha, beta, _mm_shufflelo_epi16(empties_simd, 0xc9));	// (d3d0)d2d1
            if (v < g)
                v = g;
            return v * pol;
        }

        if (v > -SCORE_INF)
            return v * pol;

        OP = _mm_shuffle_epi32(OP, SWAP64);	// pass
        int t = alpha;  alpha = -beta;  beta = -t;
    } while ((pol = -pol) < 0);

    return end_evaluate(opponent, 3);	// gameover (opponent is P here)
}

/*
    @brief Get a final min score with last 4 empties

    Only with parity-based ordering.

    @param search               search information
    @param alpha                alpha value
    @param beta                 beta value
    @return the final min score

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
int last4(Search *search, int alpha, int beta) {
    __m256i flipped;
    uint64_t opponent;
    __m128i OP = _mm_loadu_si128((__m128i*) & search->board);
    uint64_t empties = ~(search->board.player | search->board.opponent);
    uint_fast8_t p0 = first_bit(&empties);
    uint_fast8_t p1 = next_bit(&empties);
    uint_fast8_t p2 = next_bit(&empties);
    uint_fast8_t p3 = next_bit(&empties);

    // if (!global_searching || !(*searching))
    //  return SCORE_UNDEFINED;
    #if USE_LAST4_SC
        int stab_res = stability_cut_last4(search, &alpha, beta);
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

    int v = SCORE_INF;	// min stage
    int pol = 1;
    do {
        ++search->n_nodes;
        #if USE_SEARCH_STATISTICS
            ++search->n_nodes_discs[60];
        #endif
        opponent = _mm_extract_epi64(OP, 1);
        p0 = _mm_extract_epi8(empties_simd, 3);
        if ((bit_around[p0] & opponent) && !TESTZ_FLIP(flipped = Flip::calc_flip(OP, p0))) {
            v = last3(search, board_flip_next(OP, p0, flipped), alpha, beta, empties_simd);
            if (alpha >= v)
                return v * pol;
            if (beta > v)
                beta = v;
        }

        int g;
        p1 = _mm_extract_epi8(empties_simd, 7);
        if ((bit_around[p1] & opponent) && !TESTZ_FLIP(flipped = Flip::calc_flip(OP, p1))) {
            g = last3(search, board_flip_next(OP, p1, flipped), alpha, beta, _mm_srli_si128(empties_simd, 4));
            if (alpha >= g)
                return g * pol;
            if (v > g)
                v = g;
            if (beta > g)
                beta = g;
        }
 
        p2 = _mm_extract_epi8(empties_simd, 11);
        if ((bit_around[p2] & opponent) && !TESTZ_FLIP(flipped = Flip::calc_flip(OP, p2))) {
            g = last3(search, board_flip_next(OP, p2, flipped), alpha, beta, _mm_srli_si128(empties_simd, 8));
            if (alpha >= g)
                return g * pol;
            if (v > g)
                v = g;
            if (beta > g)
                beta = g;
        }
 
        p3 = _mm_extract_epi8(empties_simd, 15);
        if ((bit_around[p3] & opponent) && !TESTZ_FLIP(flipped = Flip::calc_flip(OP, p3))) {
            g = last3(search, board_flip_next(OP, p3, flipped), alpha, beta, _mm_srli_si128(empties_simd, 12));
            if (v > g)
                v = g;
            return v * pol;
        }

        if (v < SCORE_INF)
            return v * pol;

        OP = _mm_shuffle_epi32(OP, SWAP64);	// pass
        int t = alpha;  alpha = -beta;  beta = -t;
    } while ((pol = -pol) < 0);

    return end_evaluate(_mm_extract_epi64(OP, 1), 4);	// gameover
}
