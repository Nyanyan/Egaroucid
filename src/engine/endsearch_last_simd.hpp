/*
    Egaroucid Project

    @file endsearch_last_simd.hpp
        Last N Moves Optimization
        imported from Edax AVX, (C) 1998 - 2018 Richard Delorme, 2014 - 23 Toshihiko Okuhara
    @date 2021-2026
    @author Takuto Yamana
    @author Toshihiko Okuhara
    @license GPL-3.0-or-later
*/
#pragma once
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#include "setting.hpp"
#include "search.hpp"
#include "endsearch_common.hpp"

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
#if LAST_FLIP_PASS_OPT
    uint_fast16_t n_flip_both = N_LAST_FLIP_BOTH[_mm_extract_epi16(II, 4)][x];
    n_flip_both += N_LAST_FLIP[_mm_cvtsi128_si32(II)][x];
    int t = _mm_movemask_epi8(_mm_sub_epi8(_mm_setzero_si128(), _mm_and_si128(PP, M1)));
    n_flip_both += N_LAST_FLIP[t >> 8][y];
    n_flip_both += N_LAST_FLIP_BOTH[t & 0xFF][y];
    uint_fast8_t n_flip = n_flip_both & 0xff;
#else
    uint_fast8_t n_flip = N_LAST_FLIP[_mm_extract_epi16(II, 4)][x];
    n_flip += N_LAST_FLIP[_mm_cvtsi128_si32(II)][x];
    int t = _mm_movemask_epi8(_mm_sub_epi8(_mm_setzero_si128(), _mm_and_si128(PP, M1)));
    n_flip += N_LAST_FLIP[t >> 8][y];
    n_flip += N_LAST_FLIP[t & 0xFF][y];
#endif
    int score = 2 * (pop_count_ull(_mm_cvtsi128_si64(PP)) + n_flip + 1) - HW2;	// (n_P + n_flip + 1) - (HW2 - 1 - n_P - n_flip)

    if (n_flip == 0) {
        ++search->n_nodes;
        #if USE_SEARCH_STATISTICS
            ++search->n_nodes_discs[63];
        #endif
        int score2 = score - 2;	// empty for player
        if (score <= 0) {
            score = score2;
        }
        if (score > alpha) {	// lazy cut-off
            II = _mm_sad_epu8(_mm_andnot_si128(PP, M0), _mm_setzero_si128());
#if LAST_FLIP_PASS_OPT
            n_flip = n_flip_both >> 8;
            n_flip += N_LAST_FLIP[_mm_cvtsi128_si32(II)][x];
            t = _mm_movemask_epi8(_mm_sub_epi8(_mm_setzero_si128(), _mm_andnot_si128(PP, M1)));
            n_flip += N_LAST_FLIP[t >> 8][y];
#else
            n_flip = N_LAST_FLIP[_mm_extract_epi16(II, 4)][x];
            n_flip += N_LAST_FLIP[_mm_cvtsi128_si32(II)][x];
            t = _mm_movemask_epi8(_mm_sub_epi8(_mm_setzero_si128(), _mm_andnot_si128(PP, M1)));
            n_flip += N_LAST_FLIP[t >> 8][y];
            n_flip += N_LAST_FLIP[t & 0xFF][y];
#endif
            if (n_flip != 0) {
                score = score2 - 2 * n_flip;
            }
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
static int vectorcall last2(Search *search, __m128i OP, int alpha, int beta, uint32_t p0, uint32_t p1) {
    __m128i flipped;
    uint64_t opponent = _mm_extract_epi64(OP, 1);

    ++search->n_nodes;
#if USE_SEARCH_STATISTICS
    ++search->n_nodes_discs[62];
#endif
    int v;
    if (bit_around[p0] & opponent) { // p0
        if (!TESTZ_FLIP(flipped = Flip::calc_flip(OP, p0))) {
            v = last1(search, _mm_xor_si128(OP, flipped), alpha, p1);
    
            if ((v > alpha) && (bit_around[p1] & opponent) && !TESTZ_FLIP(flipped = Flip::calc_flip(OP, p1))) { // p1
                int g = last1(search, _mm_xor_si128(OP, flipped), alpha, p0);
                if (v > g) {
                    v = g;
                }
            }
            return v;
        }
    }
    if (bit_around[p1] & opponent) { // p1 only
        if (!TESTZ_FLIP(flipped = Flip::calc_flip(OP, p1))) {
            v = last1(search, _mm_xor_si128(OP, flipped), alpha, p0);
            return v;
        }
    }

    ++search->n_nodes;
#if USE_SEARCH_STATISTICS
    ++search->n_nodes_discs[62];
#endif
    alpha = -beta;
    __m128i PO = _mm_shuffle_epi32(OP, SWAP64);
    if (!TESTZ_FLIP(flipped = Flip::calc_flip(PO, p0))) { // p0
        v = last1(search, _mm_xor_si128(PO, flipped), alpha, p1);

        if ((v > alpha) && !TESTZ_FLIP(flipped = Flip::calc_flip(PO, p1))) { // p1
            int g = last1(search, _mm_xor_si128(PO, flipped), alpha, p0);
            if (v > g) {
                v = g;
            }
        }
    } else if (!TESTZ_FLIP(flipped = Flip::calc_flip(PO, p1))) { // p1 only
        v = last1(search, _mm_xor_si128(PO, flipped), alpha, p0);
    } else {	// gameover
        v = end_evaluate(_mm_extract_epi64(PO, 1), 2);
    }
    v = -v;
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
static int vectorcall last3(Search *search, __m128i OP, int alpha, int beta, uint32_t p0, uint32_t p1, uint32_t p2) {
    __m128i flipped;
    uint64_t opponent;

    uint64_t empties = ~(search->board.player | search->board.opponent);
    if (is_1empty(p2, empties)) {
        std::swap(p2, p0);
    } else if (is_1empty(p1, empties)) {
        std::swap(p1, p0);
    }

    int v = -SCORE_INF;
    int pol = 1;
    do {
        ++search->n_nodes;
#if USE_SEARCH_STATISTICS
        ++search->n_nodes_discs[61];
#endif
        opponent = _mm_extract_epi64(OP, 1);
        int x = p0;
        if (bit_around[x] & opponent) {
            if (!TESTZ_FLIP(flipped = Flip::calc_flip(OP, x))) {
                v = last2(search, board_flip_next(OP, x, flipped), alpha, beta, p1, p2);
                if (beta <= v) {
                    return v * pol;
                }
                if (alpha < v) {
                    alpha = v;
                }
            }
        }

        int g;
        x = p1;
        if (bit_around[x] & opponent) {
            if (!TESTZ_FLIP(flipped = Flip::calc_flip(OP, x))) {
                g = last2(search, board_flip_next(OP, x, flipped), alpha, beta, p2, p0);	// (d3d1)d2d0
                if (beta <= g) {
                    return g * pol;
                }
                if (v < g) {
                    v = g;
                }
                if (alpha < g) {
                    alpha = g;
                }
            }
        }

        x = p2;
        if (bit_around[x] & opponent) {
            if (!TESTZ_FLIP(flipped = Flip::calc_flip(OP, x))) {
                g = last2(search, board_flip_next(OP, x, flipped), alpha, beta, p1, p0);	// (d3d0)d2d1
                if (v < g) {
                    v = g;
                }
                return v * pol;
            }
        }

        if (v > -SCORE_INF) {
            return v * pol;
        }

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
    __m128i flipped;
    uint64_t opponent;
    __m128i OP = _mm_loadu_si128((__m128i*) & search->board);
    uint64_t empties = ~(search->board.player | search->board.opponent);
    uint_fast8_t p0, p1, p2, p3;

#if USE_LAST4_SC
    int stab_res = stability_cut_last4(search, &alpha, beta);
    if (stab_res != SCORE_UNDEFINED) {
        return stab_res;
    }
#endif
    #if USE_END_PO
        uint64_t e1 = empty1_bb(search->board.player, search->board.opponent);
        if (!e1) {
            e1 = empties;
        }
        empties &= ~e1;
    
        p0 = first_bit(&e1);
        if (!(e1 &= e1 - 1)) {
            e1 = empties;
        }
        p1 = first_bit(&e1);
        if (!(e1 &= e1 - 1)) {
            e1 = empties;
        }
        p2 = first_bit(&e1);
        if ((e1 &= e1 - 1)) {
            p3 = first_bit(&e1);
        } else {
            p3 = first_bit(&empties);
        }
    #else
        p0 = first_bit(&empties);
        p1 = next_bit(&empties);
        p2 = next_bit(&empties);
        p3 = next_bit(&empties);
    #endif

    int v = SCORE_INF;	// min stage
    int pol = 1;
    do {
        ++search->n_nodes;
#if USE_SEARCH_STATISTICS
        ++search->n_nodes_discs[60];
#endif
        opponent = _mm_extract_epi64(OP, 1);
        if (bit_around[p0] & opponent) {
            if (!TESTZ_FLIP(flipped = Flip::calc_flip(OP, p0))) {
                v = last3(search, board_flip_next(OP, p0, flipped), alpha, beta, p1, p2, p3);
                if (alpha >= v) {
                    return v * pol;
                }
                if (beta > v) {
                    beta = v;
                }
            }
        }

        int g;
        if (bit_around[p1] & opponent) {
            if (!TESTZ_FLIP(flipped = Flip::calc_flip(OP, p1))) {
                g = last3(search, board_flip_next(OP, p1, flipped), alpha, beta, p0, p2, p3);
                if (alpha >= g) {
                    return g * pol;
                }
                if (v > g) {
                    v = g;
                }
                if (beta > g) {
                    beta = g;
                }
            }
        }
 
        if (bit_around[p2] & opponent) {
            if (!TESTZ_FLIP(flipped = Flip::calc_flip(OP, p2))) {
                g = last3(search, board_flip_next(OP, p2, flipped), alpha, beta, p0, p1, p3);
                if (alpha >= g) {
                    return g * pol;
                }
                if (v > g) {
                    v = g;
                }
                if (beta > g) {
                    beta = g;
                }
            }
        }
 
        if (bit_around[p3] & opponent) {
            if (!TESTZ_FLIP(flipped = Flip::calc_flip(OP, p3))) {
                g = last3(search, board_flip_next(OP, p3, flipped), alpha, beta, p0, p1, p2);
                if (v > g) {
                    v = g;
                }
                return v * pol;
            }
        }

        if (v < SCORE_INF) {
            return v * pol;
        }

        OP = _mm_shuffle_epi32(OP, SWAP64);	// pass
        int t = alpha;  alpha = -beta;  beta = -t;
    } while ((pol = -pol) < 0);

    return end_evaluate(_mm_extract_epi64(OP, 1), 4);	// gameover
}