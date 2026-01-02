/*
    Egaroucid Project

    @file endsearch_nws_last_simd.hpp
        Last N Moves Optimization on Null Windows Search
        imported from Edax AVX, (C) 1998 - 2018 Richard Delorme, 2014 - 24 Toshihiko Okuhara
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
    @brief Get a final score with last 1 empty (NWS)

    lazy high cut-off idea was in Zebra by Gunnar Anderson (http://radagast.se/othello/zebra.html),
    but commented out because mobility check was no faster than counting flips.
    Now using AVX2, mobility check can be faster than counting flips.
    cf. http://www.amy.hi-ho.ne.jp/okuhara/edaxopt.htm#lazycutoff

    @param search               search information (board ignored)
    @param PO                   vectored board (O ignored)
    @param alpha                alpha value (beta value is alpha + 1)
    @param place                last empty
    @return the final opponent's score
*/
#if USE_AVX512

static inline int vectorcall last1_nws(Search *search, __m128i PO, int alpha, int place) {
    __m128i P2 = _mm_unpackhi_epi64(PO, PO);
    __m256i P4 = _mm256_broadcastq_epi64(P2);
    uint64_t P = _mm_cvtsi128_si64(P2);
    int score = 2 * pop_count_ull(P) - HW2 + 2;    // = (pop_count_ull(P) + 1) - (SCORE_MAX - 1 - pop_count_ull(P))
        // if player can move, final score > this score.
        // if player pass then opponent play, final score < score - 1 (cancel P) - 1 (last O).
        // if both pass, score - 1 (cancel P) - 1 (empty for O) <= final score <= score (empty for P).

    ++search->n_nodes;
#if USE_SEARCH_STATISTICS
    ++search->n_nodes_discs[63];
#endif
    __m256i lM = lrmask[place].v4[0];
    __m256i rM = lrmask[place].v4[1];
    __mmask16 lp = _mm256_test_epi64_mask(P4, lM);    // P exists on mask
    __mmask16 rp = _mm256_test_epi64_mask(P4, rM);
    __m256i F4, outflank, eraser, lmO, rmO;
    __m128i F2;
    int n_flip;

    if (score > alpha) {    // if player can move, high cut-off will occur regardress of n_flips.
        lmO = _mm256_maskz_andnot_epi64(lp, P4, lM);    // masked O, clear if all O
        rmO = _mm256_maskz_andnot_epi64(rp, P4, rM);    // (all O = all P = 0 flip)

        if (_mm256_testz_si256(_mm256_or_si256(lmO, rmO), _mm256_set1_epi64x(bit_around[place]))) {
            ++search->n_nodes;
#if USE_SEARCH_STATISTICS
            ++search->n_nodes_discs[63];
#endif
                // n_flip = last_flip(pos, ~P);
                // left: set below LS1B if O is in lM
            F4 = _mm256_maskz_add_epi64(_mm256_test_epi64_mask(lmO, lmO), lmO, _mm256_set1_epi64x(-1));
            F4 = _mm256_and_si256(_mm256_andnot_si256(lmO, F4), lM);

                // right: clear all bits lower than outflank
            eraser = _mm256_srlv_epi64(_mm256_set1_epi64x(-1),
                _mm256_maskz_lzcnt_epi64(_mm256_test_epi64_mask(rmO, rmO), rmO));
            F4 = _mm256_or_si256(F4, _mm256_andnot_si256(eraser, rM));

            F2 = _mm_or_si128(_mm256_castsi256_si128(F4), _mm256_extracti128_si256(F4, 1));
            n_flip = -pop_count_ull(_mm_cvtsi128_si64(_mm_or_si128(F2, _mm_unpackhi_epi64(F2, F2))));
                // last square for O if O can move or score <= 0
            score += (n_flip - (int)((n_flip | (score - 1)) < 0)) * 2;
        } else  score += 2;    // lazy high cut-off, return min flip

    } else {    // if player cannot move, low cut-off will occur whether opponent can move.
            // left: set below LS1B if P is in lM
        outflank = _mm256_and_si256(P4, lM);
        F4 = _mm256_maskz_add_epi64(lp, outflank, _mm256_set1_epi64x(-1));
        F4 = _mm256_and_si256(_mm256_andnot_si256(outflank, F4), lM);

            // right: clear all bits lower than outflank
        outflank = _mm256_and_si256(P4, rM);
        eraser = _mm256_srlv_epi64(_mm256_set1_epi64x(-1), _mm256_maskz_lzcnt_epi64(rp, outflank));
        F4 = _mm256_or_si256(F4, _mm256_andnot_si256(eraser, rM));

        F2 = _mm_or_si128(_mm256_castsi256_si128(F4), _mm256_extracti128_si256(F4, 1));
        n_flip = pop_count_ull(_mm_cvtsi128_si64(_mm_or_si128(F2, _mm_unpackhi_epi64(F2, F2))));
        score += n_flip * 2;
            // if n_flip == 0, score <= alpha so lazy low cut-off
    }
    return score;
}

#else // AVX2

static inline int vectorcall last1_nws(Search *search, __m128i PO, int alpha, int place) {
#if LAST_FLIP_PASS_OPT && false
    return last1(search, PO, alpha, place);
#else
    uint_fast16_t n_flip;
    uint32_t t;
    uint64_t P = _mm_extract_epi64(PO, 1);
    __m256i PP = _mm256_permute4x64_epi64(_mm256_castsi128_si256(PO), 0x55);
    int score = 2 * pop_count_ull(P) - HW2 + 2;    // = (pop_count_ull(P) + 1) - (HW2 - 1 - pop_count_ull(P))
        // if player can move, final score > score.
        // if player pass then opponent play, final score < score - 1 (cancel P) - 1 (last O).
        // if both pass, score - 1 (cancel P) - 1 (empty for O) <= final score <= score (empty for P).

    ++search->n_nodes;
#if USE_SEARCH_STATISTICS
    ++search->n_nodes_discs[63];
#endif
    if (score > alpha) {    // if player can move, high cut-off will occur regardress of n_flip.
        __m256i lM = lrmask[place].v4[0];
        __m256i rM = lrmask[place].v4[1];
        __m256i lmO = _mm256_andnot_si256(PP, lM);
        __m256i F = _mm256_andnot_si256(_mm256_cmpeq_epi64(lmO, lM), lmO); // clear if all O
        __m256i rmO = _mm256_andnot_si256(PP, rM);
        F = _mm256_or_si256(F, _mm256_andnot_si256(_mm256_cmpeq_epi64(rmO, rM), rmO));
        if (_mm256_testz_si256(F, _mm256_broadcastq_epi64(*(__m128i *) &bit_around[place]))) {    // pass
            ++search->n_nodes;
#if USE_SEARCH_STATISTICS
            ++search->n_nodes_discs[63];
#endif
                // n_flip = count_last_flip(~P, place);
            t = ~_mm256_movemask_epi8(_mm256_cmpeq_epi8(lmO, rmO));    // eq only if l = r = 0
            const int y = place >> 3;
            n_flip  = N_LAST_FLIP[(~P >> (y * 8)) & 0xFF][place & 7];    // h (both)
            n_flip += N_LAST_FLIP[(t >> 8) & 0xFF][y];    // v (both)
            n_flip += N_LAST_FLIP[(t >> 16) & 0xFF][y];    // d
            n_flip += N_LAST_FLIP[t >> 24][y];    // d
            score -= (n_flip + (int)((n_flip > 0) | (score <= 0))) * 2;
        } else { // player can move (need only min flip because already score > alpha)
            score += 2;    // min flip
        }
    } else {    // if player cannot move, low cut-off will occur whether opponent can move.
            // n_flip = count_last_flip(P, place);
        const int y = place >> 3;
        t = _mm256_movemask_epi8(_mm256_sub_epi8(_mm256_setzero_si256(), _mm256_and_si256(PP, mask_dvhd[place].v4)));
        n_flip  = N_LAST_FLIP[(P >> (y * 8)) & 0xFF][place & 7];    // h
        n_flip += N_LAST_FLIP[t & 0xFF][y];    // d
        n_flip += N_LAST_FLIP[(t >> 16) & 0xFF][y];    // v
        n_flip += N_LAST_FLIP[t >> 24][y];    // d
        score += n_flip * 2;
            // if n_flip == 0, score <= alpha so lazy low cut-off
    }
    return score;
#endif
}

#endif

/*
    @brief Get a final min score with last 2 empties (NWS)

    No move ordering. Just search it.

    @param search               search information (board ignored)
    @param OP                   vectored board
    @param alpha                alpha value (beta value is alpha + 1)
    @param empties_simd         vectored empties (2 Words)
    @return the final min score
*/
static int vectorcall last2_nws(Search *search, __m128i OP, int alpha, uint32_t p0, uint32_t p1) {
    __m128i flipped;
    uint64_t opponent = _mm_extract_epi64(OP, 1);

    ++search->n_nodes;
    #if USE_SEARCH_STATISTICS
        ++search->n_nodes_discs[62];
    #endif
    int v;
    if (bit_around[p0] & opponent) {
        if (!TESTZ_FLIP(flipped = Flip::calc_flip(OP, p0))) {
            v = last1_nws(search, _mm_xor_si128(OP, flipped), alpha, p1);
    
            if ((v > alpha) && (bit_around[p1] & opponent) && !TESTZ_FLIP(flipped = Flip::calc_flip(OP, p1))) {
                int g = last1_nws(search, _mm_xor_si128(OP, flipped), alpha, p0);
                if (v > g)
                    v = g;
            }
            return v;
        }
    }

    if (bit_around[p1] & opponent) {
        if (!TESTZ_FLIP(flipped = Flip::calc_flip(OP, p1))) {
            v = last1_nws(search, _mm_xor_si128(OP, flipped), alpha, p0);
            return v;
        }
    }
 
    // pass
    ++search->n_nodes;
    #if USE_SEARCH_STATISTICS
        ++search->n_nodes_discs[62];
    #endif
    alpha = -(alpha + 1);    // -beta
    __m128i PO = _mm_shuffle_epi32(OP, SWAP64);
    if (!TESTZ_FLIP(flipped = Flip::calc_flip(PO, p0))) {
        v = last1_nws(search, _mm_xor_si128(PO, flipped), alpha, p1);

        if ((v > alpha) && !TESTZ_FLIP(flipped = Flip::calc_flip(PO, p1))) {
            int g = last1_nws(search, _mm_xor_si128(PO, flipped), alpha, p0);
            if (v > g)
                v = g;
        }
    }

    else if (!TESTZ_FLIP(flipped = Flip::calc_flip(PO, p1)))
        v = last1_nws(search, _mm_xor_si128(PO, flipped), alpha, p0);

    else    // gameover
        v = end_evaluate(_mm_extract_epi64(PO, 1), 2);

    v = -v;
    return v;
}

/*
    @brief Get a final max score with last 3 empties (NWS)

    Only with parity-based ordering.

    @param search               search information (board ignored)
    @param OP                   vectored board
    @param alpha                alpha value (beta value is alpha + 1)
    @param empties_simd         vectored empties (3 Bytes)
    @return the final max score
*/
static int vectorcall last3_nws(Search *search, __m128i OP, int alpha, uint32_t p0, uint32_t p1, uint32_t p2) {
    __m128i flipped;
    uint64_t opponent;

    uint64_t empties = ~(search->board.player | search->board.opponent);
    if (is_1empty(p2, empties))
        std::swap(p2, p0);
    else if (is_1empty(p1, empties))
        std::swap(p1, p0);

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
                v = last2_nws(search, board_flip_next(OP, x, flipped), alpha, p1, p2);
                if (alpha < v)
                    return v * pol;
            }
        }

        int g;
        x = p1;
        if (bit_around[x] & opponent) {
            if (!TESTZ_FLIP(flipped = Flip::calc_flip(OP, x))) {
                g = last2_nws(search, board_flip_next(OP, x, flipped), alpha, p2, p0);    // (d3d1)d2d0
                if (alpha < g)
                    return g * pol;
                if (v < g)
                    v = g;
            }
        }

        x = p2;
        if (bit_around[x] & opponent) {
            if (!TESTZ_FLIP(flipped = Flip::calc_flip(OP, x))) {
                g = last2_nws(search, board_flip_next(OP, x, flipped), alpha, p1, p0);
                if (v < g)
                    v = g;
                return v * pol;
            }
        }

        if (v > -SCORE_INF)
            return v * pol;

        OP = _mm_shuffle_epi32(OP, SWAP64);    // pass
        alpha = -alpha - 1;
    } while ((pol = -pol) < 0);

    return end_evaluate(opponent, 3);    // gameover (opponent is P here)
}

/*
    @brief Get a final min score with last 4 empties (NWS)

    Only with parity-based ordering.

    @param search               search information
    @param alpha                alpha value (beta value is alpha + 1)
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
int last4_nws(Search *search, int alpha) {
    __m128i flipped;
    uint64_t opponent;
    __m128i OP = _mm_loadu_si128((__m128i*) & search->board);
    uint64_t empties = ~(search->board.player | search->board.opponent);
    uint_fast8_t p0, p1, p2, p3;

    #if USE_LAST4_SC
        int stab_res = stability_cut_last4_nws(search, alpha);
        if (stab_res != SCORE_UNDEFINED) {
            return stab_res;
        }
    #endif
    #if USE_END_PO
        uint64_t e1 = empty1_bb(search->board.player, search->board.opponent);
        if (!e1)
            e1 = empties;
        empties &= ~e1;
    
        p0 = first_bit(&e1);
        if (!(e1 &= e1 - 1))
            e1 = empties;
        p1 = first_bit(&e1);
        if (!(e1 &= e1 - 1))
            e1 = empties;
        p2 = first_bit(&e1);
        if ((e1 &= e1 - 1))
            p3 = first_bit(&e1);
        else
            p3 = first_bit(&empties);
    #else
        p0 = first_bit(&empties);
        p1 = next_bit(&empties);
        p2 = next_bit(&empties);
        p3 = next_bit(&empties);
    #endif

    int v = SCORE_INF;    // min stage
    int pol = 1;
    do {
        ++search->n_nodes;
        #if USE_SEARCH_STATISTICS
            ++search->n_nodes_discs[60];
        #endif
        opponent = _mm_extract_epi64(OP, 1);
        if (bit_around[p0] & opponent) {
            if (!TESTZ_FLIP(flipped = Flip::calc_flip(OP, p0))) {
                v = last3_nws(search, board_flip_next(OP, p0, flipped), alpha, p1, p2, p3);
                if (alpha >= v)
                    return v * pol;
            }
        }

        int g;
        if (bit_around[p1] & opponent) {
            if (!TESTZ_FLIP(flipped = Flip::calc_flip(OP, p1))) {
                g = last3_nws(search, board_flip_next(OP, p1, flipped), alpha, p0, p2, p3);
                if (alpha >= g)
                    return g * pol;
                if (v > g)
                    v = g;
            }
        }
 
        if (bit_around[p2] & opponent) {
            if (!TESTZ_FLIP(flipped = Flip::calc_flip(OP, p2))) {
                g = last3_nws(search, board_flip_next(OP, p2, flipped), alpha, p0, p1, p3);
                if (alpha >= g)
                    return g * pol;
                if (v > g)
                    v = g;
            }
        }
 
        if (bit_around[p3] & opponent) {
            if (!TESTZ_FLIP(flipped = Flip::calc_flip(OP, p3))) {
                g = last3_nws(search, board_flip_next(OP, p3, flipped), alpha, p0, p1, p2);
                if (v > g)
                    v = g;
                return v * pol;
            }
        }

        if (v < SCORE_INF)
            return v * pol;

        OP = _mm_shuffle_epi32(OP, SWAP64);    // pass
        alpha = -alpha - 1;
    } while ((pol = -pol) < 0);

    return end_evaluate(_mm_extract_epi64(OP, 1), 4);    // gameover
}