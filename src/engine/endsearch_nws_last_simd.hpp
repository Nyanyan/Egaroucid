/*
    Egaroucid Project

    @file endsearch_nws_last_simd.hpp
        Last N Moves Optimization on Null Windows Search
        imported from Edax AVX, (C) 1998 - 2018 Richard Delorme, 2014 - 23 Toshihiko Okuhara
    @date 2021-2024
    @author Takuto Yamana
    @author Toshihiko Okuhara
    @license GPL-3.0 license
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
static inline int vectorcall last1_nws(Search *search, __m128i PO, int alpha, int place) {
    uint_fast8_t n_flip;
    uint32_t t;
    uint64_t P = _mm_extract_epi64(PO, 1);
    __m256i PP = _mm256_permute4x64_epi64(_mm256_castsi128_si256(PO), 0x55);
    int score = 2 * pop_count_ull(P) - HW2 + 2;	// = (pop_count_ull(P) + 1) - (HW2 - 1 - pop_count_ull(P))
    	// if player can move, final score > score.
    	// if player pass then opponent play, final score < score - 1 (cancel P) - 1 (last O).
    	// if both pass, score - 1 (cancel P) - 1 (empty for O) <= final score <= score (empty for P).

    ++search->n_nodes;
    #if USE_SEARCH_STATISTICS
        ++search->n_nodes_discs[63];
    #endif
    if (score > alpha) {	// if player can move, high cut-off will occur regardress of n_flip.
        __m256i lM = lrmask[place].v4[0];
        __m256i rM = lrmask[place].v4[1];

    #ifdef __AVX512VL__
        __m256i F = _mm256_maskz_andnot_epi64(_mm256_test_epi64_mask(PP, lM), PP, lM);	// clear if all O
        // F = _mm256_mask_or_epi64(F, _mm256_test_epi64_mask(PP, rM), F, _mm256_andnot_si256(PP, rM));
        F = _mm256_mask_ternarylogic_epi64(F, _mm256_test_epi64_mask(PP, rM), PP, rM, 0xF2);
    #else
        __m256i lmO = _mm256_andnot_si256(PP, lM);
        __m256i F = _mm256_andnot_si256(_mm256_cmpeq_epi64(lmO, lM), lmO); // clear if all O
        __m256i rmO = _mm256_andnot_si256(PP, rM);
        F = _mm256_or_si256(F, _mm256_andnot_si256(_mm256_cmpeq_epi64(rmO, rM), rmO));
    #endif

        if (_mm256_testz_si256(F, _mm256_broadcastq_epi64(*(__m128i *) &bit_around[place]))) {	// pass
            ++search->n_nodes;
            #if USE_SEARCH_STATISTICS
                ++search->n_nodes_discs[63];
            #endif
                // n_flip = count_last_flip(~P, place);
    #ifdef __AVX512VL__
            t = _cvtmask32_u32(_mm256_test_epi8_mask(F, F));	// all O = all P = 0 flip
    #else
            t = ~_mm256_movemask_epi8(_mm256_cmpeq_epi8(lmO, rmO));	// eq only if l = r = 0
    #endif
            const int y = place >> 3;
            n_flip  = n_flip_pre_calc[(~P >> (y * 8)) & 0xFF][place & 7];	// h
            n_flip += n_flip_pre_calc[(t >> 8) & 0xFF][y];	// v
            n_flip += n_flip_pre_calc[(t >> 16) & 0xFF][y];	// d
            n_flip += n_flip_pre_calc[t >> 24][y];	// d
            score -= (n_flip + (int)((n_flip > 0) | (score <= 0))) * 2;
        } else  score += 2;	// min flip

    } else {	// if player cannot move, low cut-off will occur whether opponent can move.
        	// n_flip = count_last_flip(P, place);
        const int y = place >> 3;
        t = _mm256_movemask_epi8(_mm256_sub_epi8(_mm256_setzero_si256(), _mm256_and_si256(PP, mask_dvhd[place].v4)));
        n_flip  = n_flip_pre_calc[(P >> (y * 8)) & 0xFF][place & 7];	// h
        n_flip += n_flip_pre_calc[t & 0xFF][y];	// d
        n_flip += n_flip_pre_calc[(t >> 16) & 0xFF][y];	// v
        n_flip += n_flip_pre_calc[t >> 24][y];	// d
        score += n_flip * 2;
        	// if n_flip == 0, score <= alpha so lazy low cut-off
    }

    return score;
}

/*
    @brief Get a final min score with last 2 empties (NWS)

    No move ordering. Just search it.

    @param search               search information (board ignored)
    @param OP                   vectored board
    @param alpha                alpha value (beta value is alpha + 1)
    @param empties_simd         vectored empties (2 Words)
    @return the final min score
*/
static int vectorcall last2_nws(Search *search, __m128i OP, int alpha, __m128i empties_simd) {
    __m128i flipped;
    int p0 = _mm_extract_epi16(empties_simd, 1);
    int p1 = _mm_extract_epi16(empties_simd, 0);
    uint64_t opponent = _mm_extract_epi64(OP, 1);

    ++search->n_nodes;
    #if USE_SEARCH_STATISTICS
        ++search->n_nodes_discs[62];
    #endif
    int v;
    if ((bit_around[p0] & opponent) && !TESTZ_FLIP(flipped = Flip::calc_flip(OP, p0))) {
        v = last1_nws(search, _mm_xor_si128(OP, flipped), alpha, p1);
 
        if ((v > alpha) && (bit_around[p1] & opponent) && !TESTZ_FLIP(flipped = Flip::calc_flip(OP, p1))) {
            int g = last1_nws(search, _mm_xor_si128(OP, flipped), alpha, p0);
            if (v > g)
                v = g;
        }
    }

    else if ((bit_around[p1] & opponent) && !TESTZ_FLIP(flipped = Flip::calc_flip(OP, p1)))
        v = last1_nws(search, _mm_xor_si128(OP, flipped), alpha, p0);
 
    else {	// pass
        ++search->n_nodes;
        #if USE_SEARCH_STATISTICS
            ++search->n_nodes_discs[62];
        #endif
        alpha = -(alpha + 1);	// -beta
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

        else	// gameover
            v = end_evaluate(_mm_extract_epi64(PO, 1), 2);

        v = -v;
    }
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
static int vectorcall last3_nws(Search *search, __m128i OP, int alpha, __m128i empties_simd) {
    __m128i flipped;
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
            v = last2_nws(search, board_flip_next(OP, x, flipped), alpha, empties_simd);
            if (alpha < v)
                return v * pol;
        }

        int g;
        x = _mm_extract_epi16(empties_simd, 1);
        if ((bit_around[x] & opponent) && !TESTZ_FLIP(flipped = Flip::calc_flip(OP, x))) {
            g = last2_nws(search, board_flip_next(OP, x, flipped), alpha, _mm_shufflelo_epi16(empties_simd, 0xd8));	// (d3d1)d2d0
            if (alpha < g)
                return g * pol;
            if (v < g)
                v = g;
        }

        x = _mm_extract_epi16(empties_simd, 0);
        if ((bit_around[x] & opponent) && !TESTZ_FLIP(flipped = Flip::calc_flip(OP, x))) {
            g = last2_nws(search, board_flip_next(OP, x, flipped), alpha, _mm_shufflelo_epi16(empties_simd, 0xc9));	// (d3d0)d2d1
            if (v < g)
                v = g;
            return v * pol;
        }

        if (v > -SCORE_INF)
            return v * pol;

        OP = _mm_shuffle_epi32(OP, SWAP64);	// pass
        alpha = -alpha - 1;
    } while ((pol = -pol) < 0);

    return end_evaluate(opponent, 3);	// gameover (opponent is P here)
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
    uint_fast8_t p0 = first_bit(&empties);
    uint_fast8_t p1 = next_bit(&empties);
    uint_fast8_t p2 = next_bit(&empties);
    uint_fast8_t p3 = next_bit(&empties);

    // if (!global_searching || !(*searching))
    //  return SCORE_UNDEFINED;
    #if USE_LAST4_SC
        int stab_res = stability_cut_last4_nws(search, alpha);
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
            v = last3_nws(search, board_flip_next(OP, p0, flipped), alpha, empties_simd);
            if (alpha >= v)
                return v * pol;
        }

        int g;
        p1 = _mm_extract_epi8(empties_simd, 7);
        if ((bit_around[p1] & opponent) && !TESTZ_FLIP(flipped = Flip::calc_flip(OP, p1))) {
            g = last3_nws(search, board_flip_next(OP, p1, flipped), alpha, _mm_srli_si128(empties_simd, 4));
            if (alpha >= g)
                return g * pol;
            if (v > g)
                v = g;
        }
 
        p2 = _mm_extract_epi8(empties_simd, 11);
        if ((bit_around[p2] & opponent) && !TESTZ_FLIP(flipped = Flip::calc_flip(OP, p2))) {
            g = last3_nws(search, board_flip_next(OP, p2, flipped), alpha, _mm_srli_si128(empties_simd, 8));
            if (alpha >= g)
                return g * pol;
            if (v > g)
                v = g;
        }
 
        p3 = _mm_extract_epi8(empties_simd, 15);
        if ((bit_around[p3] & opponent) && !TESTZ_FLIP(flipped = Flip::calc_flip(OP, p3))) {
            g = last3_nws(search, board_flip_next(OP, p3, flipped), alpha, _mm_srli_si128(empties_simd, 12));
            if (v > g)
                v = g;
            return v * pol;
        }

        if (v < SCORE_INF)
            return v * pol;

        OP = _mm_shuffle_epi32(OP, SWAP64);	// pass
        alpha = -alpha - 1;
    } while ((pol = -pol) < 0);

    return end_evaluate(_mm_extract_epi64(OP, 1), 4);	// gameover
}
