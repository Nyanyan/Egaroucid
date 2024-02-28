/*
    Egaroucid Project

    @file endsearch.hpp
        Search near endgame
        last2/3/4 imported from Edax AVX, (C) 1998 - 2018 Richard Delorme, 2014 - 23 Toshihiko Okuhara
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
#include "evaluate.hpp"
#include "search.hpp"
#include "move_ordering.hpp"
#include "probcut.hpp"
#include "transposition_table.hpp"
#include "util.hpp"
#include "stability.hpp"
#include "endsearch_nws.hpp"
#include "parallel.hpp"
#include "ybwc.hpp"

#if USE_SIMD
#include "endsearch_simd.hpp"
#else
/*
    @brief Get a final min score with last 2 empties

    No move ordering. Just search it.

    @param search               search information (board ignored)
    @param alpha                alpha value
    @param beta                 beta value
    @param p0                   empty square 1/2
    @param p1                   empty square 2/2
    @param board                bitboard
    @return the final min score
*/
static int last2(Search *search, int alpha, int beta, uint_fast8_t p0, uint_fast8_t p1, Board board) {
    ++search->n_nodes;
    #if USE_SEARCH_STATISTICS
        ++search->n_nodes_discs[62];
    #endif
    int v;
    Flip flip;
    //if ((bit_around[p0] & board.player) == 0)
    //    std::swap(p0, p1);
    if ((bit_around[p0] & board.opponent) && calc_flip(&flip, &board, p0)) {
        v = last1(search, board.opponent ^ flip.flip, alpha, p1);

        if ((v > alpha) && (bit_around[p1] & board.opponent) && calc_flip(&flip, &board, p1)) {
            int g = last1(search, board.opponent ^ flip.flip, alpha, p0);
            if (v > g)
                v = g;
        }
    }

    else if ((bit_around[p1] & board.opponent) && calc_flip(&flip, &board, p1))
         v = last1(search, board.opponent ^ flip.flip, alpha, p0);

    else {	// pass
        ++search->n_nodes;
        #if USE_SEARCH_STATISTICS
            ++search->n_nodes_discs[62];
        #endif
        alpha = -beta;
        if (flip.calc_flip(board.opponent, board.player, p0)) {
            v = last1(search, board.player ^ flip.flip, alpha, p1);

            if ((v > alpha) && (flip.calc_flip(board.opponent, board.player, p1))) {
                int g = last1(search, board.player ^ flip.flip, alpha, p0);
                if (v > g)
                    v = g;
            }
        }

        else if (flip.calc_flip(board.opponent, board.player, p1))
            v = last1(search, board.player ^ flip.flip, alpha, p0);

        else	// gameover
            v = end_evaluate(&board, 2);

        v = -v;
    }
    return v;
}

/*
    @brief Get a final max score with last 3 empties

    Only with parity-based ordering.

    @param search               search information (board ignored)
    @param alpha                alpha value
    @param beta                 beta value
    @param sort3                parity sort (lower 2 bits for this node)
    @param p0                   empty square 1/3
    @param p1                   empty square 2/3
    @param p2                   empty square 3/3
    @param board                bitboard
    @return the final max score

    This board contains only 3 empty squares, so empty squares on each part will be:
        3 - 0 - 0 - 0
        2 - 1 - 0 - 0 > need to sort
        1 - 1 - 1 - 0
    then the parities for squares will be:
        1 - 1 - 1
        1 - 0 - 0 > need to sort
        1 - 1 - 1
*/
static int last3(Search *search, int alpha, int beta, int sort3, uint_fast8_t p0, uint_fast8_t p1, uint_fast8_t p2, Board board) {
    // if (!global_searching || !(*searching))
    //  return SCORE_UNDEFINED;
    #if USE_END_PO
        switch (sort3 & 3){
            case 2:
                std::swap(p0, p2);
                break;
            case 1:
                std::swap(p0, p1);
                break;
        }
    #endif

    Flip flip;
    Board board2;
    int v = -SCORE_INF;
    int pol = 1;
    do {
        ++search->n_nodes;
        #if USE_SEARCH_STATISTICS
            ++search->n_nodes_discs[61];
        #endif
        if ((bit_around[p0] & board.opponent) && calc_flip(&flip, &board, p0)) {
            board.move_copy(&flip, &board2);
            v = last2(search, alpha, beta, p1, p2, board2);
            if (beta <= v)
                return v * pol;
            if (alpha < v)
                alpha = v;
        }

        int g;
        if ((bit_around[p1] & board.opponent) && calc_flip(&flip, &board, p1)) {
            board.move_copy(&flip, &board2);
            g = last2(search, alpha, beta, p0, p2, board2);
            if (beta <= g)
                return g * pol;
            if (v < g)
                v = g;
            if (alpha < g)
                alpha = g;
        }

        if ((bit_around[p2] & board.opponent) && calc_flip(&flip, &board, p2)) {
            board.move_copy(&flip, &board2);
            g = last2(search, alpha, beta, p0, p1, board2);
            if (v < g)
                v = g;
            return v * pol;
        }

        if (v > -SCORE_INF)
            return v * pol;

        board.pass();
        int t = alpha;  alpha = -beta;  beta = -t;
    } while ((pol = -pol) < 0);

    return end_evaluate_odd(&board, 3);	// gameover
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
    #if USE_END_PO
                // parity ordering optimization
                // I referred to http://www.amy.hi-ho.ne.jp/okuhara/edaxopt.htm
        const int paritysort = parity_case[((p2 ^ p3) & 0x24) + ((((p1 ^ p3) & 0x24) * 2 + ((p0 ^ p3) & 0x24)) >> 2)];
        switch (paritysort) {
            case 8:     // case 1(p2) 1(p3) 2(p0 p1)
                std::swap(p0, p2);
                std::swap(p1, p3);
                break;
            case 7:     // case 1(p1) 1(p3) 2(p0 p2)
                std::swap(p0, p3);
                break;
            case 6:     // case 1(p1) 1(p2) 2(p0 p3)
                std::swap(p0, p2);
                break;
            case 5:     // case 1(p0) 1(p3) 2(p1 p2)
                std::swap(p1, p3);
                break;
            case 4:     // case 1(p0) 1(p2) 2(p1 p3)
                std::swap(p1, p2);
                break;
        }
        int sort3 = parity_ordering_last3[paritysort];     // for move sorting on 3 empties
    #else
        constexpr int sort3 = 0;
    #endif

    Flip flip;
    Board board3, board4;
    search->board.copy(&board4);
    int v = SCORE_INF;	// min stage
    int pol = 1;
    do {
        ++search->n_nodes;
        #if USE_SEARCH_STATISTICS
            ++search->n_nodes_discs[60];
        #endif
        if ((bit_around[p0] & board4.opponent) && calc_flip(&flip, &board4, p0)) {
            board4.move_copy(&flip, &board3);
            v = last3(search, alpha, beta, sort3, p1, p2, p3, board3);
            if (alpha >= v)
                return v * pol;
            if (beta > v)
                beta = v;
        }

        int g;
        if ((bit_around[p1] & board4.opponent) && calc_flip(&flip, &board4, p1)) {
            board4.move_copy(&flip, &board3);
            g = last3(search, alpha, beta, sort3 >> 4, p0, p2, p3, board3);
            if (alpha >= g)
                return g * pol;
            if (v > g)
                v = g;
            if (beta > g)
                beta = g;
        }

        if ((bit_around[p2] & board4.opponent) && calc_flip(&flip, &board4, p2)) {
            board4.move_copy(&flip, &board3);
            g = last3(search, alpha, beta, sort3 >> 8, p0, p1, p3, board3);
            if (alpha >= g)
                return g * pol;
            if (v > g)
                v = g;
            if (beta > g)
                beta = g;
        }

        if ((bit_around[p3] & board4.opponent) && calc_flip(&flip, &board4, p3)) {
            board4.move_copy(&flip, &board3);
            g = last3(search, alpha, beta, sort3 >> 12, p0, p1, p2, board3);
            if (v > g)
                v = g;
            return v * pol;
        }

        if (v < SCORE_INF)
            return v * pol;

        board4.pass();
        int t = alpha;  alpha = -beta;  beta = -t;
    } while ((pol = -pol) < 0);

    return -end_evaluate(&search->board, 4);	// gameover
}
#endif
