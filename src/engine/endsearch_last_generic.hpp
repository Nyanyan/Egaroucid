/*
    Egaroucid Project

    @file endsearch_last_generic.hpp
        Last N Moves Optimization
    @date 2021-2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/
#pragma once
#include "setting.hpp"
#include "search.hpp"
#include "endsearch_common.hpp"

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
inline int last1(Search *search, uint64_t player, int alpha, uint_fast8_t p0) {
    ++search->n_nodes;
#if USE_SEARCH_STATISTICS
    ++search->n_nodes_discs[63];
#endif
    int n_flip, score;
#if LAST1_OBVIOUS_PASS_OPT
    if ((~player & bit_around[p0]) == 0) { // obvious pass
        ++search->n_nodes;
#if USE_SEARCH_STATISTICS
        ++search->n_nodes_discs[63];
#endif
        score = 2 * (pop_count_ull(player) + 1) - HW2;	// (P + n_flip + 1) - (HW2 - 1 - P - n_flip)
        int score2 = score - 2;	// empty for opponent
        if (score <= 0)
            score = score2;
        if (score > alpha) {
            n_flip = count_last_flip(~player, p0);
            if (n_flip)
                score = score2 - 2 * n_flip;
        }
    } else {
#endif
#if LAST_FLIP_PASS_OPT
        const int x = p0 & 7;
        const int y = p0 >> 3;
        uint_fast8_t d7 = join_d7_line(player, x + y);
        uint_fast8_t d9 = join_d9_line(player, x + 7 - y);
        int n_flip_both = 
            N_LAST_FLIP_BOTH[join_h_line(player, y)][x] + // both h
            N_LAST_FLIP_BOTH[join_v_line(player, x)][y] + // both v
            N_LAST_FLIP[d7][x] + // player d7
            N_LAST_FLIP[d9][x];  // player d9
        n_flip = n_flip_both & 0xff;
#else
        n_flip = count_last_flip(player, p0);
#endif
        score = 2 * (pop_count_ull(player) + n_flip + 1) - HW2;	// (P + n_flip + 1) - (HW2 - 1 - P - n_flip)
        if (n_flip == 0) {
            ++search->n_nodes;
#if USE_SEARCH_STATISTICS
            ++search->n_nodes_discs[63];
#endif
            int score2 = score - 2;	// empty for opponent
            if (score <= 0)
                score = score2;
            if (score > alpha) {
#if LAST_FLIP_PASS_OPT
                n_flip = 
                    (n_flip_both >> 8) + // hv (calculated)
                    N_LAST_FLIP[DIAGONAL_LINE_MASK_T[x + y] ^ d7][x] + // d7
                    N_LAST_FLIP[DIAGONAL_LINE_MASK_T[x + 7 - y] ^ d9][x]; // d9
#else
                n_flip = count_last_flip(~player, p0);
#endif
                if (n_flip)
                    score = score2 - 2 * n_flip;
            }
        }
#if LAST1_OBVIOUS_PASS_OPT
    }
#endif
    return score;
}

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
    if ((bit_around[p0] & board.opponent) && calc_flip(&flip, &board, p0)) { // p0
        v = last1(search, board.opponent ^ flip.flip, alpha, p1);

        if ((v > alpha) && (bit_around[p1] & board.opponent) && calc_flip(&flip, &board, p1)) { // p1
            int g = last1(search, board.opponent ^ flip.flip, alpha, p0);
            if (v > g) {
                v = g;
            }
        }
    } else if ((bit_around[p1] & board.opponent) && calc_flip(&flip, &board, p1)) { // p1 only
        v = last1(search, board.opponent ^ flip.flip, alpha, p0);
    } else {	// pass
        ++search->n_nodes;
#if USE_SEARCH_STATISTICS
        ++search->n_nodes_discs[62];
#endif
        alpha = -beta;
        if (flip.calc_flip(board.opponent, board.player, p0)) { // p0
            v = last1(search, board.player ^ flip.flip, alpha, p1);

            if ((v > alpha) && (flip.calc_flip(board.opponent, board.player, p1))) { // p1
                int g = last1(search, board.player ^ flip.flip, alpha, p0);
                if (v > g) {
                    v = g;
                }
            }
        } else if (flip.calc_flip(board.opponent, board.player, p1)) { // p1 only
            v = last1(search, board.player ^ flip.flip, alpha, p0);
        } else { // gameover
            v = end_evaluate(&board, 2);
        }
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
static int last3(Search *search, int alpha, int beta, uint_fast8_t p0, uint_fast8_t p1, uint_fast8_t p2, Board board) {
#if USE_END_PO
    uint64_t empties = ~(search->board.player | search->board.opponent);
    if (is_1empty(p2, empties)) {
        std::swap(p2, p0);
    } else if (is_1empty(p1, empties)) {
        std::swap(p1, p0);
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
            if (beta <= v) {
                return v * pol;
            }
            if (alpha < v) {
                alpha = v;
            }
        }

        int g;
        if ((bit_around[p1] & board.opponent) && calc_flip(&flip, &board, p1)) {
            board.move_copy(&flip, &board2);
            g = last2(search, alpha, beta, p2, p0, board2);
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

        if ((bit_around[p2] & board.opponent) && calc_flip(&flip, &board, p2)) {
            board.move_copy(&flip, &board2);
            g = last2(search, alpha, beta, p1, p0, board2);
            if (v < g) {
                v = g;
            }
            return v * pol;
        }

        if (v > -SCORE_INF) {
            return v * pol;
        }

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
            v = last3(search, alpha, beta, p1, p2, p3, board3);
            if (alpha >= v) {
                return v * pol;
            }
            if (beta > v) {
                beta = v;
            }
        }

        int g;
        if ((bit_around[p1] & board4.opponent) && calc_flip(&flip, &board4, p1)) {
            board4.move_copy(&flip, &board3);
            g = last3(search, alpha, beta, p0, p2, p3, board3);
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

        if ((bit_around[p2] & board4.opponent) && calc_flip(&flip, &board4, p2)) {
            board4.move_copy(&flip, &board3);
            g = last3(search, alpha, beta, p0, p1, p3, board3);
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

        if ((bit_around[p3] & board4.opponent) && calc_flip(&flip, &board4, p3)) {
            board4.move_copy(&flip, &board3);
            g = last3(search, alpha, beta, p0, p1, p2, board3);
            if (v > g) {
                v = g;
            }
            return v * pol;
        }

        if (v < SCORE_INF) {
            return v * pol;
        }
        board4.pass();
        int t = alpha;  alpha = -beta;  beta = -t;
    } while ((pol = -pol) < 0);

    return -end_evaluate(&search->board, 4);	// gameover
}