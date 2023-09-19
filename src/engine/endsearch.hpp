/*
    Egaroucid Project

    @file endsearch.hpp
        Search near endgame
        last2/3/4 imported from Edax AVX, (C) 1998 - 2018 Richard Delorme, 2014 - 23 Toshihiko Okuhara
    @date 2021-2023
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

#if USE_NEGA_ALPHA_END && MID_TO_END_DEPTH > YBWC_END_SPLIT_MIN_DEPTH
    inline bool ybwc_split_end(const Search *search, int alpha, int beta, uint64_t legal, const bool *searching, uint_fast8_t policy, const int canput, const int pv_idx, const int split_count, std::vector<std::future<Parallel_task>> &parallel_tasks);
    inline void ybwc_get_end_tasks(Search *search, std::vector<std::future<Parallel_task>> &parallel_tasks, int *v, int *best_move, int *alpha);
    inline void ybwc_wait_all(Search *search, std::vector<std::future<Parallel_task>> &parallel_tasks);
    inline void ybwc_wait_all(Search *search, std::vector<std::future<Parallel_task>> &parallel_tasks, int *v, int *best_move, int *alpha, int beta, bool *searching);
#endif

#if USE_SIMD
#include "endsearch_simd.hpp"
#else
/*
    @brief Get a final score with last 2 empties

    No move ordering. Just search it.

    @param search               search information (board ignored)
    @param alpha                alpha value
    @param beta                 beta value
    @param p0                   empty square 1/2
    @param p1                   empty square 2/2
    @param board                bitboard
    @return the final score
*/
static int last2(Search *search, int alpha, int beta, uint_fast8_t p0, uint_fast8_t p1, Board board) {
    ++search->n_nodes;
    #if USE_SEARCH_STATISTICS
        ++search->n_nodes_discs[search->n_discs];
    #endif
    int v;
    Flip flip;
    //if ((bit_around[p0] & board.player) == 0)
    //    std::swap(p0, p1);
    if ((bit_around[p0] & board.opponent) && calc_flip(&flip, &board, p0)) {
        v = last1n(search, board.opponent ^ flip.flip, beta, p1);

        if ((v < beta) && (bit_around[p1] & board.opponent) && calc_flip(&flip, &board, p1)) {
            int g = last1n(search, board.opponent ^ flip.flip, beta, p0);
            if (v < g)
                v = g;
        }
    }

    else if ((bit_around[p1] & board.opponent) && calc_flip(&flip, &board, p1))
         v = last1n(search, board.opponent ^ flip.flip, beta, p0);

    else {	// pass
        ++search->n_nodes;
        beta = -alpha;
        if (flip.calc_flip(board.opponent, board.player, p0)) {
            v = last1n(search, board.player ^ flip.flip, beta, p1);

            if ((v < beta) && (flip.calc_flip(board.opponent, board.player, p1))) {
                int g = last1n(search, board.player ^ flip.flip, beta, p0);
                if (v < g)
                    v = g;
            }
        }

        else if (flip.calc_flip(board.opponent, board.player, p1))
            v = last1n(search, board.player ^ flip.flip, beta, p0);

        else	// gameover
            return end_evaluate(&board, 2);

        v = -v;
    }
    return v;
}

/*
    @brief Get a final score with last 3 empties

    Only with parity-based ordering.

    @param search               search information (board ignored)
    @param alpha                alpha value
    @param beta                 beta value
    @param sort3                parity sort (lower 2 bits for this node)
    @param p0                   empty square 1/3
    @param p1                   empty square 2/3
    @param p2                   empty square 3/3
    @param board                bitboard
    @return the final score

    This board contains only 3 empty squares, so empty squares on each part will be:
        3 - 0 - 0 - 0
        2 - 1 - 0 - 0 > need to sort
        1 - 1 - 1 - 0
    then the parities for squares will be:
        1 - 1 - 1
        1 - 0 - 0 > need to sort
        1 - 1 - 1
*/
inline int last3(Search *search, int alpha, int beta, int sort3, uint_fast8_t p0, uint_fast8_t p1, uint_fast8_t p2, Board board) {
    // if (!global_searching || !(*searching))
    //  return SCORE_UNDEFINED;
    #if USE_SEARCH_STATISTICS
        ++search->n_nodes_discs[search->n_discs];
    #endif
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
    int pol = -1;
    do {
        ++search->n_nodes;
        int t = alpha;  alpha = -beta;  beta = -t;
        int v = SCORE_INF;	// Negative score
        if ((bit_around[p0] & board.opponent) && (calc_flip(&flip, &board, p0))) {
            board.move_copy(&flip, &board2);
            v = last2(search, alpha, beta, p1, p2, board2);
            if (alpha >= v)
                return v * pol;
            if (beta > v)
                beta = v;
        }

        int g;
        if ((bit_around[p1] & board.opponent) && calc_flip(&flip, &board, p1)) {
            board.move_copy(&flip, &board2);
            g = last2(search, alpha, beta, p0, p2, board2);
            if (alpha >= g)
                return g * pol;
            if (v > g)
                v = g;
            if (beta > g)
                beta = g;
        }

        if ((bit_around[p2] & board.opponent) && calc_flip(&flip, &board, p2)) {
            board.move_copy(&flip, &board2);
            g = last2(search, alpha, beta, p0, p1, board2);
            if (v > g)
                v = g;
            return v * pol;
        }

        if (v < SCORE_INF)
            return v * pol;

	board.pass();
    } while ((pol = -pol) >= 0);

    return end_evaluate_odd(&board, 3);	// gameover
}

/*
    @brief Get a final score with last 4 empties

    Only with parity-based ordering.

    @param search               search information
    @param alpha                alpha value
    @param beta                 beta value
    @return the final score

    This board contains only 4 empty squares, so empty squares on each part will be:
        4 - 0 - 0 - 0
        3 - 1 - 0 - 0
        2 - 2 - 0 - 0
        2 - 1 - 1 - 0 > need to sort
        1 - 1 - 1 - 1
    then the parities for squares will be:
        0 - 0 - 0 - 0
        1 - 1 - 1 - 1
        0 - 0 - 0 - 0
        1 - 1 - 0 - 0 > need to sort
        1 - 1 - 1 - 1
*/
int last4(Search *search, int alpha, int beta) {
#if USE_END_PO
    int sort3;	// for move sorting on 3 empties
    static constexpr uint16_t sort3_shuf[] = {
        0x0000,	//  0: 1(p0) 3(p1 p2 p3), 1(p0) 1(p1) 2(p2 p3), 1 1 1 1, 4
        0x1100,	//  1: 1(p1) 3(p0 p2 p3)	p3p1p0p2-p2p1p0p3-p1p0p2p3-p0p1p2p3
        0x2011,	//  2: 1(p2) 3(p0 p1 p3)	p3p2p1p0-p2p0p1p3-p1p2p0p3-p0p2p1p3
        0x0222,	//  3: 1(p3) 3(p0 p1 p2)	p3p0p1p2-p2p3p1p0-p1p3p2p0-p0p3p2p1
        0x0000,	//  4: 1(p0) 1(p2) 2(p1 p3)
        0x0000,	//  5: 1(p0) 1(p3) 2(p1 p2)
        0x0000,	//  6: 1(p1) 1(p2) 2(p0 p3)
        0x0000,	//  7: 1(p1) 1(p3) 2(p0 p2)
        0x0000,	//  8: 1(p2) 1(p3) 2(p0 p1)
        0x2200,	//  9: 2(p0 p1) 2(p2 p3)	p3p2p1p0-p2p3p1p0-p1p0p2p3-p0p1p2p3
        0x1021,	// 10: 2(p0 p2) 2(p1 p3)	p3p1p0p2-p2p0p1p3-p1p3p2p0-p0p2p1p3
        0x0112	// 11: 2(p0 p3) 2(p1 p2)	p3p0p1p2-p2p1p0p3-p1p2p0p3-p0p3p2p1
    };
#endif
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
        if (stab_res != SCORE_UNDEFINED){
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
        sort3 = sort3_shuf[paritysort];     // for move sorting on 3 empties
    #endif

    Flip flip;
    Board board3, board4;
    search->board.copy(&board4);
    int pol = -1;
    do {
        ++search->n_nodes;
        int t = alpha;  alpha = -beta;  beta = -t;
        int v = SCORE_INF;	// Negative score
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
    } while ((pol = -pol) >= 0);

    return end_evaluate(&search->board, 4);	// gameover
}
#endif

#if USE_NEGA_ALPHA_END_FAST
    /*
        @brief Get a final score with few empties

        Only with parity-based ordering.

        @param search               search information
        @param alpha                alpha value
        @param beta                 beta value
        @param skipped              already passed?
        @param stab_cut             use stability cutoff?
        @param searching            flag for terminating this search
        @return the final score
    */
    int nega_alpha_end_fast(Search *search, int alpha, int beta, bool skipped, bool stab_cut, const bool *searching){
        if (!global_searching || !(*searching))
            return SCORE_UNDEFINED;
        if (alpha + 1 == beta)
            return nega_alpha_end_fast_nws(search, alpha, skipped, stab_cut, searching);
        ++search->n_nodes;
        #if USE_SEARCH_STATISTICS
            ++search->n_nodes_discs[search->n_discs];
        #endif
        #if USE_END_SC
            if (stab_cut){
                int stab_res = stability_cut(search, &alpha, &beta);
                if (stab_res != SCORE_UNDEFINED){
                    return stab_res;
                }
            }
        #endif
        uint64_t legal = search->board.get_legal();
        int v = -SCORE_INF;
        if (legal == 0ULL){
            if (skipped)
                return end_evaluate(&search->board);
            search->board.pass();
                v = -nega_alpha_end_fast(search, -beta, -alpha, true, false, searching);
            search->board.pass();
            return v;
        }
        int g;
        Flip flip;
        uint_fast8_t cell;
        #if USE_END_PO
            uint64_t legal_copy;
            if (0 < search->parity && search->parity < 15){
                uint64_t legal_mask = parity_table[search->parity];
                if (search->n_discs == 59){
                    legal_copy = legal & legal_mask;
                    for (cell = first_bit(&legal_copy); legal_copy; cell = next_bit(&legal_copy)){
                        calc_flip(&flip, &search->board, cell);
                        search->move(&flip);
                            g = -last4(search, -beta, -alpha);
                        search->undo(&flip);
                        if (v < g){
                            if (alpha < g){
                                if (beta <= g)
                                    return g;
                                alpha = g;
                            }
                            v = g;
                        }
                    }
                    legal_copy = legal & ~legal_mask;
                    for (cell = first_bit(&legal_copy); legal_copy; cell = next_bit(&legal_copy)){
                        calc_flip(&flip, &search->board, cell);
                        search->move(&flip);
                            g = -last4(search, -beta, -alpha);
                        search->undo(&flip);
                        if (v < g){
                            if (alpha < g){
                                if (beta <= g)
                                    return g;
                                alpha = g;
                            }
                            v = g;
                        }
                    }
                } else{
                    legal_copy = legal & legal_mask;
                    for (cell = first_bit(&legal_copy); legal_copy; cell = next_bit(&legal_copy)){
                        calc_flip(&flip, &search->board, cell);
                        search->move(&flip);
                            g = -nega_alpha_end_fast(search, -beta, -alpha, false, true, searching);
                        search->undo(&flip);
                        if (v < g){
                            if (alpha < g){
                                if (beta <= g)
                                    return g;
                                alpha = g;
                            }
                            v = g;
                        }
                    }
                    legal_copy = legal & ~legal_mask;
                    for (cell = first_bit(&legal_copy); legal_copy; cell = next_bit(&legal_copy)){
                        calc_flip(&flip, &search->board, cell);
                        search->move(&flip);
                            g = -nega_alpha_end_fast(search, -beta, -alpha, false, true, searching);
                        search->undo(&flip);
                        if (v < g){
                            if (alpha < g){
                                if (beta <= g)
                                    return g;
                                alpha = g;
                            }
                            v = g;
                        }
                    }
                }
            } else{
                if (search->n_discs == 59){
                    for (cell = first_bit(&legal); legal; cell = next_bit(&legal)){
                        calc_flip(&flip, &search->board, cell);
                        search->move(&flip);
                            g = -last4(search, -beta, -alpha);
                        search->undo(&flip);
                        if (v < g){
                            if (alpha < g){
                                if (beta <= g)
                                    return g;
                                alpha = g;
                            }
                            v = g;
                        }
                    }
                } else{
                    for (cell = first_bit(&legal); legal; cell = next_bit(&legal)){
                        calc_flip(&flip, &search->board, cell);
                        search->move(&flip);
                            g = -nega_alpha_end_fast(search, -beta, -alpha, false, true, searching);
                        search->undo(&flip);
                        if (v < g){
                            if (alpha < g){
                                if (beta <= g)
                                    return g;
                                alpha = g;
                            }
                            v = g;
                        }
                    }
                }
            }
        #else
            if (search->n_discs == 59){
                for (cell = first_bit(&legal); legal; cell = next_bit(&legal)){
                    calc_flip(&flip, &search->board, cell);
                    search->move(&flip);
                        g = -last4(search, -beta, -alpha);
                    search->undo(&flip);
                    if (v < g){
                        if (alpha < g){
                            if (beta <= g)
                                return g;
                            alpha = g;
                        }
                        v = g;
                    }
                }
            } else{
                for (cell = first_bit(&legal); legal; cell = next_bit(&legal)){
                    calc_flip(&flip, &search->board, cell);
                    search->move(&flip);
                        g = -nega_alpha_end_fast(search, -beta, -alpha, false, true, searching);
                    search->undo(&flip);
                    if (v < g){
                        if (alpha < g){
                            if (beta <= g)
                                return g;
                            alpha = g;
                        }
                        v = g;
                    }
                }
            }
        #endif
        return v;
    }
#endif

#if USE_NEGA_ALPHA_END
    /*
        @brief Get a final score with some empties

        Search with move ordering for endgame and transposition tables.

        @param search               search information
        @param alpha                alpha value
        @param beta                 beta value
        @param skipped              already passed?
        @param legal                for use of previously calculated legal bitboard
        @param searching            flag for terminating this search
        @return the final score
    */
    int nega_alpha_end(Search *search, int alpha, int beta, bool skipped, uint64_t legal, const bool *searching){
        if (!global_searching || !(*searching))
            return SCORE_UNDEFINED;
        if (alpha + 1 == beta)
            return nega_alpha_end_nws(search, alpha, skipped, legal, searching);
        #if USE_NEGA_ALPHA_END_FAST
            if (search->n_discs >= HW2 - END_FAST_DEPTH)
                return nega_alpha_end_fast(search, alpha, beta, skipped, false, searching);
        #else
            if (search->n_discs == 60){
                return last4(search, alpha, beta);
            }
        #endif
        ++search->n_nodes;
        #if USE_SEARCH_STATISTICS
            ++search->n_nodes_discs[search->n_discs];
        #endif
        #if USE_END_SC
            int stab_res = stability_cut(search, &alpha, &beta);
            if (stab_res != SCORE_UNDEFINED){
                return stab_res;
            }
        #endif
        if (legal == LEGAL_UNDEFINED)
            legal = search->board.get_legal();
        int v = -SCORE_INF;
        if (legal == 0ULL){
            if (skipped)
                return end_evaluate(&search->board);
            search->board.pass();
                v = -nega_alpha_end(search, -beta, -alpha, true, LEGAL_UNDEFINED, searching);
            search->board.pass();
            return v;
        }
        uint32_t hash_code = search->board.hash();
        int lower = -SCORE_MAX, upper = SCORE_MAX;
        uint_fast8_t moves[N_TRANSPOSITION_MOVES] = {TRANSPOSITION_TABLE_UNDEFINED, TRANSPOSITION_TABLE_UNDEFINED};
        if (search->n_discs <= HW2 - USE_TT_DEPTH_THRESHOLD)
            transposition_table.get(search, hash_code, HW2 - search->n_discs, &lower, &upper, moves);
        if (upper == lower)
            return upper;
        if (beta <= lower)
            return lower;
        if (upper <= alpha)
            return upper;
        if (alpha < lower)
            alpha = lower;
        if (upper < beta)
            beta = upper;
        Flip flip_best;
        int best_move = TRANSPOSITION_TABLE_UNDEFINED;
        int first_alpha = alpha;
        int g;
        #if MID_TO_END_DEPTH > YBWC_END_SPLIT_MIN_DEPTH
            int pv_idx = 0;
        #endif
        for (uint_fast8_t i = 0; i < N_TRANSPOSITION_MOVES; ++i){
            if (moves[i] == TRANSPOSITION_TABLE_UNDEFINED)
                break;
            if (1 & (legal >> moves[i])){
                calc_flip(&flip_best, &search->board, moves[i]);
                search->move(&flip_best);
                    g = -nega_alpha_end(search, -beta, -alpha, false, LEGAL_UNDEFINED, searching);
                search->undo(&flip_best);
                if (v < g){
                    v = g;
                    best_move = moves[i];
                    if (alpha < v){
                        alpha = v;
                        if (beta <= v)
                            break;
                    }
                }
                legal ^= 1ULL << moves[i];
                #if MID_TO_END_DEPTH > YBWC_END_SPLIT_MIN_DEPTH
                    ++pv_idx;
                #endif
            }
        }
        if (alpha < beta && legal){
            const int canput = pop_count_ull(legal);
            Flip_value move_list[MID_TO_END_DEPTH];
            int idx = 0;
            for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)){
                calc_flip(&move_list[idx].flip, &search->board, cell);
                if (move_list[idx].flip.flip == search->board.opponent)
                    return SCORE_MAX;
                ++idx;
            }
            move_list_evaluate_end(search, move_list, canput);
            #if MID_TO_END_DEPTH > YBWC_END_SPLIT_MIN_DEPTH
                #if USE_ALL_NODE_PREDICTION
                    const bool seems_to_be_all_node = predict_all_node(search, alpha, HW2 - search->n_discs, LEGAL_UNDEFINED, true, searching);
                #else
                    constexpr bool seems_to_be_all_node = false;
                #endif
                if (search->use_multi_thread){
                    int split_count = 0;
                    std::vector<std::future<Parallel_task>> parallel_tasks;
                    bool n_searching = true;
                    for (int move_idx = 0; move_idx < canput; ++move_idx){
                        swap_next_best_move(move_list, move_idx, canput);
                        search->move(&move_list[move_idx].flip);
                            if (ybwc_split_end(search, -beta, -alpha, move_list[move_idx].n_legal, &n_searching, move_list[move_idx].flip.pos, canput, pv_idx++, seems_to_be_all_node, split_count, parallel_tasks)){
                                ++split_count;
                            } else{
                                g = -nega_alpha_end(search, -beta, -alpha, false, move_list[move_idx].n_legal, searching);
                                if (v < g){
                                    v = g;
                                    best_move = move_list[move_idx].flip.pos;
                                    if (alpha < v){
                                        if (beta <= v){
                                            search->undo(&move_list[move_idx].flip);
                                            break;
                                        }
                                        alpha = v;
                                    }
                                }
                                if (split_count){
                                    ybwc_get_end_tasks(search, parallel_tasks, &v, &best_move, &alpha);
                                    if (alpha < v){
                                        if (beta <= v){
                                            search->undo(&move_list[move_idx].flip);
                                            break;
                                        }
                                        alpha = v;
                                    }
                                }
                            }
                        search->undo(&move_list[move_idx].flip);
                    }
                    if (split_count){
                        if (beta <= alpha || !(*searching)){
                            n_searching = false;
                            ybwc_wait_all(search, parallel_tasks);
                        } else
                            ybwc_wait_all(search, parallel_tasks, &v, &best_move, &alpha, beta, &n_searching);
                    }
                } else{
                    for (int move_idx = 0; move_idx < canput; ++move_idx){
                        swap_next_best_move(move_list, move_idx, canput);
                        search->move(&move_list[move_idx].flip);
                            g = -nega_alpha_end(search, -beta, -alpha, false, move_list[move_idx].n_legal, searching);
                        search->undo(&move_list[move_idx].flip);
                        if (v < g){
                            v = g;
                            best_move = move_list[move_idx].flip.pos;
                            if (alpha < v){
                                if (beta <= v)
                                    break;
                                alpha = v;
                            }
                        }
                    }
                }
            #else
                for (int move_idx = 0; move_idx < canput; ++move_idx){
                    swap_next_best_move(move_list, move_idx, canput);
                    search->move(&move_list[move_idx].flip);
                        g = -nega_alpha_end(search, -beta, -alpha, false, move_list[move_idx].n_legal, searching);
                    search->undo(&move_list[move_idx].flip);
                    if (v < g){
                        v = g;
                        best_move = move_list[move_idx].flip.pos;
                        if (alpha < v){
                            alpha = v;
                            if (beta <= v)
                                break;
                        }
                    }
                }
            #endif
        }
        if (*searching && global_searching && search->n_discs <= HW2 - USE_TT_DEPTH_THRESHOLD)
            transposition_table.reg(search, hash_code, HW2 - search->n_discs, first_alpha, beta, v, best_move);
        return v;
    }
#endif