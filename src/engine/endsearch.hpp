/*
    Egaroucid Project

    @file endsearch.hpp
        Search near endgame
    @date 2021-2023
    @author Takuto Yamana (a.k.a. Nyanyan)
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

/*
    @brief Get a final score with last 2 empties

    No move ordering. Just search it.

    @param search               search information
    @param alpha                alpha value
    @param beta                 beta value
    @param p0                   empty square 1/2
    @param p1                   empty square 2/2
    @param skipped              already passed?
    @return the final score
*/
inline int last2(Search *search, int alpha, int beta, uint_fast8_t p0, uint_fast8_t p1, bool skipped){
    ++search->n_nodes;
    #if USE_SEARCH_STATISTICS
        ++search->n_nodes_discs[search->n_discs];
    #endif
    int v = -SCORE_INF;
    uint64_t flip;
    if (bit_around[p0] & search->board.opponent){
        flip = calc_flip(&search->board, p0);
        if (flip){
            search->move_cell(flip, p0);
                v = -last1(search, -beta, p1);
            search->undo_cell(flip, p0);
            if (alpha < v){
                if (beta <= v)
                    return v;
                alpha = v;
            }
        }
    }
    if (bit_around[p1] & search->board.opponent){
        flip = calc_flip(&search->board, p1);
        if (flip){
            search->move_cell(flip, p1);
                int g = -last1(search, -beta, p0);
            search->undo_cell(flip, p1);
            if (v < g)
                return g;
            else
                return v;
        }
    }
    if (v == -SCORE_INF){
        if (skipped)
            v = end_evaluate(&search->board, 2);
        else{
            search->board.pass();
                v = -last2(search, -beta, -alpha, p0, p1, true);
            search->board.pass();
        }
    }
    return v;
}

/*
    @brief Get a final score with last 3 empties

    Only with parity-based ordering.

    @param search               search information
    @param alpha                alpha value
    @param beta                 beta value
    @param p0                   empty square 1/3
    @param p1                   empty square 2/3
    @param p2                   empty square 3/3
    @param skipped              already passed?
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
inline int last3(Search *search, int alpha, int beta, uint_fast8_t p0, uint_fast8_t p1, uint_fast8_t p2, bool skipped){
    ++search->n_nodes;
    #if USE_SEARCH_STATISTICS
        ++search->n_nodes_discs[search->n_discs];
    #endif
    #if USE_END_PO
        if (!skipped){
            const bool p0_parity = (search->parity & cell_div4[p0]) > 0;
            const bool p1_parity = (search->parity & cell_div4[p1]) > 0;
            const bool p2_parity = (search->parity & cell_div4[p2]) > 0;
            #if LAST_PO_OPTIMIZE
                if (!(p0_parity & p1_parity & p2_parity)){ // check necessity of sorting
                    if (p1_parity){ // 0 - 1 - 0
                        std::swap(p0, p1);
                    } else if (p2_parity){ // 0 - 0 - 1
                        std::swap(p0, p2);
                    }
                }
            #else
                if (!p0_parity && p1_parity && p2_parity){
                    std::swap(p0, p2);
                } else if (!p0_parity && !p1_parity && p2_parity){
                    std::swap(p0, p2);
                } else if (!p0_parity && p1_parity && !p2_parity){
                    std::swap(p0, p1);
                } else if (p0_parity && !p1_parity && p2_parity){
                    std::swap(p1, p2);
                }
            #endif
        }
    #endif
    int v = -SCORE_INF;
    uint64_t flip;
    if (bit_around[p0] & search->board.opponent){
        flip = calc_flip(&search->board, p0);
        if (flip){
            search->move_cell(flip, p0);
                v = -last2(search, -beta, -alpha, p1, p2, false);
            search->undo_cell(flip, p0);
            if (alpha < v){
                if (beta <= v)
                    return v;
                alpha = v;
            }
        }
    }
    int g;
    if (bit_around[p1] & search->board.opponent){
        flip = calc_flip(&search->board, p1);
        if (flip){
            search->move_cell(flip, p1);
                g = -last2(search, -beta, -alpha, p0, p2, false);
            search->undo_cell(flip, p1);
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
    if (bit_around[p2] & search->board.opponent){
        flip = calc_flip(&search->board, p2);
        if (flip){
            search->move_cell(flip, p2);
                g = -last2(search, -beta, -alpha, p0, p1, false);
            search->undo_cell(flip, p2);
            if (v < g)
                return g;
            else
                return v;
        }
    }
    if (v == -SCORE_INF){
        if (skipped)
            v = end_evaluate(&search->board, 3);
        else{
            search->board.pass();
                v = -last3(search, -beta, -alpha, p0, p1, p2, true);
            search->board.pass();
        }
    }
    return v;
}

/*
    @brief Get a final score with last 4 empties

    Only with parity-based ordering.

    @param search               search information
    @param alpha                alpha value
    @param beta                 beta value
    @param p0                   empty square 1/4
    @param p1                   empty square 2/4
    @param p2                   empty square 3/4
    @param p3                   empty square 4/4
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
        1 - 1 - 1 - 1
        0 - 0 - 0 - 0
        1 - 1 - 0 - 0 > need to sort
        1 - 1 - 1 - 1
*/
inline int last4(Search *search, int alpha, int beta, uint_fast8_t p0, uint_fast8_t p1, uint_fast8_t p2, uint_fast8_t p3, bool skipped){
    ++search->n_nodes;
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
        if (!skipped){
            const bool p0_parity = (search->parity & cell_div4[p0]) > 0;
            const bool p1_parity = (search->parity & cell_div4[p1]) > 0;
            const bool p2_parity = (search->parity & cell_div4[p2]) > 0;
            //const bool p3_parity = (search->parity & cell_div4[p3]) > 0;
            #if LAST_PO_OPTIMIZE
                if ((p0_parity | p1_parity | p2_parity) && !(p0_parity & p1_parity & p2_parity)){ // need to see only 3 squares to check necessity of sorting
                    if (p0_parity && !p1_parity){ // 1 - 0 - ? - ?
                        if (p2_parity){ // 1 - 0 - 1 - 0
                            std::swap(p1, p2);
                        } else{ // 1 - 0 - 0 - 1
                            std::swap(p1, p3);
                        }
                    } else if (!p0_parity){ // 0 - ? - ? - ?
                        if (p1_parity){ // 0 - 1 - ? - ?
                            if (p2_parity){ // 0 - 1 - 1 - 0
                                std::swap(p0, p2);
                            } else{ // 0 - 1 - 0 - 1
                                std::swap(p0, p3);
                            }
                        } else{ // 0 - 0 - 1 - 1
                            std::swap(p0, p2);
                            std::swap(p1, p3);
                        }
                    }
                }
            #else
                if (!p0_parity && p1_parity && p2_parity && p3_parity){
                    std::swap(p0, p3);
                } else if (!p0_parity && !p1_parity && p2_parity && p3_parity){
                    std::swap(p0, p2);
                    std::swap(p1, p3);
                } else if (!p0_parity && p1_parity && !p2_parity && p3_parity){
                    std::swap(p0, p3);
                } else if (!p0_parity && p1_parity && p2_parity && !p3_parity){
                    std::swap(p0, p2);
                } else if (!p0_parity && !p1_parity && !p2_parity && p3_parity){
                    std::swap(p0, p3);
                } else if (!p0_parity && !p1_parity && p2_parity && !p3_parity){
                    std::swap(p0, p2);
                } else if (!p0_parity && p1_parity && !p2_parity && !p3_parity){
                    std::swap(p0, p1);
                } else if (p0_parity && !p1_parity && p2_parity && p3_parity){
                    std::swap(p1, p3);
                } else if (p0_parity && !p1_parity && !p2_parity && p3_parity){
                    std::swap(p1, p3);
                } else if (p0_parity && !p1_parity && p2_parity && !p3_parity){
                    std::swap(p1, p2);
                } else if (p0_parity && p1_parity && !p2_parity && p3_parity){
                    std::swap(p2, p3);
                }
            #endif
        }
    #endif
    int v = -SCORE_INF;
    uint64_t flip;
    if (bit_around[p0] & search->board.opponent){
        flip = calc_flip(&search->board, p0);
        if (flip){
            search->move_cell(flip, p0);
                v = -last3(search, -beta, -alpha, p1, p2, p3, false);
            search->undo_cell(flip, p0);
            if (alpha < v){
                if (beta <= v)
                    return v;
                alpha = v;
            }
        }
    }
    int g;
    if (bit_around[p1] & search->board.opponent){
        flip = calc_flip(&search->board, p1);
        if (flip){
            search->move_cell(flip, p1);
                g = -last3(search, -beta, -alpha, p0, p2, p3, false);
            search->undo_cell(flip, p1);
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
    if (bit_around[p2] & search->board.opponent){
        flip = calc_flip(&search->board, p2);
        if (flip){
            search->move_cell(flip, p2);
                g = -last3(search, -beta, -alpha, p0, p1, p3, false);
            search->undo_cell(flip, p2);
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
    if (bit_around[p3] & search->board.opponent){
        flip = calc_flip(&search->board, p3);
        if (flip){
            search->move_cell(flip, p3);
                g = -last3(search, -beta, -alpha, p0, p1, p2, false);
            search->undo_cell(flip, p3);
            if (v < g)
                return g;
        }
    }
    if (v == -SCORE_INF){
        if (skipped)
            v = end_evaluate(&search->board, 4);
        else{
            search->board.pass();
                v = -last4(search, -beta, -alpha, p0, p1, p2, p3, true);
            search->board.pass();
        }
    }
    return v;
}

inline int last4_wrapper(Search *search, int alpha, int beta){
    uint_fast8_t p0, p1, p2, p3;
    p0 = search->empty_list[0].next->cell;
    p1 = search->empty_list[0].next->next->cell;
    p2 = search->empty_list[0].next->next->next->cell;
    p3 = search->empty_list[0].next->next->next->cell;
    return last4(search, alpha, beta, p0, p1, p2, p3, false);
}

#if USE_NEGA_ALPHA_END_FAST // CANNOT BE COMPILED
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
                    uint64_t empties;
                    uint_fast8_t p0, p1, p2, p3;
                    legal_copy = legal & legal_mask;
                    for (cell = first_bit(&legal_copy); legal_copy; cell = next_bit(&legal_copy)){
                        flip = calc_flip(&search->board, cell);
                        search->move(&flip);
                            empties = ~(search->board.player | search->board.opponent);
                            p0 = first_bit(&empties);
                            p1 = next_bit(&empties);
                            p2 = next_bit(&empties);
                            p3 = next_bit(&empties);
                            g = -last4(search, -beta, -alpha, p0, p1, p2, p3, false, searching);
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
                        flip = calc_flip(&search->board, cell);
                        search->move(&flip);
                            empties = ~(search->board.player | search->board.opponent);
                            p0 = first_bit(&empties);
                            p1 = next_bit(&empties);
                            p2 = next_bit(&empties);
                            p3 = next_bit(&empties);
                            g = -last4(search, -beta, -alpha, p0, p1, p2, p3, false, searching);
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
                        flip = calc_flip(&search->board, cell);
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
                        flip = calc_flip(&search->board, cell);
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
                    uint64_t empties;
                    uint_fast8_t p0, p1, p2, p3;
                    for (cell = first_bit(&legal); legal; cell = next_bit(&legal)){
                        flip = calc_flip(&search->board, cell);
                        search->move(&flip);
                            empties = ~(search->board.player | search->board.opponent);
                            p0 = first_bit(&empties);
                            p1 = next_bit(&empties);
                            p2 = next_bit(&empties);
                            p3 = next_bit(&empties);
                            g = -last4(search, -beta, -alpha, p0, p1, p2, p3, false, searching);
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
                        flip = calc_flip(&search->board, cell);
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
                uint64_t empties;
                uint_fast8_t p0, p1, p2, p3;
                for (cell = first_bit(&legal); legal; cell = next_bit(&legal)){
                    flip = calc_flip(&search->board, cell);
                    search->move(&flip);
                        empties = ~(search->board.player | search->board.opponent);
                        p0 = first_bit(&empties);
                        p1 = next_bit(&empties);
                        p2 = next_bit(&empties);
                        p3 = next_bit(&empties);
                        g = -last4(search, -beta, -alpha, p0, p1, p2, p3, false);
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
                    flip = calc_flip(&search->board, cell);
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

#if USE_NEGA_ALPHA_END // CANNOT BE COMPILED
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
                uint64_t empties = ~(search->board.player | search->board.opponent);
                uint_fast8_t p0 = first_bit(&empties);
                uint_fast8_t p1 = next_bit(&empties);
                uint_fast8_t p2 = next_bit(&empties);
                uint_fast8_t p3 = next_bit(&empties);
                return last4(search, alpha, beta, p0, p1, p2, p3, false, searching);
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
            for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal))
                calc_flip(&move_list[idx++].flip, &search->board, cell);
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