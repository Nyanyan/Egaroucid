/*
    Egaroucid Project

    @file midsearch_nws.hpp
        Search midgame with NWS (Null Window Search)
    @date 2021-2024
    @author Takuto Yamana
    @license GPL-3.0 license
*/

#pragma once
#include <iostream>
#include <algorithm>
#include <vector>
#include <future>
#include "setting.hpp"
#include "common.hpp"
#include "board.hpp"
#include "evaluate.hpp"
#include "search.hpp"
#include "transposition_table.hpp"
#include "endsearch.hpp"
#include "move_ordering.hpp"
#include "probcut.hpp"
#include "thread_pool.hpp"
#include "ybwc.hpp"
#include "util.hpp"
#include "stability.hpp"
#include "etc.hpp"

inline bool ybwc_split_nws(const Search *search, int alpha, int depth, uint64_t legal, bool is_end_search, const bool *searching, uint_fast8_t policy, const int pv_idx, const int move_idx, const int canput, const int running_count, std::vector<std::future<Parallel_task>> &parallel_tasks);
inline void ybwc_get_end_tasks(Search *search, std::vector<std::future<Parallel_task>> &parallel_tasks, int *v, int *best_move, int *running_count);
inline void ybwc_wait_all_nws(Search *search, std::vector<std::future<Parallel_task>> &parallel_tasks, int *v, int *best_move, int *running_count, int alpha, const bool *searching, bool *n_searching);

/*
    @brief Get a value with last move with Nega-Alpha algorithm (NWS)

    No move ordering. Just search it.

    @param search               search information
    @param alpha                alpha value (beta value is alpha + 1)
    @param skipped              already passed?
    @param searching            flag for terminating this search
    @return the value
*/
inline int nega_alpha_eval1_nws(Search *search, int alpha, bool skipped, const bool *searching){
    if (!global_searching || !(*searching))
        return SCORE_UNDEFINED;
    ++search->n_nodes;
    #if USE_SEARCH_STATISTICS
        ++search->n_nodes_discs[search->n_discs];
    #endif
    int v = -SCORE_INF;
    uint64_t legal = search->board.get_legal();
    if (legal == 0ULL){
        if (skipped)
            return end_evaluate(&search->board);
        search->pass();
            v = -nega_alpha_eval1_nws(search, -alpha - 1, true, searching);
        search->pass();
        return v;
    }
    int g;
    Flip flip;
    for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)){
        calc_flip(&flip, &search->board, cell);
        search->move(&flip);
            ++search->n_nodes;
            g = -mid_evaluate_diff(search);
        search->undo(&flip);
        ++search->n_nodes;
        if (v < g){
            if (alpha < g){
                return g;
            }
            v = g;
        }
    }
    return v;
}


/*
    @brief Get a value with given depth with Nega-Alpha algorithm (NWS)

    Search with move ordering for midgame NWS
    Parallel search (YBWC: Young Brothers Wait Concept) used.

    @param search               search information
    @param alpha                alpha value (beta value is alpha + 1)
    @param depth                remaining depth
    @param skipped              already passed?
    @param legal                for use of previously calculated legal bitboard
    @param is_end_search        search till the end?
    @param searching            flag for terminating this search
    @return the value
*/
int nega_alpha_ordering_nws(Search *search, int alpha, int depth, bool skipped, uint64_t legal, bool is_end_search, const bool *searching){
    if (!global_searching || !(*searching))
        return SCORE_UNDEFINED;
    if (is_end_search && depth <= MID_TO_END_DEPTH){
        return nega_alpha_end_nws(search, alpha, skipped, legal, searching);
    }
    if (!is_end_search){
        if (depth == 1)
            return nega_alpha_eval1_nws(search, alpha, skipped, searching);
        if (depth == 0){
            ++search->n_nodes;
            return mid_evaluate_diff(search);
        }
    }
    ++search->n_nodes;
    #if USE_SEARCH_STATISTICS
        ++search->n_nodes_discs[search->n_discs];
    #endif
    if (legal == LEGAL_UNDEFINED)
        legal = search->board.get_legal();
    int v = -SCORE_INF;
    if (legal == 0ULL){
        if (skipped)
            return end_evaluate(&search->board);
        search->pass();
            v = -nega_alpha_ordering_nws(search, -alpha - 1, depth, true, LEGAL_UNDEFINED, is_end_search, searching);
        search->pass();
        return v;
    }
    uint32_t hash_code = search->board.hash();
    uint_fast8_t moves[N_TRANSPOSITION_MOVES] = {TRANSPOSITION_TABLE_UNDEFINED, TRANSPOSITION_TABLE_UNDEFINED};
    if (transposition_cutoff_nws(search, hash_code, depth, alpha, &v, moves)){
        return v;
    }
    #if USE_MID_MPC
        if (mpc(search, alpha, alpha + 1, depth, legal, is_end_search, &v, searching)){
            //transposition_table.reg(search, hash_code, depth, alpha, alpha + 1, v, TRANSPOSITION_TABLE_UNDEFINED);
            return v;
        }
    #endif
    int best_move = TRANSPOSITION_TABLE_UNDEFINED;
    int g;
    const int canput = pop_count_ull(legal);
    std::vector<Flip_value> move_list(canput);
    int idx = 0;
    for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)){
        calc_flip(&move_list[idx].flip, &search->board, cell);
        if (move_list[idx].flip.flip == search->board.opponent)
            return SCORE_MAX;
        ++idx;
    }
    int etc_done_idx = 0;
    #if USE_MID_ETC
        if (depth >= MID_ETC_DEPTH){
            if (etc_nws(search, move_list, depth, alpha, &v, &etc_done_idx)){
                //transposition_table.reg(search, hash_code, depth, alpha, alpha + 1, v, TRANSPOSITION_TABLE_UNDEFINED);
                return v;
            }
        }
    #endif
    move_list_evaluate_nws(search, move_list, moves, depth, alpha, searching);
    #if USE_YBWC_NWS
        if (search->use_multi_thread && depth - 1 >= YBWC_MID_SPLIT_MIN_DEPTH){
            move_list_sort(move_list);
            if (move_list[0].flip.flip){
                search->move(&move_list[0].flip);
                    v = -nega_alpha_ordering_nws(search, -alpha - 1, depth - 1, false, move_list[0].n_legal, is_end_search, searching);
                search->undo(&move_list[0].flip);
                best_move = move_list[0].flip.pos;
                if (v <= alpha){
                    ybwc_search_young_brothers_nws(search, alpha, &v, &best_move, hash_code, depth, is_end_search, move_list, searching);
                }
            }
        } else{
    #endif
            for (int move_idx = 0; move_idx < canput - etc_done_idx && *searching; ++move_idx){
                swap_next_best_move(move_list, move_idx, canput);
                #if USE_MID_ETC
                    if (move_list[move_idx].flip.flip == 0ULL)
                        break;
                #endif
                if (search->need_to_see_tt_loop){
                    if (transposition_cutoff_nws_bestmove(search, hash_code, depth, alpha, &v, &best_move)){
                        break;
                    }
                }
                search->move(&move_list[move_idx].flip);
                    g = -nega_alpha_ordering_nws(search, -alpha - 1, depth - 1, false, move_list[move_idx].n_legal, is_end_search, searching);
                search->undo(&move_list[move_idx].flip);
                if (v < g){
                    v = g;
                    best_move = move_list[move_idx].flip.pos;
                    if (alpha < v)
                        break;
                }
            }
    #if USE_YBWC_NWS
        }
    #endif
    if (*searching && global_searching)
        transposition_table.reg(search, hash_code, depth, alpha, alpha + 1, v, best_move);
    return v;
}
