/*
    Egaroucid Project

    @date 2021-2022
    @author Takuto Yamana (a.k.a Nyanyan)
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
#include "transpose_table.hpp"
#include "endsearch.hpp"
#include "move_ordering.hpp"
#include "probcut.hpp"
#include "thread_pool.hpp"
#include "ybwc.hpp"
#include "util.hpp"
#include "stability.hpp"
#include "midsearch_common.hpp"

using namespace std;

inline bool ybwc_split_nws(const Search *search, const Flip *flip, int alpha, int depth, uint64_t legal, bool is_end_search, const bool *searching, uint_fast8_t policy, const int canput, const int pv_idx, const int split_count, vector<future<Parallel_task>> &parallel_tasks);
inline void ybwc_get_end_tasks_nws(Search *search, vector<future<Parallel_task>> &parallel_tasks, int *v);
inline void ybwc_get_end_tasks(Search *search, vector<future<Parallel_task>> &parallel_tasks, int *v, int *best_move);
inline void ybwc_wait_all_nws(Search *search, vector<future<Parallel_task>> &parallel_tasks, int *v, int alpha, bool *searching);
inline void ybwc_wait_all_nws(Search *search, vector<future<Parallel_task>> &parallel_tasks, int *v, int *best_move, int alpha, bool *searching);

#if MID_FAST_DEPTH > 1
    int nega_alpha_nws(Search *search, int alpha, int depth, bool skipped, const bool *searching){
        if (!global_searching || !(*searching))
            return SCORE_UNDEFINED;
        ++search->n_nodes;
        if (depth == 1)
            return nega_alpha_eval1(search, alpha, alpha + 1, skipped, searching);
        if (depth == 0)
            return mid_evaluate_diff(search);
        int v = -INF;
        uint64_t legal = search->board.get_legal();
        if (legal == 0ULL){
            if (skipped)
                return end_evaluate(&search->board);
            search->eval_feature_reversed ^= 1;
            search->board.pass();
                v = -nega_alpha_nws(search, -alpha - 1, depth, true, searching);
            search->board.pass();
            search->eval_feature_reversed ^= 1;
            return v;
        }
        Flip flip;
        int g;
        for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)){
            calc_flip(&flip, &search->board, cell);
            eval_move(search, &flip);
            search->move(&flip);
                g = -nega_alpha_nws(search, -alpha - 1, depth - 1, false, searching);
            search->undo(&flip);
            eval_undo(search, &flip);
            if (v < g){
                if (alpha < g)
                    return g;
                v = g;
            }
        }
        return v;
    }
#endif

int nega_alpha_ordering_nws(Search *search, int alpha, int depth, bool skipped, uint64_t legal, bool is_end_search, const bool *searching){
    if (!global_searching || !(*searching))
        return SCORE_UNDEFINED;
    if (is_end_search && depth <= MID_TO_END_DEPTH)
        return nega_alpha_end_nws(search, alpha, skipped, legal, searching);
    if (!is_end_search){
        #if MID_FAST_DEPTH > 1
            if (depth <= MID_FAST_DEPTH)
                return nega_alpha_nws(search, alpha, depth, skipped, searching);
        #else
            if (depth == 1)
                return nega_alpha_eval1(search, alpha, alpha + 1, skipped, searching);
            if (depth == 0)
                return mid_evaluate_diff(search);
        #endif
    }
    ++search->n_nodes;
    uint32_t hash_code = search->board.hash();
    #if MID_TO_END_DEPTH < USE_TT_DEPTH_THRESHOLD
        int l = -INF, u = INF;
        if (search->n_discs <= HW2 - USE_TT_DEPTH_THRESHOLD){
            parent_transpose_table.get(&search->board, hash_code, &l, &u, search->mpct, depth);
            if (u == l)
                return u;
            if (l < alpha && u <= alpha)
                return u;
            if (alpha < l && alpha + 1 < u)
                return l;
        }
    #else
        int l, u;
        parent_transpose_table.get(&search->board, hash_code, &l, &u, search->mpct, depth);
        if (u == l)
            return u;
        if (l < alpha && u <= alpha)
            return u;
        if (alpha < l && alpha + 1 < u)
            return l;
    #endif
    if (legal == LEGAL_UNDEFINED)
        legal = search->board.get_legal();
    int v = -INF;
    if (legal == 0ULL){
        if (skipped)
            return end_evaluate(&search->board);
        search->eval_feature_reversed ^= 1;
        search->board.pass();
            v = -nega_alpha_ordering_nws(search, -alpha - 1, depth, true, LEGAL_UNDEFINED, is_end_search, searching);
        search->board.pass();
        search->eval_feature_reversed ^= 1;
        return v;
    }
    #if USE_MID_MPC
        if (search->use_mpc){
            if (mpc(search, alpha, alpha + 1, depth, legal, is_end_search, &v, searching))
                return v;
        }
    #endif
    int best_move = child_transpose_table.get(&search->board, hash_code);
    if (best_move != TRANSPOSE_TABLE_UNDEFINED){
        if (1 & (legal >> best_move)){
            Flip flip_best;
            calc_flip(&flip_best, &search->board, best_move);
            eval_move(search, &flip_best);
            search->move(&flip_best);
                v = -nega_alpha_ordering_nws(search, -alpha - 1, depth - 1, false, LEGAL_UNDEFINED, is_end_search, searching);
            search->undo(&flip_best);
            eval_undo(search, &flip_best);
            if (alpha < v)
                return v;
            legal ^= 1ULL << best_move;
        } else
            best_move = TRANSPOSE_TABLE_UNDEFINED;
    }
    int g;
    if (legal){
        #if USE_ALL_NODE_PREDICTION
            const bool seems_to_be_all_node = is_like_all_node(search, alpha, depth, LEGAL_UNDEFINED, is_end_search, searching);
        #else
            constexpr bool seems_to_be_all_node = false;
        #endif
        const int canput = pop_count_ull(legal);
        vector<Flip_value> move_list(canput);
        int idx = 0;
        for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal))
            calc_flip(&move_list[idx++].flip, &search->board, cell);
        move_list_evaluate(search, move_list, depth, alpha, alpha + 1, is_end_search, searching);
        if (search->use_multi_thread){
            int pv_idx = 0, split_count = 0;
            if (best_move != TRANSPOSE_TABLE_UNDEFINED)
                pv_idx = 1;
            vector<future<Parallel_task>> parallel_tasks;
            bool n_searching = true;
            for (int move_idx = 0; move_idx < canput; ++move_idx){
                swap_next_best_move(move_list, move_idx, canput);
                eval_move(search, &move_list[move_idx].flip);
                search->move(&move_list[move_idx].flip);
                    if (ybwc_split_nws(search, &move_list[move_idx].flip, -alpha - 1, depth - 1, move_list[move_idx].n_legal, is_end_search, &n_searching, move_list[move_idx].flip.pos, canput, pv_idx++, seems_to_be_all_node, split_count, parallel_tasks)){
                        ++split_count;
                    } else{
                        g = -nega_alpha_ordering_nws(search, -alpha - 1, depth - 1, false, move_list[move_idx].n_legal, is_end_search, searching);
                        if (v < g){
                            v = g;
                            if (alpha < v){
                                search->undo(&move_list[move_idx].flip);
                                eval_undo(search, &move_list[move_idx].flip);
                                break;
                            }
                        }
                        if (split_count){
                            ybwc_get_end_tasks(search, parallel_tasks, &v, &best_move);
                            if (alpha < v){
                                search->undo(&move_list[move_idx].flip);
                                eval_undo(search, &move_list[move_idx].flip);
                                break;
                            }
                        }
                    }
                search->undo(&move_list[move_idx].flip);
                eval_undo(search, &move_list[move_idx].flip);
            }
            if (split_count){
                if (alpha < v || !(*searching)){
                    n_searching = false;
                    ybwc_wait_all(search, parallel_tasks);
                } else
                    ybwc_wait_all_nws(search, parallel_tasks, &v, &best_move, alpha, &n_searching);
            }
        } else{
            for (int move_idx = 0; move_idx < canput; ++move_idx){
                swap_next_best_move(move_list, move_idx, canput);
                eval_move(search, &move_list[move_idx].flip);
                search->move(&move_list[move_idx].flip);
                    g = -nega_alpha_ordering_nws(search, -alpha - 1, depth - 1, false, move_list[move_idx].n_legal, is_end_search, searching);
                search->undo(&move_list[move_idx].flip);
                eval_undo(search, &move_list[move_idx].flip);
                if (v < g){
                    v = g;
                    if (alpha < v)
                        break;
                }
            }
        }
    }
    register_tt_nws(search, depth, hash_code, alpha, v, best_move, l, u, searching);
    return v;
}
