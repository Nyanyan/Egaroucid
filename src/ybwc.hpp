#pragma once
#include <iostream>
#include "setting.hpp"
#include "common.hpp"
#include "transpose_table.hpp"
#include "search.hpp"
#if USE_CUDA
    #include "cuda_midsearch.hpp"
#else
    #include "midsearch.hpp"
#endif
#include "endsearch.hpp"
#include "thread_pool.hpp"

//#define YBWC_SPLIT_DIV 7
#define YBWC_MID_SPLIT_MIN_DEPTH 6
//#define YBWC_END_SPLIT_MIN_DEPTH 6
//#define YBWC_MAX_SPLIT_COUNT 3
//#define YBWC_PC_OFFSET 3
#define YBWC_ORDERING_MAX_OFFSET 16
#define YBWC_OFFSET_DIV_DEPTH 32

inline int depth_to_offset(const int depth){
    return depth * YBWC_ORDERING_MAX_OFFSET / YBWC_OFFSET_DIV_DEPTH;
}

int nega_alpha_ordering(Search *search, int alpha, int beta, int depth, bool skipped, uint64_t legal, bool is_end_search, const bool *searching);
int nega_alpha_end(Search *search, int alpha, int beta, bool skipped, uint64_t legal, const bool *searching);
int nega_scout(Search *search, int alpha, int beta, int depth, bool skipped, uint64_t legal, bool is_end_search, const bool *searching);

inline pair<int, uint64_t> ybwc_do_task(Search search, int alpha, int beta, int depth, uint64_t legal, bool is_end_search, const bool *searching, int policy){
    //calc_features(&search);
    int g = -nega_alpha_ordering(&search, alpha, beta, depth, false, legal, is_end_search, searching);
    if (*searching)
        return make_pair(g, search.n_nodes);
    return make_pair(SCORE_UNDEFINED, search.n_nodes);
}

inline bool ybwc_split(Search *search, const Flip *flip, int alpha, int beta, const int depth, uint64_t legal, bool is_end_search, const bool *searching, int policy, const int pv_idx, const int canput, const int split_count, vector<future<pair<int, uint64_t>>> &parallel_tasks, const int first_val){
    if (pv_idx > 0 && 
        /* pv_idx > canput / YBWC_SPLIT_DIV && */ 
        /* pv_idx < canput - 1 && */ 
        depth >= YBWC_MID_SPLIT_MIN_DEPTH &&
        flip->value < first_val - depth_to_offset(depth)
        /* split_count < YBWC_MAX_SPLIT_COUNT */ ){
        if (thread_pool.n_idle()){
            Search copy_search;
            //eval_move(search, flip);
            search->board.move(flip);
                search->board.copy(&copy_search.board);
                for (int i = 0; i < N_SYMMETRY_PATTERNS; ++i)
                    copy_search.eval_features[i] = search->eval_features[i];
                copy_search.eval_feature_reversed = search->eval_feature_reversed;
            search->board.undo(flip);
            //eval_undo(search, flip);
            copy_search.use_mpc = search->use_mpc;
            copy_search.mpct = search->mpct;
            copy_search.n_nodes = 0;
            //copy_search.p = search->p;
            parallel_tasks.emplace_back(thread_pool.push(bind(&ybwc_do_task, copy_search, alpha, beta, depth, legal, is_end_search, searching, policy)));
            return true;
        }
    }
    return false;
}

inline bool ybwc_split_without_move(Search *search, const Flip *flip, int alpha, int beta, const int depth, uint64_t legal, bool is_end_search, const bool *searching, int policy, const int pv_idx, const int canput, const int split_count, vector<future<pair<int, uint64_t>>> &parallel_tasks, const int first_val){
    if (pv_idx > 0 && 
        /* pv_idx > canput / YBWC_SPLIT_DIV && */ 
        /* pv_idx < canput - 1 && */ 
        depth >= YBWC_MID_SPLIT_MIN_DEPTH &&
        flip->value < first_val - depth_to_offset(depth)
        /* split_count < YBWC_MAX_SPLIT_COUNT */ ){
        if (thread_pool.n_idle()){
            Search copy_search;
            search->board.copy(&copy_search.board);
            for (int i = 0; i < N_SYMMETRY_PATTERNS; ++i)
                copy_search.eval_features[i] = search->eval_features[i];
            copy_search.eval_feature_reversed = search->eval_feature_reversed;
            copy_search.use_mpc = search->use_mpc;
            copy_search.mpct = search->mpct;
            copy_search.n_nodes = 0;
            //copy_search.p = search->p;
            parallel_tasks.emplace_back(thread_pool.push(bind(&ybwc_do_task, copy_search, alpha, beta, depth, legal, is_end_search, searching, policy)));
            return true;
        }
    }
    return false;
}

inline bool ybwc_split_without_move_negascout(Search *search, const Flip *flip, int alpha, int beta, const int depth, uint64_t legal, bool is_end_search, const bool *searching, int policy, const int pv_idx, const int canput, const int split_count, vector<future<pair<int, uint64_t>>> &parallel_tasks){
    if (pv_idx > 3 && 
        /* pv_idx > canput / YBWC_SPLIT_DIV && */ 
        /* pv_idx < canput - 1 && */ 
        depth >= YBWC_MID_SPLIT_MIN_DEPTH /*&&*/
        /* split_count < YBWC_MAX_SPLIT_COUNT */ ){
        if (thread_pool.n_idle()){
            Search copy_search;
            search->board.copy(&copy_search.board);
            for (int i = 0; i < N_SYMMETRY_PATTERNS; ++i)
                copy_search.eval_features[i] = search->eval_features[i];
            copy_search.eval_feature_reversed = search->eval_feature_reversed;
            copy_search.use_mpc = search->use_mpc;
            copy_search.mpct = search->mpct;
            copy_search.n_nodes = 0;
            //copy_search.p = search->p;
            parallel_tasks.emplace_back(thread_pool.push(bind(&ybwc_do_task, copy_search, alpha, beta, depth, legal, is_end_search, searching, policy)));
            return true;
        }
    }
    return false;
}

inline int ybwc_wait_all(Search *search, vector<future<pair<int, uint64_t>>> &parallel_tasks){
    int g = -INF;
    pair<int, uint64_t> got_task;
    for (future<pair<int, uint64_t>> &task: parallel_tasks){
        got_task = task.get();
        g = max(g, got_task.first);
        search->n_nodes += got_task.second;
    }
    return g;
}

inline int ybwc_negascout_wait_all(Search *search, vector<future<pair<int, uint64_t>>> &parallel_tasks, vector<Flip> &flips, int before_alpha, int alpha, int beta, int depth, bool skipped, bool is_end_search, int *best_move){
    int v = alpha, g;
    pair<int, uint64_t> got_task;
    bool searching = true;
    for (int i = 0; i < (int)parallel_tasks.size(); ++i){
        got_task = parallel_tasks[i].get();
        g = got_task.first;
        search->n_nodes += got_task.second;
        if (before_alpha < g){
            v = max(v, g);
            search->board.move(&flips[i]);
                g = -nega_scout(search, -beta, -v, depth, skipped, flips[i].n_legal, is_end_search, &searching);
            search->board.undo(&flips[i]);
            if (v < g){
                v = g;
                *best_move = flips[i].pos;
            }
        }
    }
    return v;
}
