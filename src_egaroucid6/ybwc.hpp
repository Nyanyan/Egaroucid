#pragma once
#include <iostream>
#include "setting.hpp"
#include "common.hpp"
#include "transpose_table.hpp"
#include "search.hpp"
#include "midsearch.hpp"
#include "endsearch.hpp"
#include "thread_pool.hpp"

#define YBWC_MID_SPLIT_MIN_DEPTH 6
#define YBWC_REMAIN_TASKS 1

int nega_alpha_ordering(Search *search, int alpha, int beta, int depth, bool skipped, uint64_t legal, bool is_end_search, const bool *searching);
int nega_alpha_end(Search *search, int alpha, int beta, bool skipped, uint64_t legal, const bool *searching);
int nega_scout(Search *search, int alpha, int beta, int depth, bool skipped, uint64_t legal, bool is_end_search, const bool *searching);

Parallel_task ybwc_do_task(Parallel_args arg, const bool *searching){
    Search search;
    search.board.player = arg.player;
    search.board.opponent = arg.opponent;
    search.n_discs = arg.n_discs;
    search.parity = arg.parity;
    search.use_mpc = arg.use_mpc;
    search.mpct = arg.mpct;
    search.n_nodes = 0ULL;
    calc_features(&search);
    int g = -nega_alpha_ordering(&search, arg.alpha, arg.beta, arg.depth, false, arg.legal, arg.is_end_search, searching);
    Parallel_task task;
    if (*searching)
        task.value = g;
    else
        task.value = SCORE_UNDEFINED;
    task.n_nodes = search.n_nodes;
    task.cell = arg.policy;
    return task;
}

inline bool ybwc_split(const Search *search, const Flip *flip, int alpha, int beta, const int depth, uint64_t legal, bool is_end_search, const bool *searching, int policy, const int canput, const int pv_idx, const int split_count, vector<int32_t> &parallel_tasks){
    if (pv_idx < canput - YBWC_REMAIN_TASKS && 
        pv_idx > 0 && 
        depth >= YBWC_MID_SPLIT_MIN_DEPTH){
            if (thread_pool.get_n_empty_thread()){
                Parallel_args arg;
                arg.player = search->board.player;
                arg.opponent = search->board.opponent;
                arg.n_discs = search->n_discs;
                arg.parity = search->parity;
                arg.use_mpc = search->use_mpc;
                arg.mpct = search->mpct;
                arg.alpha = alpha;
                arg.beta = beta;
                arg.depth = depth;
                arg.legal = legal;
                arg.is_end_search = is_end_search;
                arg.policy = policy;
                int32_t task_id = thread_pool.try_run(bind(&ybwc_do_task, arg, searching));
                if (task_id != -1){
                    parallel_tasks.emplace_back(task_id);
                    return true;
                }
            }
    }
    return false;
}

inline bool ybwc_get_end_tasks(Search *search, vector<int32_t> &parallel_tasks, int *v, int *best_move, int *alpha){
    Parallel_task got_task;
    bool task_remain = false;
    for (int i = 0; i < parallel_tasks.size(); ++i){
        if (parallel_tasks[i] != -1){
            if (thread_pool.get(parallel_tasks[i], &got_task)){
                if (*v < got_task.value){
                    *v = got_task.value;
                    *best_move = got_task.cell;
                }
                search->n_nodes += got_task.n_nodes;
                parallel_tasks[i] = -1;
            } else
                task_remain = true;
        }
    }
    *alpha = max((*alpha), (*v));
    return task_remain;
}

inline bool ybwc_get_end_tasks(Search *search, vector<int32_t> &parallel_tasks){
    Parallel_task got_task;
    bool task_remain = false;
    for (int i = 0; i < parallel_tasks.size(); ++i){
        if (parallel_tasks[i] != -1){
            if (thread_pool.get(parallel_tasks[i], &got_task)){
                search->n_nodes += got_task.n_nodes;
                parallel_tasks[i] = -1;
            } else
                task_remain = true;
        }
    }
    return task_remain;
}

inline void ybwc_wait_all(Search *search, vector<int32_t> &parallel_tasks, int *v, int *best_move, int *alpha, int beta, bool *searching){
    ybwc_get_end_tasks(search, parallel_tasks, v, best_move, alpha);
    if (beta <= (*alpha))
        *searching = false;
    while (ybwc_get_end_tasks(search, parallel_tasks, v, best_move, alpha));
}

inline void ybwc_wait_all(Search *search, vector<int32_t> &parallel_tasks){
    while (ybwc_get_end_tasks(search, parallel_tasks));
}
