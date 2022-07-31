#pragma once
#include <iostream>
#include "setting.hpp"
#include "common.hpp"
#include "transpose_table.hpp"
#include "search.hpp"
#include "midsearch.hpp"
#include "endsearch.hpp"
#include "thread_pool.hpp"

//#define YBWC_SPLIT_DIV 7
#define YBWC_MID_SPLIT_MIN_DEPTH 6
#define YBWC_MID_SPLIT_MAX_DEPTH 22
#define YBWC_END_SPLIT_MIN_DEPTH 11
//#define YBWC_MAX_SPLIT_COUNT 3
//#define YBWC_PC_OFFSET 3
#define YBWC_ORDERING_MAX_OFFSET 8
#define YBWC_OFFSET_DIV_DEPTH 32
#define YBWC_ORDERING_MAX_OFFSET_END 6
#define YBWC_OFFSET_DIV_DEPTH_END 40

inline int depth_to_offset(const int depth){
    return depth * YBWC_ORDERING_MAX_OFFSET / YBWC_OFFSET_DIV_DEPTH;
}

inline int depth_to_offset_end(const int depth){
    return depth * YBWC_ORDERING_MAX_OFFSET_END / YBWC_OFFSET_DIV_DEPTH_END;
}

int nega_alpha_ordering(Search *search, int alpha, int beta, int depth, bool skipped, uint64_t legal, bool is_end_search, const bool *searching);
int nega_alpha_end(Search *search, int alpha, int beta, bool skipped, uint64_t legal, const bool *searching);
int nega_scout(Search *search, int alpha, int beta, int depth, bool skipped, uint64_t legal, bool is_end_search, const bool *searching);

Parallel_task ybwc_do_task(uint64_t player, uint64_t opponent, uint_fast8_t n, uint_fast8_t parity, 
        bool use_mpc, double mpct, 
        int alpha, int beta, int depth, uint64_t legal, bool is_end_search, const bool *searching, int policy){
    Search search;
    search.board.player = player;
    search.board.opponent = opponent;
    search.n_discs = n;
    search.parity = parity;
    search.use_mpc = use_mpc;
    search.mpct = mpct;
    search.n_nodes = 0ULL;
    calc_features(&search);
    int g = -nega_alpha_ordering(&search, alpha, beta, depth, false, legal, is_end_search, searching);
    Parallel_task task;
    if (*searching)
        task.value = g;
    else
        task.value = SCORE_UNDEFINED;
    task.n_nodes = search.n_nodes;
    task.cell = policy;
    return task;
}

inline bool ybwc_split(const Search *search, const Flip *flip, int alpha, int beta, const int depth, uint64_t legal, bool is_end_search, const bool *searching, int policy, const int pv_idx, const int canput, const int split_count, vector<future<Parallel_task>> &parallel_tasks, const int first_val, const int last_val, const bool worth_searching){
    if (!worth_searching || 
        (pv_idx > 0 && 
        depth >= YBWC_MID_SPLIT_MIN_DEPTH)){
        if (thread_pool.n_idle()){
            parallel_tasks.emplace_back(thread_pool.push(bind(&ybwc_do_task, 
                search->board.player, search->board.opponent, search->n_discs, search->parity, 
                search->use_mpc, search->mpct, 
                alpha, beta, depth, legal, is_end_search, searching, policy)));
            return true;
        }
    }
    return false;
}

inline void ybwc_get_end_tasks(Search *search, vector<future<Parallel_task>> &parallel_tasks, int *v, int *best_move, int *alpha){
    Parallel_task got_task;
    for (future<Parallel_task> &task: parallel_tasks){
        if (task.valid()){
            if (task.wait_for(chrono::seconds(0)) == future_status::ready){
                got_task = task.get();
                if (*v < got_task.value){
                    *v = got_task.value;
                    *best_move = got_task.cell;
                }
                search->n_nodes += got_task.n_nodes;
            }
        }
    }
    *alpha = max((*alpha), (*v));
}

inline void ybwc_wait_all(Search *search, vector<future<Parallel_task>> &parallel_tasks, int *v, int *best_move, int *alpha, int beta, bool *searching){
    ybwc_get_end_tasks(search, parallel_tasks, v, best_move, alpha);
    if (beta <= (*alpha))
        *searching = false;
    Parallel_task got_task;
    for (future<Parallel_task> &task: parallel_tasks){
        if (task.valid()){
            got_task = task.get();
            search->n_nodes += got_task.n_nodes;
            if ((*v) < got_task.value && (*searching)){
                *best_move = got_task.cell;
                *v = got_task.value;
                if (beta <= (*v))
                    *searching = false;
            }
        }
    }
    *alpha = max((*alpha), (*v));
}

inline void ybwc_wait_all(Search *search, vector<future<Parallel_task>> &parallel_tasks){
    Parallel_task got_task;
    for (future<Parallel_task> &task: parallel_tasks){
        if (task.valid()){
            got_task = task.get();
            search->n_nodes += got_task.n_nodes;
        }
    }
}
