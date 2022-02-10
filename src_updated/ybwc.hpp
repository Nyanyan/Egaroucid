#pragma once
#include <iostream>
#include "setting.hpp"
#include "common.hpp"
#include "transpose_table.hpp"
#include "search.hpp"
#include "midsearch.hpp"
#include "thread_pool.hpp"

#define YBWC_SPLIT_DIV 6
#define YBWC_SPLIT_MIN_DEPTH 5
#define YBWC_MAX_SPLIT_COUNT 1000

int nega_alpha_ordering(Search *search, int alpha, int beta, int depth);

inline pair<int, unsigned long long> ybwc_do_task(int id, Search search, int alpha, int beta, int depth, int policy){
    int hash_code = search.board.hash() & TRANSPOSE_TABLE_MASK;
    int g = -nega_alpha_ordering(&search, alpha, beta, depth);
    child_transpose_table.reg(&search.board, hash_code, policy, g);
    return make_pair(g, search.n_nodes);
}

inline bool ybwc_split(Search *search, int alpha, int beta, const int depth, int policy, const int pv_idx, const int canput, const int split_count, vector<future<pair<int, unsigned long long>>> &parallel_tasks){
    if (pv_idx > canput / YBWC_SPLIT_DIV && pv_idx < canput - 1 && depth >= YBWC_SPLIT_MIN_DEPTH && split_count < YBWC_MAX_SPLIT_COUNT){
        if (thread_pool.n_idle()){
            Search copy_search;
            search->board.copy(&copy_search.board);
            copy_search.skipped = search->skipped;
            copy_search.use_mpc = search->use_mpc;
            copy_search.mpct = search->mpct;
            copy_search.vacant_list = search->vacant_list;
            copy_search.n_nodes = 0;
            parallel_tasks.emplace_back(thread_pool.push(ybwc_do_task, copy_search, alpha, beta, depth, policy));
            return true;
        }
    }
    return false;
}

inline int ybwc_wait(Search *search, vector<future<pair<int, unsigned long long>>> &parallel_tasks){
    int g = -INF;
    pair<int, unsigned long long> got_task;
    for (future<pair<int, unsigned long long>> &task: parallel_tasks){
        got_task = task.get();
        g = max(g, got_task.first);
        search->n_nodes += got_task.second;
    }
    return g;
}
