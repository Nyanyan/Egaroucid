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
#define YBWC_MAX_SPLIT_COUNT 3

int nega_alpha_ordering(Search *search, int alpha, int beta, int depth);

inline void ybwc_do_task(int id, Search *former_search, Search *search, int alpha, int beta, int depth, int policy, atomic<int> *state, atomic<int> *value){
    int hash_code = search->board.hash() & TRANSPOSE_TABLE_MASK;
    int g = -nega_alpha_ordering(search, alpha, beta, depth);
    child_transpose_table.reg(&search->board, hash_code, policy, g);
    state->store(state->load() + 1);
    former_search->n_nodes.store(former_search->n_nodes.load() + search->n_nodes);
    cerr << g << endl;
    if (value->load() < g)
        value->store(g);
}

inline bool ybwc_split(Search *search, int alpha, int beta, const int depth, int policy, const int pv_idx, const int canput, const int split_count, atomic<int> *state, atomic<int> *value){
    if (pv_idx > canput / YBWC_SPLIT_DIV && pv_idx < canput - 1 && depth >= YBWC_SPLIT_MIN_DEPTH && split_count < YBWC_MAX_SPLIT_COUNT){
        if (thread_pool.n_idle()){
            Search copy_search;
            search->board.copy(&copy_search.board);
            copy_search.skipped = search->skipped;
            copy_search.use_mpc = search->use_mpc;
            copy_search.mpct = search->mpct;
            copy_search.vacant_list = search->vacant_list;
            copy_search.n_nodes = 0;
            thread_pool.push(ybwc_do_task, search, &copy_search, alpha, beta, depth, policy, state, value);
            return true;
        }
    }
    return false;
}

inline void ybwc_wait(atomic<int> *state, const int split_count){
    while (state->load() < split_count);
}