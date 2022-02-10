#pragma once
#include <iostream>
#include "setting.hpp"
#include "common.hpp"
#include "search.hpp"
#include "midsearch.hpp"
#include "thread_pool.hpp"

#define YBWC_SPLIT_DIV 6
#define YBWC_SPLIT_MIN_DEPTH 5
#define YBWC_MAX_SPLIT_COUNT 3

inline void ybwc_do_task(int id, Search *search, int alpha, int beta, int depth){
    nega_alpha_ordering(search, alpha, beta, depth);
}

inline bool ybwc_split(Search *search, int alpha, int beta, const int depth, const int pv_idx, const int canput, const int split_count){
    if (pv_idx > canput / YBWC_SPLIT_DIV && pv_idx < canput - 1 && depth >= YBWC_SPLIT_MIN_DEPTH && split_count < YBWC_MAX_SPLIT_COUNT){
        if (thread_pool.n_idle()){
            Search copy_search;
            copy_search.board = search->board;
            copy_search.parent_transpose_table = search->parent_transpose_table;
            copy_search.child_transpose_table = search->child_transpose_table;
            copy_search.skipped = search->skipped;
            copy_search.use_mpc = search->use_mpc;
            copy_search.mpct = search->mpct;
            copy_search.vacant_list = search->vacant_list;
            copy_search.n_nodes = search->n_nodes;
            thread_pool.push(ybwc_do_task, copy_search, alpha, beta, depth);
            return true;
        }
    }
    return false;
}