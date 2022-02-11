#pragma once
#include <iostream>
#include "setting.hpp"
#include "common.hpp"
#include "transpose_table.hpp"
#include "search.hpp"
#include "midsearch.hpp"
#include "endsearch.hpp"
#include "thread_pool.hpp"

#define YBWC_SPLIT_DIV 5
#define YBWC_MID_SPLIT_MIN_DEPTH 6
#define YBWC_END_SPLIT_MIN_DEPTH 6
#define YBWC_MAX_SPLIT_COUNT 3

int nega_alpha_ordering(Search *search, int alpha, int beta, int depth, bool is_end_search);
int nega_alpha_end(Search *search, int alpha, int beta);
int mtd(Search *search, int l, int u, int g, int depth, bool is_end_search);

inline bool mpc_higher(Search *search, int beta, int depth);
inline bool mpc_lower(Search *search, int alpha, int depth);
inline bool mpc_end_higher(Search *search, int beta, int val);
inline bool mpc_end_lower(Search *search, int alpha, int val);

inline pair<int, unsigned long long> ybwc_do_task(int id, Search search, int alpha, int beta, int depth, bool is_end_search, int policy){
    int hash_code = search.board.hash() & TRANSPOSE_TABLE_MASK;
    int g = -nega_alpha_ordering(&search, alpha, beta, depth, is_end_search);
    child_transpose_table.reg(&search.board, hash_code, policy, g);
    return make_pair(g, search.n_nodes);
}

inline bool ybwc_split(Search *search, int alpha, int beta, const int depth, bool is_end_search, int policy, const int pv_idx, const int canput, const int split_count, vector<future<pair<int, unsigned long long>>> &parallel_tasks){
    if (pv_idx > canput / YBWC_SPLIT_DIV /* && pv_idx < canput - 1 */ && depth >= YBWC_MID_SPLIT_MIN_DEPTH /* && split_count < YBWC_MAX_SPLIT_COUNT */ ){
        if (thread_pool.n_idle()){
            /*
            if (search->use_mpc && is_end_search && END_MPC_MIN_DEPTH <= depth && depth <= END_MPC_MAX_DEPTH){
                int val = mid_evaluate(&search->board);
                if (mpc_end_lower(search, alpha, val))
                    return false;
                if (mpc_end_higher(search, beta, val))
                    return false;
            }
            */
            Search copy_search;
            search->board.copy(&copy_search.board);
            copy_search.skipped = search->skipped;
            copy_search.use_mpc = search->use_mpc;
            copy_search.mpct = search->mpct;
            copy_search.vacant_list = search->vacant_list;
            copy_search.n_nodes = 0;
            parallel_tasks.emplace_back(thread_pool.push(ybwc_do_task, copy_search, alpha, beta, depth, is_end_search, policy));
            return true;
        }
    }
    return false;
}

inline pair<int, unsigned long long> ybwc_do_task_end(int id, Search search, int alpha, int beta, int policy){
    int hash_code = search.board.hash() & TRANSPOSE_TABLE_MASK;
    int g = -nega_alpha_end(&search, alpha, beta);
    child_transpose_table.reg(&search.board, hash_code, policy, g);
    return make_pair(g, search.n_nodes);
}

inline bool ybwc_split_end(Search *search, int alpha, int beta, int policy, const int pv_idx, const int canput, const int split_count, vector<future<pair<int, unsigned long long>>> &parallel_tasks){
    if (pv_idx > canput / YBWC_SPLIT_DIV /* && pv_idx < canput - 1 */ && HW2 - search->board.n >= YBWC_END_SPLIT_MIN_DEPTH /* && split_count < YBWC_MAX_SPLIT_COUNT */ ){
        if (thread_pool.n_idle()){
            /*
            if (search->use_mpc){
                int depth = HW2 - search->board.n;
                if (END_MPC_MIN_DEPTH <= depth && depth <= END_MPC_MAX_DEPTH){
                    int val = mid_evaluate(&search->board);
                    if (mpc_end_lower(search, alpha, val))
                        return false;
                    if (mpc_end_higher(search, beta, val))
                        return false;
                }
            }
            */
            Search copy_search;
            search->board.copy(&copy_search.board);
            copy_search.skipped = search->skipped;
            copy_search.use_mpc = search->use_mpc;
            copy_search.mpct = search->mpct;
            copy_search.vacant_list = search->vacant_list;
            copy_search.n_nodes = 0;
            parallel_tasks.emplace_back(thread_pool.push(ybwc_do_task_end, copy_search, alpha, beta, policy));
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

inline pair<int, unsigned long long> parallel_mtd(int id, Search search, int alpha, int beta, int expected_value, int depth, bool is_end_search){
    search.n_nodes = 0;
    int g = -mtd(&search, alpha, beta, expected_value, depth, is_end_search);
    return make_pair(g, search.n_nodes);
}