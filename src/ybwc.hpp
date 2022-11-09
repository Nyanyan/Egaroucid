/*
    Egaroucid Project

    @date 2021-2022
    @author Takuto Yamana (a.k.a Nyanyan)
    @license GPL-3.0 license
*/

#pragma once
#include <iostream>
#include "setting.hpp"
#include "common.hpp"
#include "search.hpp"
#include "midsearch.hpp"
#include "endsearch.hpp"
#include "parallel.hpp"
#include "thread_pool.hpp"

#define YBWC_MID_SPLIT_MIN_DEPTH 5
#define YBWC_END_SPLIT_MIN_DEPTH 13

int nega_alpha_ordering_nws(Search *search, int alpha, int depth, bool skipped, uint64_t legal, bool is_end_search, const bool *searching, bool *mpc_used);
#if USE_NEGA_ALPHA_END
    int nega_alpha_end(Search *search, int alpha, int beta, bool skipped, uint64_t legal, const bool *searching);
#endif
#if MID_TO_END_DEPTH < YBWC_END_SPLIT_MIN_DEPTH
    int nega_alpha_end_nws(Search *search, int alpha, bool skipped, uint64_t legal, const bool *searching);
#endif

Parallel_task ybwc_do_task_nws(int id, uint64_t player, uint64_t opponent, int_fast8_t n_discs, uint_fast8_t parity, bool use_mpc, double mpct, int alpha, int depth, uint64_t legal, bool is_end_search, uint_fast8_t policy, const bool *searching){
    Search search;
    search.board.player = player;
    search.board.opponent = opponent;
    search.n_discs = n_discs;
    search.parity = parity;
    search.use_mpc = use_mpc;
    search.mpct = mpct;
    search.n_nodes = 0ULL;
    search.use_multi_thread = depth > YBWC_MID_SPLIT_MIN_DEPTH;
    calc_features(&search);
    Parallel_task task;
    task.mpc_used = false;
    task.value = -nega_alpha_ordering_nws(&search, alpha, depth, false, legal, is_end_search, searching, &task.mpc_used);
    if (!(*searching))
        task.value = SCORE_UNDEFINED;
    task.n_nodes = search.n_nodes;
    task.cell = policy;
    return task;
}

inline bool ybwc_split_nws(const Search *search, const Flip *flip, int alpha, int depth, uint64_t legal, bool is_end_search, const bool *searching, uint_fast8_t policy, const int canput, const int pv_idx, const bool seems_to_be_all_node, const int split_count, vector<future<Parallel_task>> &parallel_tasks){
    if (thread_pool.n_idle() &&
        (pv_idx || seems_to_be_all_node)){
            bool pushed;
            parallel_tasks.emplace_back(thread_pool.push(&pushed, &ybwc_do_task_nws, search->board.player, search->board.opponent, search->n_discs, search->parity, search->use_mpc, search->mpct, alpha, depth, legal, is_end_search, policy, searching));
            if (!pushed)
                parallel_tasks.pop_back();
            return pushed;
    }
    return false;
}

#if MID_TO_END_DEPTH < YBWC_END_SPLIT_MIN_DEPTH
    #if USE_NEGA_ALPHA_END
        Parallel_task ybwc_do_task_end(int id, uint64_t player, uint64_t opponent, int_fast8_t n_discs, uint_fast8_t parity, int alpha, int beta, uint64_t legal, uint_fast8_t policy, const bool *searching){
            Search search;
            search.board.player = player;
            search.board.opponent = opponent;
            search.n_discs = n_discs;
            search.parity = parity;
            search.use_mpc = false;
            search.mpct = NOMPC;
            search.n_nodes = 0ULL;
            search.use_multi_thread = n_discs < HW2 - YBWC_END_SPLIT_MIN_DEPTH;
            int g = -nega_alpha_end(&search, alpha, beta, false, legal, searching);
            Parallel_task task;
            if (*searching)
                task.value = g;
            else
                task.value = SCORE_UNDEFINED;
            task.n_nodes = search.n_nodes;
            task.cell = policy;
            return task;
        }

        inline bool ybwc_split_end(const Search *search, const Flip *flip, int alpha, int beta, uint64_t legal, const bool *searching, uint_fast8_t policy, const int canput, const int pv_idx, const bool seems_to_be_all_node, const int split_count, vector<future<Parallel_task>> &parallel_tasks){
            if (thread_pool.n_idle() &&
                (pv_idx || seems_to_be_all_node) && 
                search->n_discs <= HW2 - YBWC_END_SPLIT_MIN_DEPTH){
                    bool pushed;
                    parallel_tasks.emplace_back(thread_pool.push(&pushed, &ybwc_do_task_end, search->board.player, search->board.opponent, search->n_discs, search->parity, alpha, beta, legal, policy, searching));
                    if (!pushed)
                        parallel_tasks.pop_back();
                    return pushed;
            }
            return false;
        }
    #endif

    Parallel_task ybwc_do_task_end_nws(int id, uint64_t player, uint64_t opponent, int_fast8_t n_discs, uint_fast8_t parity, int alpha, uint64_t legal, uint_fast8_t policy, const bool *searching){
        Search search;
        search.board.player = player;
        search.board.opponent = opponent;
        search.n_discs = n_discs;
        search.parity = parity;
        search.use_mpc = false;
        search.mpct = NOMPC;
        search.n_nodes = 0ULL;
        search.use_multi_thread = n_discs < HW2 - YBWC_END_SPLIT_MIN_DEPTH;
        int g = -nega_alpha_end_nws(&search, alpha, false, legal, searching);
        Parallel_task task;
        if (*searching)
            task.value = g;
        else
            task.value = SCORE_UNDEFINED;
        task.n_nodes = search.n_nodes;
        task.cell = policy;
        return task;
    }

    inline bool ybwc_split_end_nws(const Search *search, const Flip *flip, int alpha, uint64_t legal, const bool *searching, uint_fast8_t policy, const int canput, const int pv_idx, const bool seems_to_be_all_node, const int split_count, vector<future<Parallel_task>> &parallel_tasks){
        if (thread_pool.n_idle() &&
            (pv_idx || seems_to_be_all_node) && 
            search->n_discs <= HW2 - YBWC_END_SPLIT_MIN_DEPTH){
                bool pushed;
                parallel_tasks.emplace_back(thread_pool.push(&pushed, &ybwc_do_task_end_nws, search->board.player, search->board.opponent, search->n_discs, search->parity, alpha, legal, policy, searching));
                if (!pushed)
                    parallel_tasks.pop_back();
                return pushed;
        }
        return false;
    }
#endif

inline void ybwc_get_end_tasks(Search *search, vector<future<Parallel_task>> &parallel_tasks, int *v, int *best_move){
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
}

inline void ybwc_get_end_tasks(Search *search, vector<future<Parallel_task>> &parallel_tasks, int *v, int *best_move, int *alpha){
    ybwc_get_end_tasks(search, parallel_tasks, v, best_move);
    *alpha = max((*alpha), (*v));
}

inline void ybwc_get_end_tasks_nws(Search *search, vector<future<Parallel_task>> &parallel_tasks, int *v){
    Parallel_task got_task;
    for (future<Parallel_task> &task: parallel_tasks){
        if (task.valid()){
            if (task.wait_for(chrono::seconds(0)) == future_status::ready){
                got_task = task.get();
                if (*v < got_task.value)
                    *v = got_task.value;
                search->n_nodes += got_task.n_nodes;
            }
        }
    }
}

inline void ybwc_get_end_tasks_nws(Search *search, vector<future<Parallel_task>> &parallel_tasks, int *v, int *best_move){
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

inline void ybwc_wait_all_nws(Search *search, vector<future<Parallel_task>> &parallel_tasks, int *v, int alpha, bool *searching){
    ybwc_get_end_tasks_nws(search, parallel_tasks, v);
    if (alpha < (*v))
        *searching = false;
    Parallel_task got_task;
    for (future<Parallel_task> &task: parallel_tasks){
        if (task.valid()){
            got_task = task.get();
            search->n_nodes += got_task.n_nodes;
            if ((*v) < got_task.value && (*searching)){
                //*best_move = got_task.cell;
                *v = got_task.value;
                if (alpha < (*v))
                    *searching = false;
            }
        }
    }
}

inline void ybwc_wait_all_nws(Search *search, vector<future<Parallel_task>> &parallel_tasks, int *v, int *best_move, int alpha, bool *searching, bool *mpc_used){
    ybwc_get_end_tasks(search, parallel_tasks, v, best_move);
    if (alpha < (*v))
        *searching = false;
    Parallel_task got_task;
    for (future<Parallel_task> &task: parallel_tasks){
        if (task.valid()){
            got_task = task.get();
            search->n_nodes += got_task.n_nodes;
            *mpc_used |= got_task.mpc_used;
            if ((*v) < got_task.value && (*searching)){
                *best_move = got_task.cell;
                *v = got_task.value;
                if (alpha < (*v))
                    *searching = false;
            }
        }
    }
}