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
#include "transpose_table.hpp"
#include "search.hpp"
#include "midsearch.hpp"
#include "endsearch.hpp"
#include "parallel.hpp"
#include "thread_pool.hpp"

#define YBWC_MID_SPLIT_MIN_DEPTH 6
//#define YBWC_REMAIN_TASKS 1

#define N_MAX_HELPER_TASKS 3

int nega_alpha_ordering(Search *search, int alpha, int beta, int depth, bool skipped, uint64_t legal, bool is_end_search, const bool *searching);
int nega_alpha_end(Search *search, int alpha, int beta, bool skipped, uint64_t legal, const bool *searching);
int nega_scout(Search *search, int alpha, int beta, int depth, bool skipped, uint64_t legal, bool is_end_search, const bool *searching);

Parallel_task ybwc_do_task(int id, uint64_t player, uint64_t opponent, int_fast8_t n_discs, uint_fast8_t parity, bool use_mpc, double mpct, int first_depth, int alpha, int beta, int depth, uint64_t legal, bool is_end_search, uint_fast8_t policy, const bool *searching, Parallel_node *parent){
    Search search;
    search.board.player = player;
    search.board.opponent = opponent;
    search.n_discs = n_discs;
    search.parity = parity;
    search.use_mpc = use_mpc;
    search.mpct = mpct;
    search.first_depth = first_depth;
    search.n_nodes = 0ULL;
    search.parallel_node.parent = parent;
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

inline bool get_helper(uint64_t player, uint64_t opponent, int_fast8_t n_discs, uint_fast8_t parity, bool use_mpc, double mpct, int first_depth, int alpha, int beta, int depth, uint64_t legal, bool is_end_search, uint_fast8_t policy, const bool *searching, Parallel_node *parent, Helper_task helper_tasks[], const int *helper_count){
    if (*helper_count >= N_MAX_HELPER_TASKS)
        return false;
    bool pushed = false;
    Parallel_node *p = parent;
    while (p != nullptr && !pushed){
        p->mtx.lock();
            if (p->is_waiting && !p->is_helping && !p->help_done){
                helper_tasks[*helper_count].node = p;
                p->help_task = bind(ybwc_do_task, -1, player, opponent, n_discs, parity, use_mpc, mpct, first_depth, alpha, beta, depth, legal, is_end_search, policy, searching, nullptr);
                p->help_res = &helper_tasks[*helper_count].task;
                p->is_helping = true;
                pushed = true;
                //cerr << "push " << p << " " << &helper_tasks[*helper_count].task << " " << *helper_count << endl;
                cerr << "push " << helper_tasks[*helper_count].node << endl;
            }
        p->mtx.unlock();
        p = p->parent;
        //p = nullptr;
    }
    return pushed;
}

inline bool ybwc_split(Search *search, const Flip *flip, int alpha, int beta, int depth, uint64_t legal, bool is_end_search, const bool *searching, uint_fast8_t policy, const int canput, const int pv_idx, const int split_count, int *helper_count, vector<future<Parallel_task>> &parallel_tasks, Helper_task helper_tasks[]){
    if (pv_idx > 0 && 
        depth >= YBWC_MID_SPLIT_MIN_DEPTH){
            if (get_helper(search->board.player, search->board.opponent, search->n_discs, search->parity, search->use_mpc, search->mpct, search->first_depth, alpha, beta, depth, legal, is_end_search, policy, searching, &search->parallel_node, helper_tasks, helper_count)){
                ++(*helper_count);
                return true;
            } else if (thread_pool.n_idle()){
                bool pushed;
                parallel_tasks.emplace_back(thread_pool.push(&pushed, &ybwc_do_task, search->board.player, search->board.opponent, search->n_discs, search->parity, search->use_mpc, search->mpct, search->first_depth, alpha, beta, depth, legal, is_end_search, policy, searching, &search->parallel_node));
                if (!pushed)
                    parallel_tasks.pop_back();
                return pushed;
            }
    }
    return false;
}

inline bool ybwc_get_end_tasks(Search *search, vector<future<Parallel_task>> &parallel_tasks, Helper_task helper_tasks[], int *v, int *best_move, int *alpha, int helper_count){
    bool task_remaining = false;
    Parallel_task got_task;
    for (future<Parallel_task> &task: parallel_tasks){
        if (task.valid()){
            if (task.wait_for(chrono::seconds(0)) == future_status::ready){
                got_task = task.get();
                if (*v < got_task.value){
                    *best_move = got_task.cell;
                    *v = got_task.value;
                }
                search->n_nodes += got_task.n_nodes;
            } else
                task_remaining = true;
        }
    }
    for (int i = 0; i < helper_count; ++i){
        if (helper_tasks[i].valid){
            if (helper_tasks[i].node->help_done){
                search->n_nodes += helper_tasks[i].task.n_nodes;
                if ((*v) < helper_tasks[i].task.value){
                    *best_move = helper_tasks[i].task.cell;
                    *v = helper_tasks[i].task.value;
                }
                helper_tasks[i].node->help_done = false;
                helper_tasks[i].node->is_helping = false;
                helper_tasks[i].valid = false;
            }
        }
    }
    *alpha = max((*alpha), (*v));
    return task_remaining;
}

inline void check_do_helper_task(Search *search){
    search->parallel_node.mtx.lock();
        if (search->parallel_node.is_helping && !search->parallel_node.help_done){
            *search->parallel_node.help_res = search->parallel_node.help_task();
            cerr << "done " << (&search->parallel_node) << endl;
            search->parallel_node.help_done = true;
        }
    search->parallel_node.mtx.unlock();
}

inline void ybwc_wait_all(Search *search, vector<future<Parallel_task>> &parallel_tasks, Helper_task helper_tasks[], int helper_count, int *v, int *best_move, int *alpha, int beta, bool *searching){
    bool task_remaining = ybwc_get_end_tasks(search, parallel_tasks, helper_tasks, v, best_move, alpha, helper_count);
    if (beta <= (*alpha))
        *searching = false;
    Parallel_task got_task;
    for (future<Parallel_task> &task: parallel_tasks){
        search->parallel_node.is_waiting = true;
        check_do_helper_task(search);
        search->parallel_node.is_waiting = false;
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
    int n_done = 0;
    int t = 0;
    while (n_done < helper_count){
        ++t;
        n_done = 0;
        for (int i = 0; i < helper_count; ++i){
            if (helper_tasks[i].valid){
                if (helper_tasks[i].node->help_done){
                    search->n_nodes += helper_tasks[i].task.n_nodes;
                    if ((*v) < helper_tasks[i].task.value && (*searching)){
                        *best_move = helper_tasks[i].task.cell;
                        *v = helper_tasks[i].task.value;
                        if (beta <= (*v))
                            *searching = false;
                    }
                    helper_tasks[i].node->help_done = false;
                    helper_tasks[i].node->is_helping = false;
                    helper_tasks[i].valid = false;
                } else if (t > 1000000000){
                    cerr << "jam " << helper_tasks[i].node << endl;
                    for(;;);
                }
            } else
                ++n_done;
        }
    }
    *alpha = max((*alpha), (*v));
}

inline void ybwc_wait_all(Search *search, vector<future<Parallel_task>> &parallel_tasks, Helper_task helper_tasks[], int helper_count){
    Parallel_task got_task;
    for (future<Parallel_task> &task: parallel_tasks){
        if (task.valid()){
            got_task = task.get();
            search->n_nodes += got_task.n_nodes;
        }
    }
    int n_done = 0;
    while (n_done < helper_count){
        n_done = 0;
        for (int i = 0; i < helper_count; ++i){
            if (helper_tasks[i].valid){
                if (helper_tasks[i].node->help_done){
                    search->n_nodes += helper_tasks[i].task.n_nodes;
                    helper_tasks[i].node->help_done = false;
                    helper_tasks[i].node->is_helping = false;
                    helper_tasks[i].valid = false;
                }
            } else
                ++n_done;
        }
    }
}
