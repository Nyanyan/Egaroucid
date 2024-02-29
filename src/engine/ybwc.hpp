/*
    Egaroucid Project

    @file ybwc.hpp
        Parallel search with YBWC (Young Brothers Wait Concept)
    @date 2021-2024
    @author Takuto Yamana
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

/*
    @brief YBWC splitting depth threshold
*/
#define YBWC_MID_SPLIT_MIN_DEPTH 6

#define YBWC_MAX_RUNNING_COUNT 5
#define YBWC_SPLIT_MIN_MOVES 2

int nega_alpha_ordering_nws(Search *search, int alpha, int depth, bool skipped, uint64_t legal, bool is_end_search, const bool *searching);

/*
    @brief Wrapper for parallel NWS (Null Window Search)

    @param player               a bitboard representing player
    @param opponent             a bitboard representing opponent
    @param n_discs              number of discs on the board
    @param parity               parity of the board
    @param mpc_level            MPC (Multi-ProbCut) probability level
    @param alpha                alpha value
    @param depth                remaining depth
    @param legal                for use of previously calculated legal bitboard
    @param is_end_search        search till the end?
    @param policy               the last move
    @param searching            flag for terminating this search
    @return the result in Parallel_task structure
*/
Parallel_task ybwc_do_task_nws(uint64_t player, uint64_t opponent, int_fast8_t n_discs, uint_fast8_t parity, uint_fast8_t mpc_level, int alpha, int depth, uint64_t legal, bool is_end_search, uint_fast8_t policy, const bool *searching){
    Search search;
    search.board.player = player;
    search.board.opponent = opponent;
    search.n_discs = n_discs;
    search.parity = parity;
    search.mpc_level = mpc_level;
    search.n_nodes = 0ULL;
    search.use_multi_thread = depth > YBWC_MID_SPLIT_MIN_DEPTH;
    calc_eval_features(&search.board, &search.eval);
    Parallel_task task;
    task.value = -nega_alpha_ordering_nws(&search, alpha, depth, false, legal, is_end_search, searching);
    if (!(*searching))
        task.value = SCORE_UNDEFINED;
    task.n_nodes = search.n_nodes;
    task.cell = policy;
    return task;
}

/*
    @brief Try to do parallel NWS (Null Window Search)

    @param search               searching information
    @param alpha                alpha value
    @param depth                remaining depth
    @param legal                for use of previously calculated legal bitboard
    @param is_end_search        search till the end?
    @param searching            flag for terminating this search
    @param policy               the last move
    @param pv_idx               the priority of this move
    @param seems_to_be_all_node     this node seems to be ALL node?
    @param parallel_tasks       vector of splitted tasks
    @return task splitted?
*/
inline bool ybwc_split_nws(const Search *search, int alpha, int depth, uint64_t legal, bool is_end_search, const bool *searching, uint_fast8_t policy, const int move_idx, const int canput, const int running_count, std::vector<std::future<Parallel_task>> &parallel_tasks){
    if (
            thread_pool.get_n_idle() &&                 // There is an idle thread
            move_idx &&                                 // The elderest brother is already searched
            move_idx < canput - YBWC_SPLIT_MIN_MOVES    // This node is not the (some) youngest brother
            //running_count < YBWC_MAX_RUNNING_COUNT      // Do not split too many nodes
    ){
            bool pushed;
            parallel_tasks.emplace_back(thread_pool.push(&pushed, std::bind(&ybwc_do_task_nws, search->board.player, search->board.opponent, search->n_discs, search->parity, search->mpc_level, alpha, depth, legal, is_end_search, policy, searching)));
            if (!pushed)
                parallel_tasks.pop_back();
            return pushed;
    }
    return false;
}

/*
    @brief Get end tasks of YBWC

    @param search               search information
    @param parallel_tasks       vector of splitted tasks
    @param v                    value to store
    @param best_move            best move to store
    @param running_count        number of running tasks
*/
inline void ybwc_get_end_tasks(Search *search, std::vector<std::future<Parallel_task>> &parallel_tasks, int *v, int *best_move, int *running_count){
    Parallel_task got_task;
    for (std::future<Parallel_task> &task: parallel_tasks){
        if (task.valid()){
            if (task.wait_for(std::chrono::seconds(0)) == std::future_status::ready){
                got_task = task.get();
                --(*running_count);
                if (*v < got_task.value){
                    *v = got_task.value;
                    *best_move = got_task.cell;
                }
                search->n_nodes += got_task.n_nodes;
            }
        }
    }
}

/*
    @brief Wait all running tasks

    For fail high

    @param search               search information
    @param parallel_tasks       vector of splitted tasks
*/
inline void ybwc_wait_all(Search *search, std::vector<std::future<Parallel_task>> &parallel_tasks){
    Parallel_task got_task;
    for (std::future<Parallel_task> &task: parallel_tasks){
        if (task.valid()){
            got_task = task.get();
            search->n_nodes += got_task.n_nodes;
        }
    }
}

/*
    @brief Wait all running tasks for NWS (Null Window Search)

    @param search               search information
    @param parallel_tasks       vector of splitted tasks
    @param v                    value to store
    @param best_move            best move to store
    @param alpha                alpha value
    @param searching            flag for terminating this search
*/
inline void ybwc_wait_all_nws(Search *search, std::vector<std::future<Parallel_task>> &parallel_tasks, int *v, int *best_move, int *running_count, int alpha, const bool *searching, bool *n_searching){
    ybwc_get_end_tasks(search, parallel_tasks, v, best_move, running_count);
    *n_searching &= alpha >= (*v);
    Parallel_task got_task;
    for (std::future<Parallel_task> &task: parallel_tasks){
        *n_searching &= (*searching);
        if (task.valid()){
            got_task = task.get();
            --(*running_count);
            search->n_nodes += got_task.n_nodes;
            if ((*v) < got_task.value && (*n_searching)){
                *best_move = got_task.cell;
                *v = got_task.value;
                if (alpha < (*v))
                    *n_searching = false;
            }
        }
    }
}

/*
    @brief Get end tasks of YBWC

    @param search               search information
    @param parallel_tasks       vector of splitted tasks
    @param v                    value to store
    @param best_move            best move to store
    @param running_count        number of running tasks
*/
inline void ybwc_get_end_tasks_negascout(Search *search, std::vector<std::future<Parallel_task>> &parallel_tasks, std::vector<int> &parallel_alphas, std::vector<int> &search_windows, int *running_count, int *best_score, int *best_idx){
    *best_score = SCORE_UNDEFINED;
    Parallel_task got_task;
    for (int i = 0; i < (int)parallel_tasks.size(); ++i){
        if (parallel_tasks[i].valid()){
            if (parallel_tasks[i].wait_for(std::chrono::seconds(0)) == std::future_status::ready){
                got_task = parallel_tasks[i].get();
                --(*running_count);
                search->n_nodes += got_task.n_nodes;
                if (*best_score < got_task.value){
                    *best_score = got_task.value;
                    *best_idx = i;
                }
                if (parallel_alphas[i] < got_task.value)
                    search_windows[i] = got_task.value;
            }
        }
    }
}

/*
    @brief Get end tasks of YBWC

    @param search               search information
    @param parallel_tasks       vector of splitted tasks
    @param v                    value to store
    @param best_move            best move to store
    @param running_count        number of running tasks
*/
inline void ybwc_wait_all_negascout(Search *search, std::vector<std::future<Parallel_task>> &parallel_tasks, std::vector<int> &parallel_alphas, std::vector<int> &search_windows, int *running_count, int *best_score, int *best_idx, int beta, const bool *searching, bool *n_searching){
    ybwc_get_end_tasks_negascout(search, parallel_tasks, parallel_alphas, search_windows, running_count, best_score, best_idx);
    if (beta <= *best_score)
        *n_searching = false;
    Parallel_task got_task;
    for (int i = 0; i < (int)parallel_tasks.size(); ++i){
        *n_searching &= (*searching);
        if (parallel_tasks[i].valid()){
            got_task = parallel_tasks[i].get();
            --(*running_count);
            search->n_nodes += got_task.n_nodes;
            if (*n_searching){
                if (*best_score < got_task.value){
                    *best_score = got_task.value;
                    *best_idx = i;
                    if (beta <= got_task.value)
                        *n_searching = false;
                }
                if (parallel_alphas[i] < got_task.value)
                    search_windows[i] = got_task.value;
            }
        }
    }
}