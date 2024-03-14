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
#include "transposition_table.hpp"

/*
    @brief YBWC parameters
*/
#define YBWC_MID_SPLIT_MIN_DEPTH 6
#define YBWC_N_ELDER_CHILD 1
#define YBWC_N_YOUNGER_CHILD 2
// #define YBWC_MAX_RUNNING_COUNT 5

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
Parallel_task ybwc_do_task_nws(uint64_t player, uint64_t opponent, int_fast8_t n_discs, uint_fast8_t parity, uint_fast8_t mpc_level, int alpha, int depth, uint64_t legal, bool is_end_search, uint_fast8_t policy, int move_idx, const bool *searching){
    Search search;
    search.board.player = player;
    search.board.opponent = opponent;
    search.n_discs = n_discs;
    search.parity = parity;
    search.mpc_level = mpc_level;
    search.n_nodes = 0ULL;
    search.use_multi_thread = depth > YBWC_MID_SPLIT_MIN_DEPTH;
    search.need_to_see_tt_loop = false; // because lazy smp sub threads are done in only single thread
    calc_eval_features(&search.board, &search.eval);
    Parallel_task task;
    task.value = -nega_alpha_ordering_nws(&search, alpha, depth, false, legal, is_end_search, searching);
    if (!(*searching))
        task.value = SCORE_UNDEFINED;
    task.n_nodes = search.n_nodes;
    task.cell = policy;
    task.move_idx = move_idx;
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
inline bool ybwc_split_nws(Search *search, int alpha, int depth, uint64_t legal, bool is_end_search, const bool *searching, uint_fast8_t policy, const int move_idx, const int canput, const int running_count, std::vector<std::future<Parallel_task>> &parallel_tasks){
    if (
            thread_pool.get_n_idle() &&                 // There is an idle thread
            move_idx >= YBWC_N_ELDER_CHILD &&           // The elderest brother is already searched
            move_idx < canput - YBWC_N_YOUNGER_CHILD    // This node is not the (some) youngest brother
            //running_count < YBWC_MAX_RUNNING_COUNT     // Do not split too many nodes
    ){
        bool pushed;
        parallel_tasks.emplace_back(thread_pool.push(&pushed, std::bind(&ybwc_do_task_nws, search->board.player, search->board.opponent, search->n_discs, search->parity, search->mpc_level, alpha, depth, legal, is_end_search, policy, move_idx, searching)));
        if (!pushed)
            parallel_tasks.pop_back();
        return pushed;
    }
    return false;
}


#if USE_YBWC_NEGASCOUT
    inline void ybwc_search_young_brothers(Search *search, int *alpha, int *beta, int *v, int *best_move, uint32_t hash_code, int depth, bool is_end_search, std::vector<Flip_value> &move_list, const bool *searching){
        std::vector<std::future<Parallel_task>> parallel_tasks;
        bool n_searching = true;
        int canput = (int)move_list.size();
        int running_count = 0;
        int g, fail_high_idx = -1;
        for (int move_idx = 0; move_idx < canput && n_searching; ++move_idx){
            n_searching &= *searching;
            if (move_list[move_idx].flip.flip){ // move is valid
                // just split moves
                search->move(&move_list[move_idx].flip);
                    if (ybwc_split_nws(search, -*alpha - 1, depth - 1, move_list[move_idx].n_legal, is_end_search, &n_searching, move_list[move_idx].flip.pos, move_idx, canput, running_count, parallel_tasks)){
                        ++running_count;
                    } else{
                        if (search->need_to_see_tt_loop){
                            if (transposition_cutoff_nomove(search, hash_code, depth, alpha, beta, v)){
                                n_searching = false;
                                break;
                            }
                        }
                        g = -nega_alpha_ordering_nws(search, -*alpha - 1, depth - 1, false, move_list[move_idx].n_legal, is_end_search, searching);
                        if (*alpha < g){
                            *alpha = g;
                            n_searching = false;
                            fail_high_idx = move_idx;
                        } else{
                            move_list[move_idx].flip.flip = 0;
                        }
                    }
                search->undo(&move_list[move_idx].flip);
            }
        }
        Parallel_task got_task;
        for (std::future<Parallel_task> &task: parallel_tasks){
            if (task.valid()){
                got_task = task.get();
                if (n_searching){
                    if (*alpha < got_task.value){
                        *alpha = got_task.value;
                        fail_high_idx = got_task.move_idx;
                        n_searching = false;
                    } else {
                        move_list[got_task.move_idx].flip.flip = 0;
                    }
                }
                search->n_nodes += got_task.n_nodes;
            }
        }
        if (*searching && fail_high_idx != -1){
            if (*alpha < *beta){
                search->move(&move_list[fail_high_idx].flip);
                    g = -nega_scout(search, -*beta, -*alpha, depth - 1, false, move_list[fail_high_idx].n_legal, is_end_search, searching);
                search->undo(&move_list[fail_high_idx].flip);
                *alpha = g;
                *v = g;
                *best_move = move_list[fail_high_idx].flip.pos;
                move_list[fail_high_idx].flip.flip = 0;
                if (*alpha < *beta){
                    ybwc_search_young_brothers(search, alpha, beta, v, best_move, hash_code, depth, is_end_search, move_list, searching);
                }
            }
        }
    }
#endif

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
inline void ybwc_wait_all_stopped(Search *search, std::vector<std::future<Parallel_task>> &parallel_tasks){
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
