/*
    Egaroucid Project

    @file ybwc.hpp
        Parallel search with YBWC (Young Brothers Wait Concept)
    @date 2021-2022
    @author Takuto Yamana (a.k.a. Nyanyan)
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
#define YBWC_MID_SPLIT_MIN_DEPTH 5
#define YBWC_END_SPLIT_MIN_DEPTH 13

int nega_alpha_ordering_nws(Search *search, int alpha, int depth, bool skipped, uint64_t legal, bool is_end_search, const bool *searching);
#if USE_NEGA_ALPHA_END
    int nega_alpha_end(Search *search, int alpha, int beta, bool skipped, uint64_t legal, const bool *searching);
#endif
#if MID_TO_END_DEPTH > YBWC_END_SPLIT_MIN_DEPTH
    int nega_alpha_end_nws(Search *search, int alpha, bool skipped, uint64_t legal, const bool *searching);
#endif

/*
    @brief Wrapper for parallel NWS (Null Window Search)

    @param id                   id for thread pool (not used at all)
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
Parallel_task ybwc_do_task_nws(int id, uint64_t player, uint64_t opponent, int_fast8_t n_discs, uint_fast8_t parity, uint_fast8_t mpc_level, int alpha, int depth, uint64_t legal, bool is_end_search, uint_fast8_t policy, const bool *searching){
    Search search;
    search.board.player = player;
    search.board.opponent = opponent;
    search.n_discs = n_discs;
    search.parity = parity;
    search.mpc_level = mpc_level;
    search.n_nodes = 0ULL;
    search.use_multi_thread = depth > YBWC_MID_SPLIT_MIN_DEPTH;
    calc_features(&search);
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
    @param canput               number of legal moves
    @param pv_idx               the priority of this move
    @param seems_to_be_all_node     this node seems to be ALL node?
    @param split_count          number of splitted nodes here
    @param parallel_tasks       vector of splitted tasks
    @return task splitted?
*/
inline bool ybwc_split_nws(const Search *search, int alpha, int depth, uint64_t legal, bool is_end_search, const bool *searching, uint_fast8_t policy, const int canput, const int pv_idx, const bool seems_to_be_all_node, const int split_count, std::vector<std::future<Parallel_task>> &parallel_tasks){
    if (thread_pool.n_idle() &&
        (pv_idx || seems_to_be_all_node)){
            bool pushed;
            parallel_tasks.emplace_back(thread_pool.push(&pushed, &ybwc_do_task_nws, search->board.player, search->board.opponent, search->n_discs, search->parity, search->mpc_level, alpha, depth, legal, is_end_search, policy, searching));
            if (!pushed)
                parallel_tasks.pop_back();
            return pushed;
    }
    return false;
}

#if MID_TO_END_DEPTH > YBWC_END_SPLIT_MIN_DEPTH
    #if USE_NEGA_ALPHA_END
        /*
            @brief Wrapper for parallel endgame search

            @param id                   id for thread pool (not used at all)
            @param player               a bitboard representing player
            @param opponent             a bitboard representing opponent
            @param n_discs              number of discs on the board
            @param parity               parity of the board
            @param alpha                alpha value
            @param beta                 beta value
            @param legal                for use of previously calculated legal bitboard
            @param policy               the last move
            @param searching            flag for terminating this search
            @return the result in Parallel_task structure
        */
        Parallel_task ybwc_do_task_end(int id, uint64_t player, uint64_t opponent, int_fast8_t n_discs, uint_fast8_t parity, int alpha, int beta, uint64_t legal, uint_fast8_t policy, const bool *searching){
            Search search;
            search.board.player = player;
            search.board.opponent = opponent;
            search.n_discs = n_discs;
            search.parity = parity;
            search.mpc_level = MPC_100_LEVEL;
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

        /*
            @brief Try to do parallel endgame search

            @param search               searching information
            @param alpha                alpha value
            @param beta                 beta value
            @param legal                for use of previously calculated legal bitboard
            @param searching            flag for terminating this search
            @param policy               the last move
            @param canput               number of legal moves
            @param pv_idx               the priority of this move
            @param seems_to_be_all_node     this node seems to be ALL node?
            @param split_count          number of splitted nodes here
            @param parallel_tasks       vector of splitted tasks
            @return task splitted?
        */
        inline bool ybwc_split_end(const Search *search, int alpha, int beta, uint64_t legal, const bool *searching, uint_fast8_t policy, const int canput, const int pv_idx, const bool seems_to_be_all_node, const int split_count, std::vector<std::future<Parallel_task>> &parallel_tasks){
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

    /*
        @brief Wrapper for parallel endgame NWS (Null Window Search)

        @param id                   id for thread pool (not used at all)
        @param player               a bitboard representing player
        @param opponent             a bitboard representing opponent
        @param n_discs              number of discs on the board
        @param parity               parity of the board
        @param alpha                alpha value
        @param legal                for use of previously calculated legal bitboard
        @param policy               the last move
        @param searching            flag for terminating this search
        @return the result in Parallel_task structure
    */
    Parallel_task ybwc_do_task_end_nws(int id, uint64_t player, uint64_t opponent, int_fast8_t n_discs, uint_fast8_t parity, int alpha, uint64_t legal, uint_fast8_t policy, const bool *searching){
        Search search;
        search.board.player = player;
        search.board.opponent = opponent;
        search.n_discs = n_discs;
        search.parity = parity;
        search.mpc_level = MPC_100_LEVEL;
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

    /*
        @brief Try to do parallel endgame NWS (Null Window Search)

        @param search               searching information
        @param alpha                alpha value
        @param legal                for use of previously calculated legal bitboard
        @param searching            flag for terminating this search
        @param policy               the last move
        @param canput               number of legal moves
        @param pv_idx               the priority of this move
        @param seems_to_be_all_node     this node seems to be ALL node?
        @param split_count          number of splitted nodes here
        @param parallel_tasks       vector of splitted tasks
        @return task splitted?
    */
    inline bool ybwc_split_end_nws(const Search *search, int alpha, uint64_t legal, const bool *searching, uint_fast8_t policy, const int canput, const int pv_idx, const bool seems_to_be_all_node, const int split_count, std::vector<std::future<Parallel_task>> &parallel_tasks){
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

/*
    @brief Get end tasks of YBWC

    @param search               search information
    @param parallel_tasks       vector of splitted tasks
    @param v                    value to store
    @param best_move            best move to store
*/
inline void ybwc_get_end_tasks(Search *search, std::vector<std::future<Parallel_task>> &parallel_tasks, int *v, int *best_move){
    Parallel_task got_task;
    for (std::future<Parallel_task> &task: parallel_tasks){
        if (task.valid()){
            if (task.wait_for(std::chrono::seconds(0)) == std::future_status::ready){
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

/*
    @brief Get end tasks of YBWC

    @param search               search information
    @param parallel_tasks       vector of splitted tasks
    @param v                    value to store
    @param best_move            best move to store
    @param alpha                alpha value to store
*/
inline void ybwc_get_end_tasks(Search *search, std::vector<std::future<Parallel_task>> &parallel_tasks, int *v, int *best_move, int *alpha){
    ybwc_get_end_tasks(search, parallel_tasks, v, best_move);
    *alpha = std::max((*alpha), (*v));
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
    @brief Wait all running tasks

    @param search               search information
    @param parallel_tasks       vector of splitted tasks
    @param v                    value to store
    @param best_move            best move to store
    @param alpha                alpha value to store
    @param beta                 beta value
    @param searching            flag for terminating this search
*/
inline void ybwc_wait_all(Search *search, std::vector<std::future<Parallel_task>> &parallel_tasks, int *v, int *best_move, int *alpha, int beta, bool *searching){
    ybwc_get_end_tasks(search, parallel_tasks, v, best_move, alpha);
    if (beta <= (*alpha))
        *searching = false;
    Parallel_task got_task;
    for (std::future<Parallel_task> &task: parallel_tasks){
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
    *alpha = std::max((*alpha), (*v));
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
inline void ybwc_wait_all_nws(Search *search, std::vector<std::future<Parallel_task>> &parallel_tasks, int *v, int *best_move, int alpha, bool *searching){
    ybwc_get_end_tasks(search, parallel_tasks, v, best_move);
    if (alpha < (*v))
        *searching = false;
    Parallel_task got_task;
    for (std::future<Parallel_task> &task: parallel_tasks){
        if (task.valid()){
            got_task = task.get();
            search->n_nodes += got_task.n_nodes;
            if ((*v) < got_task.value && (*searching)){
                *best_move = got_task.cell;
                *v = got_task.value;
                if (alpha < (*v))
                    *searching = false;
            }
        }
    }
}
