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
#include "thread_pool.hpp"

/*
    @brief YBWC parameters
*/
#define YBWC_MID_SPLIT_MIN_DEPTH 6
#define YBWC_N_ELDER_CHILD 1
#define YBWC_N_YOUNGER_CHILD 2
// #define YBWC_MAX_RUNNING_COUNT 5

#define YBWC_HELP_RUNNING_COUNT_DIFF 100

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
void ybwc_do_task_nws(Board board, int_fast8_t n_discs, uint_fast8_t parity, uint_fast8_t mpc_level, int alpha, int depth, uint64_t legal, bool is_end_search, uint_fast8_t policy, const bool *searching, YBWC_result *ybwc_result, Search *parent){
    Search search;
    search.init(&board, mpc_level, depth > YBWC_MID_SPLIT_MIN_DEPTH, parent);
    int value = -nega_alpha_ordering_nws(&search, alpha, depth, false, legal, is_end_search, searching);
    if (!(*searching))
        value = SCORE_UNDEFINED;
    {
        std::lock_guard lock(ybwc_result->mtx);
        if (value > ybwc_result->value){
            ybwc_result->value = value;
            ybwc_result->best_move = policy;
        }
        ybwc_result->n_nodes += search.n_nodes;
        ybwc_result->running_count.fetch_sub(1);
        ybwc_result->running_count.notify_all();
    }
}

bool ybwc_ask_help(Search *search, int depth, int alpha, uint_fast8_t policy, bool *searching, YBWC_result *ybwc_result){
    if (!search->ybwc_state.helping && search->parent != nullptr){
        bool pushed = false;
        {
            std::lock_guard lock(search->parent->ybwc_state.mtx);
            std::lock_guard lock2(search->parent->ybwc_task.mtx);
            if (search->parent->ybwc_state.ybwc_result != nullptr){
                std::lock_guard lock3(search->parent->ybwc_state.ybwc_result->mtx);
                if (search->parent->ybwc_state.waiting && !search->parent->ybwc_state.helping && search->parent->ybwc_state.ybwc_result->running_count < YBWC_HELP_RUNNING_COUNT_DIFF && search->parent->ybwc_state.ybwc_result->running_count > 0){
                    search->parent->ybwc_task.board = search->board.copy();
                    search->parent->ybwc_task.depth = depth;
                    search->parent->ybwc_task.alpha = alpha;
                    search->parent->ybwc_task.mpc_level = search->mpc_level;
                    search->parent->ybwc_task.policy = policy;
                    search->parent->ybwc_task.searching = searching;
                    search->parent->ybwc_task.ybwc_result = ybwc_result;
                    search->parent->ybwc_state.ybwc_result->running_count.fetch_add(YBWC_HELP_RUNNING_COUNT_DIFF);
                    search->parent->ybwc_state.ybwc_result->running_count.notify_all();
                    //std::cerr << "call " << search->parent->ybwc_state.ybwc_result->running_count << " " << &search->parent->ybwc_state.ybwc_result->running_count << std::endl;
                    pushed = true;
                }
            }
        }
        return pushed;
    }
    return false;
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
inline bool ybwc_split_nws(Search *search, int alpha, int depth, uint64_t legal, bool is_end_search, bool *searching, uint_fast8_t policy, const int move_idx, const int canput, const int running_count, YBWC_result *ybwc_result){
    if (
            move_idx >= YBWC_N_ELDER_CHILD &&           // The elderest brother(s) is already searched
            move_idx < canput - YBWC_N_YOUNGER_CHILD    // This node is not the youngest brother(s)
            //running_count < YBWC_MAX_RUNNING_COUNT     // Do not split too many nodes
    ){
        if (ybwc_ask_help(search, depth, alpha, policy, searching, ybwc_result)){
            return true;
        } else if (thread_pool.get_n_idle()){                 // There is an idle thread
            bool pushed;
            thread_pool.push(&pushed, std::bind(&ybwc_do_task_nws, search->board, search->n_discs, search->parity, search->mpc_level, alpha, depth, legal, is_end_search, policy, searching, ybwc_result, search));
            return pushed;
        }
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
inline void ybwc_get_end_tasks(Search *search, int *v, int *best_move, int *running_count, YBWC_result *ybwc_result){
    std::lock_guard lock(ybwc_result->mtx);
    if (*v < ybwc_result->value){
        *v = ybwc_result->value;
        *best_move = ybwc_result->best_move;
    }
}

/*
    @brief Wait all running tasks

    For fail high

    @param search               search information
    @param parallel_tasks       vector of splitted tasks
*/
inline void ybwc_wait_all_stopped(Search *search, int *running_count, YBWC_result *ybwc_result){
    while (*running_count){
        ybwc_result->running_count.wait(*running_count);
        {
            std::lock_guard lock(ybwc_result->mtx);
            *running_count = ybwc_result->running_count;
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
inline void ybwc_wait_all_nws(Search *search, int *running_count, int *v, int *best_move, int alpha, const bool *searching, bool *n_searching, YBWC_result *ybwc_result){
    search->ybwc_state.ybwc_result = ybwc_result;
    search->ybwc_state.waiting = true;
    while (*running_count){
        *n_searching &= (*searching);
        ybwc_result->running_count.wait(*running_count);
        {
            std::lock_guard lock(ybwc_result->mtx);
            if (ybwc_result->running_count < YBWC_HELP_RUNNING_COUNT_DIFF){
                *running_count = ybwc_result->running_count;
                if (*v < ybwc_result->value){ // get end tasks
                    *v = ybwc_result->value;
                    *best_move = ybwc_result->best_move;
                }
            }
        }
        *n_searching &= (alpha >= (*v));
        {
            std::lock_guard lock(search->ybwc_state.mtx);
            if (ybwc_result->running_count >= YBWC_HELP_RUNNING_COUNT_DIFF){
                search->ybwc_state.helping = true;
                Search help_search;
                int ybwc_alpha, ybwc_depth;
                bool ybwc_is_end_search;
                bool *ybwc_searching;
                //std::cerr << "wake " << ybwc_result->running_count << " " << &(ybwc_result->running_count) << " " << *running_count << " " << search->ybwc_state.helping << std::endl;
                help_search.init(&search->ybwc_task.board, search->ybwc_task.mpc_level, search->ybwc_task.depth > YBWC_MID_SPLIT_MIN_DEPTH, nullptr);
                help_search.ybwc_state.helping = true;
                ybwc_alpha = search->ybwc_task.alpha;
                ybwc_depth = search->ybwc_task.depth;
                ybwc_is_end_search = help_search.n_discs + search->ybwc_task.depth >= HW2;
                ybwc_searching = search->ybwc_task.searching;
                int value = -nega_alpha_ordering_nws(&help_search, ybwc_alpha, ybwc_depth, false, LEGAL_UNDEFINED, ybwc_is_end_search, ybwc_searching);
                {
                    std::lock_guard lock(search->ybwc_task.ybwc_result->mtx);
                    if (value > search->ybwc_task.ybwc_result->value && *search->ybwc_task.searching){
                        search->ybwc_task.ybwc_result->value = value;
                        search->ybwc_task.ybwc_result->best_move = search->ybwc_task.policy;
                    }
                    search->ybwc_task.ybwc_result->n_nodes += help_search.n_nodes;
                    search->ybwc_state.helping = false;
                    search->ybwc_task.ybwc_result->running_count.fetch_sub(1);
                    search->ybwc_task.ybwc_result->running_count.notify_all();
                    //std::cerr << "END " << ybwc_result->running_count << " " << &(ybwc_result->running_count) << " " << *running_count << " " << search->ybwc_state.helping << std::endl;
                }
                ybwc_result->running_count.fetch_sub(YBWC_HELP_RUNNING_COUNT_DIFF);
            }
        }
    }
    search->ybwc_state.waiting = false;
    search->ybwc_state.ybwc_result = nullptr;
    /*
    ybwc_get_end_tasks(search, parallel_tasks, v, best_move, running_count);
    *n_searching &= (alpha >= (*v));
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
    */
}


#if USE_YBWC_NEGASCOUT
    /*
        @brief Get end tasks of YBWC

        @param search               search information
        @param parallel_tasks       vector of splitted tasks
        @param v                    value to store
        @param best_move            best move to store
        @param running_count        number of running tasks
    */
    inline void ybwc_get_end_tasks_negascout(Search *search, std::vector<std::future<Parallel_task>> &parallel_tasks, int *running_count, std::vector<int> &parallel_alphas, std::vector<int> &search_windows, int *best_score, int *best_idx, YBWC_result *ybwc_result){
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
    inline void ybwc_wait_all_negascout(Search *search, std::vector<std::future<Parallel_task>> &parallel_tasks, int *running_count, std::atomic<int> *atomic_running_count, std::vector<int> &parallel_alphas, std::vector<int> &search_windows, int *best_score, int *best_idx, int beta, const bool *searching, bool *n_searching){
        search->waiting = true;
        while (*running_count){
            *n_searching &= (*searching);
            atomic_running_count->wait(*running_count);
            ybwc_get_end_tasks_negascout(search, parallel_tasks, running_count, parallel_alphas, search_windows, best_score, best_idx);
            *n_searching &= (beta > *best_score);
            if (search->helping){
                Search help_search;
                help_search.init_board(&search->task.board);
                help_search.mpc_level = search->task.mpc_level;
                help_search.n_nodes = 0;
                help_search.use_multi_thread = (search->task.depth > YBWC_MID_SPLIT_MIN_DEPTH);
                bool is_end_search = help_search.n_discs + search->task.depth >= HW2;
                search->task.value = nega_scout(&help_search, search->task.alpha, search->task.beta, search->task.depth, false, LEGAL_UNDEFINED, is_end_search, search->task.searching);
                search->helping = false;
            }
        }
        search->waiting = false;
        /*
        ybwc_get_end_tasks_negascout(search, parallel_tasks, running_count, parallel_alphas, search_windows, best_score, best_idx);
        *n_searching &= (beta > *best_score);
        //if (beta <= *best_score)
        //    *n_searching = false;
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
        */
    }
#endif
