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
#define YBWC_MID_SPLIT_MIN_DEPTH 5
//#define YBWC_MID_SPLIT_MAX_DEPTH 26
#define YBWC_END_SPLIT_MIN_DEPTH 15
//#define YBWC_END_SPLIT_MAX_DEPTH 29
#define YBWC_N_ELDER_CHILD 1
#define YBWC_N_YOUNGER_CHILD 2
// #define YBWC_MAX_RUNNING_COUNT 5
#define YBWC_NOT_PUSHED -126
#define YBWC_PUSHED 126

int nega_alpha_ordering_nws(Search *search, int alpha, const int depth, const bool skipped, uint64_t legal, const bool is_end_search, std::vector<bool*> &searchings);
inline bool is_searching(std::vector<bool*> &searchings);

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
Parallel_task ybwc_do_task_nws(uint64_t player, uint64_t opponent, int_fast8_t n_discs, uint_fast8_t parity, uint_fast8_t mpc_level, bool is_presearch, int parent_alpha, const int depth, uint64_t legal, const bool is_end_search, uint_fast8_t policy, int move_idx, std::vector<bool*> searchings, bool *n_searching) {
    Search search(player, opponent, n_discs, parity, mpc_level, (!is_end_search && depth > YBWC_MID_SPLIT_MIN_DEPTH) || (is_end_search && depth > YBWC_END_SPLIT_MIN_DEPTH), is_presearch);
    Parallel_task task;
    task.value = -nega_alpha_ordering_nws(&search, -parent_alpha - 1, depth, false, legal, is_end_search, searchings);
    if (!is_searching(searchings)) {
        task.value = SCORE_UNDEFINED;
    } else if (task.value > parent_alpha) {
        *n_searching = false; // means: *searchings.back() = false;
    }
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
    @param searching          flag for terminating this search
    @param policy               the last move
    @param pv_idx               the priority of this move
    @param seems_to_be_all_node     this node seems to be ALL node?
    @param parallel_tasks       vector of splitted tasks
    @return task splitted?
*/
inline int ybwc_split_nws(Search *search, int parent_alpha, const int depth, uint64_t legal, const bool is_end_search, std::vector<bool*> &searchings, bool *n_searching, uint_fast8_t policy, const int n_remaining_moves, const int move_idx, const int running_count, std::vector<std::future<Parallel_task>> &parallel_tasks) {
    if (
            thread_pool.get_n_idle() &&                 // There is an idle thread
            n_remaining_moves >= YBWC_N_YOUNGER_CHILD    // This node is not the (some) youngest brother
            //running_count < YBWC_MAX_RUNNING_COUNT     // Do not split too many nodes
    ) {
        int v;
        if (transposition_cutoff_nws(search, search->board.hash(), depth, -parent_alpha - 1, &v)) {
            return -v;
        }
        if (!is_end_search && search->mpc_level < MPC_100_LEVEL) {
            if (mpc(search, -parent_alpha - 1, -parent_alpha, depth, legal, is_end_search, &v, searchings)) {
                return -v;
            }
        }
        if (is_searching(searchings)) {
            bool pushed;
            parallel_tasks.emplace_back(thread_pool.push(&pushed, std::bind(&ybwc_do_task_nws, search->board.player, search->board.opponent, search->n_discs, search->parity, search->mpc_level, search->is_presearch, parent_alpha, depth, legal, is_end_search, policy, move_idx, searchings, n_searching)));
            if (pushed) {
                return YBWC_PUSHED;
            } else{
                parallel_tasks.pop_back();
            }
        }
    }
    return YBWC_NOT_PUSHED;
}


#if USE_YBWC_NWS
    inline void ybwc_search_young_brothers_nws(Search *search, int alpha, int *v, int *best_move, int n_available_moves, uint32_t hash_code, int depth, bool is_end_search, std::vector<Flip_value> &move_list, std::vector<bool*> &searchings) {
        std::vector<std::future<Parallel_task>> parallel_tasks;
        bool n_searching = true;
        searchings.emplace_back(&n_searching);
        int canput = (int)move_list.size();
        int running_count = 0;
        int g;
        bool searched;
        int n_searched = 0;
        int n_moves_seen = 0;
        for (int move_idx = 1; move_idx < canput && is_searching(searchings); ++move_idx) {
            if (move_list[move_idx].flip.flip) {
                ++n_moves_seen;
                searched = false;
                search->move(&move_list[move_idx].flip);
                    int ybwc_split_state = ybwc_split_nws(search, alpha, depth - 1, move_list[move_idx].n_legal, is_end_search, searchings, &n_searching, move_list[move_idx].flip.pos, n_available_moves - n_moves_seen, move_idx, running_count, parallel_tasks);
                    if (ybwc_split_state == YBWC_PUSHED) {
                        ++running_count;
                    } else {
                        if (ybwc_split_state == YBWC_NOT_PUSHED) {
                            g = -nega_alpha_ordering_nws(search, -alpha - 1, depth - 1, false, move_list[move_idx].n_legal, is_end_search, searchings);
                        } else{
                            g = ybwc_split_state;
                            ++search->n_nodes;
                        }
                        if (is_searching(searchings)) {
                            searched = true;
                            if (*v < g) {
                                *v = g;
                                *best_move = move_list[move_idx].flip.pos;
                                if (alpha < g) {
                                    n_searching = false;
                                }
                            }
                        }
                    }
                search->undo(&move_list[move_idx].flip);
                if (searched) {
                    move_list[move_idx].flip.flip = 0;
                    ++n_searched;
                }
            }
        }
        Parallel_task task_result;
#if USE_YBWC_SPLITTED_TASK_TERMINATION
        if (is_searching(searchings) && running_count >= 2 && ((is_end_search && depth >= 28) || (!is_end_search && depth >= 24))) {
            for (std::future<Parallel_task> &task: parallel_tasks) {
                if (task.valid()) {
                    if (task.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                        task_result = task.get();
                        --running_count;
                        search->n_nodes += task_result.n_nodes;
                        if (task_result.value != SCORE_UNDEFINED) {
                            if (*v < task_result.value) {
                                *v = task_result.value;
                                *best_move = move_list[task_result.move_idx].flip.pos;
                            }
                            move_list[task_result.move_idx].flip.flip = 0;
                            ++n_searched;
                        }
                    }
                }
            }
            if (running_count >= 2) {
                n_searching = false;
                for (std::future<Parallel_task> &task: parallel_tasks) {
                    if (task.valid()) {
                        task_result = task.get();
                        --running_count;
                        search->n_nodes += task_result.n_nodes;
                        if (task_result.value != SCORE_UNDEFINED) {
                            if (*v < task_result.value) {
                                *v = task_result.value;
                                *best_move = move_list[task_result.move_idx].flip.pos;
                            }
                            move_list[task_result.move_idx].flip.flip = 0;
                            ++n_searched;
                        }
                    }
                }
                int n_to_be_searched = n_moves_seen - n_searched;
                searchings.pop_back();
                if (*v <= alpha && n_to_be_searched && is_searching(searchings)) {
                    if (n_to_be_searched <= 1) {
                        for (int move_idx = 1; move_idx < canput && is_searching(searchings); ++move_idx) {
                            if (move_list[move_idx].flip.flip) {
                                search->move(&move_list[move_idx].flip);
                                    g = -nega_alpha_ordering_nws(search, -alpha - 1, depth - 1, false, move_list[move_idx].n_legal, is_end_search, searchings);
                                    if (is_searching(searchings)) {
                                        if (*v < g) {
                                            *v = g;
                                            *best_move = move_list[move_idx].flip.pos;
                                        }
                                    }
                                search->undo(&move_list[move_idx].flip);
                            }
                        }
                    } else {
                        ybwc_search_young_brothers_nws(search, alpha, v, best_move, n_to_be_searched, hash_code, depth, is_end_search, move_list, searchings);
                    }
                    return;
                }
            }
        }
#endif
        for (std::future<Parallel_task> &task: parallel_tasks) {
            if (task.valid()) {
                task_result = task.get();
                search->n_nodes += task_result.n_nodes;
                if (task_result.value != SCORE_UNDEFINED) {
                    if (*v < task_result.value) {
                        *v = task_result.value;
                        *best_move = move_list[task_result.move_idx].flip.pos;
                    }
                }
            }
        }
        searchings.pop_back();
    }
#endif

#if USE_YBWC_NEGASCOUT
    void ybwc_search_young_brothers(Search *search, int *alpha, int *beta, int *v, int *best_move, int n_available_moves, uint32_t hash_code, int depth, bool is_end_search, std::vector<Flip_value> &move_list, bool need_best_move, bool *searching) {
        std::vector<std::future<Parallel_task>> parallel_tasks;
        bool n_searching = true;
        std::vector<bool*> searchings = {searching, &n_searching};
        int canput = (int)move_list.size();
        int running_count = 0;
        int g;
        std::vector<int> research_idxes;
        int next_alpha = *alpha;
        int n_searched = 0;
        int n_moves_seen = 0;
        for (int move_idx = 1; move_idx < canput && n_searching && *searching; ++move_idx) {
            if (move_list[move_idx].flip.flip) {
                ++n_moves_seen;
                bool move_done = false;
                search->move(&move_list[move_idx].flip);
                    int ybwc_split_state = ybwc_split_nws(search, *alpha, depth - 1, move_list[move_idx].n_legal, is_end_search, searchings, &n_searching, move_list[move_idx].flip.pos, n_available_moves - n_moves_seen, move_idx, running_count, parallel_tasks);
                    if (ybwc_split_state == YBWC_PUSHED) {
                        ++running_count;
                    } else{
                        if (ybwc_split_state == YBWC_NOT_PUSHED) {
                            n_searching &= *searching;
                            g = -nega_alpha_ordering_nws(search, -(*alpha) - 1, depth - 1, false, move_list[move_idx].n_legal, is_end_search, searchings);
                        } else{
                            g = ybwc_split_state;
                            ++search->n_nodes;
                        }
                        if (*searching) {
                            if (*v < g) {
                                *v = g;
                                *best_move = move_list[move_idx].flip.pos;
                            }
                            if (*alpha < g) {
                                next_alpha = std::max(next_alpha, g);
                                n_searching = false;
                                research_idxes.emplace_back(move_idx);
                            } else{
                                move_done = true;
                            }
                        }
                    }
                search->undo(&move_list[move_idx].flip);
                if (move_done) {
                    move_list[move_idx].flip.flip = 0;
                    ++n_searched;
                }
            }
        }
        if (running_count) {
            Parallel_task task_result;
            for (std::future<Parallel_task> &task: parallel_tasks) {
                n_searching &= *searching;
                if (task.valid()) {
                    task_result = task.get();
                    --running_count;
                    search->n_nodes += task_result.n_nodes;
                    if (task_result.value != SCORE_UNDEFINED) {
                        if (*v < task_result.value) {
                            *v = task_result.value;
                            *best_move = move_list[task_result.move_idx].flip.pos;
                        }
                        if (*alpha < task_result.value) {
                            next_alpha = std::max(next_alpha, task_result.value);
                            research_idxes.emplace_back(task_result.move_idx);
                        } else {
                            move_list[task_result.move_idx].flip.flip = 0;
                            ++n_searched;
                        }
                    }
                }
            }
        }
        if (research_idxes.size() && next_alpha < *beta && *searching) {
            int prev_alpha = *alpha;
            *alpha = next_alpha;
            for (const int &research_idx: research_idxes) {
                search->move(&move_list[research_idx].flip);
                    g = -nega_scout(search, -(*beta), -(*alpha), depth - 1, false, move_list[research_idx].n_legal, is_end_search, searching);
                search->undo(&move_list[research_idx].flip);
                move_list[research_idx].flip.flip = 0;
                ++n_searched;
                if (*searching) {
                    if (*v < g) {
                        *v = g;
                        *best_move = move_list[research_idx].flip.pos;
                    }
                    if (*alpha < g) {
                        *alpha = g;
                        if (*alpha >= *beta) {
                            break;
                        }
                    }
                }
            }
            if (*alpha < *beta && *searching) {
                ybwc_search_young_brothers(search, alpha, beta, v, best_move, n_moves_seen - n_searched, hash_code, depth, is_end_search, move_list, need_best_move, searching);
            }
        }
    }
#endif
