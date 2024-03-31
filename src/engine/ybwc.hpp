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
#define YBWC_FAIL_HIGH -1

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
        int v;
        uint_fast8_t moves[N_TRANSPOSITION_MOVES];
        if (transposition_cutoff_nws(search, search->board.hash(), depth, alpha, &v, moves)){
            
        } else{
            bool pushed;
            parallel_tasks.emplace_back(thread_pool.push(&pushed, std::bind(&ybwc_do_task_nws, search->board.player, search->board.opponent, search->n_discs, search->parity, search->mpc_level, alpha, depth, legal, is_end_search, policy, move_idx, searching)));
            if (!pushed)
                parallel_tasks.pop_back();
            return pushed;
        }
    }
    return false;
}

#if USE_YBWC_NWS
    inline void ybwc_search_young_brothers_nws(Search *search, int alpha, int *v, int *best_move, uint32_t hash_code, int depth, bool is_end_search, std::vector<Flip_value> &move_list, const bool *searching){
        std::vector<std::future<Parallel_task>> parallel_tasks;
        bool n_searching = true;
        int canput = (int)move_list.size();
        int running_count = 0;
        int g;
        for (int move_idx = 1; move_idx < canput && n_searching; ++move_idx){
            n_searching &= *searching;
            if (move_list[move_idx].flip.flip){
                if (search->need_to_see_tt_loop){
                    if (transposition_cutoff_nws_bestmove(search, hash_code, depth, alpha, v, best_move)){
                        n_searching = false;
                        break;
                    }
                }
                bool serial_searched = false;
                search->move(&move_list[move_idx].flip);
                    if (ybwc_split_nws(search, -alpha - 1, depth - 1, move_list[move_idx].n_legal, is_end_search, &n_searching, move_list[move_idx].flip.pos, move_idx, canput, running_count, parallel_tasks)){
                        ++running_count;
                    } else{
                        g = -nega_alpha_ordering_nws(search, -alpha - 1, depth - 1, false, move_list[move_idx].n_legal, is_end_search, searching);
                        serial_searched = true;
                        if (*searching){
                            if (*v < g){
                                *v = g;
                                *best_move = move_list[move_idx].flip.pos;
                                if (alpha < g){
                                    n_searching = false;
                                }
                            }
                        }
                    }
                search->undo(&move_list[move_idx].flip);
                if (running_count && serial_searched){
                    Parallel_task got_task;
                    for (std::future<Parallel_task> &task: parallel_tasks){
                        if (task.valid()){
                            if (task.wait_for(std::chrono::seconds(0)) == std::future_status::ready){
                                got_task = task.get();
                                if (n_searching){
                                    if (*v < got_task.value){
                                        *v = got_task.value;
                                        *best_move = move_list[got_task.move_idx].flip.pos;
                                        if (alpha < got_task.value){
                                            n_searching = false;
                                        }
                                    }
                                }
                                search->n_nodes += got_task.n_nodes;
                            }
                        }
                    }
                }
            }
        }
        if (running_count){
            Parallel_task got_task;
            for (std::future<Parallel_task> &task: parallel_tasks){
                if (task.valid()){
                    got_task = task.get();
                    if (n_searching){
                        if (*v < got_task.value){
                            *v = got_task.value;
                            *best_move = move_list[got_task.move_idx].flip.pos;
                            if (alpha < got_task.value){
                                n_searching = false;
                            }
                        }
                    }
                    search->n_nodes += got_task.n_nodes;
                }
            }
        }
    }
#endif

#if USE_YBWC_NEGASCOUT
    void ybwc_search_young_brothers(Search *search, int *alpha, int *beta, int *v, int *best_move, uint32_t hash_code, int depth, bool is_end_search, std::vector<Flip_value> &move_list, bool need_best_move, const bool *searching){
        std::vector<std::future<Parallel_task>> parallel_tasks;
        bool n_searching = true;
        int canput = (int)move_list.size();
        int running_count = 0;
        int g, fail_high_idx = -1;
        for (int move_idx = 1; move_idx < canput && n_searching; ++move_idx){
            n_searching &= *searching;
            if (move_list[move_idx].flip.flip){
                if (search->need_to_see_tt_loop && !need_best_move){
                    if (transposition_cutoff_bestmove(search, hash_code, depth, alpha, beta, v, best_move)){
                        n_searching = false;
                        fail_high_idx = YBWC_FAIL_HIGH;
                        break;
                    }
                }
                bool move_done = false, serial_searched = false;
                search->move(&move_list[move_idx].flip);
                    if (ybwc_split_nws(search, -(*alpha) - 1, depth - 1, move_list[move_idx].n_legal, is_end_search, &n_searching, move_list[move_idx].flip.pos, move_idx, canput, running_count, parallel_tasks)){
                        ++running_count;
                    } else{
                        g = -nega_alpha_ordering_nws(search, -(*alpha) - 1, depth - 1, false, move_list[move_idx].n_legal, is_end_search, searching);
                        serial_searched = true;
                        if (*searching){
                            if (*alpha < g){
                                *alpha = g;
                                *v = g;
                                *best_move = move_list[move_idx].flip.pos;
                                n_searching = false;
                                fail_high_idx = move_idx;
                            } else{
                                move_done = true;
                                if (*v < g){
                                    *v = g;
                                    *best_move = move_list[move_idx].flip.pos;
                                }
                            }
                        }
                    }
                search->undo(&move_list[move_idx].flip);
                if (move_done){
                    move_list[move_idx].flip.flip = 0;
                }
                if (running_count && serial_searched){
                    Parallel_task got_task;
                    for (std::future<Parallel_task> &task: parallel_tasks){
                        if (task.valid()){
                            if (task.wait_for(std::chrono::seconds(0)) == std::future_status::ready){
                                got_task = task.get();
                                if (n_searching){
                                    if (*alpha < got_task.value){
                                        *alpha = got_task.value;
                                        *v = got_task.value;
                                        *best_move = move_list[got_task.move_idx].flip.pos;
                                        fail_high_idx = got_task.move_idx;
                                        n_searching = false;
                                    } else {
                                        if (*v < got_task.value){
                                            *v = got_task.value;
                                            *best_move = move_list[got_task.move_idx].flip.pos;
                                        }
                                        move_list[got_task.move_idx].flip.flip = 0;
                                    }
                                }
                                search->n_nodes += got_task.n_nodes;
                            }
                        }
                    }
                }
            }
        }
        if (running_count){
            Parallel_task got_task;
            for (std::future<Parallel_task> &task: parallel_tasks){
                if (task.valid()){
                    got_task = task.get();
                    if (n_searching){
                        if (*alpha < got_task.value){
                            *alpha = got_task.value;
                            *v = got_task.value;
                            *best_move = move_list[got_task.move_idx].flip.pos;
                            fail_high_idx = got_task.move_idx;
                            n_searching = false;
                        } else {
                            if (*v < got_task.value){
                                *v = got_task.value;
                                *best_move = move_list[got_task.move_idx].flip.pos;
                            }
                            move_list[got_task.move_idx].flip.flip = 0;
                        }
                    }
                    search->n_nodes += got_task.n_nodes;
                }
            }
        }
        if (*searching && fail_high_idx != YBWC_FAIL_HIGH){
            if (*alpha < *beta){
                search->move(&move_list[fail_high_idx].flip);
                    g = -nega_scout(search, -(*beta), -(*alpha), depth - 1, false, move_list[fail_high_idx].n_legal, is_end_search, searching);
                search->undo(&move_list[fail_high_idx].flip);
                *alpha = g;
                *v = g;
                *best_move = move_list[fail_high_idx].flip.pos;
                move_list[fail_high_idx].flip.flip = 0;
                if (*alpha < *beta){
                    ybwc_search_young_brothers(search, alpha, beta, v, best_move, hash_code, depth, is_end_search, move_list, need_best_move, searching);
                }
            }
        }
    }
#endif
