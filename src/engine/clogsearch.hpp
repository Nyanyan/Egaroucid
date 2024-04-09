/*
    Egaroucid Project

    @file clogsearch.hpp
        Clog search
        MPC (Multi-ProbCut) might cut a very bad move as a very good move.
        For example, with MPC, Egaroucid might be wiped out without clog search.
    @date 2021-2024
    @author Takuto Yamana
    @license GPL-3.0 license
*/

#pragma once
#include <iostream>
#include <algorithm>
#include <vector>
#include <future>
#include "setting.hpp"
#include "common.hpp"
#include "board.hpp"
#include "search.hpp"
#include "thread_pool.hpp"
#include "util.hpp"

#define CLOG_NOT_FOUND -127

// Do clog search in depth CLOG_SEARCH_DEPTH
#define CLOG_SEARCH_MAX_DEPTH 11

#if USE_PARALLEL_CLOG_SEARCH

    // Do parallel clog search until depth CLOG_SEARCH_SPLIT_DEPTH
    #define CLOG_SEARCH_SPLIT_DEPTH 6

    /*
        @brief Structure for parallel clog search

        @param n_nodes              number of nodes seen
        @param val                  the score
    */
    struct Parallel_clog_task{
        uint64_t n_nodes;
        int val;
    };

    int clog_search(Clog_search *search, bool is_enduring, int depth);

    /*
        @brief Wrapper for parallel clog search

        @param player               a bitboard representing the player
        @param opponent             a bitboard representing the opponent
        @param is_enduring          true if player is enduring from opponent's atack, else false
        @param depth                remaining depth
        @return search result in Parallel_clog_task structure
    */
    Parallel_clog_task clog_do_task(uint64_t player, uint64_t opponent, bool is_enduring, int depth){
        Clog_search search;
        search.board.player = player;
        search.board.opponent = opponent;
        search.n_nodes = 0ULL;
        Parallel_clog_task task;
        task.val = clog_search(&search, is_enduring, depth);
        task.n_nodes = search.n_nodes;
        return task;
    }

    /*
        @brief Try to do parallel clog search

        @param search               searching information
        @param canput               number of legal moves of parent node
        @param pv_idx               number of searched child nodes
        @param is_enduring          true if player is enduring from opponent's atack, else false
        @param depth                remaining depth
        @param parallel_tasks       if task splitted, push the task in this vector
        @return task splitted?
    */
    inline bool clog_split(const Clog_search *search, const int canput, const int pv_idx, bool is_enduring, int depth, std::vector<std::future<Parallel_clog_task>> &parallel_tasks){
        if (thread_pool.get_n_idle() &&
            pv_idx < canput - 1 && 
            depth >= CLOG_SEARCH_SPLIT_DEPTH){
                bool pushed;
                parallel_tasks.emplace_back(thread_pool.push(&pushed, std::bind(&clog_do_task, search->board.player, search->board.opponent, is_enduring, depth)));
                if (!pushed)
                    parallel_tasks.pop_back();
                return pushed;
        }
        return false;
    }

    /*
        @brief Wait all splitted parallel tasks

        @param search               searching information
        @param parallel_tasks       vector of splitted tasks
        @return best score
    */
    inline int clog_wait_all(Clog_search *search, std::vector<std::future<Parallel_clog_task>> &parallel_tasks){
        int res = CLOG_NOT_FOUND;
        Parallel_clog_task got_task;
        for (std::future<Parallel_clog_task> &task: parallel_tasks){
            got_task = task.get();
            if (got_task.val != CLOG_NOT_FOUND)
                res = std::max(res, -got_task.val);
            search->n_nodes += got_task.n_nodes;
        }
        return res;
    }
#endif

/*
    @brief Main algorithm for clog search

    If player is enduring, try to find at least one move that doesn't lead early game over.
    If player is attacking, try to find best move that leads early game over with highest score.
    If attacking, parallel search never increase number of visited nodes.

    @param search               searching information
    @param is_enduring          true if player is enduring from opponent's atack, else false
    @param depth                remaining depth
    @return best score
*/
int clog_search(Clog_search *search, bool is_enduring, int depth){
    if (!global_searching)
        return CLOG_NOT_FOUND;
    ++search->n_nodes;
    if (depth == 0)
        return CLOG_NOT_FOUND;
    int res = CLOG_NOT_FOUND;
    uint64_t legal = search->board.get_legal();
    if (legal == 0ULL){
        search->board.pass();
            if (search->board.get_legal() == 0ULL)
                res = search->board.score_player();
            else
                res = clog_search(search, !is_enduring, depth);
        search->board.pass();
        if (res != CLOG_NOT_FOUND)
            return -res;
        return CLOG_NOT_FOUND;
    }
    Flip flip;
    int g;
    #if USE_PARALLEL_CLOG_SEARCH
        if (is_enduring){
            for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)){
                calc_flip(&flip, &search->board, cell);
                search->board.move_board(&flip);
                    g = clog_search(search, false, depth - 1);
                search->board.undo_board(&flip);
                if (g == CLOG_NOT_FOUND)
                    return CLOG_NOT_FOUND;
                res = std::max(res, -g);
            }
        } else{
            std::vector<std::future<Parallel_clog_task>> parallel_tasks;
            const int canput = pop_count_ull(legal);
            int pv_idx = 0;
            for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)){
                calc_flip(&flip, &search->board, cell);
                search->board.move_board(&flip);
                    if (!clog_split(search, canput, pv_idx++, true, depth - 1, parallel_tasks)){
                        g = clog_search(search, true, depth - 1);
                        if (g != CLOG_NOT_FOUND)
                            res = std::max(res, -g);
                    }
                search->board.undo_board(&flip);
            }
            res = std::max(res, clog_wait_all(search, parallel_tasks));
        }
    #else
        for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)){
            calc_flip(&flip, &search->board, cell);
            search->board.move_board(&flip);
                g = clog_search(search, !is_enduring, depth - 1);
            search->board.undo_board(&flip);
            if (g != CLOG_NOT_FOUND)
                res = std::max(res, -g);
            else if (is_enduring)
                return CLOG_NOT_FOUND;
        }
    #endif
    return res;
}

/*
    @brief Wrapper of clog search algorithm for convenience

    @param board                a board to solve
    @param n_nodes              number of nodes visited
    @return vector of all moves and scores that leads early game over
*/
std::vector<Clog_result> first_clog_search(Board board, uint64_t *n_nodes, int depth, uint64_t legal){
    Clog_search search;
    search.board = board.copy();
    search.n_nodes = 0ULL;
    std::vector<Clog_result> res;
    //uint64_t legal = search.board.get_legal();
    if (legal == 0ULL)
        return res;
    Flip flip;
    int g;
    for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)){
        calc_flip(&flip, &search.board, cell);
        search.board.move_board(&flip);
            g = clog_search(&search, false, depth - 1);
            if (g != CLOG_NOT_FOUND){
                Clog_result result;
                result.pos = cell;
                result.val = -g;
                res.emplace_back(result);
            } else{
                g = clog_search(&search, true, depth - 1);
                if (g != CLOG_NOT_FOUND){
                    Clog_result result;
                    result.pos = cell;
                    result.val = -g;
                    res.emplace_back(result);
                }
            }
        search.board.undo_board(&flip);
    }
    *n_nodes = search.n_nodes;
    return res;
}

/*
    @brief Wrapper of clog search algorithm for convenience

    @param board                a board to solve
    @param n_nodes              number of nodes visited
    @return best score
*/
int clog_search(Board board, uint64_t *n_nodes, int depth){
    Clog_search search;
    search.board = board.copy();
    search.n_nodes = 0ULL;
    int res = clog_search(&search, true, depth - 1);
    if (res == CLOG_NOT_FOUND)
        res = clog_search(&search, false, depth - 1);
    *n_nodes = search.n_nodes;
    return res;
}