/*
    Egaroucid Project

    @file clogsearch.hpp
        Clog search
        MPC (Multi-ProbCut) might cut a very bad move as a very good move.
        For example, with MPC, Egaroucid might be wiped out without clog search.
    @date 2021-2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
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
#include "stability.hpp"
#include "evaluate.hpp"
#include "level.hpp"
#include "transposition_table.hpp"
#include "move_ordering.hpp"

constexpr int CLOG_NOT_FOUND = -127;

constexpr int CLOG_SEARCH_MAX_DEPTH = 5;

#if USE_PARALLEL_CLOG_SEARCH
// Do parallel clog search until depth CLOG_SEARCH_SPLIT_DEPTH
constexpr int CLOG_SEARCH_SPLIT_DEPTH = 2;

/*
    @brief Structure for parallel clog search

    @param n_nodes              number of nodes seen
    @param val                  the score
*/
struct Parallel_clog_task {
    uint64_t n_nodes;
    int val;
};

int clog_search(Search *search, int depth, bool *searching);

/*
    @brief Wrapper for parallel clog search

    @param player               a bitboard representing the player
    @param opponent             a bitboard representing the opponent
    @param is_enduring          true if player is enduring from opponent's atack, else false
    @param depth                remaining depth
    @return search result in Parallel_clog_task structure
*/
Parallel_clog_task clog_do_task(uint64_t player, uint64_t opponent, int depth, bool *searching) {
    Search search(player, opponent, MPC_100_LEVEL, true, false);
    Parallel_clog_task task;
    task.val = clog_search(&search, depth, searching);
    task.n_nodes = search.n_nodes;
    return task;
}

/*
    @brief Try to do parallel clog search

    @param search               searching information
    @param canput               number of legal moves of parent node
    @param pv_idx               number of searched child nodes
    @param depth                remaining depth
    @param searching            used to stop searching
    @param parallel_tasks       if task splitted, push the task in this vector
    @return task splitted?
*/
inline bool clog_split(const Search *search, const int canput, const int pv_idx, int depth, bool *searching, std::vector<std::future<Parallel_clog_task>> &parallel_tasks) {
    if (thread_pool.get_n_idle() &&
        pv_idx < canput - 1 && 
        depth >= CLOG_SEARCH_SPLIT_DEPTH) {
            bool pushed;
            parallel_tasks.emplace_back(thread_pool.push(&pushed, std::bind(&clog_do_task, search->board.player, search->board.opponent, depth, searching)));
            if (!pushed) {
                parallel_tasks.pop_back();
            }
            return pushed;
    }
    return false;
}
#endif

/*
    @brief Main algorithm for clog search

    @param search               searching information
    @param depth                remaining depth
    @return best score
*/
int clog_search(Search *search, int depth, bool *searching) {
    if (!global_searching) {
        return CLOG_NOT_FOUND;
    }
    ++search->n_nodes;
    if (depth == 0) {
        if (search->board.is_end()) {
            return search->board.score_player();
        }
        return CLOG_NOT_FOUND;
    }
    int res = CLOG_NOT_FOUND;
    uint64_t legal = search->board.get_legal();
    if (legal == 0ULL) {
        search->pass();
            if (search->board.get_legal() == 0ULL) {
                res = search->board.score_player();
            } else{
                res = clog_search(search, depth, searching);
            }
        search->pass();
        if (res != CLOG_NOT_FOUND) {
            return -res;
        }
        return CLOG_NOT_FOUND;
    }
    const int canput = pop_count_ull(legal);
    std::vector<Flip_value> move_list(canput);
    int idx = 0;
    for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)) {
        calc_flip(&move_list[idx].flip, &search->board, cell);
        if (move_list[idx].flip.flip == search->board.opponent) {
            return SCORE_MAX;
        }
        ++idx;
    }
    uint_fast8_t moves[N_TRANSPOSITION_MOVES] = {MOVE_UNDEFINED, MOVE_UNDEFINED};
    transposition_table.get_moves_any_level(&search->board, search->board.hash(), moves);
    move_list_evaluate(search, move_list, moves, depth, -SCORE_MAX, SCORE_MAX, searching);
    int g;
    bool uncertain_value_found = false;
    int beta = HW2 - 2 * pop_count_ull(calc_stability(search->board.opponent, search->board.player));
#if USE_PARALLEL_CLOG_SEARCH
    std::vector<std::future<Parallel_clog_task>> parallel_tasks;
    bool n_searching = true;
    for (int move_idx = 0; move_idx < canput && res < beta && *searching; ++move_idx) {
        swap_next_best_move(move_list, move_idx, canput);
        search->move(&move_list[move_idx].flip);
            if (!clog_split(search, canput, move_idx, depth - 1, &n_searching, parallel_tasks)) {
                g = clog_search(search, depth - 1, searching);
                if (g != CLOG_NOT_FOUND) {
                    res = std::max(res, -g);
                } else {
                    uncertain_value_found = true;
                }
            }
        search->undo(&move_list[move_idx].flip);
    }
    if (res < beta && *searching) {
        Parallel_clog_task got_task;
        for (std::future<Parallel_clog_task> &task: parallel_tasks) {
            got_task = task.get();
            if (got_task.val != CLOG_NOT_FOUND) {
                res = std::max(res, -got_task.val);
            } else {
                uncertain_value_found = true;
            }
            search->n_nodes += got_task.n_nodes;
        }
    } else {
        n_searching = false;
        for (std::future<Parallel_clog_task> &task: parallel_tasks) {
            search->n_nodes += task.get().n_nodes;
        }
    }
#else
    for (int move_idx = 0; move_idx < canput && res < beta && *searching && global_searching; ++move_idx) {
        swap_next_best_move(move_list, move_idx, canput);
        search->move(&move_list[move_idx].flip);
            g = clog_search(search, depth - 1, searching);
        search->undo(&move_list[move_idx].flip);
        if (g != CLOG_NOT_FOUND) {
            res = std::max(res, -g);
        } else {
            uncertain_value_found = true;
        }
    }
#endif
    if (uncertain_value_found && res < beta) {
        res = CLOG_NOT_FOUND;
    }
    return res;
}

/*
    @brief Wrapper of clog search algorithm for convenience

    @param board                a board to solve
    @param n_nodes              number of nodes visited
    @return vector of all moves and scores that leads early game over
*/
std::vector<Clog_result> first_clog_search(Board board, uint64_t *n_nodes, int depth, uint64_t legal, bool *searching) {
    Search search(&board, MPC_100_LEVEL, true, false);
    std::vector<Clog_result> res;
    Flip flip;
    int g;
    for (uint_fast8_t cell = first_bit(&legal); legal && *searching && global_searching; cell = next_bit(&legal)) {
        calc_flip(&flip, &search.board, cell);
        search.move(&flip);
            g = clog_search(&search, depth - 1, searching);
            if (g != CLOG_NOT_FOUND) {
                Clog_result result;
                result.pos = cell;
                result.val = -g;
                res.emplace_back(result);
            }
        search.undo(&flip);
    }
    *n_nodes = search.n_nodes;
    return res;
}