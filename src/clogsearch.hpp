/*
    Egaroucid Project

    @date 2021-2022
    @author Takuto Yamana (a.k.a Nyanyan)
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

using namespace std;

#define CLOG_NOT_FOUND -INF
#define CLOG_SEARCH_DEPTH 11
#define CLOG_SEARCH_SPLIT_DEPTH 6

struct Parallel_clog_task{
    uint64_t n_nodes;
    int val;
};

int clog_search(Clog_search *search, bool is_player, int depth);

Parallel_clog_task clog_do_task(int id, uint64_t player, uint64_t opponent, bool is_player, int depth){
    Clog_search search;
    search.board.player = player;
    search.board.opponent = opponent;
    search.n_nodes = 0ULL;
    Parallel_clog_task task;
    task.val = clog_search(&search, is_player, depth);
    task.n_nodes = search.n_nodes;
    return task;
}

inline bool clog_split(const Clog_search *search, const int canput, const int pv_idx, bool is_player, int depth, vector<future<Parallel_clog_task>> &parallel_tasks){
    if (thread_pool.n_idle() &&
        pv_idx < canput - 1 && 
        depth >= CLOG_SEARCH_SPLIT_DEPTH){
            bool pushed;
            parallel_tasks.emplace_back(thread_pool.push(&pushed, &clog_do_task, search->board.player, search->board.opponent, is_player, depth));
            if (!pushed)
                parallel_tasks.pop_back();
            return pushed;
    }
    return false;
}

inline int clog_wait_all(Clog_search *search, vector<future<Parallel_clog_task>> &parallel_tasks){
    int res = CLOG_NOT_FOUND;
    Parallel_clog_task got_task;
    for (future<Parallel_clog_task> &task: parallel_tasks){
        got_task = task.get();
        if (got_task.val != CLOG_NOT_FOUND)
            res = max(res, -got_task.val);
        search->n_nodes += got_task.n_nodes;
    }
    return res;
}

int clog_search(Clog_search *search, bool is_player, int depth){
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
                res = clog_search(search, !is_player, depth);
        search->board.pass();
        if (res != CLOG_NOT_FOUND)
            return -res;
        return CLOG_NOT_FOUND;
    }
    Flip flip;
    int g;
    #if USE_PARALLEL_CLOG_SEARCH
        if (!is_player){
            vector<future<Parallel_clog_task>> parallel_tasks;
            const int canput = pop_count_ull(legal);
            int pv_idx = 0;
            for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)){
                calc_flip(&flip, &search->board, cell);
                search->board.move_board(&flip);
                    if (!clog_split(search, canput, pv_idx++, true, depth - 1, parallel_tasks)){
                        g = clog_search(search, true, depth - 1);
                        if (g != CLOG_NOT_FOUND)
                            res = max(res, -g);
                    }
                search->board.undo_board(&flip);
            }
            res = max(res, clog_wait_all(search, parallel_tasks));
        } else{
            for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)){
                calc_flip(&flip, &search->board, cell);
                search->board.move_board(&flip);
                    g = clog_search(search, false, depth - 1);
                search->board.undo_board(&flip);
                if (g == CLOG_NOT_FOUND)
                    return CLOG_NOT_FOUND;
                res = max(res, -g);
            }
        }
    #else
        for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)){
            calc_flip(&flip, &search->board, cell);
            search->board.move_board(&flip);
                g = clog_search(search, !is_player, depth - 1);
            search->board.undo_board(&flip);
            if (g != CLOG_NOT_FOUND)
                res = max(res, -g);
            else if (is_player)
                return CLOG_NOT_FOUND;
        }
    #endif
    return res;
}

vector<Clog_result> first_clog_search(Board board, uint64_t *n_nodes){
    Clog_search search;
    search.board = board.copy();
    search.n_nodes = 0ULL;
    vector<Clog_result> res;
    uint64_t legal = search.board.get_legal();
    if (legal == 0ULL)
        return res;
    Flip flip;
    int g;
    for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)){
        calc_flip(&flip, &search.board, cell);
        search.board.move_board(&flip);
            g = clog_search(&search, false, CLOG_SEARCH_DEPTH - 1);
        search.board.undo_board(&flip);
        if (g != CLOG_NOT_FOUND){
            Clog_result result;
            result.pos = cell;
            result.val = -g;
            res.emplace_back(result);
        }
    }
    *n_nodes = search.n_nodes;
    return res;
}

int clog_search(Board board, uint64_t *n_nodes){
    Clog_search search;
    search.board = board.copy();
    search.n_nodes = 0ULL;
    int res = clog_search(&search, false, CLOG_SEARCH_DEPTH - 1);
    *n_nodes = search.n_nodes;
    return res;
}
