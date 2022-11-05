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
#include "evaluate.hpp"
#include "search.hpp"
#include "transpose_table.hpp"
#include "move_ordering.hpp"
#include "probcut.hpp"
#include "thread_pool.hpp"
#include "util.hpp"
#include "stability.hpp"

#define CLOG_NOT_FOUND -INF
#define CLOG_SEARCH_DEPTH 7

int clog_search(Clog_search *search, bool is_player, int depth){
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
    bool no_clog_found = false;
    for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)){
        calc_flip(&flip, &search->board, cell);
        search->board.move_board(&flip);
            g = clog_search(search, !is_player, depth - 1);
        search->board.undo_board(&flip);
        if (!is_player && g == CLOG_NOT_FOUND)
            return CLOG_NOT_FOUND;
        res = max(res, -g);
    }
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
