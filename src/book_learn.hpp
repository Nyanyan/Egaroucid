#pragma once
#include <iostream>
#include <unordered_set>
#include "evaluate.hpp"
#include "board.hpp"
#include "ai.hpp"

#define BOOK_LEARN_UNDEFINED -INF

using namespace std;

bool cmp_flip(pair<Flip, int> &a, pair<Flip, int> &b){
    return a.second > b.second;
}

inline int book_learn_calc_value(Board board, int level, bool level_minus){
    Search search;
    search.board = board;
    int g, depth;
    bool is_mid_search, searching = true;
    get_level(level, search->board.n - 4, &is_mid_search, &depth, &search.use_mpc, &search.mpct);
    if (level_minus)
        --depth;
    if (is_mid_search){
        if (depth - 1 >= 0){
            parent_transpose_table.init();
            g = nega_scout(&search, -SCORE_MAX, SCORE_MAX, depth - 1, false, LEGAL_UNDEFINED, false, &searching);
            parent_transpose_table.init();
            g += nega_scout(&search, -SCORE_MAX, SCORE_MAX, depth, false, LEGAL_UNDEFINED, false, &searching);
            g /= 2;
        } else{
            parent_transpose_table.init();
            g = nega_scout(&search, -SCORE_MAX, SCORE_MAX, depth, false, LEGAL_UNDEFINED, false, &searching);
        }
    } else{
        parent_transpose_table.init();
        g = nega_scout(&search, -SCORE_MAX, SCORE_MAX, depth, false, LEGAL_UNDEFINED, true, &searching);
    }
    return g;
}

int book_learn_search(Board board, int level, const int book_depth, const int first_value, const int expected_error, Board *board_copy){
    int g;
    if (board.n >= 4 + book_depth){
        g = book_learn_calc_value(board, level, false);
        book.reg(board, g);
        return g;
    }
    uint64_t legal = board.get();
    Flip flip;
    int v = -INF;
    for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)){
        calc_flip(&flip, &board, cell);
        board.move(&flip);
            board.copy(board_copy);
            g = -book_learn_calc_value(board, level, true);
            if (g >= first_value - expected_error){
                g = -book_learn_search(board, level, book_depth, -g, expected_error);
                v = max(v, g);
            }
        board.undo(&flip);
    }
    book.reg(board, v);
    return v;
}

inline void learn_book(Board root_board, int level, const int book_depth, const int expected_error, Board *board_copy){
    const int first_value = book_learn_calc_value(root_board, level, false);
    book_learn_search(root_board, level, book_depth, first_value, expected_error, board_copy);
}