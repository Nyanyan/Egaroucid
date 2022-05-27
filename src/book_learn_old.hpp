#pragma once
#include <iostream>
#include <unordered_set>
#include "evaluate.hpp"
#include "board.hpp"
#include "ai.hpp"

#define AUTO_BOOK_SAVE_TIME 60000

using namespace std;

bool cmp_flip(pair<Flip, int> &a, pair<Flip, int> &b){
    return a.second > b.second;
}

inline int book_learn_calc_value(Board board, int level, bool level_minus){
    Search search;
    search.board = board;
    calc_features(&search);
    int g, depth;
    bool is_mid_search, searching = true;
    get_level(level, search.board.n - 4, &is_mid_search, &depth, &search.use_mpc, &search.mpct);
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

int book_learn_search(Board board, int level, const int book_depth, const int first_value, const int expected_error, Board *board_copy, uint64_t *strt_tim, string book_file, string book_bak){
    if (tim() - *strt_tim > AUTO_BOOK_SAVE_TIME){
        book.save_bin(book_file, book_bak);
        *strt_tim = tim();
    }
    int g;
    if (board.n >= 4 + book_depth){
        g = book.get(&board);
        if (-SCORE_MAX <= g && g <= SCORE_MAX)
            return g;
        g = book_learn_calc_value(board, level / 2, false);
        if (g >= first_value - expected_error * 2){
            g = book_learn_calc_value(board, level, false);
            if (global_searching && -SCORE_MAX <= g && g <= SCORE_MAX){
                book.reg(board, -g);
                return g;
            }
        }
        return SCORE_UNDEFINED;
    }
    uint64_t legal = board.get_legal();
    if (legal == 0ULL){
        board.pass();
            g = -book_learn_search(board, level, book_depth, -first_value, expected_error, board_copy, strt_tim, book_file, book_bak);
        board.pass();
        return g;
    }
    Flip flip;
    int v = -INF;
    for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)){
        if (!global_searching)
            return SCORE_UNDEFINED;
        calc_flip(&flip, &board, cell);
        board.move(&flip);
            board.copy(board_copy);
            g = -book.get(&board);
            if (g < -SCORE_MAX || SCORE_MAX < g){
                g = -book_learn_calc_value(board, level / 2, false);
                if (g >= first_value - expected_error * 2)
                    g = -book_learn_calc_value(board, level, false);
            }
            if (-SCORE_MAX <= g && g <= SCORE_MAX){
                cerr << "pre   value " << g << " parent value " << v << " root expected value " << first_value << endl;
                if (g >= first_value - expected_error){
                    g = -book_learn_search(board, level, book_depth, -first_value, expected_error, board_copy, strt_tim, book_file, book_bak);
                    if (g != -SCORE_UNDEFINED){
                        v = max(v, g);
                        cerr << "exact value " << g << " parent value " << v << " root expected value " << first_value << endl;
                    }
                }// else if (global_searching)
                //    book.reg(board, g);
            }
        board.undo(&flip);
        if (v > first_value + expected_error)
            return v;
    }
    if (-SCORE_MAX <= v && v <= SCORE_MAX && global_searching){
        book.reg(board, -v);
        return v;
    }
    return SCORE_UNDEFINED;
}

inline void learn_book(Board root_board, int level, const int book_depth, const int expected_error, Board *board_copy, string book_file, string book_bak, bool *book_learning){
    int first_value = -book.get(&root_board);
    cerr << "book get " << first_value << endl;
    if (first_value < -SCORE_MAX || SCORE_MAX < first_value)
        first_value = book_learn_calc_value(root_board, level, false);
    cerr << "root value " << first_value << endl;
    uint64_t strt_tim = tim();
    int g = book_learn_search(root_board, level, book_depth, first_value, expected_error, board_copy, &strt_tim, book_file, book_bak);
    root_board.copy(board_copy);
    cerr << "book learn finished expected " << first_value << " got " << g << endl;
    *book_learning = false;
}