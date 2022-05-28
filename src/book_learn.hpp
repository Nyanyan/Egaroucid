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

inline int book_learn_calc_value(Board board, int level, int bound){
    Search search;
    search.board = board;
    calc_features(&search);
    int g, depth;
    bool is_mid_search, searching = true;
    get_level(level, search.board.n - 4, &is_mid_search, &depth, &search.use_mpc, &search.mpct);
    if (is_mid_search){
        if (depth - 1 >= 0){
            parent_transpose_table.init();
            g = nega_scout(&search, bound - 1, bound + 1, depth - 1, false, LEGAL_UNDEFINED, false, &searching);
            parent_transpose_table.init();
            g += nega_scout(&search, bound - 1, bound + 1, depth, false, LEGAL_UNDEFINED, false, &searching);
            g /= 2;
        } else{
            parent_transpose_table.init();
            g = nega_scout(&search, bound - 1, bound + 1, depth, false, LEGAL_UNDEFINED, false, &searching);
        }
    } else{
        parent_transpose_table.init();
        g = nega_scout(&search, bound - 1, bound + 1, depth, false, LEGAL_UNDEFINED, true, &searching);
    }
    return g;
}

int book_learn_search(Board board, int level, const int book_depth, const int best_value, int error_sum, const int expected_error, const int adopt_error_sum, Board *board_copy, uint64_t *strt_tim, string book_file, string book_bak){
    if (!global_searching)
        return SCORE_UNDEFINED;
    if (error_sum > adopt_error_sum)
        return SCORE_UNDEFINED;
    if (tim() - *strt_tim > AUTO_BOOK_SAVE_TIME){
        book.save_bin(book_file, book_bak);
        *strt_tim = tim();
    }
    int g;
    if (board.n >= 4 + book_depth){
        g = -INF;
        if (-SCORE_MAX <= best_value && best_value <= SCORE_MAX)
            g = book_learn_calc_value(board, level, best_value);
        if ((g != -INF && g <= best_value) || best_value < -SCORE_MAX || SCORE_MAX < best_value){
            g = ai(board, level, true, 0).value;
            book.reg(board, -g);
            return g;
        }
        return SCORE_UNDEFINED;
    }
    uint64_t legal = board.get_legal();
    if (legal == 0ULL){
        board.pass();
            g = -book_learn_search(board, level, book_depth, -best_value, error_sum, expected_error, adopt_error_sum, board_copy, strt_tim, book_file, book_bak);
        board.pass();
        return g;
    }
    Search_result best_move = ai(board, level, true, 0);
    if (best_move.value > best_value + expected_error && -SCORE_MAX <= best_value && best_value <= SCORE_MAX)
        return best_move.value;
    //if (-SCORE_MAX <= best_move.value && best_move.value <= SCORE_MAX)
    //    book.reg(board, -best_move.value);
    Flip flip;
    calc_flip(&flip, &board, (uint8_t)best_move.policy);
    board.move(&flip);
        board.copy(board_copy);
        g = -book_learn_search(board, level, book_depth, SCORE_UNDEFINED, error_sum, expected_error, adopt_error_sum, board_copy, strt_tim, book_file, book_bak);
        if (-SCORE_MAX <= g && g <= SCORE_MAX){
            best_move.value = g;
            cerr << "PV value " << best_move.value << endl;
        } else
            best_move.value = -INF;
    board.undo(&flip);
    legal ^= 1ULL << best_move.policy;
    bool flag;
    for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)){
        flag = false;
        calc_flip(&flip, &board, cell);
        board.move(&flip);
            board.copy(board_copy);
            g = book.get(&board);
            if (g < -SCORE_MAX || SCORE_MAX < g){
                if (-SCORE_MAX <= best_move.value && best_move.value <= SCORE_MAX)
                    g = -book_learn_calc_value(board, level, -best_move.value + expected_error);
                else
                    flag = true;
            }
            if (flag || (-SCORE_MAX <= g && g <= SCORE_MAX && g >= best_move.value - expected_error)){
                g = -book_learn_search(board, level, book_depth, -best_move.value, error_sum + max(0, best_move.value - g), expected_error, adopt_error_sum, board_copy, strt_tim, book_file, book_bak);
                if (-SCORE_MAX <= g && g <= SCORE_MAX){
                    if (g > best_move.value){
                        best_move.value = g;
                        best_move.policy = (int)cell;
                    }
                    cerr << "   value " << g << " best_move_value " << best_move.value << endl;
                }
            }
        board.undo(&flip);
        //if (flag)
        //    book.reg(board, -best_move.value);
    }
    if (-SCORE_MAX <= best_move.value && best_move.value <= SCORE_MAX && global_searching){
        book.reg(board, -best_move.value);
        return best_move.value;
    }
    return SCORE_UNDEFINED;
}

inline void learn_book(Board root_board, int level, const int book_depth, const int expected_error, Board *board_copy, string book_file, string book_bak, bool *book_learning){
    uint64_t strt_tim = tim();
    cerr << "book learn started" << endl;
    const int adopt_error_sum = max(expected_error, (book_depth + 4 - root_board.n) * expected_error / 5);
    int g = book_learn_search(root_board, level, book_depth, SCORE_UNDEFINED, 0, expected_error, adopt_error_sum, board_copy, &strt_tim, book_file, book_bak);
    root_board.copy(board_copy);
    cerr << "book learn finished " << g << endl;
    *book_learning = false;
}