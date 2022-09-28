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


inline int book_learn_calc_value(Board board, int level, int alpha, int beta){
    return ai(board, level, true, true, false).value;
}

int book_learn_search(Board board, int level, const int book_depth, int alpha, int beta, int error_sum, int expected_error, const int adopt_error_sum, Board *board_copy, uint64_t *strt_tim, string book_file, string book_bak){
    if (!global_searching)
        return SCORE_UNDEFINED;
    if (error_sum > adopt_error_sum)
        return SCORE_UNDEFINED;
    if (tim() - *strt_tim > AUTO_BOOK_SAVE_TIME){
        book.save_bin(book_file, book_bak);
        *strt_tim = tim();
    }
    int g, v = SCORE_UNDEFINED;
    if (board.n >= 4 + book_depth){
        g = book_learn_calc_value(board, level, alpha, beta);
        if (alpha <= g && g <= beta){
            cerr << "depth " << board.n - 4 << " LF value " << g << " [" << alpha << "," << beta << "]" << endl;
            book.reg(board, -g);
            return g;
        }
        return SCORE_UNDEFINED;
    }
    if (get_level_complete_depth(level) >= HW2 - board.n)
        expected_error = 0;
    uint64_t legal = board.get_legal();
    if (legal == 0ULL){
        board.pass();
            g = -book_learn_search(board, level, book_depth, -beta, -alpha, error_sum, expected_error, adopt_error_sum, board_copy, strt_tim, book_file, book_bak);
        board.pass();
        return g;
    }
    Search_result best_move = ai(board, level, true, 0, false);
    cerr << "depth " << board.n - 5 << " BM value " << best_move.value << " move " << idx_to_coord(best_move.policy) << " [" << alpha << "," << beta << "]" << endl;
    Flip flip;
    bool alpha_updated = false;
    calc_flip(&flip, &board, (uint8_t)best_move.policy);
    board.move(&flip);
        board.copy(board_copy);
        g = -book_learn_search(board, level, book_depth, -SCORE_MAX, SCORE_MAX, error_sum, expected_error, adopt_error_sum, board_copy, strt_tim, book_file, book_bak);
        if (-SCORE_MAX <= g && g <= SCORE_MAX){
            v = g;
            alpha = g;
            alpha_updated = true;
            cerr << "depth " << board.n - 5 << " PV value " << g << " move " << idx_to_coord(best_move.policy) << " [" << alpha << "," << beta << "]" << endl;
            //policies.emplace_back(best_move.policy);
        } else if (-SCORE_MAX <= g && g <= SCORE_MAX){
            v = g;
            cerr << "depth " << board.n - 5 << " PV value " << g << " move " << idx_to_coord(best_move.policy) << " [" << alpha << "," << beta << "] FAIL" << endl;
        }
    board.undo(&flip);
    legal ^= 1ULL << best_move.policy;
    for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)){
        calc_flip(&flip, &board, cell);
        board.move(&flip);
            board.copy(board_copy);
            g = book.get(&board);
            if (g < -SCORE_MAX || SCORE_MAX < g)
                g = -book_learn_calc_value(board, level, -best_move.value, -best_move.value + expected_error);
            //cerr << "PRESEARCH depth " << board.n - 5 << "    value " << g << " move " << idx_to_coord(cell) << " [" << alpha << "," << beta << "]" << endl;
            if (-SCORE_MAX <= g && g <= SCORE_MAX && g >= best_move.value - expected_error){
                g = -book_learn_search(board, level, book_depth, -beta, -alpha, error_sum + min(0, best_move.value - g), expected_error, adopt_error_sum, board_copy, strt_tim, book_file, book_bak);
                if (-SCORE_MAX <= g && g <= SCORE_MAX){
                    v = max(v, g);
                    if (alpha <= g){
                        alpha = g;
                        alpha_updated = true;
                    }
                    cerr << "depth " << board.n - 5 << " AD value " << g << " move " << idx_to_coord(cell) << " [" << alpha << "," << beta << "]" << endl;
                }
            }
        board.undo(&flip);
    }
    if (alpha_updated && global_searching){
        cerr << "depth " << board.n - 4 << " RG value " << alpha << endl;
        book.reg(board, -alpha);
        return alpha;
    }
    return v;
}

inline void learn_book(Board root_board, int level, const int book_depth, int expected_error, Board *board_copy, string book_file, string book_bak, bool *book_learning){
    uint64_t strt_tim = tim();
    cerr << "book learn started" << endl;
    const int adopt_error_sum = INF; //max(expected_error, (book_depth + 4 - root_board.n) * expected_error / 5);
    int g = book_learn_search(root_board, level, book_depth, -SCORE_MAX, SCORE_MAX, 0, expected_error, adopt_error_sum, board_copy, &strt_tim, book_file, book_bak);
    if (*book_learning)
        book.reg(root_board, -g);
    root_board.copy(board_copy);
    cerr << "book learn finished " << g << endl;
    *book_learning = false;
}
