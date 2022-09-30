#pragma once
#include <iostream>
#include <unordered_set>
#include "evaluate.hpp"
#include "board.hpp"
#include "ai.hpp"

#define AUTO_BOOK_SAVE_TIME 60000

using namespace std;

Search_result ai(Board board, int level, bool use_book, bool use_multi_thread, bool show_log);
int ai_window(Board board, int level, int alpha, int beta, bool use_multi_thread);

inline int book_learn_calc_value(Board board, int level){
    return ai(board, level, true, true, false).value;
}

int book_learn_search(Board board, int level, const int book_depth, int expected_value, int expected_error, int error_remain, Board *board_copy, int *player, uint64_t *strt_tim, string book_file, string book_bak){
    if (!global_searching || error_remain < 0)
        return SCORE_UNDEFINED;
    if (tim() - *strt_tim > AUTO_BOOK_SAVE_TIME){
        book.save_bin(book_file, book_bak);
        *strt_tim = tim();
    }
    int g, v = SCORE_UNDEFINED;
    if (board.is_end()){
        g = board.score_player();
        cerr << "depth " << board.n_discs() - 4 << " LF value " << g << endl;
        book.reg(board, -g);
        return g;
    }
    if (board.n_discs() >= 4 + book_depth){
        g = book_learn_calc_value(board, level);
        cerr << "depth " << board.n_discs() - 4 << " LF value " << g << endl;
        book.reg(board, -g);
        return g;
    }
    int book_value = book.get(&board);
    if (book_value != -INF){
        cerr << "depth " << board.n_discs() - 4 << " BK value " << book_value << endl;
        return book_value;
    }
    if (get_level_complete_depth(level) >= HW2 - board.n_discs())
        error_remain = 0;
    uint64_t legal = board.get_legal();
    if (legal == 0ULL){
        board.pass();
        *player ^= 1;
            g = -book_learn_search(board, level, book_depth, -expected_value, expected_error, error_remain, board_copy, player, strt_tim, book_file, book_bak);
        *player ^= 1;
        board.pass();
        return g;
    }
    Search_result best_move = ai(board, level, true, 0, false);
    cerr << "depth " << board.n_discs() - 4 << " BM value " << best_move.value << endl;
    Flip flip;
    bool alpha_updated = false;
    calc_flip(&flip, &board, (uint8_t)best_move.policy);
    board.move_board(&flip);
    *player ^= 1;
        board.copy(board_copy);
        g = -book_learn_search(board, level, book_depth, -expected_value, expected_error, error_remain, board_copy, player, strt_tim, book_file, book_bak);
        if (global_searching && g >= -HW2 && g <= HW2){
            v = g;
            best_move.value = g;
            cerr << "depth " << board.n_discs() - 4 << " PV value " << g << " expected " << expected_value << " remaining error " << error_remain << endl;
            //policies.emplace_back(best_move.policy);
        }
    *player ^= 1;
    board.undo_board(&flip);
    legal ^= 1ULL << best_move.policy;
    if (legal){
        int n_error_remain, alpha;
        for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)){
            calc_flip(&flip, &board, cell);
            board.move_board(&flip);
            *player ^= 1;
                board.copy(board_copy);
                alpha = best_move.value - expected_error;
                g = -ai_window(board, level, -best_move.value, -alpha + 1, true);
                if (global_searching && g >= alpha && g <= HW2){
                    n_error_remain = error_remain - max(0, best_move.value - g);
                    if (-HW2 <= expected_value && expected_value <= HW2)
                        n_error_remain -= max(0, expected_value - g);
                    g = -book_learn_search(board, level, book_depth, max(expected_value, -v), expected_error, n_error_remain, board_copy, player, strt_tim, book_file, book_bak);
                    if (global_searching && g >= -HW2 && g <= HW2){
                        v = max(v, g);
                        cerr << "depth " << board.n_discs() - 4 << " AD value " << g << " pre " << best_move.value << " best " << v << " expected " << expected_value << " remaining error " << n_error_remain << endl;
                    }
                }
            *player ^= 1;
            board.undo_board(&flip);
        }
    }
    if (global_searching && v >= -HW2 && v <= HW2){
        cerr << "depth " << board.n_discs() - 4 << " RG value " << v << endl;
        book.reg(board, -v);
    }
    return v;
}

inline void learn_book(Board root_board, int level, const int book_depth, int expected_error, Board *board_copy, int *player, string book_file, string book_bak, bool *book_learning){
    uint64_t strt_tim = tim();
    cerr << "book learn started" << endl;
    int error_remain = max(expected_error, (book_depth + 4 - root_board.n_discs()) * expected_error / 4);
    cerr << "remaining error " << error_remain << endl;
    int g = book_learn_search(root_board, level, book_depth, SCORE_UNDEFINED, expected_error, error_remain, board_copy, player, &strt_tim, book_file, book_bak);
    //if (*book_learning && global_searching)
    //    book.reg(root_board, -g);
    root_board.copy(board_copy);
    cerr << "book learn finished " << g << endl;
    *book_learning = false;
}
