/*
    Egaroucid Project

    @file book_enlarge.hpp
        Enlarging book
    @date 2021-2022
    @author Takuto Yamana (a.k.a. Nyanyan)
    @license GPL-3.0 license
*/

#pragma once
#include <iostream>
#include <unordered_set>
#include "evaluate.hpp"
#include "board.hpp"
#include "ai.hpp"

// automatically save book in this time (milliseconds)
#define AUTO_BOOK_SAVE_TIME 60000

Search_result ai(Board board, int level, bool use_book, bool use_multi_thread, bool show_log, uint8_t date);
int ai_window(Board board, int level, int alpha, int beta, bool use_multi_thread, uint8_t date);

/*
    @brief Get a value of the given board

    This function is a wrapper for convenience

    @param board                board to solve
    @param level                level to search

    @return a score of the board
*/
inline int book_enlarge_calc_value(Board board, int level, uint8_t date){
    return ai(board, level, true, true, false, date).value;
}

/*
    @brief Get adoptable error

    @param level                search level
    @param expected_error       expected error set by users
    @param book_depth           book depth
    @param n_discs              number of discs

    @return adoptable error
*/
int calc_adoptable_error(int level, int expected_error, int book_depth, int n_discs){
    if (get_level_complete_depth(level) >= HW2 - n_discs)
        return 0;
    return std::max(expected_error, (book_depth + 4 - n_discs) * expected_error / 5);
}

/*
    @brief Widen a book recursively

    This function widen the book.
    Register new boards to the book automatically.

    @param board                board to solve
    @param level                level to search
    @param book_depth           depth of the book
    @param expected_value       expected value from other branches
    @param expected_error       expected error of search set by users
    @param remaining_error      sum of errors remaining (for avoid registering many bad moves)
    @param board_copy           board pointer for screen drawing
    @param player               player information for screen drawing
    @param strt_tim             last saved time for auto-saving
    @param book_file            book file name
    @param book_bak             book backup file name
    @param strt                 time of book widen start

    @return a score of the board
*/
int book_widen_search(Board board, int level, const int book_depth, int expected_value, int expected_error, int remaining_error, Board *board_copy, int *player, uint64_t *strt_tim, std::string book_file, std::string book_bak, uint8_t *date, uint64_t strt){
    if (!global_searching || remaining_error < 0)
        return SCORE_UNDEFINED;
    if (tim() - *strt_tim > AUTO_BOOK_SAVE_TIME){
        book.save_bin(book_file, book_bak);
        *strt_tim = tim();
    }
    int g, v = SCORE_UNDEFINED;
    if (board.is_end()){
        g = board.score_player();
        std::cerr << "time " << ms_to_time_short(tim() - strt) << " depth " << board.n_discs() - 4 << " LF value " << g << std::endl;
        book.reg(board, -g);
        return g;
    }
    if (board.n_discs() >= 4 + book_depth){
        g = book_enlarge_calc_value(board, level, *date);
        ++(*date);
        *date = manage_date(*date);
        std::cerr << "time " << ms_to_time_short(tim() - strt) << " depth " << board.n_discs() - 4 << " LF value " << g << std::endl;
        book.reg(board, -g);
        return g;
    }
    if (get_level_complete_depth(level) >= HW2 - board.n_discs())
        remaining_error = 0;
    uint64_t legal = board.get_legal();
    if (legal == 0ULL){
        board.pass();
        *player ^= 1;
            g = -book_widen_search(board, level, book_depth, -expected_value, expected_error, remaining_error, board_copy, player, strt_tim, book_file, book_bak, date, strt);
        *player ^= 1;
        board.pass();
        return g;
    }
    Search_result best_move = ai(board, level, true, true, false, *date);
    if (-HW2 <= expected_value && expected_value <= HW2 && best_move.value < expected_value - expected_error)
        return SCORE_UNDEFINED;
    ++(*date);
    *date = manage_date(*date);
    std::cerr << "time " << ms_to_time_short(tim() - strt) << " depth " << board.n_discs() - 4 << " BM value " << best_move.value << std::endl;
    Flip flip;
    calc_flip(&flip, &board, (uint_fast8_t)best_move.policy);
    board.move_board(&flip);
    *player ^= 1;
        board.copy(board_copy);
        g = -book_widen_search(board, level, book_depth, -expected_value, expected_error, remaining_error, board_copy, player, strt_tim, book_file, book_bak, date, strt);
        if (global_searching && g >= -HW2 && g <= HW2){
            v = g;
            std::cerr << "time " << ms_to_time_short(tim() - strt) << " depth " << board.n_discs() - 4 << " PV value " << g << " expected " << expected_value << " remaining error " << remaining_error << std::endl;
        }
    *player ^= 1;
    board.undo_board(&flip);
    legal ^= 1ULL << best_move.policy;
    if (legal){
        int n_remaining_error, n_expected_value = v, alpha;
        if (-HW2 <= expected_value && expected_value <= HW2)
            n_expected_value = std::max(n_expected_value, expected_value);
        for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)){
            calc_flip(&flip, &board, cell);
            board.move_board(&flip);
            *player ^= 1;
            board.copy(board_copy);
                //alpha = best_move.value - expected_error;
                //g = -ai_window(board, level, -best_move.value, -alpha + 1, true);
                alpha = std::max(-HW2, v - expected_error);
                g = -ai_window(board, level, -v, -alpha + 1, true, *date);
                ++(*date);
                *date = manage_date(*date);
                if (global_searching && g >= alpha && g <= HW2){
                    //n_remaining_error = remaining_error - std::max(0, best_move.value - g);
                    n_remaining_error = remaining_error - std::max(0, v - g);
                    //if (-HW2 <= expected_value && expected_value <= HW2)
                    //    n_remaining_error -= std::max(0, expected_value - g);
                    if (n_remaining_error >= 0){
                        n_expected_value = std::max(n_expected_value, v);
                        g = -book_widen_search(board, level, book_depth, -n_expected_value, expected_error, n_remaining_error, board_copy, player, strt_tim, book_file, book_bak, date, strt);
                        if (global_searching && g >= -HW2 && g <= HW2){
                            v = std::max(v, g);
                            std::cerr << "time " << ms_to_time_short(tim() - strt) << " depth " << board.n_discs() - 4 << " AD value " << g << " pre " << best_move.value << " best " << v << " expected " << expected_value << " remaining error " << n_remaining_error << std::endl;
                        }
                    }
                }
            board.undo_board(&flip);
            board.copy(board_copy);
            *player ^= 1;
        }
    }
    if (global_searching && v >= -HW2 && v <= HW2){
        std::cerr << "time " << ms_to_time_short(tim() - strt) << " depth " << board.n_discs() - 4 << " RG value " << v << std::endl;
        book.reg(board, -v);
    }
    return v;
}

/*
    @brief Widen a book

    Users use mainly this function.

    @param root_board           the root board to register
    @param level                level to search
    @param book_depth           depth of the book
    @param expected_error       expected error of search set by users
    @param board_copy           board pointer for screen drawing
    @param player               player information for screen drawing
    @param book_file            book file name
    @param book_bak             book backup file name
    @param book_learning        a flag for screen drawing
*/
inline void book_widen(Board root_board, int level, const int book_depth, int expected_error, Board *board_copy, int *player, std::string book_file, std::string book_bak, bool *book_learning){
    uint64_t strt_tim = tim();
    uint64_t all_strt = strt_tim;
    std::cerr << "book widen started" << std::endl;
    int remaining_error = calc_adoptable_error(level, expected_error, book_depth, root_board.n_discs());
    std::cerr << "remaining error " << remaining_error << std::endl;
    uint8_t date = INIT_DATE;
    transposition_table.reset_date();
    int g = book_widen_search(root_board, level, book_depth, SCORE_UNDEFINED, expected_error, remaining_error, board_copy, player, &strt_tim, book_file, book_bak, &date, all_strt);
    root_board.copy(board_copy);
    transposition_table.reset_date();
    book.save_bin(book_file, book_bak);
    std::cerr << "time " << ms_to_time_short(tim() - all_strt) << " book widen finished value " << g << std::endl;
    *book_learning = false;
}

/*
    @brief Deepen a book recursively

    This function deepen the book.
    Register new boards to the book automatically.

    @param board                board to solve
    @param level                level to search
    @param book_depth           depth of the book
    @param expected_error       expected error of search set by users
    @param board_copy           board pointer for screen drawing
    @param player               player information for screen drawing
    @param strt_tim             last saved time for auto-saving
    @param book_file            book file name
    @param book_bak             book backup file name
    @param strt                 time of book deepen start

    @return a score of the board
*/
int book_deepen_search(Board board, int level, const int book_depth, int expected_error, Board *board_copy, int *player, uint64_t *strt_tim, std::string book_file, std::string book_bak, uint8_t *date, uint64_t strt){
    if (!global_searching)
        return SCORE_UNDEFINED;
    if (tim() - *strt_tim > AUTO_BOOK_SAVE_TIME){
        book.save_bin(book_file, book_bak);
        *strt_tim = tim();
    }
    int g, v = SCORE_UNDEFINED;
    if (board.is_end())
        return -book.get(&board);
    if (board.n_discs() >= 4 + book_depth)
        return -book.get(&board);
    uint64_t legal = board.get_legal();
    if (legal == 0ULL){
        board.pass();
        *player ^= 1;
            g = -book_deepen_search(board, level, book_depth, expected_error, board_copy, player, strt_tim, book_file, book_bak, date, strt);
        *player ^= 1;
        board.pass();
        return g;
    }
    std::vector<int> best_moves = book.get_all_best_moves(&board);
    int book_val = -book.get(&board);
    if (best_moves.size() == 0)
        return book_widen_search(board, level, book_depth, book_val, expected_error, calc_adoptable_error(level, book_val, book_depth, board.n_discs()), board_copy, player, strt_tim, book_file, book_bak, date, strt);
    Flip flip;
    for (int policy: best_moves){
        calc_flip(&flip, &board, (uint_fast8_t)policy);
        board.move_board(&flip);
        board.copy(board_copy);
        *player ^= 1;
            g = -book_deepen_search(board, level, book_depth, expected_error, board_copy, player, strt_tim, book_file, book_bak, date, strt);
        board.undo_board(&flip);
        board.copy(board_copy);
        *player ^= 1;
        v = std::max(v, g);
    }
    if (global_searching && v >= -HW2 && v <= HW2 && book_val != v){
        std::cerr << "time " << ms_to_time_short(tim() - strt) << " depth " << board.n_discs() - 4 << " RW value " << v << std::endl;
        book.reg(board, -v);
    }
    return v;
}

/*
    @brief Deepen a book

    Users use mainly this function.

    @param root_board           the root board to register
    @param level                level to search
    @param book_depth           depth of the book
    @param expected_error       expected error of search set by users
    @param board_copy           board pointer for screen drawing
    @param player               player information for screen drawing
    @param book_file            book file name
    @param book_bak             book backup file name
    @param book_learning        a flag for screen drawing
*/
inline void book_deepen(Board root_board, int level, const int book_depth, int expected_error, Board *board_copy, int *player, std::string book_file, std::string book_bak, bool *book_learning){
    uint64_t strt_tim = tim();
    uint64_t all_strt = strt_tim;
    std::cerr << "book deepen started" << std::endl;
    uint8_t date = INIT_DATE;
    transposition_table.reset_date();
    int g = book_deepen_search(root_board, level, book_depth, expected_error, board_copy, player, &strt_tim, book_file, book_bak, &date, all_strt);
    root_board.copy(board_copy);
    transposition_table.reset_date();
    book.save_bin(book_file, book_bak);
    std::cerr << "time " << ms_to_time_short(tim() - all_strt) << " book deepen finished value " << g << std::endl;
    *book_learning = false;
}
