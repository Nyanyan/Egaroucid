/*
    Egaroucid Project

    @file book_widen.hpp
        Enlarging book
    @date 2021-2023
    @author Takuto Yamana
    @license GPL-3.0 license
*/

#pragma once
#include <iostream>
#include <unordered_set>
#include "evaluate.hpp"
#include "board.hpp"
#include "ai.hpp"

// automatically save book in this time (milliseconds)
#define AUTO_BOOK_SAVE_TIME 3600000 // 1 hour

Search_result ai(Board board, int level, bool use_book, int book_acc_level, bool use_multi_thread, bool show_log);
int ai_window(Board board, int level, int alpha, int beta, bool use_multi_thread);

struct Book_enlarge_params{
    int level;
    int depth;
    int max_error_per_move;
};

/*
    @brief Get a value of the given board

    This function is a wrapper for convenience

    @param board                board to solve
    @param level                level to search

    @return a score of the board
*/
inline int book_widen_calc_value(Board board, int level){
    return ai(board, level, true, 0, true, false).value;
}

int book_widen_search(Board board, Book_enlarge_params params, int remaining_error, Board *board_copy, int *player, uint64_t *strt_tim, std::string book_file, std::string book_bak, uint64_t strt){
    if (!global_searching)
        return SCORE_UNDEFINED;
    if (tim() - *strt_tim > AUTO_BOOK_SAVE_TIME){
        book.save_bin(book_file, book_bak);
        *strt_tim = tim();
    }
    int g;
    // leaf
    if (board.is_end()){
        g = board.score_player();
        std::cerr << "time " << ms_to_time_short(tim() - strt) << " depth " << board.n_discs() - 4 << " LF value " << g << std::endl;
        book.change(board, g, params.level);
        return g;
    }
    if (board.n_discs() >= 4 + params.depth){
        g = book_widen_calc_value(board, params.level);
        std::cerr << "time " << ms_to_time_short(tim() - strt) << " depth " << board.n_discs() - 4 << " LF value " << g << std::endl;
        book.change(board, g, params.level);
        return g;
    }

    // pass
    uint64_t legal = board.get_legal();
    if (legal == 0ULL){
        board.pass();
        *player ^= 1;
            if (board.get_legal() != 0ULL)
                g = -book_widen_search(board, params, remaining_error, board_copy, player, strt_tim, book_file, book_bak, strt);
            else
                g = -board.score_player();
        *player ^= 1;
        board.pass();
        return g;
    }
    
    // main search
    // first, search best move
    Flip flip;
    int best_value;
    Search_result prev_best_move = ai(board, params.level, true, 0, true, false);
    calc_flip(&flip, &board, prev_best_move.policy);
    board.move_board(&flip);
    board.copy(board_copy);
    *player ^= 1;
        best_value = -book_widen_search(board, params, remaining_error, board_copy, player, strt_tim, book_file, book_bak, strt);
        if (global_searching && -HW2 <= best_value && best_value <= HW2){
            std::cerr << "time " << ms_to_time_short(tim() - strt) << " depth " << board.n_discs() - 4 << " BM value " << best_value << std::endl;
        } else
            return SCORE_UNDEFINED;
    board.undo_board(&flip);
    board.copy(board_copy);
    *player ^= 1;
    legal ^= 1ULL << prev_best_move.policy;

    // second, see book and search moves in book
    std::vector<Book_value> book_best_moves = book.get_all_moves_with_value(&board);
    for (Book_value &elem: book_best_moves){
        if (legal & (1ULL << elem.policy)){
            int n_remaining_error = remaining_error - std::max(0, best_value - elem.value);
            if (n_remaining_error >= 0 && best_value - elem.value <= params.max_error_per_move){
                calc_flip(&flip, &board, (uint_fast8_t)elem.policy);
                board.move_board(&flip);
                board.copy(board_copy);
                *player ^= 1;
                    g = -book_widen_search(board, params, n_remaining_error, board_copy, player, strt_tim, book_file, book_bak, strt);
                    if (global_searching && -HW2 <= g && g <= HW2){
                        std::cerr << "time " << ms_to_time_short(tim() - strt) << " depth " << board.n_discs() - 4 << " BK value " << g << " best " << best_value << " remaining error " << n_remaining_error << std::endl;
                    } else
                        return SCORE_UNDEFINED;
                board.undo_board(&flip);
                board.copy(board_copy);
                *player ^= 1;
                best_value = std::max(best_value, g);
            }
            legal ^= 1ULL << elem.policy;
        }
    }

    // third, other moves
    if (legal){
        for (uint_fast8_t policy = first_bit(&legal); legal; policy = next_bit(&legal)){
            calc_flip(&flip, &board, policy);
            board.move_board(&flip);
            board.copy(board_copy);
            *player ^= 1;
                int pre_value;
                //int alpha = best_value - params.max_error_per_move - 1;
                int alpha = std::max(-HW2, best_value - 1 - std::max(params.max_error_per_move, remaining_error));
                int beta = best_value;
                if (board.get_legal() == 0ULL){
                    board.pass();
                        if (board.get_legal() == 0ULL)
                            pre_value = board.score_player();
                        else
                            pre_value = ai_window(board, params.level, alpha, beta, true);
                    board.pass();
                } else
                    pre_value = -ai_window(board, params.level, -beta, -alpha, true);
                int n_remaining_error = remaining_error - std::max(0, best_value - pre_value);
                if (n_remaining_error >= 0 && best_value - pre_value <= params.max_error_per_move){
                    g = -book_widen_search(board, params, n_remaining_error, board_copy, player, strt_tim, book_file, book_bak, strt);
                    if (global_searching && -HW2 <= g && g <= HW2){
                        std::cerr << "time " << ms_to_time_short(tim() - strt) << " depth " << board.n_discs() - 4 << " OT value " << g << " best " << best_value << " remaining error " << n_remaining_error << std::endl;
                    } else
                        return SCORE_UNDEFINED;
                    best_value = std::max(best_value, g);
                }
            board.undo_board(&flip);
            board.copy(board_copy);
            *player ^= 1;
        }
    }

    if (global_searching && -HW2 <= best_value && best_value <= HW2){
        std::cerr << "time " << ms_to_time_short(tim() - strt) << " depth " << board.n_discs() - 4 << " RG value " << best_value << std::endl;
        book.change(board, best_value, params.level);
    }
    return best_value;
}

/*
    @brief Enlarge book

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
inline void book_widen(Board root_board, int level, int book_depth, int max_error_per_move, int remaining_error, Board *board_copy, int *player, std::string book_file, std::string book_bak, bool *book_learning){
    uint64_t strt_tim = tim();
    uint64_t all_strt = strt_tim;
    std::cerr << "book widen started" << std::endl;
    int before_player = *player;
    Book_enlarge_params params;
    params.depth = book_depth;
    params.level = level;
    params.max_error_per_move = max_error_per_move;
    int g = book_widen_search(root_board, params, remaining_error, board_copy, player, &strt_tim, book_file, book_bak, all_strt);
    root_board.copy(board_copy);
    *player = before_player;
    transposition_table.reset_date();
    //book.fix();
    book.save_bin(book_file, book_bak);
    std::cerr << "time " << ms_to_time_short(tim() - all_strt) << " book widen finished value " << book.get(root_board).value << std::endl;
    *book_learning = false;
}