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

struct Book_deviate_params{
    int level;
    int depth;
    int max_error_per_move;
};



void get_book_deviate_todo(Board board, int book_depth, int max_error_per_move, int lower, int upper, std::unordered_set<Board, Book_hash> &book_deviate_todo, uint64_t all_strt, bool *book_learning){
    if (!global_searching || !(*book_learning))
        return;
    // pass?
    if (board.get_legal() == 0){
        board.pass();
        if (board.get_legal() == 0)
            return; // game over
        get_book_deviate_todo(board, book_depth, max_error_per_move, -upper, -lower, book_deviate_todo, all_strt, book_learning);
    }
    // already searched?
    if (book_deviate_todo.find(board) != book_deviate_todo.end())
        return;
    // check depth
    if (board.n_discs() > book_depth + 4)
        return;
    Book_elem book_elem = book.get(board);
    // expand links
    if (lower <= book_elem.value && book_elem.value <= upper){
        std::vector<Book_value> links = book.get_all_moves_with_value(&board);
        Flip flip;
        for (Book_value &link: links){
            if (link.value >= book_elem.value - max_error_per_move){
                calc_flip(&flip, &board, link.policy);
                board.move_board(&flip);
                    get_book_deviate_todo(board, book_depth, max_error_per_move, -upper, -lower, book_deviate_todo, all_strt, book_learning);
                board.undo_board(&flip);
            }
        }
    }
    // check leaf
    if (book_elem.leaf.value >= book_elem.value - max_error_per_move && lower <= book_elem.leaf.value && book_elem.leaf.value <= upper){
        book_deviate_todo.emplace(board);
        if (book_deviate_todo.size() % 10 == 0)
            std::cerr << "book deviate todo " << book_deviate_todo.size() << " calculating... time " << tim() - all_strt << " ms" << std::endl;
    }
}

void expand_leaf(int level, Board board){
    Book_elem book_elem = book.get(board);
    Flip flip;
    calc_flip(&flip, &board, book_elem.leaf.move);
    board.move_board(&flip);
        if (!book.contain(&board)){ 
            Search_result search_result = ai(board, level, false, 0, true, false);
            book.change(board, search_result.value);
        }
    board.undo_board(&flip);
}

void expand_leafs(int level, std::unordered_set<Board, Book_hash> &book_deviate_todo, uint64_t all_strt, bool *book_learning){
    int n_all = book_deviate_todo.size();
    int i = 0;
    for (Board board: book_deviate_todo){
        if (!global_searching || !(*book_learning))
            break;
        expand_leaf(level, board);
        std::cerr << "book deviating " << ++i << "/" << n_all << " time " << tim() - all_strt << " ms" << std::endl;
    }
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
inline void book_deviate(Board root_board, int level, int book_depth, int max_error_per_move, int max_error_sum, Board *board_copy, int *player, std::string book_file, std::string book_bak, bool *book_learning){
    uint64_t strt_tim = tim();
    uint64_t all_strt = strt_tim;
    std::cerr << "book deviate started" << std::endl;
    int before_player = *player;
    Book_deviate_params params;
    params.depth = book_depth;
    params.level = level;
    params.max_error_per_move = max_error_per_move;
    Book_elem book_elem = book.get(root_board);
    if (book_elem.value == SCORE_UNDEFINED)
        book_elem.value = ai(root_board, level, true, 0, true, true).value;
    int lower = book_elem.value - max_error_sum;
    int upper = book_elem.value + max_error_sum;
    if (lower < -SCORE_MAX)
        lower = -SCORE_MAX;
    if (upper > SCORE_MAX)
        upper = SCORE_MAX;
    while (true){
        std::unordered_set<Board, Book_hash> book_deviate_todo;
        get_book_deviate_todo(root_board, book_depth, max_error_per_move, lower, upper, book_deviate_todo, all_strt, book_learning);
        std::cerr << "book deviate todo " << book_deviate_todo.size() << " calculated time " << tim() - all_strt << " ms" << std::endl;
        if (book_deviate_todo.size() == 0)
            break;
        expand_leafs(level, book_deviate_todo, all_strt, book_learning);
        book.add_leaf_all_search(level, book_learning);
    }
    root_board.copy(board_copy);
    *player = before_player;
    transposition_table.reset_date();
    //book.fix();
    book.save_egbk3(book_file, book_bak);
    std::cerr << "book deviate finished value " << book.get(root_board).value << " time " << ms_to_time_short(tim() - all_strt) << std::endl;
    *book_learning = false;
}