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
#define AUTO_BOOK_SAVE_TIME 3600000ULL // 1 hour

Search_result ai(Board board, int level, bool use_book, int book_acc_level, bool use_multi_thread, bool show_log);
int ai_window(Board board, int level, int alpha, int beta, bool use_multi_thread);

struct Book_deviate_todo_elem{
    Board board;
    int player;

    void move(Flip *flip){
        board.move_board(flip);
        player ^= 1;
    }

    void undo(Flip *flip){
        board.undo_board(flip);
        player ^= 1;
    }

    void pass(){
        board.pass();
        player ^= 1;
    }
};

bool operator==(const Book_deviate_todo_elem& a, const Book_deviate_todo_elem& b){
    return a.board == b.board;
}

struct Book_deviate_hash {
    size_t operator()(Book_deviate_todo_elem elem) const{
        const uint16_t *p = (uint16_t*)&elem.board.player;
        const uint16_t *o = (uint16_t*)&elem.board.opponent;
        return 
            hash_rand_player_book[0][p[0]] ^ 
            hash_rand_player_book[1][p[1]] ^ 
            hash_rand_player_book[2][p[2]] ^ 
            hash_rand_player_book[3][p[3]] ^ 
            hash_rand_opponent_book[0][o[0]] ^ 
            hash_rand_opponent_book[1][o[1]] ^ 
            hash_rand_opponent_book[2][o[2]] ^ 
            hash_rand_opponent_book[3][o[3]];
    }
};

void get_book_deviate_todo(Book_deviate_todo_elem todo_elem, int book_depth, int max_error_per_move, int lower, int upper, std::unordered_set<Book_deviate_todo_elem, Book_deviate_hash> &book_deviate_todo, uint64_t all_strt, bool *book_learning, Board *board_copy, int *player){
    if (!global_searching || !(*book_learning))
        return;
    todo_elem.board = book.get_representative_board(todo_elem.board);
    *board_copy = todo_elem.board;
    *player = todo_elem.player;
    // pass?
    if (todo_elem.board.get_legal() == 0){
        todo_elem.pass();
        if (todo_elem.board.get_legal() == 0)
            return; // game over
            get_book_deviate_todo(todo_elem, book_depth, max_error_per_move, -upper, -lower, book_deviate_todo, all_strt, book_learning, board_copy, player);
        todo_elem.pass();
        return;
    }
    // already searched?
    if (book_deviate_todo.find(todo_elem) != book_deviate_todo.end())
        return;
    // check depth
    if (todo_elem.board.n_discs() > book_depth + 4)
        return;
    Book_elem book_elem = book.get(todo_elem.board);
    // expand links
    if (lower <= book_elem.value){
        std::vector<Book_value> links = book.get_all_moves_with_value(&todo_elem.board);
        Flip flip;
        for (Book_value &link: links){
            if (link.value >= book_elem.value - max_error_per_move){
                calc_flip(&flip, &todo_elem.board, link.policy);
                todo_elem.move(&flip);
                    get_book_deviate_todo(todo_elem, book_depth, max_error_per_move, -upper, -lower, book_deviate_todo, all_strt, book_learning, board_copy, player);
                todo_elem.undo(&flip);
            }
        }
    }
    // check leaf
    if (book_elem.leaf.value >= book_elem.value - max_error_per_move && is_valid_policy(book_elem.leaf.move) && lower <= book_elem.leaf.value){
        book_deviate_todo.emplace(todo_elem);
        if (book_deviate_todo.size() % 10 == 0)
            std::cerr << "book deviate todo " << book_deviate_todo.size() << " calculating... time " << ms_to_time_short(tim() - all_strt) << std::endl;
    }
}

void expand_leaf(int level, Board board){
    Book_elem book_elem = book.get(board);
    Flip flip;
    calc_flip(&flip, &board, book_elem.leaf.move);
    board.move_board(&flip);
        if (!book.contain(&board)){ 
            Search_result search_result = ai(board, level, true, 0, true, false);
            book.change(board, search_result.value);
        }
    board.undo_board(&flip);
}

void expand_leaves(int level, std::unordered_set<Book_deviate_todo_elem, Book_deviate_hash> &book_deviate_todo, uint64_t all_strt, bool *book_learning, Board *board_copy, int *player){
    int n_all = book_deviate_todo.size();
    int i = 0;
    for (Book_deviate_todo_elem elem: book_deviate_todo){
        if (!global_searching || !(*book_learning))
            break;
        *board_copy = elem.board;
        *player = elem.player;
        expand_leaf(level, elem.board);
        ++i;
        int percent = 100ULL * i / n_all;
        std::cerr << "book deviating " << percent << "% " <<  i << "/" << n_all << " time " << ms_to_time_short(tim() - all_strt) << std::endl;
    }
}

/*
    @brief Deviate book

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
    Book_deviate_todo_elem root_elem;
    root_elem.board = root_board;
    root_elem.player = *player;
    int n_saved = 1;
    while (true){
        if (tim() - all_strt > AUTO_BOOK_SAVE_TIME * n_saved){
            ++n_saved;
            book.save_egbk3(book_file, book_bak);
        }
        Book_elem book_elem = book.get(root_board);
        if (book_elem.value == SCORE_UNDEFINED)
            book_elem.value = ai(root_board, level, true, 0, true, true).value;
        int lower = book_elem.value - max_error_sum;
        int upper = book_elem.value + max_error_sum;
        if (lower < -SCORE_MAX)
            lower = -SCORE_MAX;
        if (upper > SCORE_MAX)
            upper = SCORE_MAX;
        bool stop = false;
        book.add_leaf_all_search(level / 2, &stop);
        std::unordered_set<Book_deviate_todo_elem, Book_deviate_hash> book_deviate_todo;
        get_book_deviate_todo(root_elem, book_depth, max_error_per_move, lower, upper, book_deviate_todo, all_strt, book_learning, board_copy, player);
        std::cerr << "book deviate todo " << book_deviate_todo.size() << " calculated time " << ms_to_time_short(tim() - all_strt) << std::endl;
        if (book_deviate_todo.size() == 0)
            break;
        expand_leaves(level, book_deviate_todo, all_strt, book_learning, board_copy, player);
        //book.fix();
        std::cerr << "book deviated size " << book.size() << std::endl;
    }
    root_board.copy(board_copy);
    *player = before_player;
    transposition_table.reset_date();
    book.save_egbk3(book_file, book_bak);
    std::cerr << "book deviate finished value " << book.get(root_board).value << " time " << ms_to_time_short(tim() - all_strt) << std::endl;
    *book_learning = false;
}



void get_book_recalculate_leaf_todo(Book_deviate_todo_elem todo_elem, int book_depth, int max_error_per_move, int lower, int upper, std::unordered_set<Book_deviate_todo_elem, Book_deviate_hash> &todo_list, uint64_t all_strt, bool *book_learning, Board *board_copy, int *player){
    if (!global_searching || !(*book_learning))
        return;
    todo_elem.board = book.get_representative_board(todo_elem.board);
    *board_copy = todo_elem.board;
    *player = todo_elem.player;
    // pass?
    if (todo_elem.board.get_legal() == 0){
        todo_elem.pass();
        if (todo_elem.board.get_legal() == 0)
            return; // game over
            get_book_recalculate_leaf_todo(todo_elem, book_depth, max_error_per_move, -upper, -lower, todo_list, all_strt, book_learning, board_copy, player);
        todo_elem.pass();
        return;
    }
    // already searched?
    if (todo_list.find(todo_elem) != todo_list.end())
        return;
    // check depth
    if (todo_elem.board.n_discs() > book_depth + 4)
        return;
    Book_elem book_elem = book.get(todo_elem.board);
    // add to list
    todo_list.emplace(todo_elem);
    if (todo_list.size() % 10 == 0)
        std::cerr << "book recalculate leaf todo " << todo_list.size() << " calculating... time " << ms_to_time_short(tim() - all_strt) << std::endl;
    // expand links
    if (lower <= book_elem.value){
        std::vector<Book_value> links = book.get_all_moves_with_value(&todo_elem.board);
        Flip flip;
        for (Book_value &link: links){
            if (link.value >= book_elem.value - max_error_per_move){
                calc_flip(&flip, &todo_elem.board, link.policy);
                todo_elem.move(&flip);
                    get_book_recalculate_leaf_todo(todo_elem, book_depth, max_error_per_move, -upper, -lower, todo_list, all_strt, book_learning, board_copy, player);
                todo_elem.undo(&flip);
            }
        }
    }
}

void book_recalculate_leaves(int level, std::unordered_set<Book_deviate_todo_elem, Book_deviate_hash> &todo_list, uint64_t all_strt, bool *book_learning, Board *board_copy, int *player){
    int n_all = todo_list.size();
    int i = 0;
    for (Book_deviate_todo_elem elem: todo_list){
        if (!global_searching || !(*book_learning))
            break;
        *board_copy = elem.board;
        *player = elem.player;
        book.search_leaf(elem.board, level);
        ++i;
        int percent = 100ULL * i / n_all;
        std::cerr << "book recalculating leaves " << percent << "% " <<  i << "/" << n_all << " time " << ms_to_time_short(tim() - all_strt) << std::endl;
    }
}

inline void book_recalculate_leaf(Board root_board, int level, int book_depth, int max_error_per_move, int max_error_sum, Board *board_copy, int *player, bool *book_learning){
    uint64_t strt_tim = tim();
    uint64_t all_strt = strt_tim;
    int before_player = *player;
    Book_deviate_todo_elem root_elem;
    root_elem.board = root_board;
    root_elem.player = *player;
    int n_saved = 1;
    Book_elem book_elem = book.get(root_board);
    if (book_elem.value == SCORE_UNDEFINED)
        book_elem.value = ai(root_board, level, true, 0, true, true).value;
    int lower = book_elem.value - max_error_sum;
    int upper = book_elem.value + max_error_sum;
    if (lower < -SCORE_MAX)
        lower = -SCORE_MAX;
    if (upper > SCORE_MAX)
        upper = SCORE_MAX;
    bool stop = false;
    std::unordered_set<Book_deviate_todo_elem, Book_deviate_hash> book_recalculate_leaf_todo;
    get_book_recalculate_leaf_todo(root_elem, book_depth, max_error_per_move, lower, upper, book_recalculate_leaf_todo, all_strt, book_learning, board_copy, player);
    book_recalculate_leaves(level, book_recalculate_leaf_todo, all_strt, book_learning, board_copy, player);
    book.add_leaf_all_search(level, &stop);
    root_board.copy(board_copy);
    *player = before_player;
    std::cerr << "recalculate leaf finished value " << book.get(root_board).value << " time " << ms_to_time_short(tim() - all_strt) << std::endl;
    *book_learning = false;
}