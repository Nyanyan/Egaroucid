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



void get_book_recalculate_leaf_todo(Book_deviate_todo_elem todo_elem, int book_depth, int max_error_per_move, int lower, int upper, int level, std::unordered_set<Book_deviate_todo_elem, Book_deviate_hash> &todo_list, uint64_t all_strt, bool *book_learning, Board *board_copy, int *player, bool only_illegal){
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
            get_book_recalculate_leaf_todo(todo_elem, book_depth, max_error_per_move, -upper, -lower, level, todo_list, all_strt, book_learning, board_copy, player, only_illegal);
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
    // already searched?
    if (book_elem.seen)
        return;
    book.flag_book_elem(todo_elem.board);
    // add to list
    std::vector<Book_value> links = book.get_all_moves_with_value(&todo_elem.board);
    bool illegal_leaf = false;
    uint64_t remaining_legal = todo_elem.board.get_legal();
    for (Book_value &link: links)
        remaining_legal ^= 1ULL << link.policy;
    if (remaining_legal)
        illegal_leaf = (remaining_legal & (1ULL << book_elem.leaf.move)) == 0;
    else
        illegal_leaf = book_elem.leaf.move != MOVE_NOMOVE;
    if ((!only_illegal && book_elem.leaf.level < level) || illegal_leaf){
        todo_list.emplace(todo_elem);
        if (todo_list.size() % 100 == 0)
            std::cerr << "book recalculate leaf todo " << todo_list.size() << " calculating... time " << ms_to_time_short(tim() - all_strt) << std::endl;
    }
    // expand links
    if (lower <= book_elem.value){
        Flip flip;
        for (Book_value &link: links){
            if (link.value >= book_elem.value - max_error_per_move){
                calc_flip(&flip, &todo_elem.board, link.policy);
                todo_elem.move(&flip);
                    get_book_recalculate_leaf_todo(todo_elem, book_depth, max_error_per_move, -upper, -lower, level, todo_list, all_strt, book_learning, board_copy, player, only_illegal);
                todo_elem.undo(&flip);
            }
        }
    }
}

void book_search_leaf(Board board, int level, bool use_multi_thread){
    book.search_leaf(board, level, use_multi_thread);
}

void book_recalculate_leaves(int level, std::unordered_set<Book_deviate_todo_elem, Book_deviate_hash> &todo_list, uint64_t all_strt, uint64_t strt, bool *book_learning, Board *board_copy, int *player){
    int n_all = todo_list.size();
    if (n_all == 0)
        return;
    int n_done = 0, n_doing = 0;
    std::vector<std::future<void>> tasks;
    for (Book_deviate_todo_elem elem: todo_list){
        if (!global_searching || !(*book_learning))
            break;
        *board_copy = elem.board;
        *player = elem.player;
        bool use_multi_thread = (n_all - n_doing) < thread_pool.size() * 2 / 3;
        bool pushed;
        ++n_doing;
        tasks.emplace_back(thread_pool.push(&pushed, std::bind(&book_search_leaf, elem.board, level, use_multi_thread)));
        if (!pushed){
            tasks.pop_back();
            book_search_leaf(elem.board, level, true);
            ++n_done;
            if (n_done % 10 == 0){
                int percent = 100ULL * n_done / n_all;
                uint64_t eta = (tim() - strt) * ((double)n_all / n_done - 1.0);
                std::cerr << "book recalculating leaves " << percent << "% " <<  n_done << "/" << n_all << " time " << ms_to_time_short(tim() - all_strt) << " ETA " << ms_to_time_short(eta) << std::endl;
            }
        }
        int tasks_size = tasks.size();
        for (int i = 0; i < tasks_size; ++i){
            if (tasks[i].valid()){
                if (tasks[i].wait_for(std::chrono::seconds(0)) == std::future_status::ready){
                    tasks[i].get();
                    ++n_done;
                    if (n_done % 10 == 0){
                        int percent = 100ULL * n_done / n_all;
                        uint64_t eta = (tim() - strt) * ((double)n_all / n_done - 1.0);
                        std::cerr << "book recalculating leaves " << percent << "% " <<  n_done << "/" << n_all << " time " << ms_to_time_short(tim() - all_strt) << " ETA " << ms_to_time_short(eta) << std::endl;
                    }
                }
            }
        }
        for (int i = 0; i < tasks_size; ++i){
            if (i >= tasks.size())
                break;
            if (!tasks[i].valid()){
                tasks.erase(tasks.begin() + i);
                --i;
            }
        }
    }
    int tasks_size = tasks.size();
    for (int i = 0; i < tasks_size; ++i){
        if (tasks[i].valid()){
            tasks[i].get();
            ++n_done;
            if (n_done % 10 == 0){
                int percent = 100ULL * n_done / n_all;
                uint64_t eta = (tim() - strt) * ((double)n_all / n_done - 1.0);
                std::cerr << "book recalculating leaves " << percent << "% " <<  n_done << "/" << n_all << " time " << ms_to_time_short(tim() - all_strt) << " ETA " << ms_to_time_short(eta) << std::endl;
            }
        }
    }
    int percent = 100ULL * n_done / n_all;
    uint64_t eta = (tim() - strt) * ((double)n_all / n_done - 1.0);
    std::cerr << "book recalculating leaves finished " << percent << "% " <<  n_done << "/" << n_all << " time " << ms_to_time_short(tim() - all_strt) << " ETA " << ms_to_time_short(eta) << std::endl;
}

inline void book_recalculate_leaf(Board root_board, int level, int book_depth, int max_error_per_move, int max_error_sum, Board *board_copy, int *player, bool *book_learning, bool only_illegal){
    uint64_t strt_tim = tim();
    uint64_t all_strt = strt_tim;
    int before_player = *player;
    Book_deviate_todo_elem root_elem;
    root_elem.board = root_board;
    root_elem.player = *player;
    int n_saved = 1;
    Book_elem book_elem = book.get(root_board);
    if (book_elem.value == SCORE_UNDEFINED){
        std::cerr << "board not registered, searching..." << std::endl;
        book_elem.value = ai(root_board, level, true, 0, true, false).value;
    }
    int lower = book_elem.value - max_error_sum;
    int upper = book_elem.value + max_error_sum;
    if (lower < -SCORE_MAX)
        lower = -SCORE_MAX;
    if (upper > SCORE_MAX)
        upper = SCORE_MAX;
    bool stop = false;
    book.reset_seen();
    std::unordered_set<Book_deviate_todo_elem, Book_deviate_hash> book_recalculate_leaf_todo;
    get_book_recalculate_leaf_todo(root_elem, book_depth, max_error_per_move, lower, upper, level, book_recalculate_leaf_todo, all_strt, book_learning, board_copy, player, only_illegal);
    std::cerr << "book recalculate leaf todo " << book_recalculate_leaf_todo.size() << " calculated time " << ms_to_time_short(tim() - all_strt) << std::endl;
    if (book_recalculate_leaf_todo.size()){
        uint64_t strt = tim();
        book_recalculate_leaves(level, book_recalculate_leaf_todo, all_strt, strt, book_learning, board_copy, player);
    }
    book.reset_seen();
    root_board.copy(board_copy);
    *player = before_player;
    std::cerr << "recalculate leaf finished value " << (int)book.get(root_board).value << " time " << ms_to_time_short(tim() - all_strt) << std::endl;
    *book_learning = false;
}




void get_book_deviate_todo(Book_deviate_todo_elem todo_elem, int book_depth, int max_error_per_move, int lower, int upper, std::unordered_set<Book_deviate_todo_elem, Book_deviate_hash> &book_deviate_todo, uint64_t all_strt, bool *book_learning, Board *board_copy, int *player, int n_loop){
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
            get_book_deviate_todo(todo_elem, book_depth, max_error_per_move, -upper, -lower, book_deviate_todo, all_strt, book_learning, board_copy, player, n_loop);
        todo_elem.pass();
        return;
    }
    // already searched?
    if (book_deviate_todo.find(todo_elem) != book_deviate_todo.end())
        return;
    // check depth
    if (todo_elem.board.n_discs() >= book_depth + 4)
        return;
    Book_elem book_elem = book.get(todo_elem.board);
    // already seen
    if (book_elem.seen)
        return;
    book.flag_book_elem(todo_elem.board);
    if (lower <= book_elem.value){
        // expand links
        if (todo_elem.board.n_discs() < book_depth + 4){
            std::vector<Book_value> links = book.get_all_moves_with_value(&todo_elem.board);
            Flip flip;
            for (Book_value &link: links){
                if (link.value >= book_elem.value - max_error_per_move){
                    calc_flip(&flip, &todo_elem.board, link.policy);
                    todo_elem.move(&flip);
                        get_book_deviate_todo(todo_elem, book_depth, max_error_per_move, -upper, -lower, book_deviate_todo, all_strt, book_learning, board_copy, player, n_loop);
                    todo_elem.undo(&flip);
                }
            }
        }
        // check leaf
        if (book_elem.leaf.value >= book_elem.value - max_error_per_move && lower <= book_elem.leaf.value && is_valid_policy(book_elem.leaf.move)){
            if (todo_elem.board.get_legal() & (1ULL << book_elem.leaf.move)){ // is leaf legal?
                book_deviate_todo.emplace(todo_elem);
                if (book_deviate_todo.size() % 10 == 0)
                    std::cerr << "loop " << n_loop << " book deviate todo " << book_deviate_todo.size() << " calculating... time " << ms_to_time_short(tim() - all_strt) << std::endl;
            }
        }
    }
}

void expand_leaf(int book_depth, int level, Board board, bool use_multi_thread){
    Book_elem book_elem = book.get(board);
    Flip flip;
    calc_flip(&flip, &board, book_elem.leaf.move);
    board.move_board(&flip);
        if (!book.contain(&board)){ 
            Search_result search_result = ai(board, level, true, 0, use_multi_thread, false);
            if (-HW2 <= search_result.value && search_result.value <= HW2){
                book.change(board, search_result.value, level);
                if (search_result.depth >= HW2 - board.n_discs() && search_result.probability == 100){ // perfect serach
                    Board board_copy = board.copy();
                    Flip flip;
                    int val, best_move;
                    while (board_copy.n_discs() <= book_depth + 4 && transposition_table.get_if_perfect(&board_copy, board_copy.hash(), &val, &best_move)){
                        if (board_copy != board)
                            book.change(&board_copy, val, level);
                        if ((board_copy.get_legal() & (1ULL << best_move)) == 0)
                            break;
                        book.add_leaf(&board_copy, val, best_move, level);
                        calc_flip(&flip, &board_copy, best_move);
                        board_copy.move_board(&flip);
                    }
                } else if (0 <= search_result.policy && search_result.policy < HW2 && (board.get_legal() & (1ULL << search_result.policy)))
                    book.add_leaf(&board, search_result.value, search_result.policy, level);
            }
        }
    board.undo_board(&flip);
}

void expand_leaves(int book_depth, int level, std::unordered_set<Book_deviate_todo_elem, Book_deviate_hash> &book_deviate_todo, uint64_t all_strt, uint64_t strt, bool *book_learning, Board *board_copy, int *player, int n_loop, std::string file, std::string bak_file){
    int n_all = book_deviate_todo.size();
    if (n_all == 0)
        return;
    int n_done = 0;
    int n_doing = 0;
    const int n_threads = thread_pool.size();
    std::vector<std::future<void>> tasks;
    uint64_t s = tim();
    for (Book_deviate_todo_elem elem: book_deviate_todo){
        if (tim() - s >= AUTO_BOOK_SAVE_TIME){
            book.save_egbk3(file, bak_file);
            s = tim();
        }
        if (!global_searching || !(*book_learning))
            break;
        *board_copy = elem.board;
        *player = elem.player;
        bool use_multi_thread = (n_all - n_doing) < thread_pool.size() * 2 / 3;
        bool pushed;
        ++n_doing;
        //std::cerr << use_multi_thread << std::endl;
        //elem.board.print();
        if (use_multi_thread){
            expand_leaf(book_depth, level, elem.board, true);
            ++n_done;
            int percent = 100ULL * n_done / n_all;
            uint64_t eta = (tim() - strt) * ((double)n_all / n_done - 1.0);
            std::cerr << "loop " << n_loop << " book deviating " << percent << "% " <<  n_done << "/" << n_all << " time " << ms_to_time_short(tim() - all_strt) << " ETA " << ms_to_time_short(eta) << std::endl;
        } else{
            tasks.emplace_back(thread_pool.push(&pushed, std::bind(&expand_leaf, book_depth, level, elem.board, use_multi_thread)));
            if (!pushed){
                //std::cerr << "deleted " << true << std::endl;
                //elem.board.print();
                tasks.pop_back();
                expand_leaf(book_depth, level, elem.board, true);
                ++n_done;
                int percent = 100ULL * n_done / n_all;
                uint64_t eta = (tim() - strt) * ((double)n_all / n_done - 1.0);
                std::cerr << "loop " << n_loop << " book deviating " << percent << "% " <<  n_done << "/" << n_all << " time " << ms_to_time_short(tim() - all_strt) << " ETA " << ms_to_time_short(eta) << std::endl;
            }
        }
        int tasks_size = tasks.size();
        for (int i = 0; i < tasks_size; ++i){
            if (tasks[i].valid()){
                if (tasks[i].wait_for(std::chrono::seconds(0)) == std::future_status::ready){
                    tasks[i].get();
                    ++n_done;
                    int percent = 100ULL * n_done / n_all;
                    uint64_t eta = (tim() - strt) * ((double)n_all / n_done - 1.0);
                    std::cerr << "loop " << n_loop << " book deviating " << percent << "% " <<  n_done << "/" << n_all << " time " << ms_to_time_short(tim() - all_strt) << " ETA " << ms_to_time_short(eta) << std::endl;
                }
            }
        }
        for (int i = 0; i < tasks_size; ++i){
            if (i >= tasks.size())
                break;
            if (!tasks[i].valid()){
                tasks.erase(tasks.begin() + i);
                --i;
            }
        }
    }
    int tasks_size = tasks.size();
    for (int i = 0; i < tasks_size; ++i){
        if (tasks[i].valid()){
            tasks[i].get();
            ++n_done;
            int percent = 100ULL * n_done / n_all;
            uint64_t eta = (tim() - strt) * ((double)n_all / n_done - 1.0);
            std::cerr << "loop " << n_loop << " book deviating " << percent << "% " <<  n_done << "/" << n_all << " time " << ms_to_time_short(tim() - all_strt) << " ETA " << ms_to_time_short(eta) << std::endl;
        }
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
    int n_loop = 0;
    while (true){
        ++n_loop;
        if (tim() - all_strt > AUTO_BOOK_SAVE_TIME * n_saved){
            ++n_saved;
            book.save_egbk3(book_file, book_bak);
        }
        Book_elem book_elem = book.get(root_board);
        if (book_elem.value == SCORE_UNDEFINED){
            std::cerr << "board not registered, searching..." << std::endl;
            book_elem.value = ai(root_board, level, true, 0, true, false).value;
        }
        int lower = book_elem.value - max_error_sum;
        int upper = book_elem.value + max_error_sum;
        if (lower < -SCORE_MAX)
            lower = -SCORE_MAX;
        if (upper > SCORE_MAX)
            upper = SCORE_MAX;
        std::cerr << "search [" << lower << ", " << (int)book_elem.value << ", " << upper << "]" << std::endl;
        bool book_recalculating = true;
        book_recalculate_leaf(root_board, std::max(1, level * 2 / 3), std::max(1, book_depth - 1), max_error_per_move, max_error_sum, board_copy, player, &book_recalculating, true);
        //bool stop = false;
        //book.check_add_leaf_all_search(std::max(1, level * 2 / 3), &stop);
        std::unordered_set<Book_deviate_todo_elem, Book_deviate_hash> book_deviate_todo;
        book.reset_seen();
        get_book_deviate_todo(root_elem, book_depth, max_error_per_move, lower, upper, book_deviate_todo, all_strt, book_learning, board_copy, player, n_loop);
        book.reset_seen();
        std::cerr << "loop " << n_loop << " book deviate todo " << book_deviate_todo.size() << " calculated time " << ms_to_time_short(tim() - all_strt) << std::endl;
        if (book_deviate_todo.size() == 0)
            break;
        uint64_t strt = tim();
        expand_leaves(book_depth, level, book_deviate_todo, all_strt, strt, book_learning, board_copy, player, n_loop, book_file, book_bak);
        std::cerr << "loop " << n_loop << " book deviated size " << book.size() << std::endl;
    }
    //bool stop = false;
    //book.check_add_leaf_all_search(std::max(1, level / 2), &stop);
    root_board.copy(board_copy);
    *player = before_player;
    transposition_table.reset_date();
    //book.fix();
    book.save_egbk3(book_file, book_bak);
    std::cerr << "book deviate finished value " << (int)book.get(root_board).value << " time " << ms_to_time_short(tim() - all_strt) << std::endl;
    *book_learning = false;
}