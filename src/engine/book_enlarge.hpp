/*
    Egaroucid Project

    @file book_widen.hpp
        Enlarging book
    @date 2021-2024
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
    int max_error_per_move;
    int remaining_error;

    void move(Flip *flip, int error){
        board.move_board(flip);
        player ^= 1;
        if (error > 0)
            remaining_error -= error;
    }

    void undo(Flip *flip, int error){
        board.undo_board(flip);
        player ^= 1;
        if (error > 0)
            remaining_error += error;
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



void get_book_recalculate_leaf_todo(Book_deviate_todo_elem todo_elem, int book_depth, int level, std::unordered_set<Book_deviate_todo_elem, Book_deviate_hash> &todo_list, uint64_t all_strt, bool *book_learning, Board *board_copy, int *player, bool only_illegal){
    if (!global_searching || !(*book_learning))
        return;
    // pass?
    if (todo_elem.board.get_legal() == 0){
        todo_elem.pass();
        if (todo_elem.board.get_legal() == 0)
            return; // game over
    }
    // check depth
    if (todo_elem.board.n_discs() > book_depth + 4)
        return;
    // already searched?
    if (todo_list.find(todo_elem) != todo_list.end())
        return;
    Book_elem book_elem = book.get(todo_elem.board);
    // not registered?
    if (book_elem.value == SCORE_UNDEFINED)
        return;
    // already registered?
    if (book_elem.seen)
        return;
    book.flag_book_elem(todo_elem.board);
    todo_elem.board = book.get_representative_board(todo_elem.board);
    *board_copy = todo_elem.board;
    *player = todo_elem.player;
    // check leaf recalculation necessity
    std::vector<Book_value> links = book.get_all_moves_with_value(&todo_elem.board);
    uint64_t remaining_legal = todo_elem.board.get_legal();
    for (Book_value &link: links)
        remaining_legal &= ~(1ULL << link.policy);
    bool illegal_leaf = false;
    if (remaining_legal){
        if (is_valid_policy(book_elem.leaf.move)){
            if ((remaining_legal & (1ULL << book_elem.leaf.move)) == 0)
                illegal_leaf = true;
        } else
            illegal_leaf = true;
    } else if (book_elem.leaf.move != MOVE_NOMOVE){
        illegal_leaf = true;
    }
    // add to list
    if (illegal_leaf || (!only_illegal && book_elem.leaf.level < level)){
        todo_list.emplace(todo_elem);
        if (todo_list.size() % 100 == 0)
            std::cerr << "book recalculate leaf todo " << todo_list.size() << " calculating... time " << ms_to_time_short(tim() - all_strt) << std::endl;
    }
    // expand links
    Flip flip;
    for (Book_value &link: links){
        int link_error = book_elem.value - link.value;
        if (link_error <= todo_elem.max_error_per_move && link_error <= todo_elem.remaining_error){
            calc_flip(&flip, &todo_elem.board, link.policy);
            todo_elem.move(&flip, link_error);
                get_book_recalculate_leaf_todo(todo_elem, book_depth, level, todo_list, all_strt, book_learning, board_copy, player, only_illegal);
            todo_elem.undo(&flip, link_error);
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
        bool use_multi_thread = (n_all - n_doing) < thread_pool.size() * 2;
        bool pushed;
        ++n_doing;
        if (use_multi_thread){
            book_search_leaf(elem.board, level, true);
            ++n_done;
            if (n_done % 10 == 0){
                int percent = 100ULL * n_done / n_all;
                uint64_t eta = (tim() - strt) * ((double)n_all / n_done - 1.0);
                std::cerr << "book recalculating leaves " << percent << "% " <<  n_done << "/" << n_all << " time " << ms_to_time_short(tim() - all_strt) << " ETA " << ms_to_time_short(eta) << std::endl;
            }
        } else{
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
    root_elem.max_error_per_move = max_error_per_move;
    root_elem.remaining_error = max_error_sum;
    int n_saved = 1;
    Book_elem book_elem = book.get(root_board);
    if (book_elem.value == SCORE_UNDEFINED){
        std::cerr << "board not registered, searching..." << std::endl;
        book_elem.value = ai(root_board, level, true, 0, true, false).value;
    }

    bool stop = false;
    book.reset_seen();
    std::unordered_set<Book_deviate_todo_elem, Book_deviate_hash> book_recalculate_leaf_todo;
    get_book_recalculate_leaf_todo(root_elem, book_depth, level, book_recalculate_leaf_todo, all_strt, book_learning, board_copy, player, only_illegal);
    std::cerr << "book recalculate leaf todo " << book_recalculate_leaf_todo.size() << " calculated time " << ms_to_time_short(tim() - all_strt) << std::endl;
    if (book_recalculate_leaf_todo.size()){
        uint64_t strt = tim();
        book_recalculate_leaves(level, book_recalculate_leaf_todo, all_strt, strt, book_learning, board_copy, player);
    }
    book.reset_seen();
    root_board.copy(board_copy);
    *player = before_player;
    std::cerr << "recalculate leaf finished " << book_recalculate_leaf_todo.size() << " boards time " << ms_to_time_short(tim() - all_strt) << std::endl;
    *book_learning = false;
}




void get_book_deviate_todo(Book_deviate_todo_elem todo_elem, int book_depth, std::unordered_set<Book_deviate_todo_elem, Book_deviate_hash> &book_deviate_todo, uint64_t all_strt, bool *book_learning, Board *board_copy, int *player, int n_loop){
    if (!global_searching || !(*book_learning))
        return;
    // pass?
    if (todo_elem.board.get_legal() == 0){
        todo_elem.pass();
        if (todo_elem.board.get_legal() == 0)
            return; // game over
    }
    // check depth
    if (todo_elem.board.n_discs() >= book_depth + 4)
        return;
    // already searched?
    if (book_deviate_todo.find(todo_elem) != book_deviate_todo.end())
        return;
    todo_elem.board = book.get_representative_board(todo_elem.board);
    *board_copy = todo_elem.board;
    *player = todo_elem.player;
    Book_elem book_elem = book.get(todo_elem.board);
    // not registered?
    if (book_elem.value == SCORE_UNDEFINED)
        return;
    // already seen
    if (book_elem.seen)
        return;
    book.flag_book_elem(todo_elem.board);
    // check leaf
    std::vector<Book_value> links = book.get_all_moves_with_value(&todo_elem.board);
    int leaf_error = book_elem.value - book_elem.leaf.value;
    if (((leaf_error <= todo_elem.max_error_per_move && leaf_error <= todo_elem.remaining_error) || links.size() == 0) && is_valid_policy(book_elem.leaf.move)){
        if (todo_elem.board.get_legal() & (1ULL << book_elem.leaf.move)){ // is leaf legal?
            bool leaf_in_link = false;
            for (Book_value &link: links){
                leaf_in_link |= link.policy == book_elem.leaf.move;
            }
            if (!leaf_in_link){
                book_deviate_todo.emplace(todo_elem);
                if (book_deviate_todo.size() % 100 == 0)
                    std::cerr << "loop " << n_loop << " book deviate todo " << book_deviate_todo.size() << " calculating... time " << ms_to_time_short(tim() - all_strt) << std::endl;
            }
        }
    }
    // expand links
    if (todo_elem.board.n_discs() + 1 < book_depth + 4){
        Flip flip;
        for (Book_value &link: links){
            calc_flip(&flip, &todo_elem.board, link.policy);
            int link_value = link.value;
            todo_elem.board.move_board(&flip);
                Book_elem link_elem = book.get(todo_elem.board);
                if (link_elem.value != SCORE_UNDEFINED && link_value < -link_elem.value){
                    link_value = link_elem.value;
                }
            todo_elem.board.undo_board(&flip);
            int link_error = book_elem.value - link_value;
            if (link_error <= todo_elem.max_error_per_move && link_error <= todo_elem.remaining_error){
                todo_elem.move(&flip, link_error);
                    get_book_deviate_todo(todo_elem, book_depth, book_deviate_todo, all_strt, book_learning, board_copy, player, n_loop);
                todo_elem.undo(&flip, link_error);
            }
        }
    }
}

uint64_t expand_leaf(Book_deviate_todo_elem todo_elem, int book_depth, int level, bool use_multi_thread, bool *book_learning){
    Book_elem book_elem = book.get(todo_elem.board);
    Flip flip;
    calc_flip(&flip, &todo_elem.board, book_elem.leaf.move);
    todo_elem.move(&flip, 0);
    if (todo_elem.board.get_legal() == 0){ // check pass
        todo_elem.board.pass();
        if (todo_elem.board.get_legal() == 0){
            todo_elem.board.pass();
            book.change(todo_elem.board, todo_elem.board.score_player(), 60);
            return 1;
        }
    }
    uint64_t n_add = 0;
    int prev_value = book_elem.value;
    while (todo_elem.board.n_discs() <= book_depth + 4 && !book.contain(&todo_elem.board) && (*book_learning) && todo_elem.remaining_error >= 0){
        Search_result search_result = ai(todo_elem.board, level, true, 0, use_multi_thread, false);
        if (-HW2 <= search_result.value && search_result.value <= HW2){
            book.change(todo_elem.board, search_result.value, level);
            ++n_add;
            if (is_valid_policy(search_result.policy) && (todo_elem.board.get_legal() & (1ULL << search_result.policy))){
                int error = 0;
                if (prev_value != SCORE_UNDEFINED){
                    error = -prev_value - search_result.value;
                }
                prev_value = search_result.value;
                if (
                    todo_elem.board.n_discs() < book_depth + 4 && 
                    error <= todo_elem.max_error_per_move && 
                    error <= todo_elem.remaining_error
                ){
                    calc_flip(&flip, &todo_elem.board, search_result.policy);
                    todo_elem.board.move_board(&flip);
                    if (todo_elem.board.get_legal() == 0){ // check pass
                        todo_elem.board.pass();
                        prev_value *= -1;
                        if (todo_elem.board.get_legal() == 0){
                            todo_elem.board.pass();
                            book.change(todo_elem.board, todo_elem.board.score_player(), 60);
                            ++n_add;
                            break;
                        }
                    }
                } else{
                    book.add_leaf(&todo_elem.board, search_result.value, search_result.policy, level);
                    break;
                }
            } else
                break;
        }
    }
    return n_add;
}

uint64_t expand_leaves(int book_depth, int level, int max_error_per_move, std::unordered_set<Book_deviate_todo_elem, Book_deviate_hash> &book_deviate_todo, uint64_t all_strt, uint64_t strt, bool *book_learning, Board *board_copy, int *player, int n_loop, std::string file, std::string bak_file){
    int n_all = book_deviate_todo.size();
    if (n_all == 0)
        return 0;
    int n_done = 0;
    int n_doing = 0;
    std::vector<std::future<uint64_t>> tasks;
    uint64_t s = tim();
    uint64_t n_add = 0;
    for (Book_deviate_todo_elem elem: book_deviate_todo){
        if (tim() - s >= AUTO_BOOK_SAVE_TIME){
            book.save_egbk3(file, bak_file);
            s = tim();
        }
        if (!global_searching || !(*book_learning))
            break;
        *board_copy = elem.board;
        *player = elem.player;
        bool use_multi_thread = (n_all - n_doing) < thread_pool.size() * 2;
        bool pushed;
        ++n_doing;
        if (use_multi_thread){
            n_add += expand_leaf(elem, book_depth, level, true, book_learning);
            ++n_done;
            int percent = 100ULL * n_done / n_all;
            uint64_t eta = (tim() - strt) * ((double)n_all / n_done - 1.0);
            std::string time_short = ms_to_time_short(tim() - all_strt);
            std::string eta_short = ms_to_time_short(eta);
            std::cerr << "loop " << n_loop << " book deviating " << percent << "% " <<  n_done << "/" << n_all << " registered " << n_add << " time " << time_short << " ETA " << eta_short << std::endl;
        } else{
            tasks.emplace_back(thread_pool.push(&pushed, std::bind(&expand_leaf, elem, book_depth, level, use_multi_thread, book_learning)));
            if (!pushed){
                //std::cerr << "deleted " << true << std::endl;
                //elem.board.print();
                tasks.pop_back();
                expand_leaf(elem, book_depth, level, true, book_learning);
                ++n_done;
                int percent = 100ULL * n_done / n_all;
                uint64_t eta = (tim() - strt) * ((double)n_all / n_done - 1.0);
                std::string time_short = ms_to_time_short(tim() - all_strt);
                std::string eta_short = ms_to_time_short(eta);
                std::cerr << "loop " << n_loop << " book deviating " << percent << "% " <<  n_done << "/" << n_all << " registered " << n_add << " time " << time_short << " ETA " << eta_short << std::endl;
            }
        }
        int tasks_size = tasks.size();
        for (int i = 0; i < tasks_size; ++i){
            if (tasks[i].valid()){
                if (tasks[i].wait_for(std::chrono::seconds(0)) == std::future_status::ready){
                    n_add += tasks[i].get();
                    ++n_done;
                    int percent = 100ULL * n_done / n_all;
                    uint64_t eta = (tim() - strt) * ((double)n_all / n_done - 1.0);
                    std::string time_short = ms_to_time_short(tim() - all_strt);
                    std::string eta_short = ms_to_time_short(eta);
                    std::cerr << "loop " << n_loop << " book deviating " << percent << "% " <<  n_done << "/" << n_all << " registered " << n_add << " time " << time_short << " ETA " << eta_short << std::endl;
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
            n_add += tasks[i].get();
            ++n_done;
            int percent = 100ULL * n_done / n_all;
            uint64_t eta = (tim() - strt) * ((double)n_all / n_done - 1.0);
            std::string time_short = ms_to_time_short(tim() - all_strt);
            std::string eta_short = ms_to_time_short(eta);
            std::cerr << "loop " << n_loop << " book deviating " << percent << "% " <<  n_done << "/" << n_all << " registered " << n_add << " time " << time_short << " ETA " << eta_short << std::endl;
        }
    }
    return n_add;
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
    uint64_t all_strt = tim();
    uint64_t s = all_strt;
    std::cerr << "book deviate started" << std::endl;
    int before_player = *player;
    Book_deviate_todo_elem root_elem;
    root_elem.board = root_board;
    root_elem.player = *player;
    root_elem.max_error_per_move = max_error_per_move;
    root_elem.remaining_error = max_error_sum;
    int n_loop = 0;
    uint64_t n_registered = 0;
    while (true){
        ++n_loop;
        if (tim() - s > AUTO_BOOK_SAVE_TIME && *book_learning){
            book.save_egbk3(book_file, book_bak);
            s = tim();
        }
        Book_elem book_elem = book.get(root_board);
        if (book_elem.value == SCORE_UNDEFINED){
            std::cerr << "board not registered, searching..." << std::endl;
            book_elem.value = ai(root_board, level, true, 0, true, false).value;
        }
        bool book_recalculating = true;
        book_recalculate_leaf(root_board, std::min(20 + (level & 1), std::max(1, level - 2)), std::max(1, book_depth - 1), max_error_per_move, max_error_sum, board_copy, player, &book_recalculating, true);
        std::unordered_set<Book_deviate_todo_elem, Book_deviate_hash> book_deviate_todo;
        book.reset_seen();
        get_book_deviate_todo(root_elem, book_depth, book_deviate_todo, all_strt, book_learning, board_copy, player, n_loop);
        book.reset_seen();
        std::cerr << "loop " << n_loop << " book deviate todo " << book_deviate_todo.size() << " calculated time " << ms_to_time_short(tim() - all_strt) << std::endl;
        uint64_t strt = tim();
        uint64_t n_add = expand_leaves(book_depth, level, max_error_per_move, book_deviate_todo, all_strt, strt, book_learning, board_copy, player, n_loop, book_file, book_bak);
        if (n_add == 0)
            break;
        n_registered += n_add;
        std::cerr << "loop " << n_loop << " book deviated registered " << n_registered << "board size " << book.size() << std::endl;
    }
    *player = before_player;
    root_board.copy(board_copy);
    //book.fix();
    //book.save_egbk3(book_file, book_bak);
    std::cerr << "book deviate finished registered " << n_registered << " time " << ms_to_time_short(tim() - all_strt) << std::endl;
    *book_learning = false;
}
