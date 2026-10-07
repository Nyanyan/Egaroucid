/*
    Egaroucid Project

    @file book_widen.hpp
        Enlarging book
    @date 2021-2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once
#include <iostream>
#include <new>
#include <atomic>
#include <future>
#include <mutex>
#include <thread>
#include <vector>
#include "evaluate.hpp"
#include "board.hpp"
#include "ai.hpp"

// automatically save book in this time (milliseconds)
constexpr uint64_t AUTO_BOOK_SAVE_TIME = 3600000ULL; // 1 hour

constexpr int BOOK_DEVIATE_MAX_N_LOOPS_INF = 100000000;

Search_result ai(Board board, int level, bool use_book, int book_acc_level, bool use_multi_thread, bool show_log);

struct Book_deviate_todo_elem {
    Board board;
    int player;
    int max_error_per_move;
    int remaining_error;
    int max_leaf_error;

    void move(Flip *flip, int error) {
        board.move_board(flip);
        player ^= 1;
        if (error > 0)
            remaining_error -= error;
    }

    void undo(Flip *flip, int error) {
        board.undo_board(flip);
        player ^= 1;
        if (error > 0)
            remaining_error += error;
    }

    void pass() {
        board.pass();
        player ^= 1;
    }
};



/*
    Walk of the book for book learning, with all threads

    The walks of book learning (positions to expand, leaves to recalculate) start at a root board and
    follow the registered moves while the error of each move and the sum of the errors are within the limits.
    They were recursive functions on one thread: with a book of 650M boards, one walk for the positions
    to expand visited 170M boards and took 6 minutes in every loop of book_deviate(), whatever the number of threads.

    Here the boards are walked in order of the sum of the errors (0 first), and the boards with the same sum are
    walked round by round: the boards of a round are processed by all threads, and the boards after their
    moves without error make the next round. The boards after a move with an error wait for their sum.

    A board is walked once (flag "seen" of the book), by the first walk which reaches it:
    - a board which can be reached with several sums of errors is walked with the smallest one
      (the recursive function walked it with the sum of the path which reached it first, which depended
      on the order of the moves: with a larger sum than necessary, fewer moves were followed from it),
    - in a round, the boards with BLACK to move are processed before the boards with WHITE to move
      (the same board can be reached with both only by lines with a different number of passes),
    so the result does not depend on the threads.
*/

// number of boards given to a thread at once
constexpr size_t BOOK_WALK_CHUNK_SIZE = 256;

/*
    @brief what a thread found in the boards of a chunk

    same:   boards after a move without error (by player to move): walked in the next round
    later:  boards after a move with an error of 1 or more (by player to move and error): walked later
    found:  elements for the caller (positions to expand, leaves to recalculate)
*/
template<class Found>
struct Book_walk_output {
    std::vector<Board> same[2];
    std::vector<std::vector<Board>> later[2];
    std::vector<Found> found;

    /*
        @brief add the board after a move

        @param board                board after the move (as it is: the walk passes if needed and makes it representative when the board is walked)
        @param player               player to move
        @param error                error of the move (0 or less: no error)
    */
    void add_child(const Board &board, int player, int error) {
        if (error <= 0) {
            same[player].emplace_back(board);
        } else {
            if ((int)later[player].size() <= error) {
                later[player].resize(error + 1);
            }
            later[player][error].emplace_back(board);
        }
    }
};

/*
    @brief walk the book from a board in order of the sum of the errors, with all threads

    Boards must not be registered / deleted during the walk. The flag "seen" must be reset before.

    @param root_board           board to start from
    @param root_player          player to move of root_board
    @param max_error_sum        maximum of the sum of the errors
    @param book_learning        the walk stops if this becomes false
    @param board_copy           board pointer for screen drawing
    @param player               player information for screen drawing
    @param visit                visit(const Board &board, int player, int remaining_error, Book_walk_output<Found> &output) is called for each board
                                reached (from several threads at the same time). board is the representative board, after a pass if the
                                player had no move (game-over boards are not visited), player is the player to move.
                                It checks and flags the board with book.flag_if_not_seen(),
                                and gives the boards after the moves to follow to output.add_child(), what it found to output.found
    @param collect              collect(int error_sum, std::vector<Found> &found) is called on the calling thread for the output of each chunk
*/
template<class Found, class Visit, class Collect>
void book_walk_by_error(Board root_board, int root_player, int max_error_sum, bool *book_learning, Board *board_copy, int *player, Visit &&visit, Collect &&collect) {
    const int n_error_sums = std::max(0, max_error_sum) + 1;
    std::vector<std::vector<Board>> waiting[2]; // [player][sum of errors]
    waiting[BLACK].resize(n_error_sums);
    waiting[WHITE].resize(n_error_sums);
    waiting[root_player][0].emplace_back(root_board);
    for (int error_sum = 0; error_sum < n_error_sums; ++error_sum) {
        const int remaining_error = max_error_sum - error_sum;
        std::vector<Board> round[2] = {std::move(waiting[BLACK][error_sum]), std::move(waiting[WHITE][error_sum])};
        while (!round[BLACK].empty() || !round[WHITE].empty()) {
            std::vector<Board> next_round[2];
            for (int round_player = BLACK; round_player <= WHITE; ++round_player) {
                if (!global_searching || !(*book_learning)) {
                    return;
                }
                const std::vector<Board> &boards = round[round_player];
                const size_t n_boards = boards.size();
                if (n_boards == 0) {
                    continue;
                }
                *board_copy = boards[0];
                *player = round_player;
                const int n_chunks = (int)((n_boards + BOOK_WALK_CHUNK_SIZE - 1) / BOOK_WALK_CHUNK_SIZE);
                std::vector<Book_walk_output<Found>> outputs(n_chunks);
                auto walk_chunk = [&boards, &outputs, &visit, n_boards, round_player, remaining_error, book_learning](int chunk_idx) {
                    const size_t chunk_start = (size_t)chunk_idx * BOOK_WALK_CHUNK_SIZE;
                    const size_t chunk_end = std::min(n_boards, chunk_start + BOOK_WALK_CHUNK_SIZE);
                    // pass if needed and representative board, for all boards of the chunk first: the memory of the book
                    // which the visits read is asked in advance (the table is much larger than the cache, and the
                    // visits of a chunk would wait for the memory one after the other)
                    Board chunk_boards[BOOK_WALK_CHUNK_SIZE];
                    int chunk_players[BOOK_WALK_CHUNK_SIZE]; // -1: game over (not visited)
                    for (size_t i = chunk_start; i < chunk_end; ++i) {
                        Board board = boards[i];
                        int board_player = round_player;
                        if (board.get_legal() == 0) { // pass?
                            board.pass();
                            board_player ^= 1;
                            if (board.get_legal() == 0) { // game over
                                chunk_players[i - chunk_start] = -1;
                                continue;
                            }
                        }
                        board = representative_board(board);
                        chunk_boards[i - chunk_start] = board;
                        chunk_players[i - chunk_start] = board_player;
                        book.prefetch_unlocked(board);
                    }
                    for (size_t i = chunk_start; i < chunk_end; ++i) {
                        if (chunk_players[i - chunk_start] != -1) {
                            book.prefetch_entry_unlocked(chunk_boards[i - chunk_start]);
                        }
                    }
                    for (size_t i = chunk_start; i < chunk_end && global_searching && *book_learning; ++i) {
                        if (chunk_players[i - chunk_start] != -1) {
                            visit(chunk_boards[i - chunk_start], chunk_players[i - chunk_start], remaining_error, outputs[chunk_idx]);
                        }
                    }
                };
                if (n_chunks == 1) { // a few boards: no task
                    walk_chunk(0);
                } else {
                    book.parallel_for(n_chunks, walk_chunk);
                }
                // the outputs are joined in the order of the chunks: the order of the result does not depend on the threads
                for (Book_walk_output<Found> &output: outputs) {
                    for (int child_player = BLACK; child_player <= WHITE; ++child_player) {
                        next_round[child_player].insert(next_round[child_player].end(), output.same[child_player].begin(), output.same[child_player].end());
                        for (int error = 1; error < (int)output.later[child_player].size(); ++error) {
                            if (error_sum + error < n_error_sums) { // (always true for the moves which the visitors follow)
                                std::vector<Board> &dst = waiting[child_player][error_sum + error];
                                dst.insert(dst.end(), output.later[child_player][error].begin(), output.later[child_player][error].end());
                            }
                        }
                    }
                    collect(error_sum, output.found);
                }
            }
            round[BLACK] = std::move(next_round[BLACK]);
            round[WHITE] = std::move(next_round[WHITE]);
        }
    }
}

/*
    @brief find the boards whose leaf has to be recalculated

    The tests of each board are the ones of the former recursive function; the walk is done by
    book_walk_by_error() (all threads; a board is walked with its smallest sum of errors).
    (The former test "already searched?" looked for the representative board in the list: a board in the list
    was always flagged "seen" too, so the test of the flag is enough.)

    @param root_elem            board to start from, player and limits of the errors
    @param book_depth           depth of the book
    @param level                level of the leaves
    @param todo_list            the boards are added here, in order of the sum of the errors
    @param all_strt             start time (for the log)
    @param book_learning        the walk stops if this becomes false
    @param board_copy           board pointer for screen drawing
    @param player               player information for screen drawing
    @param only_illegal         true: only the invalid leaves, false: also the leaves with a lower level
*/
void get_book_recalculate_leaf_todo(Book_deviate_todo_elem root_elem, int book_depth, int level, std::vector<Book_deviate_todo_elem> &todo_list, uint64_t all_strt, bool *book_learning, Board *board_copy, int *player, bool only_illegal) {
    const int max_error_per_move = root_elem.max_error_per_move;
    const int max_leaf_error = root_elem.max_leaf_error;
    auto visit = [book_depth, level, only_illegal, max_error_per_move, max_leaf_error](const Board &reached_board, int reached_player, int remaining_error, Book_walk_output<Book_deviate_todo_elem> &output) {
        Book_deviate_todo_elem todo_elem;
        todo_elem.board = reached_board;
        todo_elem.player = reached_player;
        todo_elem.max_error_per_move = max_error_per_move;
        todo_elem.remaining_error = remaining_error;
        todo_elem.max_leaf_error = max_leaf_error;
        // (pass and game over: done by book_walk_by_error())
        // check depth
        if (todo_elem.board.n_discs() > book_depth + 4) {
            return;
        }
        // (the board is already the representative board)
        // not registered, or already seen?
        Book_elem book_elem;
        if (!book.flag_if_not_seen(todo_elem.board, &book_elem)) {
            return;
        }
        if (book_elem.value == SCORE_UNDEFINED) {
            return;
        }
        // check leaf recalculation necessity
        std::vector<Book_value> links = book.get_all_moves_with_value_unlocked(&todo_elem.board);
        uint64_t remaining_legal = todo_elem.board.get_legal();
        for (Book_value &link: links) {
            remaining_legal &= ~(1ULL << link.policy);
        }
        bool illegal_leaf = false;
        if (remaining_legal) {
            if (is_valid_policy(book_elem.leaf.move)) {
                if ((remaining_legal & (1ULL << book_elem.leaf.move)) == 0) {
                    illegal_leaf = true;
                }
                // A value out of range is also an invalid leaf (the bundled v7.8.3 book has leaves with 65-73, 126 and 127,
                // probably values of stopped searches). Such leaves were never recalculated here.
                if (!is_valid_score(book_elem.leaf.value)) {
                    illegal_leaf = true;
                }
            } else {
                illegal_leaf = true;
            }
        } else if (book_elem.leaf.move != MOVE_NOMOVE) {
            illegal_leaf = true;
        }
        // add to list
        if (illegal_leaf || (!only_illegal && book_elem.leaf.level < level)) {
            output.found.emplace_back(todo_elem);
        }
        // expand links
        Flip flip;
        for (Book_value &link: links) {
            calc_flip(&flip, &todo_elem.board, link.policy);
            int link_value = link.value;
            Board moved_board = todo_elem.board;
            moved_board.move_board(&flip);
            int link_elem_value = book.get_value_unlocked(moved_board);
            if (link_elem_value != SCORE_UNDEFINED && link_value < -link_elem_value) {
                link_value = link_elem_value;
            }
            int link_error = book_elem.value - link_value;
            if (link_error <= todo_elem.max_error_per_move && link_error <= todo_elem.remaining_error) {
                output.add_child(moved_board, todo_elem.player ^ 1, link_error);
            }
        }
    };
    auto collect = [&todo_list](int error_sum, std::vector<Book_deviate_todo_elem> &found) {
        todo_list.insert(todo_list.end(), found.begin(), found.end());
    };
    book_walk_by_error<Book_deviate_todo_elem>(root_elem.board, root_elem.player, root_elem.remaining_error, book_learning, board_copy, player, visit, collect);
}

void book_search_leaf(Board board, int level, bool use_multi_thread) {
    book.search_leaf(board, level, use_multi_thread);
}

/*
    @brief run func(idx, use_multi_thread) for idx = 0, 1, ..., n - 1 in book learning, with all threads

    Used for the expansions and for the recalculation of the leaves: many independent searches.
    Every thread (the threads of the pool and the calling thread) takes the next idx and runs func with a
    one-thread search (use_multi_thread = false), until the last elements; the last elements
    (thread_pool.size() * 2 - 1 of them, as before) are run one after the other by the calling thread with a
    multi-thread search, which uses the threads as they finish.

    Before, the calling thread pushed one task for each element to the pool, and when no thread was idle it
    ran the element itself with a multi-thread search. That search got almost no thread, and until it ended
    no task was pushed: the threads which finished in the meantime waited. With 32 threads (book of 650M
    boards, level 18) the calling thread spent 98% of the time in such searches and 20% of the CPU was idle.

    func is called from several threads at the same time.

    @param n                    number of elements
    @param func                 func(int idx, bool use_multi_thread)
*/
template<class F>
void book_learning_for_each(int n, F &&func) {
    const int n_pool_threads = thread_pool.size();
    if (n_pool_threads == 0) { // one thread: as before (the push to the empty pool failed, and the element was run with use_multi_thread = true)
        for (int idx = 0; idx < n; ++idx) {
            func(idx, true);
        }
        return;
    }
    const int n_single = std::clamp(n - n_pool_threads * 2 + 1, 0, n); // elements run with a one-thread search
    std::atomic<int> next_idx(0);
    auto worker = [&next_idx, &func, n_single]() {
        for (;;) {
            int idx = next_idx.fetch_add(1);
            if (idx >= n_single) {
                break;
            }
            func(idx, false);
        }
    };
    std::vector<std::future<void>> tasks;
    for (;;) {
        // a worker on each thread of the pool (a thread which was not idle yet gets its worker at a later turn)
        while ((int)tasks.size() < n_pool_threads && next_idx.load() < n_single) {
            bool pushed = false;
            std::future<void> task = thread_pool.push(&pushed, worker);
            if (!pushed) {
                break;
            }
            tasks.emplace_back(std::move(task));
        }
        // the calling thread works too
        int idx = next_idx.fetch_add(1);
        if (idx >= n_single) {
            break;
        }
        func(idx, false);
    }
    for (int idx = n_single; idx < n; ++idx) {
        func(idx, true);
    }
    for (std::future<void> &task: tasks) {
        task.get();
    }
}

void book_recalculate_leaves(int level, std::vector<Book_deviate_todo_elem> &todo_list, uint64_t all_strt, uint64_t strt, bool *book_learning, Board *board_copy, int *player) {
    const int n_all = todo_list.size();
    if (n_all == 0) {
        return;
    }
    int n_done = 0;
    std::mutex progress_mtx; // n_done and the log
    const std::thread::id calling_thread = std::this_thread::get_id();
    book_learning_for_each(n_all, [&](int idx, bool use_multi_thread) {
        if (!global_searching || !(*book_learning)) {
            return;
        }
        const Book_deviate_todo_elem &elem = todo_list[idx];
        if (std::this_thread::get_id() == calling_thread) { // (screen drawing: the boards of one thread)
            *board_copy = elem.board;
            *player = elem.player;
        }
        book_search_leaf(elem.board, level, use_multi_thread);
        // count a finished search, print the progress every 10 searches
        std::lock_guard<std::mutex> lock(progress_mtx);
        ++n_done;
        if (n_done % 10 == 0) {
            int percent = 100ULL * n_done / n_all;
            uint64_t eta = (tim() - strt) * ((double)n_all / n_done - 1.0);
            std::cerr << "book recalculating leaves " << percent << "% " <<  n_done << "/" << n_all << " time " << ms_to_time_short(tim() - all_strt) << " ETA " << ms_to_time_short(eta) << std::endl;
        }
    });
    int percent = 100ULL * n_done / n_all;
    uint64_t eta = n_done ? (uint64_t)((tim() - strt) * ((double)n_all / n_done - 1.0)) : 0;
    std::cerr << "book recalculating leaves finished " << percent << "% " <<  n_done << "/" << n_all << " time " << ms_to_time_short(tim() - all_strt) << " ETA " << ms_to_time_short(eta) << std::endl;
}

inline void book_recalculate_leaf(Board root_board, int level, int book_depth, int max_error_per_move, int max_error_sum, Board *board_copy, int *player, bool *book_learning, bool only_illegal, uint64_t all_strt) {
    int before_player = *player;
    Book_deviate_todo_elem root_elem;
    root_elem.board = root_board;
    root_elem.player = *player;
    root_elem.max_error_per_move = max_error_per_move;
    root_elem.remaining_error = max_error_sum;
    Book_elem book_elem = book.get(root_board);
    if (book_elem.value == SCORE_UNDEFINED) {
        std::cerr << "board not registered, searching..." << std::endl;
        book_elem.value = ai(root_board, level, true, 0, true, false).value;
    }

    book.reset_seen();
    std::vector<Book_deviate_todo_elem> book_recalculate_leaf_todo;
    get_book_recalculate_leaf_todo(root_elem, book_depth, level, book_recalculate_leaf_todo, all_strt, book_learning, board_copy, player, only_illegal);
    std::cerr << "book recalculate leaf todo " << book_recalculate_leaf_todo.size() << " calculated time " << ms_to_time_short(tim() - all_strt) << std::endl;
    if (book_recalculate_leaf_todo.size()) {
        uint64_t strt = tim();
        book_recalculate_leaves(level, book_recalculate_leaf_todo, all_strt, strt, book_learning, board_copy, player);
    }
    book.reset_seen();
    root_board.copy(board_copy);
    *player = before_player;
    std::cerr << "recalculate leaf finished " << book_recalculate_leaf_todo.size() << " boards time " << ms_to_time_short(tim() - all_strt) << std::endl;
    *book_learning = false;
}




/*
    @brief find the positions to expand in book learning (one walk; book_deviate() does two of them)

    The player best_move_only_player follows only the best moves, the other player the moves within the errors.
    The tests of each board are the ones of the former recursive function. Differences:
    - the walk is done by book_walk_by_error() (all threads; a board is walked with its smallest sum of errors),
    - the former first test "already searched?" looked for the board in the list of positions to expand before
      the board was made representative, so it matched only when the board was reached in its representative
      orientation (and then the moves of a position found by the first walk were not followed by the second walk).
      Whether it matched depended on the path which reached the board first. It is not done any more: with the
      bundled book, 2,145 positions instead of 2,142 were found without it (v7.8.3, level 7, depth 18).

    @param root_elem            board to start from, player and limits of the errors
    @param best_move_only_player    the player who follows only the best moves
    @param book_depth           depth of the book
    @param todo_by_error        the positions to expand are added here: todo_by_error[sum of the errors]
    @param all_strt             start time of book learning (for the log)
    @param book_learning        the walk stops if this becomes false
    @param board_copy           board pointer for screen drawing
    @param player               player information for screen drawing
    @param n_loop               loop of book_deviate() (for the log)
*/
void get_book_deviate_todo(Book_deviate_todo_elem root_elem, int best_move_only_player, int book_depth, std::vector<std::vector<Book_deviate_todo_elem>> &todo_by_error, uint64_t all_strt, bool *book_learning, Board *board_copy, int *player, int n_loop) {
    const int max_error_per_move = root_elem.max_error_per_move;
    const int max_leaf_error = root_elem.max_leaf_error;
    auto visit = [best_move_only_player, book_depth, max_error_per_move, max_leaf_error](const Board &reached_board, int reached_player, int remaining_error, Book_walk_output<Book_deviate_todo_elem> &output) {
        Book_deviate_todo_elem todo_elem;
        todo_elem.board = reached_board;
        todo_elem.player = reached_player;
        todo_elem.max_error_per_move = max_error_per_move;
        todo_elem.remaining_error = remaining_error;
        todo_elem.max_leaf_error = max_leaf_error;
        // (pass and game over: done by book_walk_by_error())
        // check depth
        if (todo_elem.board.n_discs() >= book_depth + 4) {
            return;
        }
        // (the board is already the representative board)
        // not registered, or already seen?
        Book_elem book_elem;
        if (!book.flag_if_not_seen(todo_elem.board, &book_elem)) {
            return;
        }
        if (book_elem.value == SCORE_UNDEFINED) {
            return;
        }
        std::vector<Book_value> links = book.get_all_moves_with_value_unlocked(&todo_elem.board);
        const bool best_move_only = (todo_elem.player == best_move_only_player);
        int best_score = -SCORE_MAX;
        if (best_move_only) {
            for (Book_value &link: links) {
                best_score = std::max(best_score, link.value);
            }
            if (-SCORE_MAX <= book_elem.leaf.value && book_elem.leaf.value <= SCORE_MAX && is_valid_policy(book_elem.leaf.move)) {
                best_score = std::max(best_score, (int)book_elem.leaf.value);
            }
        }
        // check leaf (best_move_only: only if the leaf is the best move)
        if (!best_move_only || book_elem.leaf.value >= best_score) {
            int leaf_error = book_elem.value - book_elem.leaf.value;
            if (leaf_error <= todo_elem.max_leaf_error && is_valid_policy(book_elem.leaf.move)) {
                if (todo_elem.board.get_legal() & (1ULL << book_elem.leaf.move)) { // is leaf legal?
                    bool leaf_in_link = false;
                    for (Book_value &link: links) {
                        leaf_in_link |= link.policy == book_elem.leaf.move;
                    }
                    if (!leaf_in_link) {
                        output.found.emplace_back(todo_elem);
                    }
                }
            }
        }
        // expand links (best_move_only: only the best moves)
        if (todo_elem.board.n_discs() + 1 < book_depth + 4) {
            Flip flip;
            for (Book_value &link: links) {
                if (!best_move_only || link.value == best_score) {
                    int link_error = book_elem.value - link.value;
                    if (link_error <= todo_elem.max_error_per_move && link_error <= todo_elem.remaining_error) {
                        calc_flip(&flip, &todo_elem.board, link.policy);
                        Board moved_board = todo_elem.board;
                        moved_board.move_board(&flip);
                        output.add_child(moved_board, todo_elem.player ^ 1, link_error);
                    }
                }
            }
        }
    };
    uint64_t n_todo = 0;
    auto collect = [&todo_by_error, &n_todo](int error_sum, std::vector<Book_deviate_todo_elem> &found) {
        todo_by_error[error_sum].insert(todo_by_error[error_sum].end(), found.begin(), found.end());
        n_todo += found.size();
    };
    book_walk_by_error<Book_deviate_todo_elem>(root_elem.board, root_elem.player, root_elem.remaining_error, book_learning, board_copy, player, visit, collect);
    std::cerr << "loop " << n_loop << " book deviate todo " << n_todo << " found (" << (best_move_only_player == BLACK ? "black" : "white") << " follows the best moves) time " << ms_to_time_short(tim() - all_strt) << std::endl;
}

/*
    @brief make the list of the positions to expand from the results of the two walks of book_deviate()

    A position found by both walks is kept once (the one of the first walk, as before).
    The positions are in order of the sum of the errors (the lines without error first), first walk before second walk.
    (Formerly: the order of the std::unordered_set which held them.)
    The flag "seen" must be reset before, and is used: reset it afterwards.

    @param first                result of the first walk
    @param second               result of the second walk
    @return the positions to expand
*/
std::vector<Book_deviate_todo_elem> merge_book_deviate_todo(const std::vector<std::vector<Book_deviate_todo_elem>> &first, const std::vector<std::vector<Book_deviate_todo_elem>> &second) {
    // flag the positions of the first walk
    for (const std::vector<Book_deviate_todo_elem> &elems: first) {
        const int n_chunks = (int)((elems.size() + BOOK_WALK_CHUNK_SIZE - 1) / BOOK_WALK_CHUNK_SIZE);
        book.parallel_for(n_chunks, [&elems](int chunk_idx) {
            const size_t chunk_end = std::min(elems.size(), (size_t)(chunk_idx + 1) * BOOK_WALK_CHUNK_SIZE);
            for (size_t i = (size_t)chunk_idx * BOOK_WALK_CHUNK_SIZE; i < chunk_end; ++i) {
                book.flag_book_elem(elems[i].board);
            }
        });
    }
    std::vector<Book_deviate_todo_elem> res;
    std::vector<char> is_new;
    for (size_t error_sum = 0; error_sum < first.size(); ++error_sum) {
        res.insert(res.end(), first[error_sum].begin(), first[error_sum].end());
        // positions of the second walk which the first walk did not find
        const std::vector<Book_deviate_todo_elem> &elems = second[error_sum];
        is_new.assign(elems.size(), 0);
        const int n_chunks = (int)((elems.size() + BOOK_WALK_CHUNK_SIZE - 1) / BOOK_WALK_CHUNK_SIZE);
        book.parallel_for(n_chunks, [&elems, &is_new](int chunk_idx) {
            const size_t chunk_end = std::min(elems.size(), (size_t)(chunk_idx + 1) * BOOK_WALK_CHUNK_SIZE);
            for (size_t i = (size_t)chunk_idx * BOOK_WALK_CHUNK_SIZE; i < chunk_end; ++i) {
                is_new[i] = !book.get(elems[i].board).seen;
            }
        });
        for (size_t i = 0; i < elems.size(); ++i) {
            if (is_new[i]) {
                res.emplace_back(elems[i]);
            }
        }
    }
    return res;
}

/*
    @brief register a board in book learning

    If the memory for the board can't be allocated (std::bad_alloc from the book), book learning is stopped
    and the boards registered so far are kept, so that the book can be saved.
    Otherwise the exception of a task of the thread pool is only stored in its future: the thread of book learning
    ends with the exception while the other tasks go on, and the caller never sees the end of the learning.

    @param board                board to register
    @param value                value of the board
    @param level                level of the value
    @param book_learning        set to false if the board can't be registered
    @return registered?
*/
inline bool book_learning_change(Board board, int value, int level, bool *book_learning) {
    try {
        book.change(board, value, level);
    } catch (const std::bad_alloc&) {
        if (*book_learning) {
            std::cerr << "[ERROR] book memory allocation failed. book learning stopped" << std::endl;
        }
        *book_learning = false;
        return false;
    }
    return true;
}

uint64_t expand_leaf(Book_deviate_todo_elem todo_elem, int book_depth, int level, bool use_multi_thread, bool *book_learning) {
    Book_elem book_elem = book.get(todo_elem.board);
    Flip flip;
    calc_flip(&flip, &todo_elem.board, book_elem.leaf.move);
    todo_elem.move(&flip, 0);
    int prev_value = book_elem.value;
    if (todo_elem.board.get_legal() == 0) { // check pass
        todo_elem.board.pass();
        // The player to move is the same as the board of book_elem: change the sign as in the pass in the loop below.
        // (Without this, the error of the first move was calculated with the wrong sign after a pass.)
        prev_value *= -1;
        if (todo_elem.board.get_legal() == 0) {
            todo_elem.board.pass();
            return book_learning_change(todo_elem.board, todo_elem.board.score_player(), 60, book_learning) ? 1 : 0;
        }
    }
    uint64_t n_add = 0;
    while (todo_elem.board.n_discs() <= book_depth + 4 && !book.contain(&todo_elem.board) && (*book_learning) && todo_elem.remaining_error >= 0) {
        Search_result search_result = ai(todo_elem.board, level, true, 0, use_multi_thread, false);
        if (-HW2 <= search_result.value && search_result.value <= HW2) {
            if (!book_learning_change(todo_elem.board, search_result.value, level, book_learning)) {
                break;
            }
            ++n_add;
            if (is_valid_policy(search_result.policy) && (todo_elem.board.get_legal() & (1ULL << search_result.policy))) {
                int error = 0;
                if (prev_value != SCORE_UNDEFINED) {
                    error = -prev_value - search_result.value;
                }
                prev_value = search_result.value;
                if (
                    todo_elem.board.n_discs() < book_depth + 4 && 
                    error <= todo_elem.max_error_per_move && 
                    error <= todo_elem.remaining_error
                ) {
                    calc_flip(&flip, &todo_elem.board, search_result.policy);
                    todo_elem.board.move_board(&flip);
                    if (todo_elem.board.get_legal() == 0) { // check pass
                        todo_elem.board.pass();
                        prev_value *= -1;
                        if (todo_elem.board.get_legal() == 0) { // game over
                            todo_elem.board.pass();
                            if (book_learning_change(todo_elem.board, todo_elem.board.score_player(), 60, book_learning)) {
                                ++n_add;
                            }
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

uint64_t expand_leaves(int book_depth, int level, int max_error_per_move, std::vector<Book_deviate_todo_elem> &book_deviate_todo, uint64_t all_strt, uint64_t strt, bool *book_learning, Board *board_copy, int *player, int n_loop, std::string file, std::string bak_file) {
    const int n_all = book_deviate_todo.size();
    if (n_all == 0) {
        return 0;
    }
    int n_done = 0;
    uint64_t n_add = 0;
    std::mutex progress_mtx; // n_done, n_add and the log
    std::atomic<uint64_t> last_save(tim());
    std::mutex save_mtx; // one thread saves the book
    const std::thread::id calling_thread = std::this_thread::get_id();
    book_learning_for_each(n_all, [&](int idx, bool use_multi_thread) {
        // automatic save: by the thread which sees that it is time; the other threads go on (as the tasks did before)
        if (tim() - last_save.load() >= AUTO_BOOK_SAVE_TIME && save_mtx.try_lock()) {
            if (tim() - last_save.load() >= AUTO_BOOK_SAVE_TIME) {
                book.save_egbk3(file, bak_file);
                last_save.store(tim());
            }
            save_mtx.unlock();
        }
        if (!global_searching || !(*book_learning)) {
            return;
        }
        const Book_deviate_todo_elem &elem = book_deviate_todo[idx];
        if (std::this_thread::get_id() == calling_thread) { // (screen drawing: the boards of one thread)
            *board_copy = elem.board;
            *player = elem.player;
        }
        uint64_t n_add_elem = expand_leaf(elem, book_depth, level, use_multi_thread, book_learning);
        // count a finished expansion and print the progress
        std::lock_guard<std::mutex> lock(progress_mtx);
        n_add += n_add_elem;
        ++n_done;
        int percent = 100ULL * n_done / n_all;
        uint64_t eta = (tim() - strt) * ((double)n_all / n_done - 1.0);
        std::string time_short = ms_to_time_short(tim() - all_strt);
        std::string eta_short = ms_to_time_short(eta);
        std::cerr << "loop " << n_loop << " book deviating " << percent << "% " <<  n_done << "/" << n_all << " registered " << n_add << " time " << time_short << " ETA " << eta_short << std::endl;
    });
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
inline void book_deviate(Board root_board, int level, int book_depth, int max_error_per_move, int max_error_sum, int max_leaf_error, int max_n_loops, Board *board_copy, int *player, std::string book_file, std::string book_bak, bool *book_learning) {
    uint64_t all_strt = tim();
    uint64_t s = all_strt;
    std::cerr << "book deviate started" << std::endl;
    int before_player = *player;
    Book_deviate_todo_elem root_elem;
    root_elem.board = root_board;
    root_elem.player = *player;
    root_elem.max_error_per_move = max_error_per_move;
    root_elem.remaining_error = max_error_sum;
    root_elem.max_leaf_error = max_leaf_error;
    int n_loop = 0;
    uint64_t n_registered = 0;
    while (n_loop < max_n_loops) {
        ++n_loop;
        if (tim() - s > AUTO_BOOK_SAVE_TIME && *book_learning) {
            book.save_egbk3(book_file, book_bak);
            s = tim();
        }
        Book_elem book_elem = book.get(root_board);
        if (book_elem.value == SCORE_UNDEFINED) {
            std::cerr << "board not registered, searching..." << std::endl;
            book_elem.value = ai(root_board, level, true, 0, true, false).value;
            if (!book_learning_change(root_board, book_elem.value, level, book_learning)) {
                break;
            }
        }
        bool book_recalculating = true;
        book_recalculate_leaf(root_board, std::min(20 + (level & 1), std::max(1, level - 2)), std::max(1, book_depth - 1), max_error_per_move, max_error_sum, board_copy, player, &book_recalculating, true, all_strt);
        // positions to expand: two walks (black, then white follows only the best moves), by sum of the errors
        const int n_error_sums = std::max(0, max_error_sum) + 1;
        std::vector<std::vector<Book_deviate_todo_elem>> todo_black_best(n_error_sums), todo_white_best(n_error_sums);
        book.reset_seen();
        get_book_deviate_todo(root_elem, BLACK, book_depth, todo_black_best, all_strt, book_learning, board_copy, player, n_loop);
        book.reset_seen();
        get_book_deviate_todo(root_elem, WHITE, book_depth, todo_white_best, all_strt, book_learning, board_copy, player, n_loop);
        book.reset_seen();
        std::vector<Book_deviate_todo_elem> book_deviate_todo = merge_book_deviate_todo(todo_black_best, todo_white_best);
        book.reset_seen();
        todo_black_best = std::vector<std::vector<Book_deviate_todo_elem>>(); // (release the memory)
        todo_white_best = std::vector<std::vector<Book_deviate_todo_elem>>();
        std::cerr << "loop " << n_loop << " book deviate todo " << book_deviate_todo.size() << " calculated time " << ms_to_time_short(tim() - all_strt) << std::endl;
        uint64_t strt = tim();
        uint64_t n_add = expand_leaves(book_depth, level, max_error_per_move, book_deviate_todo, all_strt, strt, book_learning, board_copy, player, n_loop, book_file, book_bak);
        if (n_add == 0) {
            break;
        }
        n_registered += n_add;
        std::cerr << "loop " << n_loop << " book deviated registered " << n_registered << " board size " << book.size() << std::endl;
    }
    *player = before_player;
    root_board.copy(board_copy);
    //book.fix();
    //book.save_egbk3(book_file, book_bak);
    std::cerr << "book deviate finished registered " << n_registered << " time " << ms_to_time_short(tim() - all_strt) << std::endl;
    *book_learning = false;
}


inline void book_store(std::vector<std::pair<Board, int>> tasks, int level, int book_depth, Board *board_copy, int *player, std::string book_file, std::string book_bak, bool *book_learning) {
    uint64_t all_strt = tim();
    Board before_board = board_copy->copy();
    int before_player = *player;
    std::cerr << "book store started" << std::endl;
    Flip flip;
    *player = BLACK;
    int n_registered = 0;
    for (std::pair<Board, int> task: tasks) {
        if (!(*book_learning)) {
            break;
        }
        Board board = task.first;
        if (board.n_discs() > 4 + book_depth) {
            continue;
        }
        board.copy(board_copy);
        *player = task.second;
        Search_result search_result = ai_searching(board, level, true, 0, true, false, book_learning);
        if (is_valid_policy(search_result.policy) && (board.get_legal() & (1ULL << search_result.policy)) && is_valid_score(search_result.value)) {
            std::cerr << idx_to_coord(search_result.policy) << " " << search_result.value << std::endl;
            board.print();
            std::cerr << std::endl;
            if (!book_learning_change(board, search_result.value, level, book_learning)) {
                break;
            }
            calc_flip(&flip, &board, search_result.policy);
            board.move_board(&flip);
                bool passed = board.get_legal() == 0;
                if (passed) {
                    board.pass();
                }
                Search_result child_search_result = ai_searching(board, level, true, 0, true, false, book_learning);
                std::cerr << "child " << child_search_result.value << std::endl;
                board.print();
                std::cerr << std::endl;
                book_learning_change(board, child_search_result.value, level, book_learning); // the loop ends if this fails
                if (passed) {
                    board.pass();
                }
            board.undo_board(&flip);
            n_registered += 2;
        }
    }
    *player = before_player;
    before_board.copy(board_copy);
    //book.fix();
    //book.save_egbk3(book_file, book_bak);
    std::cerr << "book store finished registered " << n_registered << " time " << ms_to_time_short(tim() - all_strt) << std::endl;
    *book_learning = false;
}