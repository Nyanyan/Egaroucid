/*
    Egaroucid Project

    @file ai.hpp
        Main algorithm of Egaroucid
    @date 2021-2025
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once
#include <iostream>
#include <future>
#include <unordered_set>
#include <iomanip>
#include "level.hpp"
#include "setting.hpp"
#include "midsearch.hpp"
#include "book.hpp"
#include "util.hpp"
#include "clogsearch.hpp"
#include "time_management.hpp"

constexpr int AI_TYPE_BOOK = 1000;

constexpr int IDSEARCH_ENDSEARCH_PRESEARCH_OFFSET = 10;
constexpr int IDSEARCH_ENDSEARCH_PRESEARCH_OFFSET_TIMELIMIT = 8;
constexpr int PONDER_ENDSEARCH_PRESEARCH_OFFSET_TIMELIMIT = 4;

constexpr int PONDER_START_SELFPLAY_DEPTH = 19;

constexpr int AI_TL_EARLY_BREAK_THRESHOLD = 6;

constexpr double AI_TL_ADDITIONAL_SEARCH_THRESHOLD = 2.25;

struct Lazy_SMP_task {
    uint_fast8_t mpc_level;
    int depth;
    bool is_end_search;
};

struct Ponder_elem {
    Flip flip;
    double value;
    int count;
    int depth;
    uint_fast8_t mpc_level;
    bool is_endgame_search;
    bool is_complete_search;
};

std::vector<Ponder_elem> ai_ponder(Board board, bool show_log, thread_id_t thread_id, bool *searching);
std::vector<Ponder_elem> ai_get_values(Board board, bool show_log, uint64_t time_limit, thread_id_t thread_id);
std::pair<int, int> ponder_selfplay(Board board_start, int mid_depth, bool show_log, bool use_multi_thread, bool *searching);
std::vector<Ponder_elem> ai_align_move_levels(Board board, bool show_log, std::vector<Ponder_elem> move_list, int n_good_moves, uint64_t time_limit, thread_id_t thread_id, int aligned_min_level);
std::vector<Ponder_elem> ai_search_moves(Board board, bool show_log, std::vector<Ponder_elem> move_list, int n_good_moves, uint64_t time_limit, thread_id_t thread_id);
Search_result ai_legal_window(Board board, int alpha, int beta, int level, bool use_book, int book_acc_level, bool use_multi_thread, bool show_log, uint64_t use_legal);

inline uint64_t get_this_search_time_limit(uint64_t time_limit, uint64_t elapsed) {
    if (time_limit <= elapsed) {
        return 0;
    }
    return time_limit - elapsed;
}

void iterative_deepening_search(Board board, int alpha, int beta, int depth, uint_fast8_t mpc_level, bool show_log, std::vector<Clog_result> clogs, uint64_t use_legal, bool use_multi_thread, thread_id_t thread_id, Search_result *result, bool *searching) {
    uint64_t strt = tim();
    result->value = SCORE_UNDEFINED;
    int main_depth = 1;
    int main_mpc_level = mpc_level;
    const int max_depth = HW2 - board.n_discs();
    depth = std::min(depth, max_depth);
    bool is_end_search = (depth == max_depth);
    if (is_end_search) {
        main_mpc_level = MPC_74_LEVEL;
    }
    if (show_log) {
        std::cerr << "thread pool size " << thread_pool.size() << " n_idle " << thread_pool.get_n_idle() << std::endl;
    }
#if USE_LAZY_SMP
    std::vector<Search> searches(thread_pool.size() + 1);
#endif
    while (main_depth <= depth && main_mpc_level <= mpc_level && global_searching && *searching) {
#if USE_LAZY_SMP
        for (Search &search: searches) {
            search.n_nodes = 0;
            search.thread_id = thread_id;
        }
#endif
        bool main_is_end_search = false;
        if (main_depth >= max_depth) {
            main_is_end_search = true;
            main_depth = max_depth;
        }
        bool is_last_search = (main_depth == depth) && (main_mpc_level == mpc_level);
#if USE_LAZY_SMP
        std::vector<std::future<int>> parallel_tasks;
        std::vector<int> sub_depth_arr;
        int sub_max_mpc_level[61];
        bool sub_searching = true;
        int sub_depth = main_depth;
        if (use_multi_thread && !(is_end_search && main_depth == depth) && main_depth <= 10) {
            int max_thread_size = std::min(thread_pool.size(), thread_pool.get_max_thread_size(thread_id));
            for (int i = 0; i < main_depth - 14; ++i) {
                max_thread_size *= 0.9;
            }
            sub_max_mpc_level[main_depth] = main_mpc_level + 1;
            for (int i = main_depth + 1; i < 61; ++i) {
                sub_max_mpc_level[i] = MPC_74_LEVEL;
            }
            for (int sub_thread_idx = 0; sub_thread_idx < max_thread_size && sub_thread_idx < searches.size() && global_searching && *searching; ++sub_thread_idx) {
                int ntz = ctz_uint32(sub_thread_idx + 1);
                int sub_depth = std::min(max_depth, main_depth + ntz);
                uint_fast8_t sub_mpc_level = sub_max_mpc_level[sub_depth];
                bool sub_is_end_search = (sub_depth == max_depth);
                if (sub_mpc_level <= MPC_100_LEVEL) {
                    //std::cerr << sub_thread_idx << " " << sub_depth << " " << SELECTIVITY_PERCENTAGE[sub_mpc_level] << std::endl;
                    searches[sub_thread_idx] = Search{&board, sub_mpc_level, false, true};
                    bool pushed = false;
                    parallel_tasks.emplace_back(thread_pool.push(thread_id, &pushed, std::bind(&nega_scout, &searches[sub_thread_idx], alpha, beta, sub_depth, false, use_legal, sub_is_end_search, &sub_searching)));
                    sub_depth_arr.emplace_back(sub_depth);
                    ++sub_max_mpc_level[sub_depth];
                    if (!pushed) {
                        parallel_tasks.pop_back();
                        sub_depth_arr.pop_back();
                    }
                }
            }
            int max_sub_search_depth = -1;
            int max_sub_main_mpc_level = 0;
            bool max_is_only_one = false;
            for (int i = 0; i < (int)parallel_tasks.size(); ++i) {
                if (sub_depth_arr[i] > max_sub_search_depth) {
                    max_sub_search_depth = sub_depth_arr[i];
                    max_sub_main_mpc_level = searches[i].mpc_level;
                    max_is_only_one = true;
                } else if (sub_depth_arr[i] == max_sub_search_depth && max_sub_main_mpc_level < searches[i].mpc_level) {
                    max_sub_main_mpc_level = searches[i].mpc_level;
                    max_is_only_one = true;
                } else if (sub_depth_arr[i] == max_sub_search_depth && searches[i].mpc_level == max_sub_main_mpc_level) {
                    max_is_only_one = false;
                }
            }
        }
#endif
        Search main_search(&board, main_mpc_level, use_multi_thread, !is_last_search);
        main_search.thread_id = thread_id;
        std::pair<int, int> id_result = first_nega_scout_legal(&main_search, alpha, beta, main_depth, main_is_end_search, clogs, use_legal, strt, searching);
#if USE_LAZY_SMP
        sub_searching = false;
        for (std::future<int> &task: parallel_tasks) {
            task.get();
        }
        for (Search &search: searches) {
            result->nodes += search.n_nodes;
        }
#endif
        result->nodes += main_search.n_nodes;
        if (*searching) {
            if (result->value != SCORE_UNDEFINED && !main_is_end_search) {
                double n_value = (0.9 * result->value + 1.1 * id_result.first) / 2.0;
                result->value = round(n_value);
            } else{
                result->value = id_result.first;
            }
            result->policy = id_result.second;
            result->depth = main_depth;
            result->is_end_search = main_is_end_search;
            result->probability = SELECTIVITY_PERCENTAGE[main_mpc_level];
        }
        result->time = tim() - strt;
        result->nps = calc_nps(result->nodes, result->time);
        if (show_log) {
            if (is_last_search) {
                std::cerr << "main ";
            } else{
                std::cerr << "pre ";
            }
            if (main_is_end_search) {
                std::cerr << "end ";
            } else{
                std::cerr << "mid ";
            }
#if USE_LAZY_SMP
            std::cerr << "depth " << result->depth << "@" << SELECTIVITY_PERCENTAGE[main_mpc_level] << "%" << " value " << result->value << " (raw " << id_result.first << ") policy " << idx_to_coord(id_result.second) << " n_worker " << parallel_tasks.size() << " n_nodes " << result->nodes << " time " << result->time << " NPS " << result->nps << std::endl;
#else
            std::cerr << "depth " << result->depth << "@" << SELECTIVITY_PERCENTAGE[main_mpc_level] << "%" << " value " << result->value << " (raw " << id_result.first << ") policy " << idx_to_coord(id_result.second) << " n_nodes " << result->nodes << " time " << result->time << " NPS " << result->nps << std::endl;
#endif
        }
        if (is_end_search && main_depth >= depth - IDSEARCH_ENDSEARCH_PRESEARCH_OFFSET) {
            if (main_depth < depth) {
                main_depth = depth;
                if (depth <= 27 && mpc_level >= MPC_88_LEVEL) {
                    main_mpc_level = MPC_88_LEVEL;
                } else {
                    main_mpc_level = MPC_74_LEVEL;
                }
            } else{
                if (main_mpc_level < mpc_level) {
                    if (
                        (main_mpc_level >= MPC_74_LEVEL && mpc_level > MPC_74_LEVEL && depth <= 22) || 
                        (main_mpc_level >= MPC_88_LEVEL && mpc_level > MPC_88_LEVEL && depth <= 25) || 
                        (main_mpc_level >= MPC_93_LEVEL && mpc_level > MPC_93_LEVEL && depth <= 29) || 
                        (main_mpc_level >= MPC_98_LEVEL && mpc_level > MPC_98_LEVEL)
                    ) {
                        main_mpc_level = mpc_level;
                    } else{
                        ++main_mpc_level;
                    }
                } else{
                    break;
                }
            }
        } else {
            if (main_depth <= 15 && main_depth < depth - 3) {
                main_depth += 3;
            } else{
                ++main_depth;
            }
        }
    }
}

void iterative_deepening_search_time_limit(Board board, int alpha, int beta, bool show_log, std::vector<Clog_result> clogs, uint64_t use_legal, bool use_multi_thread, thread_id_t thread_id, Search_result *result, uint64_t time_limit, bool *searching) {
    uint64_t strt = tim();
    result->value = SCORE_UNDEFINED;
    int main_depth = 1;
    int main_mpc_level = MPC_100_LEVEL;
    const int max_depth = HW2 - board.n_discs();
    if (show_log) {
        std::cerr << "thread pool size " << thread_pool.size() << " n_idle " << thread_pool.get_n_idle() << std::endl;
    }
    int before_raw_value = -100;
    bool policy_changed_before = true;
    while (global_searching && (*searching) && ((tim() - strt < time_limit) || main_depth <= 1)) {
        bool main_is_end_search = false;
        if (main_depth >= max_depth) {
            main_is_end_search = true;
            main_depth = max_depth;
        }
        bool main_is_complete_search = main_is_end_search && main_mpc_level == MPC_100_LEVEL;
        if (show_log) {
            if (main_is_end_search) {
                std::cerr << "end ";
            } else{
                std::cerr << "mid ";
            }
            std::cerr << "depth " << main_depth << "@" << SELECTIVITY_PERCENTAGE[main_mpc_level] << "% " << std::flush;
        }
        Search main_search(&board, main_mpc_level, use_multi_thread, false);
        main_search.thread_id = thread_id;
        std::pair<int, int> id_result;
        bool search_success = false;
        bool main_searching = true;
        uint64_t time_limit_this_search = get_this_search_time_limit(time_limit, tim() - strt);
        std::future<std::pair<int, int>> f = std::async(std::launch::async, first_nega_scout_legal, &main_search, alpha, beta, main_depth, main_is_end_search, clogs, use_legal, strt, &main_searching);
        if (f.wait_for(std::chrono::milliseconds(time_limit_this_search)) == std::future_status::ready) {
            id_result = f.get();
            search_success = true;
        } else {
            main_searching = false;
            try {
                f.get();
            } catch (const std::exception &e) {
            }
            if (show_log) {
                std::cerr << "terminated " << tim() - strt << " ms" << std::endl;
            }
        }
        result->nodes += main_search.n_nodes;
        result->time = tim() - strt;
        result->nps = calc_nps(result->nodes, result->time);
        if (search_success) {
            if (result->value != SCORE_UNDEFINED && !main_is_end_search) {
                double n_value = (0.9 * result->value + 1.1 * id_result.first) / 2.0;
                result->value = round(n_value);
            } else{
                result->value = id_result.first;
            }
            bool policy_changed = result->policy != id_result.second;
            result->policy = id_result.second;
            result->depth = main_depth;
            result->is_end_search = main_is_end_search;
            result->probability = SELECTIVITY_PERCENTAGE[main_mpc_level];
            if (show_log) {
                std::cerr << "value " << result->value << " (raw " << id_result.first << ") policy " << idx_to_coord(id_result.second) << " n_nodes " << result->nodes << " time " << result->time << " NPS " << result->nps << std::endl;
            }
            uint64_t legal_without_bestmove = use_legal ^ (1ULL << result->policy);
            if (
                (!main_is_end_search && main_depth >= 30 && main_depth <= 31) && 
                !policy_changed && 
                !policy_changed_before && 
                main_mpc_level == MPC_74_LEVEL && 
                legal_without_bestmove != 0
            ) {
                int nws_alpha = result->value - AI_TL_EARLY_BREAK_THRESHOLD;
                if (nws_alpha >= -SCORE_MAX) {
                    Search nws_search(&board, main_mpc_level, use_multi_thread, false);
                    nws_search.thread_id = thread_id;
                    bool nws_searching = true;
                    uint64_t time_limit_nws = get_this_search_time_limit(time_limit, tim() - strt);
                    std::future<std::pair<int, int>> nws_f = std::async(std::launch::async, first_nega_scout_legal, &nws_search, nws_alpha, nws_alpha + 1, main_depth, main_is_end_search, clogs, legal_without_bestmove, strt, &nws_searching);
                    int nws_value = SCORE_INF;
                    int nws_move = MOVE_NOMOVE;
                    bool nws_success = false;
                    if (nws_f.wait_for(std::chrono::milliseconds(time_limit_nws)) == std::future_status::ready) {
                        std::pair<int, int> nws_result = nws_f.get();
                        nws_value = nws_result.first;
                        nws_move = nws_result.second;
                        nws_success = true;
                    } else {
                        nws_searching = false;
                        try {
                            nws_f.get();
                        } catch (const std::exception &e) {
                        }
                        if (show_log) {
                            std::cerr << "terminate early cut nws by time limit " << tim() - strt << " ms" << std::endl;
                        }
                    }
                    result->nodes += nws_search.n_nodes;
                    result->time = tim() - strt;
                    result->nps = calc_nps(result->nodes, result->time);
                    if (nws_success) {
                        if (nws_value <= nws_alpha) {
                            if (show_log) {
                                std::cerr << "early break second best " << idx_to_coord(nws_move) << " value <= " << nws_value << " time " << tim() - strt << std::endl;
                            }
                            break;
                        } else if (nws_searching) {
                            if (show_log) {
                                std::cerr << "no early break second best " << idx_to_coord(nws_move) << " value >= " << nws_value << " time " << tim() - strt << std::endl;
                            }
                        }
                    }
                }
            }
            // check second best move
            /*
            if (!main_is_complete_search && main_depth >= 30 && legal_without_bestmove != 0) {
                Search second_search(&board, main_mpc_level, use_multi_thread, false);
                std::pair<int, int> second_id_result;
                bool second_search_success = false;
                bool second_searching = true;
                uint64_t time_limit_second_search = get_this_search_time_limit(time_limit, tim() - strt);
                std::future<std::pair<int, int>> second_f = std::async(std::launch::async, first_nega_scout_legal, &second_search, alpha, beta, main_depth, main_is_end_search, clogs, legal_without_bestmove, strt, &second_searching);
                if (second_f.wait_for(std::chrono::milliseconds(time_limit_second_search)) == std::future_status::ready) {
                    std::pair<int, int> nws_result = second_f.get();
                    second_search_success = true;
                } else {
                    second_searching = false;
                    second_f.get();
                    if (show_log) {
                        std::cerr << "terminate second search by time limit " << tim() - strt << " ms" << std::endl;
                    }
                }
            }
            */
            before_raw_value = id_result.first;
            policy_changed_before = policy_changed;
        }
        //if (main_depth > 10 && pop_count_ull(board.get_legal()) == 1) { // not use_legal
        //    break; // there is only 1 move
        //}
        if (main_depth < max_depth - IDSEARCH_ENDSEARCH_PRESEARCH_OFFSET_TIMELIMIT) { // next: midgame search
            if (main_depth <= 15 && main_depth < max_depth - 3) {
                main_depth += 3;
                if (main_depth > 13 && main_mpc_level == MPC_100_LEVEL) {
                    main_mpc_level = MPC_74_LEVEL;
                }
            } else {
                if (main_depth + 1 < 23) {
                    ++main_depth;
                    if (main_depth > 13 && main_mpc_level == MPC_100_LEVEL) {
                        main_mpc_level = MPC_74_LEVEL;
                    }
                } else {
                    ++main_depth;
                    /*
                    if (main_mpc_level < MPC_88_LEVEL) {
                        ++main_mpc_level;
                    } else {
                        ++main_depth;
                        main_mpc_level = MPC_74_LEVEL;
                    }
                    */
                }
            }
        } else { // next: endgame search
//#if IS_GGS_TOURNAMENT
            if (max_depth >= 38) {
                if (main_depth < max_depth - 8) { // midgame -> midgame
                    ++main_depth;
                } else if (main_depth < max_depth) { // midgame -> endgame
                    main_depth = max_depth;
                    main_mpc_level = MPC_74_LEVEL;
                } else if (main_mpc_level < MPC_100_LEVEL) { // endgame -> endgame
                    ++main_mpc_level;
                } else {
                    break;
                }
            } else
//#endif
            if (main_depth < max_depth) {
                main_depth = max_depth;
                main_mpc_level = MPC_74_LEVEL;
            } else {
                if (main_mpc_level < MPC_100_LEVEL) {
                    ++main_mpc_level;
                } else {
                    // if (show_log) {
                    //     std::cerr << "completely searched" << std::endl;
                    // }
                    break;
                }
            }
        }
    }
    if (show_log && result->is_end_search && result->probability == 100) {
        std::cerr << "completely searched" << std::endl;
    }
}


/*
    @brief Get a result of a search

    Firstly, if using MPC, execute clog search for finding special endgame.
    Then do some pre-search, and main search.

    @param board                board to solve
    @param depth                depth to search
    @param mpc_level            MPC level
    @param show_log             show log?
    @param use_multi_thread     search in multi thread?
    @return the result in Search_result structure
*/
inline Search_result tree_search_legal(Board board, int alpha, int beta, int depth, uint_fast8_t mpc_level, bool show_log, uint64_t use_legal, bool use_multi_thread, uint64_t time_limit, thread_id_t thread_id, bool *searching) {
    //thread_pool.tell_start_using();
    Search_result res;
    depth = std::min(HW2 - board.n_discs(), depth);
    bool is_end_search = (HW2 - board.n_discs() == depth);
    bool use_time_limit = (time_limit != TIME_LIMIT_INF);
    std::vector<Clog_result> clogs;
    uint64_t clog_nodes = 0;
    uint64_t clog_time = 0;
    if (mpc_level != MPC_100_LEVEL || use_time_limit) {
        uint64_t strt = tim();
        int clog_depth = std::min(depth, CLOG_SEARCH_MAX_DEPTH);
        if (use_time_limit) {
            clog_depth = CLOG_SEARCH_MAX_DEPTH;
        }
        clogs = first_clog_search(board, &clog_nodes, clog_depth, use_legal, searching);
        clog_time = tim() - strt;
        if (show_log) {
            std::cerr << "clog search depth " << clog_depth << " time " << clog_time << " nodes " << clog_nodes << " nps " << calc_nps(clog_nodes, clog_time) << std::endl;
            for (int i = 0; i < (int)clogs.size(); ++i) {
                std::cerr << "clogsearch " << i + 1 << "/" << clogs.size() << " " << idx_to_coord(clogs[i].pos) << " value " << clogs[i].val << std::endl;
            }
        }
        res.clog_nodes = clog_nodes;
        res.clog_time = clog_time;
    }
    if (use_legal) {
        uint64_t time_limit_proc = TIME_LIMIT_INF;
        if (use_time_limit) {
            if (time_limit > clog_time) {
                time_limit_proc = time_limit - clog_time;
            } else {
                time_limit_proc = 1;
            }
        }
        if (use_time_limit) {
            /*
            if (HW2 - board.n_discs() >= 30) {
                uint64_t strt_selfplay = tim();
                uint64_t selfplay_time = time_limit_proc * 0.3;
                time_management_selfplay(board, show_log, use_legal, selfplay_time);
                time_limit_proc -= tim() - strt_selfplay;
            }
            */
            iterative_deepening_search_time_limit(board, alpha, beta, show_log, clogs, use_legal, use_multi_thread, thread_id, &res, time_limit_proc, searching);
        } else {
            iterative_deepening_search(board, alpha, beta, depth, mpc_level, show_log, clogs, use_legal, use_multi_thread, thread_id, &res, searching);
        }
    }
    //thread_pool.tell_finish_using();
    //thread_pool.reset_unavailable();
    //delete_tt(&board, 6);
    return res;
}

/*
    @brief Get a result of a search with book or search

    Firstly, check if the given board is in the book.
    Then search the board and get the result.

    @param board                board to solve
    @param level                level of AI
    @param alpha                search window (alpha)
    @param beta                 search window (beta)
    @param use_book             use book?
	@param book_acc_level		book accuracy level
    @param use_multi_thread     search in multi thread?
    @param show_log             show log?
    @return the result in Search_result structure
*/
Search_result ai_common(Board board, int alpha, int beta, int level, bool use_book, int book_acc_level, bool use_multi_thread, bool show_log, uint64_t use_legal, bool use_specified_move_book, uint64_t time_limit, thread_id_t thread_id, bool *searching) {
    Search_result res;
    int value_sign = 1;
    if (board.get_legal() == 0ULL) {
        board.pass();
        if (board.get_legal() == 0ULL) {
            res.level = MAX_LEVEL;
            res.policy = 64;
            res.value = -board.score_player();
            res.depth = 0;
            res.nps = 0;
            res.is_end_search = true;
            res.probability = 100;
            return res;
        } else{
            if (show_log) {
                std::cerr << "pass found in ai_common!" << std::endl;
            }
            value_sign = -1;
        }
    }
    Book_value book_result;
    if (use_specified_move_book) {
        book_result = book.get_specified_best_move(&board, use_legal);
    } else{
        book_result = book.get_random(&board, book_acc_level, use_legal);
    }
    if (is_valid_policy(book_result.policy) && use_book) {
        if (show_log) {
            std::cerr << "book found " << value_sign * book_result.value << " " << idx_to_coord(book_result.policy) << std::endl;
        }
        res.level = LEVEL_TYPE_BOOK;
        res.policy = book_result.policy;
        res.value = value_sign * book_result.value;
        res.depth = SEARCH_BOOK;
        res.nps = 0;
        res.is_end_search = false;
        res.probability = 100;
        if (book_acc_level == 0) { // accurate book level
            std::vector<Book_value> book_moves = book.get_all_moves_with_value(&board);
            for (const Book_value &move: book_moves) {
                use_legal &= ~(1ULL << move.policy);
            }
            if (use_legal != 0) { // there is moves out of book
                if (show_log) {
                    std::cerr << "there are good moves out of book" << std::endl;
                }
                bool need_to_check = false;
                Flip flip;
                calc_flip(&flip, &board, book_result.policy);
                board.move_board(&flip);
                    bool passed = false;
                    bool game_over = false;
                    if (board.get_legal() == 0) {
                        passed = true;
                        board.pass();
                        if (board.get_legal() == 0) {
                            game_over = true;
                        }
                    }
                    if (!game_over) {
                        Book_elem book_elem = book.get(board);
                        if (book_elem.level < level) {
                            need_to_check = true;
                        }
                    }
                    if (passed) {
                        board.pass();
                    }
                board.undo_board(&flip);
                if (need_to_check) {
                    int n_alpha = std::max(alpha, book_result.value + 1);
                    int level_proc = level;
                    if (time_limit != TIME_LIMIT_INF) {
                        level_proc = 25;
                    }
                    Search_result additional_result = ai_legal_window(board, n_alpha, beta, level_proc, true, 0, true, false, use_legal);
                    if (value_sign * additional_result.value >= res.value + 2) {
                        if (show_log) {
                            std::cerr << "better move found out of book " << idx_to_coord(additional_result.policy) << "@" << value_sign * additional_result.value << " book " << idx_to_coord(res.policy) << "@" << res.value << std::endl;
                        }
                        res = additional_result;
                        res.level = level;
                        res.value *= value_sign;
                    }
                }
            } else {
                if (show_log) {
                    std::cerr << "all moves are in book" << std::endl;
                }
            }
        }
    } else { // no move in book
        int depth;
        bool is_mid_search;
        uint_fast8_t mpc_level;
        get_level(level, board.n_discs() - 4, &is_mid_search, &depth, &mpc_level);
        if (show_log && time_limit == TIME_LIMIT_INF) {
            std::cerr << "level status " << level << " " << board.n_discs() - 4 << " discs depth " << depth << "@" << SELECTIVITY_PERCENTAGE[mpc_level] << "%" << std::endl;
        }
        //thread_pool.tell_start_using();
        res = tree_search_legal(board, alpha, beta, depth, mpc_level, show_log, use_legal, use_multi_thread, time_limit, thread_id, searching);
        //thread_pool.tell_finish_using();
        res.level = level;
        res.value *= value_sign;
    }
    return res;
}

/*
    @brief Get a result of a search with book or search

    Firstly, check if the given board is in the book.
    Then search the board and get the result.

    @param board                board to solve
    @param level                level of AI
    @param use_book             use book?
	@param book_acc_level		book accuracy level
    @param use_multi_thread     search in multi thread?
    @param show_log             show log?
    @return the result in Search_result structure
*/
Search_result ai(Board board, int level, bool use_book, int book_acc_level, bool use_multi_thread, bool show_log) {
    bool searching = true;
    return ai_common(board, -SCORE_MAX, SCORE_MAX, level, use_book, book_acc_level, use_multi_thread, show_log, board.get_legal(), false, TIME_LIMIT_INF, THREAD_ID_NONE, &searching);
}

Search_result ai_searching(Board board, int level, bool use_book, int book_acc_level, bool use_multi_thread, bool show_log, bool *searching) {
    return ai_common(board, -SCORE_MAX, SCORE_MAX, level, use_book, book_acc_level, use_multi_thread, show_log, board.get_legal(), false, TIME_LIMIT_INF, THREAD_ID_NONE, searching);
}

Search_result ai_searching_thread_id(Board board, int level, bool use_book, int book_acc_level, bool use_multi_thread, bool show_log, thread_id_t thread_id, bool *searching) {
    return ai_common(board, -SCORE_MAX, SCORE_MAX, level, use_book, book_acc_level, use_multi_thread, show_log, board.get_legal(), false, TIME_LIMIT_INF, thread_id, searching);
}

Search_result ai_legal(Board board, int level, bool use_book, int book_acc_level, bool use_multi_thread, bool show_log, uint64_t use_legal) {
    bool searching = true;
    return ai_common(board, -SCORE_MAX, SCORE_MAX, level, use_book, book_acc_level, use_multi_thread, show_log, use_legal, false, TIME_LIMIT_INF, THREAD_ID_NONE, &searching);
}

Search_result ai_legal_window(Board board, int alpha, int beta, int level, bool use_book, int book_acc_level, bool use_multi_thread, bool show_log, uint64_t use_legal) {
    bool searching = true;
    return ai_common(board, alpha, beta, level, use_book, book_acc_level, use_multi_thread, show_log, use_legal, false, TIME_LIMIT_INF, THREAD_ID_NONE, &searching);
}

Search_result ai_legal_searching(Board board, int level, bool use_book, int book_acc_level, bool use_multi_thread, bool show_log, uint64_t use_legal, bool *searching) {
    return ai_common(board, -SCORE_MAX, SCORE_MAX, level, use_book, book_acc_level, use_multi_thread, show_log, use_legal, false, TIME_LIMIT_INF, THREAD_ID_NONE, searching);
}

Search_result ai_window_legal(Board board, int alpha, int beta, int level, bool use_book, int book_acc_level, bool use_multi_thread, bool show_log, uint64_t use_legal) {
    bool searching = true;
    return ai_common(board, alpha, beta, level, use_book, book_acc_level, use_multi_thread, show_log, use_legal, false, TIME_LIMIT_INF, THREAD_ID_NONE, &searching);
}

Search_result ai_window_legal_searching(Board board, int alpha, int beta, int level, bool use_book, int book_acc_level, bool use_multi_thread, bool show_log, uint64_t use_legal, bool *searching) {
    return ai_common(board, alpha, beta, level, use_book, book_acc_level, use_multi_thread, show_log, use_legal, false, TIME_LIMIT_INF, THREAD_ID_NONE, searching);
}

Search_result ai_window_searching(Board board, int alpha, int beta, int level, bool use_book, int book_acc_level, bool use_multi_thread, bool show_log, bool *searching) {
    return ai_common(board, alpha, beta, level, use_book, book_acc_level, use_multi_thread, show_log, board.get_legal(), false, TIME_LIMIT_INF, THREAD_ID_NONE, searching);
}

Search_result ai_specified(Board board, int level, bool use_book, int book_acc_level, bool use_multi_thread, bool show_log) {
    bool searching = true;
    return ai_common(board, -SCORE_MAX, SCORE_MAX, level, use_book, book_acc_level, use_multi_thread, show_log, board.get_legal(), true, TIME_LIMIT_INF, THREAD_ID_NONE, &searching);
}

std::vector<Search_result> ai_best_n_moves_searching(Board board, int level, bool use_book, int book_acc_level, bool use_multi_thread, bool show_log, int n_moves, bool *searching) {
    Search_result best = ai_searching(board, level, true, 0, true, show_log, searching);
    std::vector<Search_result> search_results;
    search_results.emplace_back(best);
    int alpha = -SCORE_MAX;
    int beta = best.value;
    uint64_t legal = board.get_legal() ^ (1ULL << best.policy);
    while (legal && search_results.size() < n_moves) {
        Search_result result_loss = ai_window_legal_searching(board, alpha, beta, level, true, 0, true, show_log, legal, searching);
        legal ^= 1ULL << result_loss.policy;
        search_results.emplace_back(result_loss);
    }
    return search_results;
}

std::vector<Search_result> ai_best_moves_loss_searching(Board board, int level, bool use_book, int book_acc_level, bool use_multi_thread, bool show_log, int loss_max, bool *searching) {
    Search_result best = ai_searching(board, level, true, 0, true, show_log, searching);
    std::vector<Search_result> search_results;
    search_results.emplace_back(best);
    int alpha = best.value - loss_max - 1;
    int beta = best.value;
    uint64_t legal = board.get_legal() ^ (1ULL << best.policy);
    while (legal && (*searching)) {
        Search_result result_loss = ai_window_legal_searching(board, alpha, beta, level, true, 0, true, show_log, legal, searching);
        legal ^= 1ULL << result_loss.policy;
        if (*searching) {
            if (result_loss.value >= best.value - loss_max) {
                search_results.emplace_back(result_loss);
            } else {
                break;
            }
        }
    }
    return search_results;
}

Search_result ai_loss_searching(Board board, int level, bool use_book, int book_acc_level, bool use_multi_thread, bool show_log, int loss_max, bool *searching) {
    std::vector<Search_result> search_results = ai_best_moves_loss_searching(board, level, use_book, book_acc_level, use_multi_thread, show_log, loss_max, searching);
    if (show_log){
        std::cerr << "play loss candidate " << search_results.size() << " moves: ";
        for (const Search_result &elem: search_results) {
            std::cerr << idx_to_coord(elem.policy) << "@" << elem.value << " ";
        }
        std::cerr << std::endl;
    }
    return search_results[myrandrange(0, (int)search_results.size())];
}

Search_result ai_loss(Board board, int level, bool use_book, int book_acc_level, bool use_multi_thread, bool show_log, int loss_max) {
    bool searching = true;
    return ai_loss_searching(board, level, use_book, book_acc_level, use_multi_thread, show_log, loss_max, &searching);
}

Search_result ai_time_limit(Board board, bool use_book, int book_acc_level, bool use_multi_thread, bool show_log, uint64_t remaining_time_msec, thread_id_t thread_id, bool *searching) {
    uint64_t time_limit = calc_time_limit_ply(board, remaining_time_msec, show_log);
    if (show_log) {
        std::cerr << "ai_time_limit start! tl " << time_limit << " remaining " << remaining_time_msec << " n_empties " << HW2 - board.n_discs() << " " << board.to_str() << std::endl;
    }
    uint64_t strt = tim();
    int n_empties = HW2 - board.n_discs();
    if (time_limit > 10000ULL && n_empties >= 33) { // additional search
        bool need_request_more_time = false;
        bool get_values_searching = true;
        uint64_t get_values_tl = 1000ULL;
        if (show_log) {
            std::cerr << "getting values tl " << get_values_tl << std::endl;
        }
        std::vector<Ponder_elem> get_values_move_list = ai_get_values(board, show_log, get_values_tl, thread_id);
        if (get_values_move_list.size()) {
            double best_value = get_values_move_list[0].value;
            int n_good_moves = 0;
            for (const Ponder_elem &elem: get_values_move_list) {
                if (elem.value >= best_value - AI_TL_ADDITIONAL_SEARCH_THRESHOLD) {
                    ++n_good_moves;
                } else {
                    break; // because sorted
                }
            }
            if (n_good_moves >= 2) {
                if (show_log) {
                    std::cerr << "need to search good moves :";
                    for (int i = 0; i < n_good_moves; ++i) {
                        std::cerr << " " << idx_to_coord(get_values_move_list[i].flip.pos);
                    }
                    std::cerr << std::endl;
                }
                uint64_t elapsed_till_get_values = tim() - strt;
                if (elapsed_till_get_values > 3000) {
                    elapsed_till_get_values = 3000;
                }
                uint64_t align_moves_tl = std::max<uint64_t>(3000ULL - elapsed_till_get_values, (uint64_t)((time_limit - elapsed_till_get_values) * std::min(0.3, 0.1 * n_good_moves)));
                std::vector<Ponder_elem> after_move_list = ai_align_move_levels(board, show_log, get_values_move_list, n_good_moves, align_moves_tl, thread_id, 27);
                need_request_more_time = true;

                double new_best_value = after_move_list[0].value;
                int new_n_good_moves = 0;
                for (const Ponder_elem &elem: after_move_list) {
                    if (elem.value >= best_value - AI_TL_ADDITIONAL_SEARCH_THRESHOLD) {
                        ++new_n_good_moves;
                    } else {
                        break; // because sorted
                    }
                }
                if (new_n_good_moves >= 2) {
                    uint64_t elapsed_till_align_level = tim() - strt;
                    if (time_limit > elapsed_till_align_level) {
                        uint64_t self_play_tl = (uint64_t)((time_limit - elapsed_till_align_level) * std::min(0.7, 0.3 * new_n_good_moves));
                        if (show_log) {
                            std::cerr << "need to search good moves (self play) :";
                            for (int i = 0; i < new_n_good_moves; ++i) {
                                std::cerr << " " << idx_to_coord(after_move_list[i].flip.pos);
                            }
                            std::cerr << std::endl;
                        }
                        std::vector<Ponder_elem> after_move_list2 = ai_search_moves(board, show_log, after_move_list, new_n_good_moves, self_play_tl, thread_id);
                    }
                }
            }
        }
        uint64_t elapsed_special_search = tim() - strt;
        if (time_limit > elapsed_special_search) {
            time_limit -= elapsed_special_search;
        } else {
            time_limit = 1;
        }
        if (need_request_more_time) {
            uint64_t remaining_time_msec_p = 1;
            if (remaining_time_msec > elapsed_special_search) {
                remaining_time_msec_p = remaining_time_msec - elapsed_special_search;
            }
            time_limit = request_more_time(board, remaining_time_msec_p, time_limit, show_log);
        }
        if (show_log) {
            std::cerr << "additional calculation elapsed " << elapsed_special_search << " reduced time limit " << time_limit << std::endl;
        }
    }
    if (show_log) {
        std::cerr << "ai_common main search tl " << time_limit << std::endl;
    }
    Search_result search_result = ai_common(board, -SCORE_MAX, SCORE_MAX, MAX_LEVEL, use_book, book_acc_level, use_multi_thread, show_log, board.get_legal(), false, time_limit, thread_id, searching);
    if (show_log) {
        std::cerr << "ai_time_limit selected " << idx_to_coord(search_result.policy) << " value " << search_result.value << " depth " << search_result.depth << "@" << search_result.probability << "%" << " time " << tim() - strt << " " << board.to_str() << std::endl;
    }
    return search_result;
}

/*
    @brief Search for analyze command

    @param board                board to solve
    @param level                level of AI
    @param use_multi_thread     search in multi thread?
    @return the result in Search_result structure
*/

Analyze_result ai_analyze(Board board, int level, bool use_multi_thread, uint_fast8_t played_move) {
    int depth;
    bool is_mid_search;
    uint_fast8_t mpc_level;
    get_level(level, board.n_discs() - 4, &is_mid_search, &depth, &mpc_level);
    depth = std::min(HW2 - board.n_discs(), depth);
    bool is_end_search = (HW2 - board.n_discs() == depth);
    bool searching = true;
    std::vector<Clog_result> clogs;
    uint64_t clog_nodes = 0;
    uint64_t clog_time = 0;
    int clog_depth = std::min(depth, CLOG_SEARCH_MAX_DEPTH);
    if (mpc_level != MPC_100_LEVEL) {
        uint64_t clog_strt = tim();
        clogs = first_clog_search(board, &clog_nodes, clog_depth, board.get_legal(), &searching);
        clog_time = tim() - clog_strt;
    }
    Search search(&board, mpc_level, use_multi_thread, false);
    uint64_t strt = tim();
    //thread_pool.tell_start_using();
    Analyze_result res = first_nega_scout_analyze(&search, -SCORE_MAX, SCORE_MAX, depth, is_end_search, clogs, clog_depth, played_move, strt, &searching);
    //thread_pool.tell_finish_using();
    return res;
}



Search_result ai_accept_loss(Board board, int level, int acceptable_loss) {
    uint64_t strt = tim();
    Flip flip;
    int v = SCORE_UNDEFINED;
    uint64_t legal = board.get_legal();
    std::vector<std::pair<int, int>> moves;
    //thread_pool.tell_start_using();
    for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)) {
        calc_flip(&flip, &board, cell);
        board.move_board(&flip);
            int g = -ai(board, level, true, 0, true, false).value;
        board.undo_board(&flip);
        v = std::max(v, g);
        moves.emplace_back(std::make_pair(g, cell));
    }
    //thread_pool.tell_finish_using();
    std::vector<std::pair<int, int>> acceptable_moves;
    for (std::pair<int, int> move: moves) {
        if (move.first >= v - acceptable_loss)
            acceptable_moves.emplace_back(move);
    }
    int rnd_idx = myrandrange(0, (int)acceptable_moves.size());
    int use_policy = acceptable_moves[rnd_idx].second;
    int use_value = acceptable_moves[rnd_idx].first;
    Search_result res;
    res.depth = 1;
    res.nodes = 0;
    res.time = tim() - strt;
    res.nps = calc_nps(res.nodes, res.time);
    res.policy = use_policy;
    res.value = use_value;
    res.is_end_search = board.n_discs() == HW2 - 1;
    res.probability = SELECTIVITY_PERCENTAGE[MPC_100_LEVEL];
    return res;
}

void ai_hint(Board board, int level, bool use_book, int book_acc_level, bool use_multi_thread, bool show_log, int n_display, double values[], int hint_types[]) {
    uint64_t legal = board.get_legal();
    if (use_book) {
        std::vector<Book_value> links = book.get_all_moves_with_value(&board);
        for (Book_value &link: links) {
            values[link.policy] = link.value;
            hint_types[link.policy] = AI_TYPE_BOOK;
            legal ^= 1ULL << link.policy;
            --n_display;
        }
    }
    //thread_pool.tell_start_using();
    for (int search_level = 1; search_level <= level && global_searching; ++search_level) {
        //if (show_log) {
        //    std::cerr << "hint level " << search_level << " calculating" << std::endl;
        //}
        uint64_t search_legal = legal;
        for (int i = 0; i < n_display && search_legal && global_searching; ++i) {
            Search_result elem = ai_legal(board, search_level, use_book, book_acc_level, use_multi_thread, false, search_legal);
            if (global_searching && (search_legal & (1ULL << elem.policy))) {
                search_legal ^= 1ULL << elem.policy;
                values[elem.policy] = elem.value;
                if (elem.is_end_search) {
                    hint_types[elem.policy] = elem.probability;
                } else{
                    hint_types[elem.policy] = search_level;
                }
            }
        }
    }
    //thread_pool.tell_finish_using();
}

std::pair<int, int> ponder_selfplay(Board board_start, int root_depth, bool show_log, bool use_multi_thread, bool *searching) { // used for ponder
    int l = -SCORE_MAX, u = SCORE_MAX, former_val = SCORE_UNDEFINED;
    Search ttsearch(&board_start, MPC_74_LEVEL, use_multi_thread, false);
    transposition_table.get_bounds(&ttsearch, board_start.hash(), root_depth - 1, &l, &u);
    if (l == u) {
        former_val = -l;
        //std::cerr << "former " << root_depth - 1 << " " << l << " " << u << " " << former_val << std::endl;
    }
    Flip flip;
    Board board = board_start.copy();
    int root_n_discs = board.n_discs();
    std::vector<Board> boards;
    int depth_arr[HW2];
    uint_fast8_t mpc_level_arr[HW2];
    int start_n_discs = board_start.n_discs();
    for (int n_discs = 4; n_discs < HW2; ++n_discs) {
        int n_empties = HW2 - n_discs;
        if (n_empties <= 24) { // complete
            depth_arr[n_discs] = n_empties;
            mpc_level_arr[n_discs] = MPC_100_LEVEL;
        } else {
            depth_arr[n_discs] = std::max(root_depth - (n_discs - root_n_discs), 21); // depth 21 or more
            mpc_level_arr[n_discs] = MPC_74_LEVEL;
        }
    }
    if (show_log) {
        std::cerr << "root_depth " << root_depth << " " << board_start.to_str() << " ";
    }
    std::vector<Clog_result> clogs;
    bool is_player = false; // opponent first
    bool move_masked = false;
    while (*searching) { // self play
        uint64_t legal = board.get_legal();
        if (legal == 0) {
            board.pass();
            is_player ^= 1;
            legal = board.get_legal();
            if (legal == 0) {
                break;
            }
        }
        int n_discs = board.n_discs();
        int n_empties = HW2 - n_discs;
        if (n_empties >= 24) {
            boards.emplace_back(board);
        }
        Search search(&board, mpc_level_arr[n_discs], use_multi_thread, false);
        uint64_t use_legal = legal;
        bool is_complete_search = (depth_arr[n_discs] == n_empties) && (mpc_level_arr[n_discs] == MPC_100_LEVEL);
        bool is_end_search = (depth_arr[n_discs] == n_empties);
        // if (pop_count_ull(use_legal) > 1 && !is_player && !is_complete_search && myrandom() < 0.5) {
        //     int n_off = myrandrange(0, 3);
        //     for (int i = 0; i < n_off; ++i) {
        //         std::pair<int, int> presearch_result = first_nega_scout_legal(&search, -SCORE_MAX, SCORE_MAX, 11, (11 == n_empties), clogs, use_legal, tim(), searching);
        //         uint64_t n_legal = use_legal & ~(1ULL << presearch_result.second); // off best move
        //         std::pair<int, int> presearch_masked_result = first_nega_scout_legal(&search, -SCORE_MAX, SCORE_MAX, 11, (11 == n_empties), clogs, n_legal, tim(), searching);
        //         if (presearch_masked_result.first >= presearch_result.first - 4) {
        //             use_legal = n_legal;
        //             move_masked = true;
        //         }
        //     }
        // }
        std::pair<int, int> result = first_nega_scout_legal(&search, -SCORE_MAX, SCORE_MAX, depth_arr[n_discs], is_end_search, clogs, use_legal, tim(), searching);
        if (show_log) {
            std::cerr << idx_to_coord(result.second);
        }
        calc_flip(&flip, &board, result.second);
        board.move_board(&flip);
        is_player ^= 1;
    }
    int score = SCORE_UNDEFINED;
    if (*searching) {
        score = board.score_player();
        if (!is_player) {
            score *= -1;
        }
    }
    if (show_log) {
        if (!(*searching)) {
            std::cerr << " terminated";
        } else {
            std::cerr << " score " << score;
        }
    }
    int analyzed_value = SCORE_UNDEFINED;
    for (int i = (int)boards.size() - 1; i >= 0 && (*searching); --i) { // analyze
        int n_discs = boards[i].n_discs();
        int n_empties = HW2 - n_discs;
        int n_depth = depth_arr[n_discs];
        if (i == 0) {
            n_depth = std::min(n_empties, std::max(21, root_depth));
        }
        Search search(&boards[i], mpc_level_arr[n_discs], use_multi_thread, false);
        bool is_end_search = (n_depth == n_empties);
        std::pair<int, int> result = first_nega_scout(&search, -SCORE_MAX, SCORE_MAX, n_depth, is_end_search, clogs, tim(), searching);
        if (i == 0 && (*searching) && global_searching) {
            analyzed_value = -result.first;
            if (show_log) {
                std::cerr << " analyzed " << analyzed_value;
            }
            double registered_value = analyzed_value;
            if (score < analyzed_value && n_depth < n_empties) {
                registered_value = (double)score * 0.6 + (double)analyzed_value * 0.4;
            }
            if (former_val < registered_value && former_val != SCORE_UNDEFINED) {
                registered_value = (double)former_val * 0.5 + (double)registered_value * 0.5;
            }
            int registered_value_int = round(registered_value);
            transposition_table.reg_overwrite(&search, boards[i].hash(), n_depth, -SCORE_MAX, SCORE_MAX, -registered_value_int, MOVE_UNDEFINED);
            if (show_log) {
                std::cerr << " tt " << registered_value_int;
                if (analyzed_value != registered_value_int) {
                    std::cerr << " updated";
                }
            }
        }
    }
    if (show_log) {
        std::cerr << std::endl;
    }
    return std::make_pair(score, analyzed_value);
}

bool comp_ponder_elem(Ponder_elem &a, Ponder_elem &b) {
    if (a.count == b.count) {
        return a.value > b.value;
    }
    return a.count > b.count;
}

std::vector<Ponder_elem> ai_ponder(Board board, bool show_log, thread_id_t thread_id, bool *searching) {
    uint64_t strt = tim();
    uint64_t legal = board.get_legal();
    if (legal == 0) {
        board.pass();
        legal = board.get_legal();
        if (legal == 0) {
            if (show_log) {
                std::cerr << "no ponder needed because of game over" << std::endl;
            }
            std::vector<Ponder_elem> empty_list;
            return empty_list;
        } else {
            std::cerr << "ponder pass found" << std::endl;
        }
    }
    const int canput = pop_count_ull(legal);
    std::vector<Ponder_elem> move_list(canput);
    int idx = 0;
    for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)) {
        calc_flip(&move_list[idx].flip, &board, cell);
        move_list[idx].value = INF;
        move_list[idx].count = 0;
        move_list[idx].depth = 0;
        move_list[idx].mpc_level = MPC_74_LEVEL;
        move_list[idx].is_endgame_search = false;
        move_list[idx].is_complete_search = false;
        ++idx;
    }
    const int max_depth = HW2 - board.n_discs() - 1;
    /*
    for (Ponder_elem &elem: move_list) {
        bool depth_updated = true;
        while (depth_updated && elem.depth < std::min(21, max_depth)) {
            depth_updated = false;
            Board n_board = board.copy();
            n_board.move_board(&elem.flip);
            Search tt_search(&n_board, MPC_74_LEVEL, true, false);
            int l = -SCORE_MAX, u = SCORE_MAX;
            if (transposition_table.get_bounds(&tt_search, n_board.hash(), elem.depth + 1, &l, &u)) {
                if (u - l < 5) {
                    ++elem.depth;
                    depth_updated = true;
                }
            }
        }
    }
    if (show_log) {
        std::cerr << "starting depth" << std::endl;
        for (Ponder_elem &elem: move_list) {
            std::cerr << idx_to_coord(elem.flip.pos) << " " << elem.depth << std::endl;
        }
    }
    */
    int n_searched_all = 0;
    while (*searching) {
        int selected_idx = -1;
        double max_ucb = -INF - 1;
        for (int i = 0; i < canput; ++i) {
            double ucb = -INF;
            if (n_searched_all == 0) { // not searched at all
                ucb = move_list[i].value;
            } else if (move_list[i].count == 0) { // this node is not searched
                ucb = INF;
            } else if (move_list[i].is_complete_search) { // fully searched
                ucb = -INF;
            } else {
                //double depth_weight = (double)std::min(10, move_list[i].depth) / (double)std::min(10, max_depth);
                ucb = move_list[i].value / (double)HW2 + 1.0 * sqrt(log(2.0 * (double)n_searched_all) / (double)move_list[i].count);
            }
            if (ucb > max_ucb) {
                selected_idx = i;
                max_ucb = ucb;
            }
        }
        //std::cerr << std::endl;
        if (selected_idx == -1) {
            if (show_log) {
                std::cerr << "ponder: no move selected n_moves " << canput << std::endl;
            }
            break;
        }
        if (move_list[selected_idx].is_complete_search) {
            if (show_log) {
                std::cerr << "ponder completely searched" << std::endl;
            }
            break;
        }
        Board n_board = board.copy();
        n_board.move_board(&move_list[selected_idx].flip);
        int max_depth = HW2 - n_board.n_discs();
        int new_depth = move_list[selected_idx].depth + 1;
        uint_fast8_t new_mpc_level = move_list[selected_idx].mpc_level;
        if (new_depth > max_depth) {
            new_depth = max_depth;
            ++new_mpc_level;
        } else if (new_depth > max_depth - PONDER_ENDSEARCH_PRESEARCH_OFFSET_TIMELIMIT) {
            new_depth = max_depth;
        }
        bool new_is_end_search = (new_depth == max_depth);
        bool new_is_complete_search = new_is_end_search && new_mpc_level == MPC_100_LEVEL;
        Search search(&n_board, new_mpc_level, true, false);
        search.thread_id = thread_id;
        int v = SCORE_UNDEFINED;
        //if (new_depth < PONDER_START_SELFPLAY_DEPTH || new_is_end_search) {
        v = -nega_scout(&search, -SCORE_MAX, SCORE_MAX, new_depth, false, LEGAL_UNDEFINED, new_is_end_search, searching);
        //}
        if (new_depth >= PONDER_START_SELFPLAY_DEPTH && !new_is_complete_search) { // selfplay
            std::cerr << "ponder selfplay " << idx_to_coord(move_list[selected_idx].flip.pos) << " depth " << new_depth << std::endl;
            std::pair<int, int> random_played_scores = ponder_selfplay(n_board, new_depth, false, true, searching); // no -1 (opponent first)
            if (*searching) {
                if (v == SCORE_UNDEFINED) {
                    v = random_played_scores.second;
                } else {
                    v = ((double)v * 1.1 + (double)random_played_scores.second * 0.9) / 2.0;
                }
            }
        }
        if (*searching) {
            if (move_list[selected_idx].value == INF || new_is_end_search) {
                move_list[selected_idx].value = v;
            } else {
                move_list[selected_idx].value = (0.9 * move_list[selected_idx].value + 1.1 * v) / 2.0;
            }
            move_list[selected_idx].depth = new_depth;
            move_list[selected_idx].mpc_level = new_mpc_level;
            move_list[selected_idx].is_endgame_search = new_is_end_search;
            move_list[selected_idx].is_complete_search = new_is_complete_search;
            ++move_list[selected_idx].count;
            ++n_searched_all;
        }
    }
    std::sort(move_list.begin(), move_list.end(), comp_ponder_elem);
    if (show_log && n_searched_all) {
        std::cerr << "ponder loop " << n_searched_all << " in " << tim() - strt << " ms" << std::endl;
        std::cerr << "ponder board " << board.to_str() << std::endl;
        for (int i = 0; i < canput; ++i) {
            std::cerr << "pd " << idx_to_coord(move_list[i].flip.pos) << " value " << std::fixed << std::setprecision(2) << move_list[i].value;
            std::cerr << " count " << move_list[i].count << " depth " << move_list[i].depth << "@" << SELECTIVITY_PERCENTAGE[move_list[i].mpc_level] << "%";
            if (move_list[i].is_complete_search) {
                std::cerr << " complete";
            } else if (move_list[i].is_endgame_search) {
                std::cerr << " endgame";
            }
            std::cerr << std::endl;
        }
    }
    return move_list;
}

bool comp_get_values_elem(Ponder_elem &a, Ponder_elem &b) {
    if (a.value == b.value) {
        if (a.depth == b.depth) {
            return a.mpc_level > b.mpc_level;
        }
        return a.depth > b.depth;
    }
    return a.value > b.value;
}

std::vector<Ponder_elem> ai_get_values(Board board, bool show_log, uint64_t time_limit, thread_id_t thread_id) {
    uint64_t strt = tim();
    uint64_t legal = board.get_legal();
    if (legal == 0) {
        board.pass();
        legal = board.get_legal();
        if (legal == 0) {
            if (show_log) {
                std::cerr << "get values game overgame over" << std::endl;
            }
            std::vector<Ponder_elem> empty_list;
            return empty_list;
        } else {
            std::cerr << "get values pass found" << std::endl;
        }
    }
    const int canput = pop_count_ull(legal);
    std::vector<Ponder_elem> move_list(canput);
    int idx = 0;
    for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)) {
        calc_flip(&move_list[idx].flip, &board, cell);
        move_list[idx].value = INF;
        move_list[idx].count = 0;
        move_list[idx].depth = 0;
        move_list[idx].mpc_level = MPC_74_LEVEL;
        move_list[idx].is_endgame_search = false;
        move_list[idx].is_complete_search = false;
        ++idx;
    }
    uint64_t tl_per_move = time_limit / canput;
    if (show_log) {
        std::cerr << "get values tl per move " << tl_per_move << std::endl;
    }
    const int max_depth = HW2 - board.n_discs() - 1;
    for (Ponder_elem &elem: move_list) {
        uint64_t elem_strt = tim();
        while (tim() - elem_strt < tl_per_move && !elem.is_complete_search && global_searching) {
            Board n_board = board.copy();
            n_board.move_board(&elem.flip);
            int max_depth = HW2 - n_board.n_discs();
            int new_depth = elem.depth + 1;
            uint_fast8_t new_mpc_level = elem.mpc_level;
            if (new_depth > max_depth) {
                new_depth = max_depth;
                ++new_mpc_level;
            } else if (new_depth > max_depth - PONDER_ENDSEARCH_PRESEARCH_OFFSET_TIMELIMIT) {
                new_depth = max_depth;
            }
            bool new_is_end_search = (new_depth == max_depth);
            bool new_is_complete_search = new_is_end_search && new_mpc_level == MPC_100_LEVEL;
            Search search(&n_board, new_mpc_level, true, false);
            search.thread_id = thread_id;
            bool n_searching = true;
            uint64_t time_limit_this_search = get_this_search_time_limit(tl_per_move, tim() - elem_strt);
            std::future<int> v_future = std::async(std::launch::async, nega_scout, &search, -SCORE_MAX, SCORE_MAX, new_depth, false, LEGAL_UNDEFINED, new_is_end_search, &n_searching);
            if (v_future.wait_for(std::chrono::milliseconds(time_limit_this_search)) == std::future_status::ready) {
                int v = -v_future.get();
                if (global_searching) {
                    if (elem.value == INF || new_is_end_search) {
                        elem.value = v;
                    } else {
                        elem.value = (0.9 * elem.value + 1.1 * v) / 2.0;
                    }
                    elem.depth = new_depth;
                    elem.mpc_level = new_mpc_level;
                    elem.is_endgame_search = new_is_end_search;
                    elem.is_complete_search = new_is_complete_search;
                    ++elem.count;
                }
            } else {
                n_searching = false;
                try {
                    v_future.get();
                } catch (const std::exception &e) {
                }
            }
        }
    }
    std::sort(move_list.begin(), move_list.end(), comp_get_values_elem);
    if (show_log) {
        std::cerr << "ai_get_values searched in " << tim() - strt << " ms" << std::endl;
        std::cerr << "ai_get_values board " << board.to_str() << std::endl;
        for (int i = 0; i < canput; ++i) {
            std::cerr << "gb " << idx_to_coord(move_list[i].flip.pos) << " value " << std::fixed << std::setprecision(2) << move_list[i].value;
            std::cerr << " count " << move_list[i].count << " depth " << move_list[i].depth << "@" << SELECTIVITY_PERCENTAGE[move_list[i].mpc_level] << "%";
            if (move_list[i].is_complete_search) {
                std::cerr << " complete";
            } else if (move_list[i].is_endgame_search) {
                std::cerr << " endgame";
            }
            std::cerr << std::endl;
        }
    }
    return move_list;
}

std::vector<Ponder_elem> ai_align_move_levels(Board board, bool show_log, std::vector<Ponder_elem> move_list, int n_good_moves, uint64_t time_limit, thread_id_t thread_id, int aligned_min_level) {
    uint64_t strt = tim();
    if (show_log) {
        std::cerr << "align levels tl " << time_limit << " n_good_moves " << n_good_moves << std::endl;
    }
    const int max_depth = HW2 - board.n_discs() - 1;
    while (tim() - strt < time_limit) {
        int min_depth = 100;
        for (int i = 0; i < n_good_moves; ++i) {
            min_depth = std::min(min_depth, move_list[i].depth);
        }
        if (min_depth >= aligned_min_level) {
            std::cerr << "min depth >= " << aligned_min_level << std::endl;
            break;
        }
        bool level_aligned = true;
        for (int i = 0; i < n_good_moves; ++i) {
            if (move_list[i].depth != move_list[0].depth || move_list[i].mpc_level != move_list[0].mpc_level) {
                level_aligned = false;
                break;
            }
        }
        if (level_aligned && min_depth >= aligned_min_level) {
            std::cerr << "level aligned & min depth >= " << aligned_min_level << std::endl;
            break;
        }
        int min_depth2 = INF;
        uint_fast8_t min_mpc_level = 100;
        int selected_idx = -1;
        for (int i = 0; i < n_good_moves; ++i) {
            if (move_list[i].depth < min_depth2) {
                min_depth2 = move_list[i].depth;
                min_mpc_level = move_list[i].mpc_level;
                selected_idx = i;
            } else if (move_list[i].depth == min_depth2 && move_list[i].mpc_level < min_mpc_level) {
                min_mpc_level = move_list[i].mpc_level;
                selected_idx = i;
            }
        }
        if (move_list[selected_idx].is_complete_search) {
            if (show_log) {
                std::cerr << "completely searched" << std::endl;
            }
            break;
        }
        Board n_board = board.copy();
        n_board.move_board(&move_list[selected_idx].flip);
        int max_depth = HW2 - n_board.n_discs();
        int new_depth = move_list[selected_idx].depth + 1;
        uint_fast8_t new_mpc_level = move_list[selected_idx].mpc_level;
        if (new_depth > max_depth) {
            new_depth = max_depth;
            ++new_mpc_level;
        } else if (new_depth > max_depth - PONDER_ENDSEARCH_PRESEARCH_OFFSET_TIMELIMIT) {
            new_depth = max_depth;
        }
        bool new_is_end_search = (new_depth == max_depth);
        bool new_is_complete_search = new_is_end_search && new_mpc_level == MPC_100_LEVEL;
        Search search(&n_board, new_mpc_level, true, false);
        search.thread_id = thread_id;
        bool n_searching = true;
        uint64_t time_limit_this_search = get_this_search_time_limit(time_limit, tim() - strt);
        std::future<int> v_future = std::async(std::launch::async, nega_scout, &search, -SCORE_MAX, SCORE_MAX, new_depth, false, LEGAL_UNDEFINED, new_is_end_search, &n_searching);
        if (v_future.wait_for(std::chrono::milliseconds(time_limit_this_search)) == std::future_status::ready) {
            int v = -v_future.get();
            if (global_searching) {
                if (move_list[selected_idx].value == INF || new_is_end_search) {
                    move_list[selected_idx].value = v;
                } else {
                    move_list[selected_idx].value = (0.9 * move_list[selected_idx].value + 1.1 * v) / 2.0;
                }
                move_list[selected_idx].depth = new_depth;
                move_list[selected_idx].mpc_level = new_mpc_level;
                move_list[selected_idx].is_endgame_search = new_is_end_search;
                move_list[selected_idx].is_complete_search = new_is_complete_search;
                ++move_list[selected_idx].count;
            }
        } else {
            n_searching = false;
            try {
                v_future.get();
            } catch (const std::exception &e) {
            }
        }
    }
    std::sort(move_list.begin(), move_list.end(), comp_get_values_elem);
    if (show_log) {
        std::cerr << "ai_align_move_levels searched in " << tim() - strt << " ms" << std::endl;
        std::cerr << "ai_align_move_levels board " << board.to_str() << std::endl;
        for (int i = 0; i < n_good_moves; ++i) {
            std::cerr << "ag " << idx_to_coord(move_list[i].flip.pos) << " value " << std::fixed << std::setprecision(2) << move_list[i].value;
            std::cerr << " count " << move_list[i].count << " depth " << move_list[i].depth << "@" << SELECTIVITY_PERCENTAGE[move_list[i].mpc_level] << "%";
            if (move_list[i].is_complete_search) {
                std::cerr << " complete";
            } else if (move_list[i].is_endgame_search) {
                std::cerr << " endgame";
            }
            std::cerr << std::endl;
        }
    }
    return move_list;
}

std::vector<Ponder_elem> ai_search_moves(Board board, bool show_log, std::vector<Ponder_elem> move_list, int n_good_moves, uint64_t time_limit, thread_id_t thread_id) {
    uint64_t strt = tim();
    if (show_log) {
        std::cerr << "search moves tl " << time_limit << " n_good_moves " << n_good_moves << " out of " << move_list.size() << std::endl;
    }
    const int max_depth = HW2 - board.n_discs() - 1;
    int initial_level = 19;
    std::vector<int> levels;
    for (int i = 0; i < n_good_moves; ++i) {
        levels.emplace_back(initial_level);
    }
    std::vector<bool> is_first_searches;
    for (int i = 0; i < n_good_moves; ++i) {
        is_first_searches.emplace_back(true);
    }
    // selfplay & analyze
    std::vector<Clog_result> clogs;
    std::vector<Board> n_boards;
    Flip flip;
    while (tim() - strt < time_limit) {
        double max_val = -INF;
        int selected_idx = -1;
        for (int i = 0; i < n_good_moves; ++i) {
            //if (!move_list[i].is_complete_search) {
            if (!(move_list[i].is_endgame_search && move_list[i].mpc_level >= MPC_99_LEVEL)) {
                double val = move_list[i].value + myrandom() * AI_TL_ADDITIONAL_SEARCH_THRESHOLD * 2.0;
                if (val > max_val) {
                    max_val = val;
                    selected_idx = i;
                }
            }
        }
        if (selected_idx == -1) {
            if (show_log) {
                std::cerr << "enough searched" << std::endl;
            }
            break;
        }
        int level = levels[selected_idx];
        std::cerr << "move " << idx_to_coord(move_list[selected_idx].flip.pos) << " selfplay lv." << level << " ";
        Board n_board = board.copy();
        n_board.move_board(&move_list[selected_idx].flip);
        bool searching = true;
        // selfplay
        n_boards.clear();
        bool terminated = false;
        while (n_board.check_pass()) {
            if (n_board.n_discs() >= HW2 - 23 && n_boards.size()) { // complete search with last 24 empties in lv.21
                break;
            }
            n_boards.emplace_back(n_board);
            uint64_t elapsed = tim() - strt;
            if (elapsed < time_limit) {
                uint64_t tl_this_search = time_limit - elapsed;
                bool is_mid_search;
                int depth;
                uint_fast8_t mpc_level;
                get_level(level, n_board.n_discs() - 4, &is_mid_search, &depth, &mpc_level);
                {
                    Search search(&n_board, mpc_level, true, false);
                    search.thread_id = thread_id;
                    searching = true;
                    std::future<std::pair<int, int>> sp_future = std::async(std::launch::async, first_nega_scout, &search, -SCORE_MAX, SCORE_MAX, depth, !is_mid_search, clogs, strt, &searching);
                    if (sp_future.wait_for(std::chrono::milliseconds(tl_this_search)) == std::future_status::ready) {
                        int policy = sp_future.get().second;
                        if (is_valid_policy(policy)) {
                            std::cerr << idx_to_coord(policy);
                            calc_flip(&flip, &n_board, policy);
                            n_board.move_board(&flip);
                        } else {
                            std::cerr << " ERR " << policy << " ";
                            break;
                        }
                    } else {
                        searching = false;
                        try {
                            sp_future.get();
                        } catch (const std::exception &e) {
                        }
                        std::cerr << " selfplay " << tim() - strt << " ms" << std::endl;
                        terminated = true;
                        break;
                    }
                }
            } else { // time limit
                break;
            }
        }
        if (terminated) {
            break;
        }
        std::cerr << " selfplay " << tim() - strt << " ms";
        // analyze
        for (int i = n_boards.size() - 1; i >= 0; --i) {
            uint64_t elapsed = tim() - strt;
            if (elapsed < time_limit) {
                uint64_t tl_this_search = time_limit - elapsed;
                int max_depth = HW2 - n_boards[i].n_discs();
                bool level_is_mid_search;
                int level_depth;
                uint_fast8_t level_mpc_level;
                get_level(level, n_boards[i].n_discs() - 4, &level_is_mid_search, &level_depth, &level_mpc_level);
                int new_depth = move_list[selected_idx].depth - i;
                //new_depth = move_list[selected_idx].depth;
                if (!is_first_searches[selected_idx] && move_list[selected_idx].depth < 34) { // limit root depth min(34, move_list[selected_idx].depth)
                    ++new_depth; // increase depth
                }
                new_depth = std::max(new_depth, level_depth);
                // if (i != 0) {
                //     new_depth = std::min(new_depth, std::max(level_depth, 29)); // limit depth for non-root
                // }
                uint_fast8_t new_mpc_level = level_mpc_level;
                if (i == 0) {
                    new_mpc_level = move_list[selected_idx].mpc_level;
                }
                if (i == 0) {
                    if (new_depth > max_depth) {
                        new_depth = max_depth;
                        if (new_mpc_level < MPC_100_LEVEL) {
                            ++new_mpc_level;
                        }
                    }
                } else {
                    if (new_depth >= max_depth) {
                        new_depth = max_depth;
                        new_mpc_level = MPC_74_LEVEL;
                        new_mpc_level = move_list[selected_idx].mpc_level;
                        if (move_list[selected_idx].is_endgame_search) {
                            new_mpc_level = move_list[selected_idx].mpc_level;
                            if (new_mpc_level < MPC_100_LEVEL) {
                                ++new_mpc_level;
                            }
                        }
                    }
                }
                //std::cerr << i << "-" << new_depth << "@" << SELECTIVITY_PERCENTAGE[new_mpc_level] << "% ";
                bool new_is_end_search = (new_depth == max_depth);
                bool new_is_complete_search = new_is_end_search && new_mpc_level == MPC_100_LEVEL;
                Search search(&n_boards[i], new_mpc_level, true, false);
                search.thread_id = thread_id;
                searching = true;
                std::future<int> nega_scout_future = std::async(std::launch::async, nega_scout, &search, -SCORE_MAX, SCORE_MAX, new_depth, false, LEGAL_UNDEFINED, new_is_end_search, &searching);
                if (nega_scout_future.wait_for(std::chrono::milliseconds(tl_this_search)) == std::future_status::ready) {
                    int v = nega_scout_future.get();
                    if (i == 0) {
                        if (new_is_end_search) {
                            move_list[selected_idx].value = -v;
                        } else {
                            move_list[selected_idx].value = (0.9 * move_list[selected_idx].value + 1.1 * -v) / 2.0;
                        }
                        move_list[selected_idx].depth = new_depth;
                        move_list[selected_idx].mpc_level = new_mpc_level;
                        move_list[selected_idx].is_endgame_search = new_is_end_search;
                        move_list[selected_idx].is_complete_search = new_is_complete_search;
                        ++move_list[selected_idx].count;
                        std::cerr << " value " << move_list[selected_idx].value << " raw " << -v << " depth " << new_depth << "@" << SELECTIVITY_PERCENTAGE[new_mpc_level] << "% " << tim() - strt << " ms" << std::endl;
                    }
                } else {
                    searching = false;
                    try {
                        nega_scout_future.get();
                    } catch (const std::exception &e) {
                    }
                    std::cerr << std::endl;
                    break;
                }
            } else { // time limit
                break;
            }
        }
        is_first_searches[selected_idx] = false;
        ++levels[selected_idx];
    }
    std::sort(move_list.begin(), move_list.end(), comp_get_values_elem);
    if (show_log) {
        std::cerr << "ai_search_moves searched in " << tim() - strt << " ms" << std::endl;
        std::cerr << "ai_search_moves board " << board.to_str() << std::endl;
        for (int i = 0; i < n_good_moves; ++i) {
            std::cerr << "sm " << idx_to_coord(move_list[i].flip.pos) << " value " << std::fixed << std::setprecision(2) << move_list[i].value;
            std::cerr << " count " << move_list[i].count << " depth " << move_list[i].depth << "@" << SELECTIVITY_PERCENTAGE[move_list[i].mpc_level] << "%";
            if (move_list[i].is_complete_search) {
                std::cerr << " complete";
            } else if (move_list[i].is_endgame_search) {
                std::cerr << " endgame";
            }
            std::cerr << std::endl;
        }
    }
    return move_list;
}


/*
std::vector<Ponder_elem> ai_search_moves(Board board, bool show_log, std::vector<Ponder_elem> move_list, int n_good_moves, uint64_t time_limit, thread_id_t thread_id) {
    uint64_t strt = tim();
    if (show_log) {
        std::cerr << "search moves tl " << time_limit << std::endl;
    }
    const int max_depth = HW2 - board.n_discs() - 1;
    while (tim() - strt < time_limit) {
        int min_depth = INF;
        uint_fast8_t min_mpc_level = 100;
        int selected_idx = -1;
        for (int i = 0; i < n_good_moves; ++i) {
            if (move_list[i].depth < min_depth) {
                min_depth = move_list[i].depth;
                min_mpc_level = move_list[i].mpc_level;
                selected_idx = i;
            } else if (move_list[i].depth == min_depth && move_list[i].mpc_level < min_mpc_level) {
                min_mpc_level = move_list[i].mpc_level;
                selected_idx = i;
            }
        }
        if (move_list[selected_idx].is_complete_search) {
            if (show_log) {
                std::cerr << "completely searched" << std::endl;
            }
            break;
        }
        Board n_board = board.copy();
        n_board.move_board(&move_list[selected_idx].flip);
        int max_depth = HW2 - n_board.n_discs();
        int new_depth = move_list[selected_idx].depth + 1;
        uint_fast8_t new_mpc_level = move_list[selected_idx].mpc_level;
        if (new_depth > max_depth) {
            new_depth = max_depth;
            ++new_mpc_level;
        } else if (new_depth > max_depth - PONDER_ENDSEARCH_PRESEARCH_OFFSET_TIMELIMIT) {
            new_depth = max_depth;
        }
        bool new_is_end_search = (new_depth == max_depth);
        bool new_is_complete_search = new_is_end_search && new_mpc_level == MPC_100_LEVEL;
        Search search(&n_board, new_mpc_level, true, false);
        search.thread_id = thread_id;
        bool n_searching = true;
        uint64_t time_limit_this_search = get_this_search_time_limit(time_limit, tim() - strt);
        std::future<int> v_future = std::async(std::launch::async, nega_scout, &search, -SCORE_MAX, SCORE_MAX, new_depth, false, LEGAL_UNDEFINED, new_is_end_search, &n_searching);
        if (v_future.wait_for(std::chrono::milliseconds(time_limit_this_search)) == std::future_status::ready) {
            int v = -v_future.get();
            if (global_searching) {
                if (move_list[selected_idx].value == INF || new_is_end_search) {
                    move_list[selected_idx].value = v;
                } else {
                    move_list[selected_idx].value = (0.9 * move_list[selected_idx].value + 1.1 * v) / 2.0;
                }
                move_list[selected_idx].depth = new_depth;
                move_list[selected_idx].mpc_level = new_mpc_level;
                move_list[selected_idx].is_endgame_search = new_is_end_search;
                move_list[selected_idx].is_complete_search = new_is_complete_search;
                ++move_list[selected_idx].count;
            }
        } else {
            n_searching = false;
            v_future.get();
        }
    }
    std::sort(move_list.begin(), move_list.end(), comp_get_values_elem);
    if (show_log) {
        std::cerr << "ai_search_moves searched in " << tim() - strt << " ms" << std::endl;
        std::cerr << "ai_search_moves board " << board.to_str() << std::endl;
        for (int i = 0; i < n_good_moves; ++i) {
            std::cerr << "sm " << idx_to_coord(move_list[i].flip.pos) << " value " << std::fixed << std::setprecision(2) << move_list[i].value;
            std::cerr << " count " << move_list[i].count << " depth " << move_list[i].depth << "@" << SELECTIVITY_PERCENTAGE[move_list[i].mpc_level] << "%";
            if (move_list[i].is_complete_search) {
                std::cerr << " complete";
            } else if (move_list[i].is_endgame_search) {
                std::cerr << " endgame";
            }
            std::cerr << std::endl;
        }
    }
    return move_list;
}
*/