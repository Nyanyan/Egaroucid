/*
    Egaroucid Project

    @file ai.hpp
        Main algorithm of Egaroucid
    @date 2021-2024
    @author Takuto Yamana
    @license GPL-3.0 license
*/

#pragma once
#include <iostream>
#include <future>
#include <unordered_set>
#include "level.hpp"
#include "setting.hpp"
#include "midsearch.hpp"
#include "book.hpp"
#include "util.hpp"
#include "clogsearch.hpp"
#include "time_management.hpp"

#define SEARCH_BOOK -1

#define AI_TYPE_BOOK 1000

#define IDSEARCH_ENDSEARCH_PRESEARCH_OFFSET 10

struct Lazy_SMP_task {
    uint_fast8_t mpc_level;
    int depth;
    bool is_end_search;
};

void iterative_deepening_search(Board board, int depth, uint_fast8_t mpc_level, bool show_log, std::vector<Clog_result> clogs, uint64_t use_legal, bool use_multi_thread, Search_result *result) {
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
    std::vector<Search> searches(thread_pool.size() + 1);
    while (main_depth <= depth && main_mpc_level <= mpc_level && global_searching) {
        for (Search &search: searches) {
            search.n_nodes = 0;
        }
        bool main_is_end_search = false;
        if (main_depth >= max_depth) {
            main_is_end_search = true;
            main_depth = max_depth;
        }
        bool is_last_search = (main_depth == depth) && (main_mpc_level == mpc_level);
        std::vector<std::future<int>> parallel_tasks;
        std::vector<int> sub_depth_arr;
        int sub_max_mpc_level[61];
        bool sub_searching = true;
        int sub_depth = main_depth;
        if (use_multi_thread && !(is_end_search && main_depth == depth) && main_depth <= 10) {
            int max_thread_size = thread_pool.size();
            for (int i = 0; i < main_depth - 14; ++i) {
                max_thread_size *= 0.9;
            }
            sub_max_mpc_level[main_depth] = main_mpc_level + 1;
            for (int i = main_depth + 1; i < 61; ++i) {
                sub_max_mpc_level[i] = MPC_74_LEVEL;
            }
            for (int sub_thread_idx = 0; sub_thread_idx < max_thread_size && sub_thread_idx < searches.size() && global_searching; ++sub_thread_idx) {
                int ntz = ctz_uint32(sub_thread_idx + 1);
                int sub_depth = std::min(max_depth, main_depth + ntz);
                uint_fast8_t sub_mpc_level = sub_max_mpc_level[sub_depth];
                bool sub_is_end_search = (sub_depth == max_depth);
                if (sub_mpc_level <= MPC_100_LEVEL) {
                    //std::cerr << sub_thread_idx << " " << sub_depth << " " << SELECTIVITY_PERCENTAGE[sub_mpc_level] << std::endl;
                    searches[sub_thread_idx] = Search{&board, sub_mpc_level, false, true, true};
                    bool pushed = false;
                    parallel_tasks.emplace_back(thread_pool.push(&pushed, std::bind(&nega_scout, &searches[sub_thread_idx], -SCORE_MAX, SCORE_MAX, sub_depth, false, LEGAL_UNDEFINED, sub_is_end_search, &sub_searching)));
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
            if (max_is_only_one) {
                for (int i = 0; i < (int)parallel_tasks.size(); ++i) {
                    if (sub_depth_arr[i] == max_sub_search_depth && searches[i].mpc_level == max_sub_main_mpc_level) {
                        searches[i].need_to_see_tt_loop = false; // off the inside-loop tt lookup in the max level thread
                    }
                }
            }
        }
        Search main_search(&board, main_mpc_level, use_multi_thread, parallel_tasks.size() != 0, !is_last_search);
        bool searching = true;
        std::pair<int, int> id_result = first_nega_scout_legal(&main_search, -SCORE_MAX, SCORE_MAX, main_depth, main_is_end_search, clogs, use_legal, strt, &searching);
        sub_searching = false;
        for (std::future<int> &task: parallel_tasks) {
            task.get();
        }
        for (Search &search: searches) {
            result->nodes += search.n_nodes;
        }
        result->nodes += main_search.n_nodes;
        if (result->value != SCORE_UNDEFINED && !main_is_end_search) {
            double n_value = (0.9 * result->value + 1.1 * id_result.first) / 2.0;
            result->value = round(n_value);
        } else{
            result->value = id_result.first;
        }
        result->policy = id_result.second;
        result->depth = main_depth;
        result->time = tim() - strt;
        result->nps = calc_nps(result->nodes, result->time);
        result->is_end_search = main_is_end_search;
        result->probability = SELECTIVITY_PERCENTAGE[main_mpc_level];
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
            std::cerr << "depth " << result->depth << "@" << SELECTIVITY_PERCENTAGE[main_mpc_level] << "%" << " value " << result->value << " (raw " << id_result.first << ") policy " << idx_to_coord(id_result.second) << " n_worker " << parallel_tasks.size() << " n_nodes " << result->nodes << " time " << result->time << " NPS " << result->nps << std::endl;
        }
        if (!is_end_search || main_depth < depth - IDSEARCH_ENDSEARCH_PRESEARCH_OFFSET) {
            if (main_depth <= 15 && main_depth < depth - 3) {
                main_depth += 3;
            } else{
                ++main_depth;
            }
        } else{
            if (main_depth < depth) {
                main_depth = depth;
                if (depth <= 30 && mpc_level >= MPC_88_LEVEL) {
                    main_mpc_level = MPC_88_LEVEL;
                } else{
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
        }
    }
}

void iterative_deepening_search_time_limit(Board board, bool show_log, std::vector<Clog_result> clogs, uint64_t use_legal, bool use_multi_thread, Search_result *result, uint64_t time_limit) {
    uint64_t strt = tim();
    result->value = SCORE_UNDEFINED;
    int main_depth = 1;
    int main_mpc_level = MPC_100_LEVEL;
    const int max_depth = HW2 - board.n_discs();
    if (show_log) {
        std::cerr << "thread pool size " << thread_pool.size() << " n_idle " << thread_pool.get_n_idle() << std::endl;
    }
    std::vector<Search> searches(thread_pool.size() + 1);
    int before_raw_value = -100;
    while (global_searching && tim() - strt < time_limit) {
        for (Search &search: searches) {
            search.n_nodes = 0;
        }
        bool main_is_end_search = false;
        if (main_depth >= max_depth) {
            main_is_end_search = true;
            main_depth = max_depth;
        }
        std::vector<std::future<int>> parallel_tasks;
        std::vector<int> sub_depth_arr;
        int sub_max_mpc_level[61];
        bool sub_searching = true;
        int sub_depth = main_depth;
        if (use_multi_thread && main_depth <= 10) {
            int max_thread_size = thread_pool.size();
            for (int i = 0; i < main_depth - 14; ++i) {
                max_thread_size *= 0.9;
            }
            sub_max_mpc_level[main_depth] = main_mpc_level + 1;
            for (int i = main_depth + 1; i < 61; ++i) {
                sub_max_mpc_level[i] = MPC_74_LEVEL;
            }
            for (int sub_thread_idx = 0; sub_thread_idx < max_thread_size && sub_thread_idx < searches.size() && global_searching; ++sub_thread_idx) {
                int ntz = ctz_uint32(sub_thread_idx + 1);
                int sub_depth = std::min(max_depth, main_depth + ntz);
                uint_fast8_t sub_mpc_level = sub_max_mpc_level[sub_depth];
                bool sub_is_end_search = (sub_depth == max_depth);
                if (sub_mpc_level <= MPC_100_LEVEL) {
                    //std::cerr << sub_thread_idx << " " << sub_depth << " " << SELECTIVITY_PERCENTAGE[sub_mpc_level] << std::endl;
                    searches[sub_thread_idx] = Search{&board, sub_mpc_level, false, true, true};
                    bool pushed = false;
                    parallel_tasks.emplace_back(thread_pool.push(&pushed, std::bind(&nega_scout, &searches[sub_thread_idx], -SCORE_MAX, SCORE_MAX, sub_depth, false, LEGAL_UNDEFINED, sub_is_end_search, &sub_searching)));
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
            if (max_is_only_one) {
                for (int i = 0; i < (int)parallel_tasks.size(); ++i) {
                    if (sub_depth_arr[i] == max_sub_search_depth && searches[i].mpc_level == max_sub_main_mpc_level) {
                        searches[i].need_to_see_tt_loop = false; // off the inside-loop tt lookup in the max level thread
                    }
                }
            }
        }
        Search main_search(&board, main_mpc_level, use_multi_thread, parallel_tasks.size() != 0, false);
        bool searching = true;
        std::pair<int, int> id_result;
        bool search_success = true;
        if (time_limit == TIME_LIMIT_INF) {
            id_result = first_nega_scout_legal(&main_search, -SCORE_MAX, SCORE_MAX, main_depth, main_is_end_search, clogs, use_legal, strt, &searching);
        } else {
            std::future<std::pair<int, int>> f = std::async(std::launch::async, first_nega_scout_legal, &main_search, -SCORE_MAX, SCORE_MAX, main_depth, main_is_end_search, clogs, use_legal, strt, &searching);
            for (;;) {
                if (f.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                    id_result = f.get();
                    break;
                }
                if (tim() - strt >= time_limit) {
                    if (show_log) {
                        std::cerr << "terminate search by time limit " << tim() - strt << " ms" << std::endl;
                    }
                    searching = false;
                    f.get();
                    search_success = false;
                    //std::cerr << "got main " << tim() - strt << " ms" << std::endl;
                    break;
                }
            }
        }
        sub_searching = false;
        for (std::future<int> &task: parallel_tasks) {
            task.get();
        }
        for (Search &search: searches) {
            result->nodes += search.n_nodes;
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
                if (main_is_end_search) {
                    std::cerr << "end ";
                } else{
                    std::cerr << "mid ";
                }
                std::cerr << "depth " << result->depth << "@" << SELECTIVITY_PERCENTAGE[main_mpc_level] << "%" << " value " << result->value << " (raw " << id_result.first << ") policy " << idx_to_coord(id_result.second) << " n_worker " << parallel_tasks.size() << " n_nodes " << result->nodes << " time " << result->time << " NPS " << result->nps << std::endl;
            }
            if (
                !main_is_end_search && 
                main_depth >= 23 && 
                tim() - strt > time_limit * 0.35 && 
                result->nodes >= 100000000ULL && 
                !policy_changed && 
                abs(before_raw_value - id_result.first) <= 0
            ) {
                if (show_log) {
                    std::cerr << "early break" << std::endl;
                }
                break;
            }
            before_raw_value = id_result.first;
        }
        if (main_depth < max_depth - IDSEARCH_ENDSEARCH_PRESEARCH_OFFSET) { // next: midgame search
            if (main_depth <= 15 && main_depth < max_depth - 3) {
                main_depth += 3;
            } else{
                ++main_depth;
            }
            if (main_depth > 13 && main_mpc_level == MPC_100_LEVEL) {
                main_mpc_level = MPC_74_LEVEL;
            }
            if (main_depth > 23) {
                if (main_mpc_level < MPC_88_LEVEL) {
                    --main_depth;
                    ++main_mpc_level;
                } else {
                    main_mpc_level = MPC_74_LEVEL;
                }
            }
        } else{ // next: endgame search
            if (main_depth < max_depth) {
                main_depth = max_depth;
                main_mpc_level = MPC_74_LEVEL;
            } else{
                if (main_mpc_level < MPC_100_LEVEL) {
                    ++main_mpc_level;
                } else{
                    if (show_log) {
                        std::cerr << "all searched" << std::endl;
                    }
                    break;
                }
            }
        }
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
inline Search_result tree_search_legal(Board board, int depth, uint_fast8_t mpc_level, bool show_log, uint64_t use_legal, bool use_multi_thread, uint64_t time_limit) {
    //thread_pool.tell_start_using();
    Search_result res;
    depth = std::min(HW2 - board.n_discs(), depth);
    bool is_end_search = (HW2 - board.n_discs() == depth);
    bool use_time_limit = (time_limit != TIME_LIMIT_INF);
    std::vector<Clog_result> clogs;
    uint64_t clog_nodes = 0;
    uint64_t clog_time = 0;
    if (mpc_level != MPC_100_LEVEL) {
        uint64_t strt = tim();
        int clog_depth = std::min(depth, CLOG_SEARCH_MAX_DEPTH);
        clogs = first_clog_search(board, &clog_nodes, clog_depth, use_legal);
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
            iterative_deepening_search_time_limit(board, show_log, clogs, use_legal, use_multi_thread, &res, time_limit_proc);
        } else {
            iterative_deepening_search(board, depth, mpc_level, show_log, clogs, use_legal, use_multi_thread, &res);
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
    @param use_book             use book?
	@param book_acc_level		book accuracy level
    @param use_multi_thread     search in multi thread?
    @param show_log             show log?
    @return the result in Search_result structure
*/
Search_result ai_common(Board board, int level, bool use_book, int book_acc_level, bool use_multi_thread, bool show_log, uint64_t use_legal, bool use_specified_move_book, uint64_t time_limit) {
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
            value_sign = -1;
        }
    }
    Book_value book_result;
    if (use_specified_move_book) {
        book_result = book.get_specified_best_move(&board);
    } else{
        book_result = book.get_random_specified_moves(&board, book_acc_level, use_legal);
    }
    if (is_valid_policy(book_result.policy) && use_book) {
        if (show_log)
            std::cerr << "book " << idx_to_coord(book_result.policy) << " " << book_result.value << " at book error level " << book_acc_level << std::endl;
        res.level = LEVEL_TYPE_BOOK;
        res.policy = book_result.policy;
        res.value = value_sign * book_result.value;
        res.depth = SEARCH_BOOK;
        res.nps = 0;
        res.is_end_search = false;
        res.probability = 100;
    } else{
        int depth;
        bool is_mid_search;
        uint_fast8_t mpc_level;
        get_level(level, board.n_discs() - 4, &is_mid_search, &depth, &mpc_level);
        if (show_log)
            std::cerr << "level status " << level << " " << board.n_discs() - 4 << " discs depth " << depth << "@" << SELECTIVITY_PERCENTAGE[mpc_level] << "%" << std::endl;
        //thread_pool.tell_start_using();
        res = tree_search_legal(board, depth, mpc_level, show_log, use_legal, use_multi_thread, time_limit);
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
    return ai_common(board, level, use_book, book_acc_level, use_multi_thread, show_log, board.get_legal(), false, TIME_LIMIT_INF);
}

Search_result ai_legal(Board board, int level, bool use_book, int book_acc_level, bool use_multi_thread, bool show_log, uint64_t use_legal) {
    return ai_common(board, level, use_book, book_acc_level, use_multi_thread, show_log, use_legal, false, TIME_LIMIT_INF);
}

Search_result ai_specified(Board board, int level, bool use_book, int book_acc_level, bool use_multi_thread, bool show_log) {
    return ai_common(board, level, use_book, book_acc_level, use_multi_thread, show_log, board.get_legal(), true, TIME_LIMIT_INF);
}

Search_result ai_time_limit(Board board, int level, bool use_book, int book_acc_level, bool use_multi_thread, bool show_log, uint64_t remaining_time_msec) {
    uint64_t time_limit = calc_time_limit_ply(board, remaining_time_msec, show_log);
    if (show_log) {
        std::cerr << "time limit: " << time_limit << std::endl;
    }
    return ai_common(board, level, use_book, book_acc_level, use_multi_thread, show_log, board.get_legal(), false, time_limit);
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
    std::vector<Clog_result> clogs;
    uint64_t clog_nodes = 0;
    uint64_t clog_time = 0;
    int clog_depth = std::min(depth, CLOG_SEARCH_MAX_DEPTH);
    if (mpc_level != MPC_100_LEVEL) {
        uint64_t clog_strt = tim();
        clogs = first_clog_search(board, &clog_nodes, clog_depth, board.get_legal());
        clog_time = tim() - clog_strt;
    }
    Search search(&board, mpc_level, use_multi_thread, false, false);
    uint64_t strt = tim();
    bool searching = true;
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


void ai_ponder(Board board, bool *searching) {
    uint64_t legal = board.get_legal();
    const int canput = pop_count_ull(legal);
    std::vector<Flip_value> move_list(canput);
    std::vector<int> searched_levels(canput);
    std::vector<int> searched_counts(canput);
    int idx = 0;
    for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)) {
        calc_flip(&move_list[idx].flip, &board, cell);
        move_list[idx].value = INF;
        searched_levels[idx] = 0;
        searched_counts[idx] = 0;
        ++idx;
    }
    int n_searched_all = 1;
    while (*searching) {
        int selected_idx = 0;
        double max_ucb = -1000;
        for (int i = 0; i < canput; ++i) {
            double ucb = move_list[i].value + sqrt(log((double)n_searched_all) / 2.0 / (double)searched_counts[i]);
            if (ucb > max_ucb) {
                selected_idx = i;
                max_ucb = ucb;
            }
        }
        Board n_board = board.copy();
        n_board.move_board(&move_list[selected_idx].flip);
        int new_level = std::min(60, searched_levels[selected_idx] + 1);
        int depth;
        bool is_mid_search;
        uint_fast8_t mpc_level;
        get_level(new_level, n_board.n_discs() - 4, &is_mid_search, &depth, &mpc_level);
        depth = std::min(HW2 - board.n_discs(), depth);
        Search search(&n_board, mpc_level, true, false, false);
        int v = nega_scout(&search, -SCORE_MAX, SCORE_MAX, depth, false, LEGAL_UNDEFINED, !is_mid_search, searching);
        if (move_list[selected_idx].value != INF) {
            move_list[selected_idx].value = v;
        } else {
            double n_value = (0.9 * move_list[selected_idx].value + 1.1 * v) / 2.0;
            move_list[selected_idx].value = round(n_value);
        }
        searched_levels[selected_idx] = new_level;
        ++searched_counts[selected_idx];
        for (int i = 0; i < canput; ++i) {
            std::cerr << idx_to_coord(move_list[i].flip.pos) << " " << move_list[i].value << " " << searched_counts[i] << std::endl;
        }
        ++n_searched_all;
    }
}
