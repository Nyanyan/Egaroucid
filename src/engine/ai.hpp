/*
    Egaroucid Project

    @file ai.hpp
        Main algorithm of Egaroucid
    @date 2021-2023
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

#define SEARCH_BOOK -1

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
inline Search_result tree_search(Board board, int depth, uint_fast8_t mpc_level, bool show_log, bool use_multi_thread){
    uint64_t strt;
    Search_result res;
    depth = std::min(HW2 - board.n_discs(), depth);
    bool is_end_search = (HW2 - board.n_discs() == depth);
    std::vector<Clog_result> clogs;
    res.clog_nodes = 0ULL;
    if (!is_end_search || mpc_level != MPC_100_LEVEL){
        strt = tim();
        clogs = first_clog_search(board, &res.clog_nodes);
        res.clog_time = tim() - strt;
        if (show_log){
            std::cerr << "clog search time " << res.clog_time << " nodes " << res.clog_nodes << " nps " << calc_nps(res.clog_nodes, res.clog_time) << std::endl;
            for (int i = 0; i < (int)clogs.size(); ++i){
                std::cerr << i + 1 << "/" << clogs.size() << " " << idx_to_coord(clogs[i].pos) << " value " << clogs[i].val << std::endl;
            }
        }
    }
    Search search;
    int g = SCORE_UNDEFINED, policy = -1;
    std::pair<int, int> result;
    search.init_board(&board);
    search.n_nodes = 0ULL;
    #if USE_SEARCH_STATISTICS
        for (int i = 0; i < HW2; ++i)
            search.n_nodes_discs[i] = 0;
    #endif
    search.use_multi_thread = use_multi_thread;
    search.mpc_level = 0;
    calc_features(&search);
    int search_depth;
    if (is_end_search){
        strt = tim();
        if (show_log)
            std::cerr << "start!" << std::endl;
        if (depth >= 17){
            search_depth = depth;
            search.mpc_level = std::max(0, mpc_level - 4);
            result = first_nega_scout(&search, -SCORE_MAX, SCORE_MAX, SCORE_UNDEFINED, search_depth, true, false, clogs, strt);
            g = result.first;
            if (show_log)
                std::cerr << "presearch depth " << search_depth << "@" << SELECTIVITY_PERCENTAGE[search.mpc_level] << "% value " << g << " policy " << idx_to_coord(result.second) << " nodes " << search.n_nodes << " time " << (tim() - strt) << " nps " << calc_nps(search.n_nodes, tim() - strt) << std::endl;
        }
        if (depth >= 23 && search.mpc_level < std::max(0, mpc_level - 3)){
            search_depth = depth;
            search.mpc_level = std::max(0, mpc_level - 3);
            result = first_nega_scout(&search, -SCORE_MAX, SCORE_MAX, g, search_depth, true, false, clogs, strt);
            g = result.first;
            if (show_log)
                std::cerr << "presearch depth " << search_depth << "@" << SELECTIVITY_PERCENTAGE[search.mpc_level] << "% value " << g << " policy " << idx_to_coord(result.second) << " nodes " << search.n_nodes << " time " << (tim() - strt) << " nps " << calc_nps(search.n_nodes, tim() - strt) << std::endl;
        }
        if (depth >= 27 && search.mpc_level < std::max(0, mpc_level - 2)){
            search_depth = depth;
            search.mpc_level = std::max(0, mpc_level - 2);
            result = first_nega_scout(&search, -SCORE_MAX, SCORE_MAX, g, search_depth, true, false, clogs, strt);
            g = result.first;
            if (show_log)
                std::cerr << "presearch depth " << search_depth << "@" << SELECTIVITY_PERCENTAGE[search.mpc_level] << "% value " << g << " policy " << idx_to_coord(result.second) << " nodes " << search.n_nodes << " time " << (tim() - strt) << " nps " << calc_nps(search.n_nodes, tim() - strt) << std::endl;
        }
        if (depth >= 30 && search.mpc_level < std::max(0, mpc_level - 1)){
            search_depth = depth;
            search.mpc_level = std::max(0, mpc_level - 1);
            result = first_nega_scout(&search, -SCORE_MAX, SCORE_MAX, g, search_depth, true, false, clogs, strt);
            g = result.first;
            if (show_log)
                std::cerr << "presearch depth " << search_depth << "@" << SELECTIVITY_PERCENTAGE[search.mpc_level] << "% value " << g << " policy " << idx_to_coord(result.second) << " nodes " << search.n_nodes << " time " << (tim() - strt) << " nps " << calc_nps(search.n_nodes, tim() - strt) << std::endl;
        }
        search_depth = depth;
        search.mpc_level = mpc_level;
        result = first_nega_scout(&search, -SCORE_MAX, SCORE_MAX, g, search_depth, true, show_log, clogs, strt);
        g = result.first;
        policy = result.second;
        if (show_log)
            std::cerr << "mainsearch depth " << search_depth << "@" << SELECTIVITY_PERCENTAGE[search.mpc_level] << "% value " << g << " policy " << idx_to_coord(policy) << " nodes " << search.n_nodes << " time " << (tim() - strt) << " nps " << calc_nps(search.n_nodes, tim() - strt) << std::endl;
    } else{
        strt = tim();
        result.second = TRANSPOSITION_TABLE_UNDEFINED;
        g = -INF;
        if (depth >= 22){
            search_depth = depth - 6;
            search.mpc_level = std::max(0, mpc_level - 2);
            result = first_nega_scout(&search, -SCORE_MAX, SCORE_MAX, SCORE_UNDEFINED, search_depth, false, false, clogs, strt);
            g = result.first;
            policy = result.second;
            if (show_log)
                std::cerr << "presearch depth " << search_depth << "@" << SELECTIVITY_PERCENTAGE[search.mpc_level] << "% value " << g << " policy " << idx_to_coord(policy) << " nodes " << search.n_nodes << " time " << (tim() - strt) << " nps " << calc_nps(search.n_nodes, tim() - strt) << std::endl;
        }
        if (depth >= 17){
            search_depth = depth - 4;
            search.mpc_level = std::max(0, mpc_level - 2);
            result = first_nega_scout(&search, -SCORE_MAX, SCORE_MAX, SCORE_UNDEFINED, search_depth, false, false, clogs, strt);
            g = result.first;
            policy = result.second;
            if (show_log)
                std::cerr << "presearch depth " << search_depth << "@" << SELECTIVITY_PERCENTAGE[search.mpc_level] << "% value " << g << " policy " << idx_to_coord(policy) << " nodes " << search.n_nodes << " time " << (tim() - strt) << " nps " << calc_nps(search.n_nodes, tim() - strt) << std::endl;
        }
        if (depth - 1 >= 1){
            search_depth = depth - 1;
            search.mpc_level = std::max(0, mpc_level - 2);
            result = first_nega_scout(&search, -SCORE_MAX, SCORE_MAX, SCORE_UNDEFINED, search_depth, false, false, clogs, strt);
            g = result.first;
            policy = result.second;
            if (show_log)
                std::cerr << "presearch depth " << search_depth << "@" << SELECTIVITY_PERCENTAGE[search.mpc_level] << "% value " << g << " policy " << idx_to_coord(policy) << " nodes " << search.n_nodes << " time " << (tim() - strt) << " nps " << calc_nps(search.n_nodes, tim() - strt) << std::endl;
        }
        search_depth = depth;
        search.mpc_level = mpc_level;
        result = first_nega_scout(&search, -SCORE_MAX, SCORE_MAX, SCORE_UNDEFINED, search_depth, false, show_log, clogs, strt);
        if (g == -INF)
            g = result.first;
        else
            g = round((0.8 * g + 1.2 * result.first) / 2.0);
        policy = result.second;
        if (show_log)
            std::cerr << "mainsearch depth " << search_depth << "@" << SELECTIVITY_PERCENTAGE[search.mpc_level] << "% value " << g << " policy " << idx_to_coord(policy) << " nodes " << search.n_nodes << " time " << (tim() - strt) << " nps " << calc_nps(search.n_nodes, tim() - strt) << std::endl;
    }
    res.depth = depth;
    res.nodes = search.n_nodes;
    res.time = tim() - strt;
    res.nps = calc_nps(res.nodes, res.time);
    res.policy = policy;
    res.value = g;
    res.is_end_search = is_end_search;
    res.probability = SELECTIVITY_PERCENTAGE[mpc_level];
    #if USE_SEARCH_STATISTICS
        std::cerr << "statistics:" << std::endl;
        for (int i = 0; i < HW2; ++i)
            std::cerr << search.n_nodes_discs[i] << " ";
        std::cerr << std::endl;
    #endif
    return res;
}

/*
    @brief Get a result of a search with no pre-search

    No clog search needed because this function is used for iterative deepning.
    This function is used for hint calculation.

    @param board                board to solve
    @param depth                depth to search
    @param mpc_level            MPC level
    @param show_log             show log?
    @param use_multi_thread     search in multi thread?
    @return the result in Search_result structure
*/
inline Search_result tree_search_iterative_deepening(Board board, int depth, uint_fast8_t mpc_level, bool show_log, bool use_multi_thread){
    Search search;
    int g = 0, policy = -1;
    std::pair<int, int> result;
    depth = std::min(HW2 - board.n_discs(), depth);
    bool is_end_search = (HW2 - board.n_discs() == depth);
    search.init_board(&board);
    search.n_nodes = 0ULL;
    search.mpc_level = mpc_level;
    search.use_multi_thread = use_multi_thread;
    calc_features(&search);
    std::vector<Clog_result> clogs;
    uint64_t strt = tim();
    result = first_nega_scout(&search, -SCORE_MAX, SCORE_MAX, SCORE_UNDEFINED, depth, is_end_search, show_log, clogs, strt);
    g = result.first;
    policy = result.second;
    if (show_log){
        if (is_end_search)
            std::cerr << "depth " << depth << "@" << SELECTIVITY_PERCENTAGE[search.mpc_level] << "% value " << g << " policy " << idx_to_coord(policy) << " nodes " << search.n_nodes << " time " << (tim() - strt) << " nps " << calc_nps(search.n_nodes, tim() - strt) << std::endl;
        else
            std::cerr << "midsearch time " << tim() - strt << " depth " << depth << "@" << SELECTIVITY_PERCENTAGE[search.mpc_level] << "% value " << g << " policy " << idx_to_coord(policy) << " nodes " << search.n_nodes << " time " << (tim() - strt) << " nps " << calc_nps(search.n_nodes, tim() - strt) << std::endl;
    }
    Search_result res;
    res.depth = depth;
    res.nodes = search.n_nodes;
    res.time = tim() - strt;
    res.nps = calc_nps(res.nodes, res.time);
    res.policy = policy;
    res.value = g;
    res.is_end_search = is_end_search;
    res.probability = SELECTIVITY_PERCENTAGE[mpc_level];
    return res;
}

/*
    @brief Get a result of a search with given window with no pre-search

    This function is used for book expanding.

    @param board                board to solve
    @param depth                depth to search
    @param alpha                alpha of the search window
    @param beta                 beta of the search window
    @param mpc_level            MPC probability
    @param use_multi_thread     search in multi thread?
    @return the score of the board
*/
inline int tree_search_window(Board board, int depth, int alpha, int beta, uint_fast8_t mpc_level, bool use_multi_thread){
    Search search;
    depth = std::min(HW2 - board.n_discs(), depth);
    bool is_end_search = (HW2 - board.n_discs() == depth);
    search.init_board(&board);
    search.n_nodes = 0ULL;
    search.mpc_level = mpc_level;
    search.use_multi_thread = use_multi_thread;
    calc_features(&search);
    bool searching = true;
    uint64_t clog_n_nodes = 0ULL;
    int clog_res = clog_search(board, &clog_n_nodes);
    if (clog_res != CLOG_NOT_FOUND)
        return clog_res;
    return nega_scout(&search, alpha, beta, depth, false, LEGAL_UNDEFINED, is_end_search, &searching);
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
Search_result ai(Board board, int level, bool use_book, int book_acc_level, bool use_multi_thread, bool show_log){
    Search_result res;
    int value_sign = 1;
    if (board.get_legal() == 0ULL){
        board.pass();
        if (board.get_legal() == 0ULL){
            res.policy = -1;
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
    Book_value book_result = book.get_random(&board, book_acc_level);
    if (book_result.policy != -1 && use_book){
        if (show_log)
            std::cerr << "book " << idx_to_coord(book_result.policy) << " " << book_result.value << std::endl;
        res.policy = book_result.policy;
        res.value = value_sign * book_result.value;
        res.depth = SEARCH_BOOK;
        res.nps = 0;
        res.is_end_search = false;
        res.probability = 100;
    } else if (level == 0){
        uint64_t legal = board.get_legal();
        std::vector<int> move_lst;
        for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal))
            move_lst.emplace_back(cell);
        res.policy = move_lst[myrandrange(0, (int)move_lst.size())];
        res.value = value_sign * mid_evaluate(&board);
        res.depth = 0;
        res.nps = 0;
        res.is_end_search = false;
        res.probability = 0;
    } else{
        int depth;
        bool is_mid_search;
        uint_fast8_t mpc_level;
        get_level(level, board.n_discs() - 4, &is_mid_search, &depth, &mpc_level);
        if (show_log)
            std::cerr << "level status " << level << " " << board.n_discs() - 4 << " discs depth " << depth << "@" << SELECTIVITY_PERCENTAGE[mpc_level] << "%" << std::endl;
        res = tree_search(board, depth, mpc_level, show_log, use_multi_thread);
        res.value *= value_sign;
    }
    return res;
}

/*
    @brief Get a result of a search with book or search

    This function is used for hint calculation.

    @param board                board to solve
    @param level                level of AI
    @param use_book             use book?
    @param use_multi_thread     search in multi thread?
    @param show_log             show log?
    @return the result in Search_result structure
*/
Search_result ai_hint(Board board, int level, bool use_book, bool use_multi_thread, bool show_log){
    Search_result res;
    int value_sign = 1;
    if (board.get_legal() == 0ULL){
        board.pass();
        if (board.get_legal() == 0ULL){
            res.policy = -1;
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
    int book_result = book.get(&board);
    if (book_result != -INF && use_book){
        if (show_log)
            std::cerr << "book " << idx_to_coord(book_result) << std::endl;
        res.policy = -1;
        res.value = -value_sign * book_result;
        res.depth = SEARCH_BOOK;
        res.nps = 0;
        res.is_end_search = false;
        res.probability = 100;
    } else if (level == 0){
        uint64_t legal = board.get_legal();
        std::vector<int> move_lst;
        for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal))
            move_lst.emplace_back(cell);
        res.policy = move_lst[myrandrange(0, (int)move_lst.size())];
        res.value = value_sign * mid_evaluate(&board);
        res.depth = 0;
        res.nps = 0;
        res.is_end_search = false;
        res.probability = 0;
    } else{
        int depth;
        bool is_mid_search;
        uint_fast8_t mpc_level;
        get_level(level, board.n_discs() - 4, &is_mid_search, &depth, &mpc_level);
        if (show_log)
            std::cerr << "level status " << level << " " << board.n_discs() - 4 << " discs depth " << depth << "@" << SELECTIVITY_PERCENTAGE[mpc_level] << "%" << std::endl;
        res = tree_search_iterative_deepening(board, depth, mpc_level, show_log, use_multi_thread);
        res.value *= value_sign;
    }
    return res;
}

/*
    @brief Get a result of a search with given window with book or search

    This function is used for book expanding.

    @param board                board to solve
    @param level                level of AI
    @param alpha                alpha of the search window
    @param beta                 beta of the search window
    @param use_multi_thread     search in multi thread?
    @return the score of the board
*/
int ai_window(Board board, int level, int alpha, int beta, bool use_multi_thread){
    int value_sign = 1;
    if (board.get_legal() == 0ULL){
        board.pass();
        if (board.get_legal() == 0ULL)
            return -board.score_player();
        else
            value_sign = -1;
    }
    int book_result = book.get(&board);
    if (book_result != -INF)
        return -value_sign * book_result;
    else if (level == 0)
        return value_sign * mid_evaluate(&board);
    int depth;
        bool is_mid_search;
        uint_fast8_t mpc_level;
        get_level(level, board.n_discs() - 4, &is_mid_search, &depth, &mpc_level);
    return value_sign * tree_search_window(board, depth, alpha, beta, mpc_level, use_multi_thread);
}

/*
    @brief Get places where opponent want to put

    @param board                board to solve
    @param res                  array to store result
*/
void ai_opponent_move(Board board, double res[]){
    uint64_t legal = board.get_legal();
    Flip flip;
    int best_move;
    for (int i = 0;  i < HW2; ++i)
        res[i] = 0.0;
    int cnt = 0;
    std::vector<int> best_moves;
    for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)){
        best_move = TRANSPOSITION_TABLE_UNDEFINED;
        calc_flip(&flip, &board, cell);
        board.move_board(&flip);
            best_moves = book.get_all_best_moves(&board);
            if (best_moves.size()){
                for (int cell: best_moves){
                    res[cell] += 1.0;
                    ++cnt;
                }
            } else
                best_move = transposition_table.get_best_move(&board, board.hash());
        board.undo_board(&flip);
        if (best_move != TRANSPOSITION_TABLE_UNDEFINED){
            res[best_move] += 1.0;
            ++cnt;
        }
    }
    for (int i = 0;  i < HW2; ++i)
        res[i] /= cnt;
}

/*
    @brief Search for analyze command

    @param board                board to solve
    @param level                level of AI
    @param use_multi_thread     search in multi thread?
    @return the result in Search_result structure
*/
Analyze_result ai_analyze(Board board, int level, bool use_multi_thread, uint_fast8_t played_move){
    Analyze_result res;
    res.played_move = played_move;
    int got_depth, depth;
    bool is_mid_search;
    uint_fast8_t mpc_level;
    get_level(level, board.n_discs() - 4, &is_mid_search, &got_depth, &mpc_level);
    if (got_depth - 1 >= 0)
        depth = got_depth - 1;
    else
        depth = got_depth;
    Search search;
    search.init_board(&board);
    search.n_nodes = 0ULL;
    search.mpc_level = mpc_level;
    #if USE_SEARCH_STATISTICS
        for (int i = 0; i < HW2; ++i)
            search.n_nodes_discs[i] = 0;
    #endif
    search.use_multi_thread = use_multi_thread;
    calc_features(&search);
    Flip flip;
    calc_flip(&flip, &search.board, played_move);
    eval_move(&search, &flip);
    search.move(&flip);
        res.played_score = book.get(&search.board);
        if (res.played_score != -INF){
            res.played_depth = SEARCH_BOOK;
            res.played_probability = SELECTIVITY_PERCENTAGE[MPC_100_LEVEL];
        } else{
            res.played_score = -first_nega_scout_value(&search, -SCORE_MAX, SCORE_MAX, depth, !is_mid_search, false, false, search.board.get_legal());
            res.played_depth = got_depth;
            res.played_probability = SELECTIVITY_PERCENTAGE[mpc_level];
        }
    search.undo(&flip);
    eval_undo(&search, &flip);
    uint64_t legal = search.board.get_legal() ^ (1ULL << played_move);
    if (legal){
        uint64_t legal_copy = legal;
        bool book_found = false;
        Flip flip;
        int g, v = -INF, best_move = -1;
        for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)){
            calc_flip(&flip, &search.board, cell);
            search.board.move_board(&flip);
                g = book.get(&search.board);
                if (g != -INF){
                    book_found = true;
                    if (v < g){
                        v = g;
                        best_move = flip.pos;
                    }
                }
            search.board.undo_board(&flip);
        }
        if (book_found){
            res.alt_move = best_move;
            res.alt_score = v;
            res.alt_depth = SEARCH_BOOK;
            res.alt_probability = SELECTIVITY_PERCENTAGE[MPC_100_LEVEL];
        } else{
            std::vector<Clog_result> clogs;
            uint64_t strt = tim();
            std::pair<int, int> nega_scout_res = first_nega_scout(&search, -SCORE_MAX, SCORE_MAX, SCORE_UNDEFINED, got_depth, !is_mid_search, false, clogs, legal_copy, strt);
            res.alt_move = nega_scout_res.second;
            res.alt_score = nega_scout_res.first;
            res.alt_depth = got_depth;
            res.alt_probability = SELECTIVITY_PERCENTAGE[mpc_level];
        }
    } else{
        res.alt_move = -1;
        res.alt_score = SCORE_UNDEFINED;
        res.alt_depth = -1;
        res.alt_probability = -1;
    }
    return res;
}