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
#include "lazy_smp.hpp"

#define SEARCH_BOOK -1

#ifndef HINT_TYPE_BOOK
#define HINT_TYPE_BOOK 1000
#endif

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
inline Search_result tree_search_legal(Board board, int depth, uint_fast8_t mpc_level, bool show_log, uint64_t use_legal, bool use_multi_thread){
    depth = std::min(HW2 - board.n_discs(), depth);
    bool is_end_search = (HW2 - board.n_discs() == depth);
    std::vector<Clog_result> clogs;
    uint64_t clog_nodes = 0;
    uint64_t clog_time = 0;
    if (mpc_level != MPC_100_LEVEL){
        uint64_t strt = tim();
        clogs = first_clog_search(board, &clog_nodes, std::min(depth, CLOG_SEARCH_MAX_DEPTH));
        clog_time = tim() - strt;
        if (show_log){
            std::cerr << "clog search time " << clog_time << " nodes " << clog_nodes << " nps " << calc_nps(clog_nodes, clog_time) << std::endl;
            for (int i = 0; i < (int)clogs.size(); ++i){
                std::cerr << "clogsearch " << i + 1 << "/" << clogs.size() << " " << idx_to_coord(clogs[i].pos) << " value " << clogs[i].val << std::endl;
            }
        }
    }
    Search_result res;
    res = lazy_smp(board, depth, mpc_level, show_log, clogs, use_legal, use_multi_thread);
    res.clog_nodes = clog_nodes;
    res.clog_time = clog_time;
    thread_pool.reset_unavailable();
    return res;
}

inline void tree_search_hint(Board board, int depth, uint_fast8_t mpc_level, bool use_multi_thread, bool show_log, uint64_t use_legal, int n_display, double values[], int hint_types[]){
    depth = std::min(HW2 - board.n_discs(), depth);
    bool is_end_search = (HW2 - board.n_discs() == depth);
    std::vector<Clog_result> clogs;
    uint64_t clog_nodes = 0;
    uint64_t clog_time = 0;
    if (mpc_level != MPC_100_LEVEL){
        uint64_t strt = tim();
        clogs = first_clog_search(board, &clog_nodes, std::min(depth, CLOG_SEARCH_MAX_DEPTH));
        clog_time = tim() - strt;
        for (Clog_result &clog: clogs){
            if (1 & (use_legal >> clog.pos)){
                values[clog.pos] = clog.val;
                hint_types[clog.pos] = 100;
                use_legal ^= 1ULL << clog.pos;
                --n_display;
            }
        }
        if (show_log){
            std::cerr << "clog search time " << clog_time << " nodes " << clog_nodes << " nps " << calc_nps(clog_nodes, clog_time) << std::endl;
            for (int i = 0; i < (int)clogs.size(); ++i){
                std::cerr << "clogsearch " << i + 1 << "/" << clogs.size() << " " << idx_to_coord(clogs[i].pos) << " value " << clogs[i].val << std::endl;
            }
        }
    }
    bool searching = true;
    Search search;
    search.init(&board, mpc_level, use_multi_thread, false);
    if (n_display < 0){
        return;
    }
    lazy_smp_hint(board, depth, mpc_level, show_log, use_legal, use_multi_thread, n_display, values, hint_types);
    thread_pool.reset_unavailable();
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
Search_result ai_legal(Board board, int level, bool use_book, int book_acc_level, bool use_multi_thread, bool show_log, uint64_t use_legal){
    Search_result res;
    int value_sign = 1;
    if (board.get_legal() == 0ULL){
        board.pass();
        if (board.get_legal() == 0ULL){
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
    Book_value book_result = book.get_random_specified_moves(&board, book_acc_level, use_legal);
    if (book_result.policy != -1 && use_book){
        if (show_log)
            std::cerr << "book " << idx_to_coord(book_result.policy) << " " << book_result.value << " at book error level " << book_acc_level << std::endl;
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
        res = tree_search_legal(board, depth, mpc_level, show_log, use_legal, use_multi_thread);
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
Search_result ai(Board board, int level, bool use_book, int book_acc_level, bool use_multi_thread, bool show_log){
    return ai_legal(board, level, use_book, book_acc_level, use_multi_thread, show_log, board.get_legal());
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
    Search_result played_result = ai_legal(board, level, true, 0, true, false, 1ULL << played_move);
    res.played_move = played_move;
    res.played_score = played_result.value;
    res.played_depth = played_result.depth;
    res.played_probability = played_result.probability;

    uint64_t alt_legal = board.get_legal() ^ (1ULL << played_move);
    if (alt_legal){
        Search_result alt_result = ai_legal(board, level, true, 0, true, false, alt_legal);
        res.alt_move = alt_result.policy;
        res.alt_score = alt_result.value;
        res.alt_depth = alt_result.depth;
        res.alt_probability = alt_result.probability;
    } else{
        res.alt_move = -1;
        res.alt_score = -SCORE_INF;
        res.alt_depth = -1;
        res.alt_probability = 0;
    }
    return res;
}



Search_result ai_accept_loss(Board board, int level, int acceptable_loss){
    uint64_t strt = tim();
    Flip flip;
    int v = SCORE_UNDEFINED;
    uint64_t legal = board.get_legal();
    std::vector<std::pair<int, int>> moves;
    for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)){
        calc_flip(&flip, &board, cell);
        board.move_board(&flip);
            int g = -ai(board, level, true, 0, true, false).value;
        board.undo_board(&flip);
        v = std::max(v, g);
        moves.emplace_back(std::make_pair(g, cell));
    }
    std::vector<std::pair<int, int>> acceptable_moves;
    for (std::pair<int, int> move: moves){
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

void ai_hint(Board board, int level, bool use_book, int book_acc_level, bool use_multi_thread, bool show_log, int n_display, double values[], int hint_types[]){
    uint64_t legal = board.get_legal();
    if (use_book){
        std::vector<Book_value> links = book.get_all_moves_with_value(&board);
        for (Book_value &link: links){
            values[link.policy] = link.value;
            hint_types[link.policy] = HINT_TYPE_BOOK;
            legal ^= 1ULL << link.policy;
            --n_display;
        }
    }
    int depth;
    bool is_mid_search;
    uint_fast8_t mpc_level;
    get_level(level, board.n_discs() - 4, &is_mid_search, &depth, &mpc_level);
    if (show_log)
        std::cerr << "level status " << level << " " << board.n_discs() - 4 << " discs depth " << depth << "@" << SELECTIVITY_PERCENTAGE[mpc_level] << "%" << std::endl;
    tree_search_hint(board, depth, mpc_level, use_multi_thread, show_log, legal, n_display, values, hint_types);
}
