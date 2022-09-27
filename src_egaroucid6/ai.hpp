#pragma once
#include <iostream>
#include <future>
#include <unordered_set>
#include "level.hpp"
#include "setting.hpp"
#include "midsearch.hpp"
#include "book.hpp"
#include "util.hpp"

#define SEARCH_FINAL 100
#define SEARCH_BOOK -1
#define BOOK_CUT_THRESHOLD_DIV 2
#define USE_DEFAULT_MPC -1.0
#define PRESEARCH_OFFSET 6
#define PARALLEL_SPLIT_DIV 6

inline Search_result tree_search(Board board, int depth, bool use_mpc, double mpct, bool show_log){
    Search search;
    int g = 0, alpha, beta, policy = -1;
    pair<int, int> result;
    depth = min(HW2 - pop_count_ull(board.player | board.opponent), depth);
    bool is_end_search = (HW2 - pop_count_ull(board.player | board.opponent) == depth);
    search.init_board(&board);
    search.n_nodes = 0ULL;
    calc_features(&search);
    uint64_t strt;

    if (is_end_search){
        //child_transpose_table.init();
        //parent_transpose_table.init();

        strt = tim();

        if (show_log)
            cerr << "start!" << endl;
        if (depth >= 14){
            search.first_depth = depth / 2;
            search.mpct = 0.8;
            search.use_mpc = true;
            //search.p = (search.board.p + depth / 2) % 2;
            result = first_nega_scout(&search, -SCORE_MAX, SCORE_MAX, search.first_depth, false, false, false, TRANSPOSE_TABLE_UNDEFINED);
            g = result.first;
            if (show_log)
                cerr << "presearch d=" << depth / 2 << " t=" << search.mpct << " [-64,64] " << g << " " << idx_to_coord(result.second) << endl;
        }
        if (depth >= 23){
            double presearch_mpct;
            if (use_mpc)
                presearch_mpct = mpct - 0.5;
            else
                presearch_mpct = 0.9 + 0.1 * (depth - 20);
            //parent_transpose_table.init();
            search.first_depth = depth;
            search.mpct = presearch_mpct;
            search.use_mpc = true;
            alpha = -SCORE_MAX;
            beta = SCORE_MAX;
            result = first_nega_scout(&search, alpha, beta, search.first_depth, false, true, false, result.second);
            g = result.first;
            if (show_log)
                cerr << "presearch d=" << depth << " t=" << search.mpct << " [" << alpha << "," << beta << "] " << g << " " << idx_to_coord(result.second) << endl;
        }

        //if (show_log)
        //    cerr << "presearch n_nodes " << search.n_nodes << " nps " << search.n_nodes * 1000 / max(1ULL, tim() - strt) << endl;

        //parent_transpose_table.init();
        search.first_depth = depth;
        search.use_mpc = use_mpc;
        search.mpct = mpct;
        if (!use_mpc){
            alpha = -INF;
            beta = -INF;
            while ((g <= alpha || beta <= g) && global_searching){
                if (g % 2){
                    alpha = max(-SCORE_MAX, g - 2);
                    beta = min(SCORE_MAX, g + 2);
                } else{
                    alpha = max(-SCORE_MAX, g - 1);
                    beta = min(SCORE_MAX, g + 1);
                }
                result = first_nega_scout(&search, alpha, beta, search.first_depth, false, true, show_log, result.second);
                g = result.first;
                //cerr << alpha << " " << g << " " << beta << endl;
                if (show_log)
                    cerr << "mainsearch d=" << search.first_depth << " t=" << search.mpct << " [" << alpha << "," << beta << "] " << g << " " << idx_to_coord(result.second) << endl;
                if (alpha == -SCORE_MAX && g == -SCORE_MAX)
                    break;
                if (beta == SCORE_MAX && g == SCORE_MAX)
                    break;
            }
        } else{
            if (show_log)
                cerr << "main search" << endl;
            result = first_nega_scout(&search, -SCORE_MAX, SCORE_MAX, search.first_depth, false, true, show_log, result.second);
            g = result.first;
        }
        policy = result.second;
        if (show_log)
            cerr << "depth " << depth << " value " << g << " policy " << idx_to_coord(policy) << " nodes " << search.n_nodes << " time " << (tim() - strt) << " nps " << search.n_nodes * 1000 / max(1ULL, tim() - strt) << endl;
    
    } else{
        //child_transpose_table.init();
        //parent_transpose_table.init();
        strt = tim();
        result.second = TRANSPOSE_TABLE_UNDEFINED;

        if (depth >= 15){
            search.first_depth = depth - 1;
            search.use_mpc = true;
            search.mpct = 0.6;
            result = first_nega_scout(&search, -SCORE_MAX, SCORE_MAX, search.first_depth, false, false, false, TRANSPOSE_TABLE_UNDEFINED);
            g = result.first;
            policy = result.second;
            if (show_log)
                cerr << "presearch time " << tim() - strt << " depth " << search.first_depth << " value " << g << " policy " << idx_to_coord(policy) << " nodes " << search.n_nodes << " time " << (tim() - strt) << " nps " << search.n_nodes * 1000 / max(1ULL, tim() - strt) << endl;
        }
        search.use_mpc = true;
        search.mpct = 0.9;
        g = -INF;
        if (depth - 1 >= 1){
            search.first_depth = depth - 1;
            result = first_nega_scout(&search, -SCORE_MAX, SCORE_MAX, search.first_depth, false, false, false, result.second);
            g = result.first;
            policy = result.second;
            if (show_log)
                cerr << "presearch time " << tim() - strt << " depth " << search.first_depth << " value " << g << " policy " << idx_to_coord(policy) << " nodes " << search.n_nodes << " time " << (tim() - strt) << " nps " << search.n_nodes * 1000 / max(1ULL, tim() - strt) << endl;
        }
        search.first_depth = depth;
        search.use_mpc = use_mpc;
        search.mpct = mpct;
        result = first_nega_scout(&search, -SCORE_MAX, SCORE_MAX, search.first_depth, false, false, show_log, result.second);
        if (g == -INF)
            g = result.first;
        else
            g = round((0.8 * g + 1.2 * result.first) / 2.0);
        policy = result.second;
        if (show_log)
            cerr << "midsearch time " << tim() - strt << " depth " << search.first_depth << " value " << g << " policy " << idx_to_coord(policy) << " nodes " << search.n_nodes << " time " << (tim() - strt) << " nps " << search.n_nodes * 1000 / max(1ULL, tim() - strt) << endl;
    }
    Search_result res;
    res.depth = depth;
    res.nodes = search.n_nodes;
    res.time = tim() - strt;
    res.nps = search.n_nodes * 1000 / max(1ULL, res.time);
    res.policy = policy;
    res.value = g;
    res.is_end_search = is_end_search;
    res.probability = calc_probability(mpct);
    return res;
}

inline Search_result tree_search_iterative_deepening(Board board, int depth, bool use_mpc, double mpct, bool show_log){
    Search search;
    int g = 0, alpha, beta, policy = -1;
    pair<int, int> result;
    depth = min(HW2 - pop_count_ull(board.player | board.opponent), depth);
    bool is_end_search = (HW2 - pop_count_ull(board.player | board.opponent) == depth);
    search.init_board(&board);
    search.n_nodes = 0ULL;
    search.use_mpc = use_mpc;
    //search.mpct = max(0.6, mpct - 0.2);
    search.mpct = mpct;
    calc_features(&search);
    //first_nega_scout(&search, -SCORE_MAX, SCORE_MAX, depth, false, is_end_search, show_log, result.second);
    //search.mpct = mpct;
    uint64_t strt = tim();
    result = first_nega_scout(&search, -SCORE_MAX, SCORE_MAX, depth, false, is_end_search, show_log, result.second);
    g = result.first;
    policy = result.second;
    if (show_log){
        if (is_end_search)
            cerr << "depth " << depth << " value " << g << " policy " << idx_to_coord(policy) << " nodes " << search.n_nodes << " time " << (tim() - strt) << " nps " << search.n_nodes * 1000 / max(1ULL, tim() - strt) << endl;
        else
            cerr << "midsearch time " << tim() - strt << " depth " << depth << " value " << g << " policy " << idx_to_coord(policy) << " nodes " << search.n_nodes << " time " << (tim() - strt) << " nps " << search.n_nodes * 1000 / max(1ULL, tim() - strt) << endl;
    }
    Search_result res;
    res.depth = depth;
    res.nodes = search.n_nodes;
    res.time = tim() - strt;
    res.nps = search.n_nodes * 1000 / max(1ULL, res.time);
    res.policy = policy;
    res.value = g;
    res.is_end_search = is_end_search;
    res.probability = calc_probability(mpct);
    return res;
}

Search_result ai(Board board, int level, bool use_book, bool show_log){
    Search_result res;
    Book_value book_result = book.get_random(&board, 0);
    if (book_result.policy != -1 && use_book){
        if (show_log)
            cerr << "BOOK " << book_result.policy << " " << book_result.value << endl;
        res.policy = book_result.policy;
        res.value = book_result.value;
        res.depth = SEARCH_BOOK;
        res.nps = 0;
        res.is_end_search = false;
        res.probability = 100;
    } else if (level == 0){
        uint64_t legal = board.get_legal();
        vector<int> move_lst;
        for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal))
            move_lst.emplace_back(cell);
        res.policy = move_lst[myrandrange(0, (int)move_lst.size())];
        res.value = mid_evaluate(&board);
        res.depth = 0;
        res.nps = 0;
        res.is_end_search = false;
        res.probability = 0;
    } else{
        int depth;
        bool use_mpc, is_mid_search;
        double mpct;
        get_level(level, pop_count_ull(board.player | board.opponent) - 4, &is_mid_search, &depth, &use_mpc, &mpct);
        if (show_log)
            cerr << "level status " << level << " " << pop_count_ull(board.player | board.opponent) - 4 << " " << depth << " " << use_mpc << " " << mpct << endl;
        res = tree_search(board, depth, use_mpc, mpct, show_log);
    }
    return res;
}

Search_result ai_hint(Board board, int level, bool use_book, bool show_log){
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
            cerr << "BOOK " << book_result << endl;
        res.policy = -1;
        res.value = -value_sign * book_result;
        res.depth = SEARCH_BOOK;
        res.nps = 0;
        res.is_end_search = false;
        res.probability = 100;
    } else if (level == 0){
        uint64_t legal = board.get_legal();
        vector<int> move_lst;
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
        bool use_mpc, is_mid_search;
        double mpct;
        get_level(level, pop_count_ull(board.player | board.opponent) - 4, &is_mid_search, &depth, &use_mpc, &mpct);
        if (show_log)
            cerr << "level status " << level << " " << pop_count_ull(board.player | board.opponent) - 4 << " " << depth << " " << use_mpc << " " << mpct << endl;
        res = tree_search_iterative_deepening(board, depth, use_mpc, mpct, show_log);
        res.value *= value_sign;
    }
    return res;
}