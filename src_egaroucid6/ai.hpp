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
    int g, alpha, beta, policy = -1;
    pair<int, int> result;
    depth = min(HW2 - pop_count_ull(board.player | board.opponent), depth);
    bool is_end_search = (HW2 - pop_count_ull(board.player | board.opponent) == depth);
    search.init_board(&board);
    search.n_nodes = 0ULL;
    calc_features(&search);
    uint64_t strt;

    if (is_end_search){
        child_transpose_table.init();
        parent_transpose_table.init();

        strt = tim();

        if (show_log)
            cerr << "start!" << endl;
        search.mpct = 0.6;
        search.use_mpc = true;
        //search.p = (search.board.p + depth / 2) % 2;
        result = first_nega_scout(&search, -SCORE_MAX, SCORE_MAX, depth / 2, false, false, false, TRANSPOSE_TABLE_UNDEFINED);
        g = result.first;
        if (show_log)
            cerr << "presearch d=" << depth / 2 << " t=" << search.mpct << " [-64,64] " << g << " " << idx_to_coord(result.second) << endl;

        //search.p = (search.board.p + depth) % 2;
        if (depth >= 22 && 1.0 < mpct){
            parent_transpose_table.init();
            search.mpct = 1.0;
            //search.mpct = 0.0;
            search.use_mpc = true;
            result = first_nega_scout(&search, -SCORE_MAX, SCORE_MAX, depth, false, true, false, result.second);
            g = result.first;
            if (show_log)
                cerr << "presearch d=" << depth << " t=" << search.mpct << " [-64,64] " << g << " " << idx_to_coord(result.second) << endl;

            if (depth >= 25 && 2.0 < mpct){
                parent_transpose_table.init();
                search.mpct = 2.0;
                search.use_mpc = true;
                alpha = -SCORE_MAX;
                beta = SCORE_MAX;
                result = first_nega_scout(&search, alpha, beta, depth, false, true, false, result.second);
                g = result.first;
                if (show_log)
                    cerr << "presearch d=" << depth << " t=" << search.mpct << " [" << alpha << "," << beta << "] " << g << " " << idx_to_coord(result.second) << endl;
            }
        }

        //if (show_log)
        //    cerr << "presearch n_nodes " << search.n_nodes << " nps " << search.n_nodes * 1000 / max(1ULL, tim() - strt) << endl;

        parent_transpose_table.init();
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
                result = first_nega_scout(&search, alpha, beta, depth, false, true, show_log, result.second);
                g = result.first;
                //cerr << alpha << " " << g << " " << beta << endl;
                if (show_log)
                    cerr << "mainsearch d=" << depth << " t=" << search.mpct << " [" << alpha << "," << beta << "] " << g << " " << idx_to_coord(result.second) << endl;
                if (alpha == -SCORE_MAX && g == -SCORE_MAX)
                    break;
                if (beta == SCORE_MAX && g == SCORE_MAX)
                    break;
            }
        } else{
            if (show_log)
                cerr << "main search" << endl;
            result = first_nega_scout(&search, -SCORE_MAX, SCORE_MAX, depth, false, true, show_log, result.second);
            g = result.first;
        }
        policy = result.second;
        if (show_log)
            cerr << "depth " << depth << " value " << g << " policy " << idx_to_coord(policy) << " nodes " << search.n_nodes << " time " << (tim() - strt) << " nps " << search.n_nodes * 1000 / max(1ULL, tim() - strt) << endl;
    
    } else{
        child_transpose_table.init();
        parent_transpose_table.init();
        strt = tim();
        result.second = TRANSPOSE_TABLE_UNDEFINED;

        if (depth >= 15){
            search.use_mpc = true;
            search.mpct = 0.6;
            //search.p = (search.board.p + depth) % 2;
            parent_transpose_table.init();
            result = first_nega_scout(&search, -SCORE_MAX, SCORE_MAX, depth - 1, false, false, false, TRANSPOSE_TABLE_UNDEFINED);
            g = result.first;
            policy = result.second;
            if (show_log)
                cerr << "presearch time " << tim() - strt << " depth " << depth - 1 << " value " << g << " policy " << idx_to_coord(policy) << " nodes " << search.n_nodes << " time " << (tim() - strt) << " nps " << search.n_nodes * 1000 / max(1ULL, tim() - strt) << endl;
        }
        search.use_mpc = 1;
        search.mpct = 0.9;
        g = -INF;
        if (depth - 1 >= 1){
            //search.p = (search.board.p + depth - 1) % 2;
            parent_transpose_table.init();
            result = first_nega_scout(&search, -SCORE_MAX, SCORE_MAX, depth - 1, false, false, false, result.second);
            g = result.first;
            policy = result.second;
            if (show_log)
                cerr << "presearch time " << tim() - strt << " depth " << depth - 1 << " value " << g << " policy " << idx_to_coord(policy) << " nodes " << search.n_nodes << " time " << (tim() - strt) << " nps " << search.n_nodes * 1000 / max(1ULL, tim() - strt) << endl;
        }
        search.use_mpc = use_mpc;
        search.mpct = mpct;
        //search.p = (search.board.p + depth) % 2;
        parent_transpose_table.init();
        result = first_nega_scout(&search, -SCORE_MAX, SCORE_MAX, depth, false, false, show_log, result.second);
        if (g == -INF)
            g = result.first;
        else
            g = (g + result.first) / 2;
        policy = result.second;
        if (show_log)
            cerr << "midsearch time " << tim() - strt << " depth " << depth << " value " << g << " policy " << idx_to_coord(policy) << " nodes " << search.n_nodes << " time " << (tim() - strt) << " nps " << search.n_nodes * 1000 / max(1ULL, tim() - strt) << endl;
    }
    Search_result res;
    res.depth = depth;
    res.nodes = search.n_nodes;
    res.time = tim() - strt;
    res.nps = search.n_nodes * 1000 / max(1ULL, res.time);
    res.policy = policy;
    res.value = g;
    return res;
}

inline bool cache_search(Board board, int *val, int *best_move){
    int l, u;
    bak_parent_transpose_table.get(&board, board.hash() & TRANSPOSE_TABLE_MASK, &l, &u);
    int best_move_child_tt = bak_child_transpose_table.get(&board, board.hash() & TRANSPOSE_TABLE_MASK);
    cerr << "cache get " << l << " " << u << endl;
    if (l != u || best_move_child_tt == TRANSPOSE_TABLE_UNDEFINED)
        return false;
    *val = l;
    *best_move = TRANSPOSE_TABLE_UNDEFINED;
    uint64_t legal = board.get_legal();
    Flip flip;
    for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)){
        calc_flip(&flip, &board, cell);
        board.move(&flip);
            bak_parent_transpose_table.get(&board, board.hash() & TRANSPOSE_TABLE_MASK, &l, &u);
        board.undo(&flip);
        cerr << idx_to_coord(cell) << " " << l << " " << u << endl;
        if (l == u && -l == *val && cell == best_move_child_tt)
            *best_move = cell;
    }
    if (*best_move != TRANSPOSE_TABLE_UNDEFINED)
        cerr << "cache search value " << *val << " policy " << idx_to_coord(*best_move) << endl;
    return *best_move != TRANSPOSE_TABLE_UNDEFINED;
}

double val_to_prob(int val, int error_level, int min_val, int max_val){
    double dval = (double)(val - min_val + 1) / (max_val - min_val + 1);
    return exp((26.0 - error_level) * dval);
}

int prob_to_val(double val, int error_level, int min_val, int max_val){
    return round(log(val) / (26.0 - error_level) * (max_val - min_val + 1) + min_val - 1);
}

Search_result ai(Board board, int level, bool use_book, int error_level, bool show_log){
    Search_result res;
    Book_value book_result = book.get_random(&board, 0);
    if (book_result.policy != -1 && use_book){
        cerr << "BOOK " << book_result.policy << " " << book_result.value << endl;
        res.policy = book_result.policy;
        res.value = book_result.value;
        res.depth = SEARCH_BOOK;
        res.nps = 0;
    } else if (level == 0){
        uint64_t legal = board.get_legal();
        vector<int> move_lst;
        for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal))
            move_lst.emplace_back(cell);
        res.policy = move_lst[myrandrange(0, (int)move_lst.size())];
        res.value = mid_evaluate(&board);
        res.depth = 0;
        res.nps = 0;
    } else{
        int depth;
        bool use_mpc, is_mid_search;
        double mpct;
        get_level(level, pop_count_ull(board.player | board.opponent) - 4, &is_mid_search, &depth, &use_mpc, &mpct);
        if (show_log)
            cerr << "level status " << level << " " << pop_count_ull(board.player | board.opponent) - 4 << " " << depth << " " << use_mpc << " " << mpct << endl;
        bool cache_hit = false;
        if (!is_mid_search && !use_mpc){
            int val, best_move;
            if (cache_search(board, &val, &best_move)){
                res.depth = depth;
                res.nodes = 0;
                res.nps = 0;
                res.policy = best_move;
                res.value = val;
                cache_hit = true;
                if (show_log)
                    cerr << "cache hit depth " << depth << " value " << val << " policy " << idx_to_coord(best_move) << endl;
            }
        }
        if (!cache_hit){
            res = tree_search(board, depth, use_mpc, mpct, show_log);
            if (!is_mid_search && !use_mpc && depth > CACHE_SAVE_EMPTY){
                parent_transpose_table.copy(&bak_parent_transpose_table);
                child_transpose_table.copy(&bak_child_transpose_table);
                if (show_log)
                    cerr << "cache saved" << endl;
            }
        }
    }
    return res;
}
