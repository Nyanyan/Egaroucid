#pragma once
#include <iostream>
#include <future>
#include "level.hpp"
#include "midsearch.hpp"
#include "book.hpp"
#include "util.hpp"

#define SEARCH_FINAL 100
#define SEARCH_BOOK -1
#define BOOK_CUT_THRESHOLD_DIV 2
#define USE_DEFAULT_MPC -1.0
#define PRESEARCH_OFFSET 6
#define PARALLEL_SPLIT_DIV 6

inline Search_result tree_search(Board board, int depth, bool use_mpc, double mpct){
    Search search;
    int g, alpha, beta, policy;
    pair<int, int> result;
    depth = min(HW2 - board.n, depth);
    bool is_end_search = (HW2 - board.n == depth);
    search.board = board;
    search.n_nodes = 0ULL;

    child_transpose_table.init();

    g = -INF;

    uint64_t strt = tim(), strt2, search_time = 0ULL;
    if (depth >= 10 && 1.0 < mpct){
        strt2 = tim();
        parent_transpose_table.init();
        search.mpct = 1.0;
        search.use_mpc = true;
        result = first_nega_scout(&search, -HW2, HW2, depth, false, is_end_search);
        g = result.first;
        policy = result.second;
        if (is_end_search)
            g = g / 2 * 2;
        cerr << "presearch t=" << search.mpct << " [-64,64] " << g << " " << idx_to_coord(policy) << endl;
        search_time += tim() - strt2;

        if (depth >= 24 && 1.5 < mpct){
            parent_transpose_table.init();
            strt2 = tim();
            search.mpct = 1.5;
            search.use_mpc = true;
            alpha = max(-HW2, g - 3);
            beta = min(HW2, g + 3);
            result = first_nega_scout(&search, alpha, beta, depth, false, is_end_search);
            g = result.first;
            policy = result.second;
            if (is_end_search)
                g = g / 2 * 2;
            cerr << "presearch t=" << search.mpct << " [" << alpha << "," << beta << "] " << g << " " << idx_to_coord(policy) << endl;
            search_time += tim() - strt2;

            if (depth >= 28 && 1.8 < mpct){
                parent_transpose_table.init();
                strt2 = tim();
                search.mpct = 1.8;
                search.use_mpc = true;
                alpha = max(-HW2, g - 1);
                beta = min(HW2, g + 1);
                result = first_nega_scout(&search, alpha, beta, depth, false, is_end_search);
                g = result.first;
                policy = result.second;
                cerr << "presearch t=" << search.mpct << " [" << alpha << "," << beta << "] " << g << " " << idx_to_coord(policy) << endl;
                search_time += tim() - strt2;
            }
        }
    }

    parent_transpose_table.init();
    strt2 = tim();
    alpha = -INF;
    beta = -INF;
    while (g <= alpha || beta <= g){
        if (g == -INF){
            alpha = -HW2;
            beta = HW2;
        } else{
            alpha = max(-HW2, g - 1);
            beta = min(HW2, g + 1);
        }
        search.use_mpc = use_mpc;
        search.mpct = mpct;
        result = first_nega_scout(&search, alpha, beta, depth, false, is_end_search);
        g = result.first;
        policy = result.second;
        cerr << "[" << alpha << "," << beta << "] " << g << " " << idx_to_coord(policy) << endl;
        if (alpha == -HW2 && g == -HW2)
            break;
        if (beta == HW2 && g == HW2)
            break;
    }
    search_time += tim() - strt2;
    
    cerr << "depth " << depth << " value " << g << " policy " << idx_to_coord(policy) << " nodes " << search.n_nodes << " whole time " << (tim() - strt) << " search time " << search_time << " nps " << search.n_nodes * 1000 / max(1ULL, search_time) << endl;

    Search_result res;
    res.depth = depth;
    res.nodes = search.n_nodes;
    res.nps = search.n_nodes * 1000 / max(1ULL, search_time);
    res.policy = policy;
    res.value = g;
    return res;
}

Search_result ai(Board b, int level, int book_error){
    Search_result res;
    Book_value book_result = book.get_random(&b, book_error);
    if (book_result.policy != -1){
        cerr << "BOOK " << book_result.policy << " " << book_result.value << endl;
        res.policy = book_result.policy;
        res.value = book_result.value;
        res.depth = -1;
        res.nps = 0;
    }
    else if (level == 0){
        uint64_t legal = b.get_legal();
        vector<int> move_lst;
        for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal))
            move_lst.emplace_back(cell);
        res.policy = move_lst[myrandrange(0, (int)move_lst.size())];
        res.value = mid_evaluate(&b);
        res.depth = 0;
        res.nps = 0;
    } else{
        int depth;
        bool use_mpc, is_mid_search;
        double mpct;
        get_level(level, b.n - 4, &is_mid_search, &depth, &use_mpc, &mpct);
        cerr << "level status " << level << " " << b.n - 4 << " " << depth << " " << use_mpc << " " << mpct << endl;
        res = tree_search(b, depth, use_mpc, mpct);
    }
    return res;
}
