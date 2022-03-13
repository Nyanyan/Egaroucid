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

inline Search_result tree_search(Board board, int depth, bool use_mpc, double mpct, bool show_log){
    Search search;
    int g, alpha, beta, policy = -1;
    pair<int, int> result;
    depth = min(HW2 - board.n, depth);
    bool is_end_search = (HW2 - board.n == depth);
    search.board = board;
    search.n_nodes = 0ULL;
    uint64_t strt = tim();

    if (is_end_search){
        child_transpose_table.init();

        parent_transpose_table.init();
        if (show_log)
            cerr << "start!" << endl;
        search.mpct = 0.6;
        search.use_mpc = true;
        result = first_nega_scout(&search, -HW2, HW2, depth / 2, false, false);
        g = result.first;
        if (show_log)
            cerr << "presearch d=" << depth / 2 << " t=" << search.mpct << " [-64,64] " << g << " " << idx_to_coord(result.second) << endl;

        if (depth >= 24 && 1.2 < mpct){
            parent_transpose_table.init();
            search.mpct = 1.2;
            //search.mpct = 0.0;
            search.use_mpc = true;
            result = first_nega_scout(&search, -HW2, HW2, depth, false, true);
            g = result.first;
            if (show_log)
                cerr << "presearch d=" << depth << " t=" << search.mpct << " [-64,64] " << g << " " << idx_to_coord(result.second) << endl;

            if (depth >= 26 && 1.7 < mpct){
                parent_transpose_table.init();
                search.mpct = 1.7;
                search.use_mpc = true;
                alpha = max(-HW2, g - 3);
                beta = min(HW2, g + 3);
                result = first_nega_scout(&search, alpha, beta, depth, false, true);
                g = result.first;
                if (show_log)
                    cerr << "presearch d=" << depth << " t=" << search.mpct << " [" << alpha << "," << beta << "] " << g << " " << idx_to_coord(result.second) << endl;

                if (depth >= 28 && 2.3 < mpct){
                    parent_transpose_table.init();
                    search.mpct = 2.3;
                    search.use_mpc = true;
                    alpha = max(-HW2, g - 2);
                    beta = min(HW2, g + 2);
                    result = first_nega_scout(&search, alpha, beta, depth, false, true);
                    g = result.first;
                    if (show_log)
                        cerr << "presearch d=" << depth << " t=" << search.mpct << " [" << alpha << "," << beta << "] " << g << " " << idx_to_coord(result.second) << endl;
                }
            }
        }

        if (show_log)
            cerr << "presearch n_nodes " << search.n_nodes << " nps " << search.n_nodes * 1000 / max(1ULL, tim() - strt) << endl;

        parent_transpose_table.init();
        search.use_mpc = use_mpc;
        search.mpct = mpct;
        if (!use_mpc){
            alpha = -INF;
            beta = -INF;
            while (g <= alpha || beta <= g){
                if (g % 2){
                    alpha = max(-HW2, g - 2);
                    beta = min(HW2, g + 2);
                } else{
                    alpha = max(-HW2, g - 1);
                    beta = min(HW2, g + 1);
                }
                result = first_nega_scout(&search, alpha, beta, depth, false, true);
                g = result.first;
                if (show_log)
                    cerr << "mainsearch d=" << depth << " t=" << search.mpct << " [" << alpha << "," << beta << "] " << g << " " << idx_to_coord(result.second) << endl;
                if (alpha == -HW2 && g == -HW2)
                    break;
                if (beta == HW2 && g == HW2)
                    break;
            }
        } else{
            cerr << "main search with probcut" << endl;
            result = first_nega_scout(&search, -HW2, HW2, depth, false, true);
            g = result.first;
        }
        policy = result.second;
        if (show_log)
            cerr << "depth " << depth << " value " << g << " policy " << idx_to_coord(policy) << " nodes " << search.n_nodes << " time " << (tim() - strt) << " nps " << search.n_nodes * 1000 / max(1ULL, tim() - strt) << endl;
    
    } else{
        child_transpose_table.init();
        search.use_mpc = use_mpc;
        search.mpct = mpct;
        g = -INF;
        if (depth - 1 >= 1){
            parent_transpose_table.init();
            result = first_nega_scout(&search, -HW2, HW2, depth - 1, false, false);
            g = result.first;
            if (show_log)
                cerr << "presearch time " << tim() - strt << " depth " << depth - 1 << " value " << g << " policy " << idx_to_coord(policy) << " nodes " << search.n_nodes << " time " << (tim() - strt) << " nps " << search.n_nodes * 1000 / max(1ULL, tim() - strt) << endl;
        }
        parent_transpose_table.init();
        result = first_nega_scout(&search, -HW2, HW2, depth, false, false);
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
    res.nps = search.n_nodes * 1000 / max(1ULL, tim() - strt);
    res.policy = policy;
    res.value = g;
    return res;
}

inline int tree_search_noid(Board board, int depth, bool use_mpc, double mpct){
    int g;
    Search search;
    pair<int, int> result;
    depth = min(HW2 - board.n, depth);
    bool is_end_search = (HW2 - board.n == depth);
    search.board = board;
    search.n_nodes = 0ULL;
    search.use_mpc = use_mpc;
    search.mpct = mpct;
    g = nega_scout(&search, -HW2, HW2, depth, false, LEGAL_UNDEFINED, is_end_search);
    return g;
}

Search_result ai(Board b, int level, int book_error){
    Search_result res;
    Book_value book_result = book.get_random(&b, book_error);
    if (book_result.policy != -1){
        cerr << "BOOK " << book_result.policy << " " << book_result.value << endl;
        res.policy = book_result.policy;
        res.value = book_result.value;
        res.depth = SEARCH_BOOK;
        res.nps = 0;
    } else if (level == 0){
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
        res = tree_search(b, depth, use_mpc, mpct, true);
    }
    return res;
}

bool ai_hint(Board b, int level, int max_level, int res[], int info[], bool best_moves[], const int pre_searched_values[], uint64_t legal){
    Flip flip;
    Board nb;
    future<int> val_future[HW2];
    int depth;
    bool use_mpc, is_mid_search;
    double mpct;
    get_level(level, b.n - 4, &is_mid_search, &depth, &use_mpc, &mpct);
    if (!is_mid_search && level != max_level && !use_mpc)
        return false;
    if (depth - 1 >= 0){
        parent_transpose_table.init();
        for (int i = 0; i < HW2; ++i){
            if (1 & (legal >> i)){
                calc_flip(&flip, &b, i);
                b.move_copy(&flip, &nb);
                res[i] = book.get(&nb);
                if (res[i] == -INF){
                    val_future[i] = async(launch::async, tree_search_noid, nb, depth - 1, use_mpc, mpct);
                    if (!is_mid_search && !use_mpc)
                        info[i] = SEARCH_FINAL;
                    else
                        info[i] = level;
                } else
                    info[i] = SEARCH_BOOK;
            }
        }
        for (int i = 0; i < HW2; ++i){
            if (1 & (legal >> i)){
                if (res[i] == -INF){
                    res[i] = -val_future[i].get();
                }
            }
        }
    } else{
        for (int i = 0; i < HW2; ++i){
            if (1 & (legal >> i)){
                calc_flip(&flip, &b, i);
                b.move_copy(&flip, &nb);
                res[i] = book.get(&nb);
                if (res[i] == -INF){
                    res[i] = -mid_evaluate(&nb);
                    info[i] = level;
                } else
                    info[i] = SEARCH_BOOK;
            }
        }
    }
    int best_score = -INF;
    for (int i = 0; i < HW2; ++i){
        if (1 & (legal >> i))
            best_score = max(best_score, res[i]);
    }
    for (int i = 0; i < HW2; ++i){
        if (1 & (legal >> i))
            best_moves[i] = (res[i] == best_score);
    }
    return true;
}