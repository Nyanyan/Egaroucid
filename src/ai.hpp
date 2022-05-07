#pragma once
#include <iostream>
#include <future>
#include <unordered_set>
#include "level.hpp"
#include "setting.hpp"
#if USE_CUDA
    #include "cuda_midsearch.hpp"
#else
    #include "midsearch.hpp"
#endif
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
    search.init();
    int g, alpha, beta, policy = -1;
    pair<int, int> result;
    depth = min(HW2 - board.n, depth);
    bool is_end_search = (HW2 - board.n == depth);
    board.copy(&search.board);
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
        result = first_nega_scout(&search, -SCORE_MAX, SCORE_MAX, depth / 2, false, false);
        g = result.first;
        if (show_log)
            cerr << "presearch d=" << depth / 2 << " t=" << search.mpct << " [-64,64] " << value_to_score_double(g) << " " << idx_to_coord(result.second) << endl;

        //search.p = (search.board.p + depth) % 2;
        if (depth >= 22 && 1.0 < mpct){
            parent_transpose_table.init();
            search.mpct = 1.0;
            //search.mpct = 0.0;
            search.use_mpc = true;
            result = first_nega_scout(&search, -SCORE_MAX, SCORE_MAX, depth, false, true);
            g = result.first;
            if (show_log)
                cerr << "presearch d=" << depth << " t=" << search.mpct << " [-64,64] " << value_to_score_double(g) << " " << idx_to_coord(result.second) << endl;

            if (depth >= 24 && 1.8 < mpct){
                parent_transpose_table.init();
                search.mpct = 1.8;
                search.use_mpc = true;
                alpha = -SCORE_MAX; //max(-SCORE_MAX, score_to_value(value_to_score_double(g) - 3.0));
                beta = SCORE_MAX; //min(SCORE_MAX, score_to_value(value_to_score_double(g) + 3.0));
                result = first_nega_scout(&search, alpha, beta, depth, false, true);
                g = result.first;
                if (show_log)
                    cerr << "presearch d=" << depth << " t=" << search.mpct << " [" << value_to_score_double(alpha) << "," << value_to_score_double(beta) << "] " << value_to_score_double(g) << " " << idx_to_coord(result.second) << endl;
                /*
                if (depth >= 26 && 1.8 < mpct){
                    parent_transpose_table.init();
                    search.mpct = 1.8;
                    search.use_mpc = true;
                    alpha = max(-SCORE_MAX, score_to_value(value_to_score_double(g) - 2.0));
                    beta = min(SCORE_MAX, score_to_value(value_to_score_double(g) + 2.0));
                    result = first_nega_scout(&search, alpha, beta, depth, false, true);
                    g = result.first;
                    if (show_log)
                        cerr << "presearch d=" << depth << " t=" << search.mpct << " [" << value_to_score_double(alpha) << "," << value_to_score_double(beta) << "] " << value_to_score_double(g) << " " << idx_to_coord(result.second) << endl;
                }
                */
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
            while ((g <= alpha || beta <= g) && global_searching){
                #if EVALUATION_STEP_WIDTH_MODE == 1
                    alpha = max(-SCORE_MAX, score_to_value(value_to_score_double(g) - 2.0));
                    beta = min(SCORE_MAX, score_to_value(value_to_score_double(g) + 2.0));
                #else
                    if (value_to_score_int(g) % 2){
                        alpha = max(-SCORE_MAX, score_to_value(value_to_score_double(g) - 2.0));
                        beta = min(SCORE_MAX, score_to_value(value_to_score_double(g) + 2.0));
                    } else{
                        alpha = max(-SCORE_MAX, score_to_value(value_to_score_double(g) - 1.0));
                        beta = min(SCORE_MAX, score_to_value(value_to_score_double(g) + 1.0));
                    }
                #endif
                result = first_nega_scout(&search, alpha, beta, depth, false, true);
                g = result.first;
                //cerr << alpha << " " << g << " " << beta << endl;
                if (show_log)
                    cerr << "mainsearch d=" << depth << " t=" << search.mpct << " [" << value_to_score_double(alpha) << "," << value_to_score_double(beta) << "] " << value_to_score_double(g) << " " << idx_to_coord(result.second) << endl;
                if (alpha == -SCORE_MAX && g == -SCORE_MAX)
                    break;
                if (beta == SCORE_MAX && g == SCORE_MAX)
                    break;
            }
        } else{
            cerr << "main search" << endl;
            result = first_nega_scout(&search, -SCORE_MAX, SCORE_MAX, depth, false, true);
            g = result.first;
        }
        policy = result.second;
        if (show_log)
            cerr << "depth " << depth << " value " << value_to_score_double(g) << " policy " << idx_to_coord(policy) << " nodes " << search.n_nodes << " time " << (tim() - strt) << " nps " << search.n_nodes * 1000 / max(1ULL, tim() - strt) << endl;
    
    } else{
        child_transpose_table.init();
        parent_transpose_table.init();
        strt = tim();

        if (depth >= 15){
            search.use_mpc = true;
            search.mpct = 0.6;
            //search.p = (search.board.p + depth) % 2;
            parent_transpose_table.init();
            result = first_nega_scout(&search, -SCORE_MAX, SCORE_MAX, depth, false, false);
            g = result.first;
            policy = result.second;
            if (show_log)
                cerr << "presearch time " << tim() - strt << " depth " << depth << " value " << value_to_score_double(g) << " policy " << idx_to_coord(policy) << " nodes " << search.n_nodes << " time " << (tim() - strt) << " nps " << search.n_nodes * 1000 / max(1ULL, tim() - strt) << endl;
        }
        search.use_mpc = 1;
        search.mpct = 0.9;
        g = -INF;
        if (depth - 1 >= 1){
            //search.p = (search.board.p + depth - 1) % 2;
            parent_transpose_table.init();
            result = first_nega_scout(&search, -SCORE_MAX, SCORE_MAX, depth - 1, false, false);
            g = result.first;
            policy = result.second;
            if (show_log)
                cerr << "presearch time " << tim() - strt << " depth " << depth - 1 << " value " << value_to_score_double(g) << " policy " << idx_to_coord(policy) << " nodes " << search.n_nodes << " time " << (tim() - strt) << " nps " << search.n_nodes * 1000 / max(1ULL, tim() - strt) << endl;
        }
        search.use_mpc = use_mpc;
        search.mpct = mpct;
        //search.p = (search.board.p + depth) % 2;
        parent_transpose_table.init();
        result = first_nega_scout(&search, -SCORE_MAX, SCORE_MAX, depth, false, false);
        if (g == -INF)
            g = result.first;
        else
            g = (g + result.first) / 2;
        policy = result.second;
        if (show_log)
            cerr << "midsearch time " << tim() - strt << " depth " << depth << " value " << value_to_score_double(g) << " policy " << idx_to_coord(policy) << " nodes " << search.n_nodes << " time " << (tim() - strt) << " nps " << search.n_nodes * 1000 / max(1ULL, tim() - strt) << endl;
    }
    search.del();
    Search_result res;
    res.depth = depth;
    res.nodes = search.n_nodes;
    res.time = tim() - strt;
    res.nps = search.n_nodes * 1000 / max(1ULL, res.time);
    res.policy = policy;
    res.value = value_to_score_int(g);
    return res;
}

inline bool cache_search(Board b, int *val, int *best_move){
    int l, u;
    bak_parent_transpose_table.get(&b, b.hash() & TRANSPOSE_TABLE_MASK, &l, &u);
    cerr << "cache get " << l << " " << u << endl;
    if (l != u)
        return false;
    *val = l;
    *best_move = TRANSPOSE_TABLE_UNDEFINED;
    uint64_t legal = b.get_legal();
    Flip flip;
    for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)){
        calc_flip(&flip, &b, cell);
        b.move(&flip);
            bak_parent_transpose_table.get(&b, b.hash() & TRANSPOSE_TABLE_MASK, &l, &u);
        b.undo(&flip);
        cerr << idx_to_coord(cell) << " " << l << " " << u << endl;
        if (l == u && -l == *val)
            *best_move = cell;
    }
    if (*best_move != TRANSPOSE_TABLE_UNDEFINED)
        cerr << "cache search value " << value_to_score_int(*val) << " policy " << idx_to_coord(*best_move) << endl;
    return *best_move != TRANSPOSE_TABLE_UNDEFINED;
}

Search_result ai(Board b, int level, bool use_book, int book_error){
    Search_result res;
    Book_value book_result = book.get_random(&b, book_error);
    if (book_result.policy != -1 && use_book){
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
        bool cache_hit = false;
        if (!is_mid_search && !use_mpc){
            int val, best_move;
            if (cache_search(b, &val, &best_move)){
                res.depth = depth;
                res.nodes = 0;
                res.nps = 0;
                res.policy = best_move;
                res.value = value_to_score_int(val);
                cache_hit = true;
                cerr << "cache hit depth " << depth << " value " << value_to_score_double(val) << " policy " << idx_to_coord(best_move) << endl;
            }
        }
        if (!cache_hit){
            res = tree_search(b, depth, use_mpc, mpct, true);
            if (!is_mid_search && !use_mpc && depth > END_FAST_DEPTH){
                parent_transpose_table.copy(&bak_parent_transpose_table);
                child_transpose_table.copy(&bak_child_transpose_table);
                cerr << "cache saved" << endl;
            }
        }
    }
    return res;
}
/*
inline double tree_search_noid(Board board, int depth, bool use_mpc, double mpct){
    int g;
    Search search;
    search.init();
    pair<int, int> result;
    depth = min(HW2 - board.n, depth);
    bool is_end_search = (HW2 - board.n == depth);
    board.copy(&search.board);
    search.n_nodes = 0ULL;
    search.use_mpc = use_mpc;
    search.mpct = mpct;
    calc_features(&search);
    bool searching = true;
    g = nega_scout(&search, -SCORE_MAX, SCORE_MAX, depth, false, LEGAL_UNDEFINED, is_end_search, &searching);
    search.del();
    return value_to_score_double(g);
}
*/
bool ai_hint(Board b, int level, int max_level, int res[], int info[], bool best_moves[], const int pre_searched_values[], uint64_t legal){
    Flip flip;
    int depth;
    bool use_mpc, is_mid_search;
    double mpct;
    double value_double, max_value = -INF;
    unordered_set<int> best_moves_set;
    get_level(level, b.n - 4, &is_mid_search, &depth, &use_mpc, &mpct);
    if (!is_mid_search && level != max_level)
        return false;
    Search search;
    search.init();
    search.board = b;
    search.use_mpc = use_mpc;
    search.mpct = mpct;
    search.n_nodes = 0;
    calc_features(&search);
    bool searching = true;
    if (depth - 1 >= 1){
        parent_transpose_table.init();
        for (int i = 0; i < HW2; ++i){
            if (1 & (legal >> i)){
                calc_flip(&flip, &b, i);
                search.board.move(&flip);
                eval_move(&search, &flip);
                    res[i] = book.get(&search.board);
                    if (res[i] == -INF){
                        res[i] = -first_nega_scout(&search, -SCORE_MAX, SCORE_MAX, depth - 1, false, !is_mid_search).first;
                        if (!is_mid_search && !use_mpc)
                            info[i] = SEARCH_FINAL;
                        else
                            info[i] = level;
                    } else
                        info[i] = SEARCH_BOOK;
                eval_undo(&search, &flip);
                search.board.undo(&flip);
                if (max_value < (double)res[i]){
                    max_value = (double)res[i];
                    best_moves_set.clear();
                    best_moves_set.emplace(i);
                } else if (max_value == (double)res[i])
                    best_moves_set.emplace(i);
            }
        }
        /*
        if (!is_mid_search && !use_mpc){
            parent_transpose_table.copy(&bak_parent_transpose_table);
            cerr << "cache saved" << endl;
        }
        */
    } else{
        for (int i = 0; i < HW2; ++i){
            if (1 & (legal >> i)){
                calc_flip(&flip, &b, i);
                search.board.move(&flip);
                eval_move(&search, &flip);
                    res[i] = book.get(&search.board);
                    if (res[i] == -INF){
                        res[i] = value_to_score_int(-mid_evaluate_diff(&search, &searching));
                        info[i] = level;
                    } else
                        info[i] = SEARCH_BOOK;
                eval_undo(&search, &flip);
                search.board.undo(&flip);
                if (max_value < (double)res[i]){
                    max_value = (double)res[i];
                    best_moves_set.clear();
                    best_moves_set.emplace(i);
                } else if (max_value == (double)res[i])
                    best_moves_set.emplace(i);
            }
        }
    }
    for (int i = 0; i < HW2; ++i){
        if (1 & (legal >> i))
            best_moves[i] = (best_moves_set.find(i) != best_moves_set.end());
    }
    search.del();
    return true;
}

int ai_value(Board b, int level){
    int res = book.get(&b);
    if (res != -INF){
        cerr << "BOOK " << res << endl;
        return -res;
    } else if (level == 0){
        res = mid_evaluate(&b);
        cerr << "level 0 " << res << endl;
        return res;
    } else{
        int depth;
        bool use_mpc, is_mid_search;
        double mpct;
        get_level(level, b.n - 4, &is_mid_search, &depth, &use_mpc, &mpct);
        cerr << "level status " << level << " " << b.n - 4 << " " << depth << " " << use_mpc << " " << mpct << endl;
        bool cache_hit = false;
        if (!is_mid_search && !use_mpc){
            int l, u;
            bak_parent_transpose_table.get(&b, b.hash() & TRANSPOSE_TABLE_MASK, &l, &u);
            if (l == u){
                res = l;
                cache_hit = true;
            }
        }
        if (!cache_hit){
            res = tree_search(b, depth, use_mpc, mpct, true).value;
            if (!is_mid_search && !use_mpc){
                parent_transpose_table.copy(&bak_parent_transpose_table);
                cerr << "cache saved" << endl;
            }
        }
    }
    return res;
}
