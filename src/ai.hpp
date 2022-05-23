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
        result = first_nega_scout(&search, -SCORE_MAX, SCORE_MAX, depth / 2, false, false, false);
        g = result.first;
        if (show_log)
            cerr << "presearch d=" << depth / 2 << " t=" << search.mpct << " [-64,64] " << value_to_score_double(g) << " " << idx_to_coord(result.second) << endl;

        //search.p = (search.board.p + depth) % 2;
        if (depth >= 22 && 1.0 < mpct){
            parent_transpose_table.init();
            search.mpct = 1.0;
            //search.mpct = 0.0;
            search.use_mpc = true;
            result = first_nega_scout(&search, -SCORE_MAX, SCORE_MAX, depth, false, true, false);
            g = result.first;
            if (show_log)
                cerr << "presearch d=" << depth << " t=" << search.mpct << " [-64,64] " << value_to_score_double(g) << " " << idx_to_coord(result.second) << endl;

            if (depth >= 24 && 1.8 < mpct){
                parent_transpose_table.init();
                search.mpct = 1.8;
                search.use_mpc = true;
                alpha = -SCORE_MAX; //max(-SCORE_MAX, score_to_value(value_to_score_double(g) - 3.0));
                beta = SCORE_MAX; //min(SCORE_MAX, score_to_value(value_to_score_double(g) + 3.0));
                result = first_nega_scout(&search, alpha, beta, depth, false, true, false);
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
                result = first_nega_scout(&search, alpha, beta, depth, false, true, true);
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
            result = first_nega_scout(&search, -SCORE_MAX, SCORE_MAX, depth, false, true, true);
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
            result = first_nega_scout(&search, -SCORE_MAX, SCORE_MAX, depth, false, false, false);
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
            result = first_nega_scout(&search, -SCORE_MAX, SCORE_MAX, depth - 1, false, false, false);
            g = result.first;
            policy = result.second;
            if (show_log)
                cerr << "presearch time " << tim() - strt << " depth " << depth - 1 << " value " << value_to_score_double(g) << " policy " << idx_to_coord(policy) << " nodes " << search.n_nodes << " time " << (tim() - strt) << " nps " << search.n_nodes * 1000 / max(1ULL, tim() - strt) << endl;
        }
        search.use_mpc = use_mpc;
        search.mpct = mpct;
        //search.p = (search.board.p + depth) % 2;
        parent_transpose_table.init();
        result = first_nega_scout(&search, -SCORE_MAX, SCORE_MAX, depth, false, false, true);
        if (g == -INF)
            g = result.first;
        else
            g = (g + result.first) / 2;
        policy = result.second;
        if (show_log)
            cerr << "midsearch time " << tim() - strt << " depth " << depth << " value " << value_to_score_double(g) << " policy " << idx_to_coord(policy) << " nodes " << search.n_nodes << " time " << (tim() - strt) << " nps " << search.n_nodes * 1000 / max(1ULL, tim() - strt) << endl;
    }
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

double val_to_prob(int val, int error_level, int min_val, int max_val){
    double dval = (double)(val - min_val + 1) / (max_val - min_val + 1);
    return exp((26.0 - error_level) * dval);
}

int prob_to_val(double val, int error_level, int min_val, int max_val){
    return round(log(val) / (26.0 - error_level) * (max_val - min_val + 1) + min_val - 1);
}

Search_result ai(Board b, int level, bool use_book, int error_level){
    Search_result res;
    if (error_level == 0){
        Book_value book_result = book.get_random(&b, 0);
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
    } else{
        uint64_t legal = b.get_legal();
        int n_legal = pop_count_ull(legal);
        vector<pair<int, double>> probabilities;
        Flip flip;
        int v;
        int depth;
        bool use_mpc, is_mid_search;
        double mpct;
        get_level(level, b.n - 4, &is_mid_search, &depth, &use_mpc, &mpct);
        bool searching = true;
        Search search;
        search.n_nodes = 0;
        search.use_mpc = use_mpc;
        search.mpct = mpct;
        double p_sum = 0.0;
        int min_val = INF;
        int max_val = -INF;
        for (uint8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)){
            calc_flip(&flip, &b, cell);
            b.move(&flip);
                v = book.get(&b);
                if (v == -INF){
                    search.board = b;
                    calc_features(&search);
                    v = -nega_scout(&search, -SCORE_MAX, SCORE_MAX, max(0, depth - 1), false, LEGAL_UNDEFINED, !is_mid_search, &searching);
                }
                //cerr << idx_to_coord((int)cell) << " " << v << endl;
            b.undo(&flip);
            probabilities.emplace_back(make_pair((int)cell, v));
            max_val = max(max_val, v);
            min_val = min(min_val, v);
        }
        for (pair<int, double> &elem: probabilities){
            elem.second = val_to_prob(elem.second, error_level, min_val, max_val);
            p_sum += elem.second;
        }
        double prob = myrandom();
        double p = 0.0;
        for (pair<int, double> &elem: probabilities){
            //cerr << idx_to_coord(elem.first) << " " << elem.second / p_sum << endl;
            p += elem.second / p_sum;
            if (p >= prob){
                res.depth = depth;
                res.nodes = search.n_nodes;
                res.nps = 0;
                res.policy = elem.first;
                res.value = (b.p ? -1 : 1) * value_to_score_int(prob_to_val(elem.second, error_level, min_val, max_val));
                break;
            }
        }
    }
    return res;
}

inline double tree_search_noid(Board board, int depth, bool use_mpc, double mpct, bool use_multi_thread){
    int g;
    Search search;
    pair<int, int> result;
    depth = min(HW2 - board.n, depth);
    bool is_end_search = (HW2 - board.n == depth);
    board.copy(&search.board);
    search.n_nodes = 0ULL;
    search.use_mpc = use_mpc;
    search.mpct = mpct;
    calc_features(&search);
    bool searching = true;
    if (use_multi_thread)
        g = nega_scout(&search, -SCORE_MAX, SCORE_MAX, depth, false, LEGAL_UNDEFINED, is_end_search, &searching);
    else
        g = nega_scout_single_thread(&search, -SCORE_MAX, SCORE_MAX, depth, false, LEGAL_UNDEFINED, is_end_search, &searching);
    return value_to_score_double(g);
}

bool ai_hint(Board b, int level, int max_level, int res[], int info[], bool best_moves[], const int pre_searched_values[], uint64_t legal){
    Flip flip;
    Board nb;
    future<double> val_future[HW2];
    int depth;
    bool use_mpc, is_mid_search;
    double mpct;
    double value_double, max_value = -INF;
    unordered_set<int> best_moves_set;
    get_level(level, b.n - 4, &is_mid_search, &depth, &use_mpc, &mpct);
    if (!is_mid_search && level != max_level)
        return false;
    int n_legal = 0;
    for (int i = 0; i < HW2; ++i){
        if (1 & (legal >> i)){
            calc_flip(&flip, &b, i);
            b.move_copy(&flip, &nb);
            res[i] = book.get(&nb);
            if (res[i] == -INF)
                ++n_legal;
        }
    }
    int thread_size = 0;
    #if USE_MULTI_THREAD
        thread_size = (int)thread_pool.size();
    #endif
    bool use_multi_thread = (n_legal < thread_size);
    if (depth - 1 >= 0){
        //int l, u;
        bool cache_hit;
        parent_transpose_table.init();
        for (int i = 0; i < HW2; ++i){
            if (1 & (legal >> i)){
                calc_flip(&flip, &b, i);
                b.move_copy(&flip, &nb);
                cache_hit = false;
                res[i] = book.get(&nb);
                if (res[i] == -INF){
                    /*
                    if (!is_mid_search && !use_mpc){
                        bak_parent_transpose_table.get(&nb, nb.hash() & TRANSPOSE_TABLE_MASK, &l, &u);
                        if (l == u){
                            res[i] = -value_to_score_int(l);
                            if (max_value < (double)res[i]){
                                max_value = (double)res[i];
                                best_moves_set.clear();
                                best_moves_set.emplace(i);
                            } else if (max_value == (double)res[i])
                                best_moves_set.emplace(i);
                            cache_hit = true;
                        }
                    }
                    */
                    if (!cache_hit){
                        #if USE_MULTI_THREAD
                            val_future[i] = thread_pool.push(bind(&tree_search_noid, nb, depth - 1, use_mpc, mpct, use_multi_thread));
                        #else
                            val_future[i] = async(launch::async, tree_search_noid, nb, depth - 1, use_mpc, mpct, false);
                        #endif
                        //val_future[i] = async(launch::async, tree_search_noid, nb, depth - 1, use_mpc, mpct);
                    }
                    if (!is_mid_search && !use_mpc)
                        info[i] = SEARCH_FINAL;
                    else
                        info[i] = level;
                } else{
                    if (max_value < (double)res[i]){
                        max_value = (double)res[i];
                        best_moves_set.clear();
                        best_moves_set.emplace(i);
                    } else if (max_value == (double)res[i])
                        best_moves_set.emplace(i);
                    info[i] = SEARCH_BOOK;
                }
            }
        }
        for (int i = 0; i < HW2; ++i){
            if (1 & (legal >> i)){
                if (res[i] == -INF){
                    value_double = -val_future[i].get();
                    //cerr << idx_to_coord(i) << " " << value_double << endl;
                    if (max_value < value_double){
                        max_value = value_double;
                        best_moves_set.clear();
                        best_moves_set.emplace(i);
                    } else if (max_value == value_double)
                        best_moves_set.emplace(i);
                    res[i] = round(value_double);
                }
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
                b.move_copy(&flip, &nb);
                res[i] = book.get(&nb);
                if (res[i] == -INF){
                    res[i] = value_to_score_int(-mid_evaluate(&nb));
                    info[i] = level;
                } else
                    info[i] = SEARCH_BOOK;
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