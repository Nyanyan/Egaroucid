#pragma once
#include <iostream>
#include <algorithm>
#include <vector>
#include "setting.hpp"
#include "common.hpp"
#include "board.hpp"
#include "evaluate.hpp"
#include "search.hpp"
#include "transpose_table.hpp"
#include "level.hpp"
#if USE_MULTI_THREAD
    #include "thread_pool.hpp"
#endif

using namespace std;

int nega_alpha(board *b, bool skipped, int depth, int alpha, int beta);
inline bool mpc_higher(board *b, bool skipped, int depth, int beta, double t);
inline bool mpc_lower(board *b, bool skipped, int depth, int alpha, double t);

int nega_alpha_ordering_nomemo(board *b, bool skipped, int depth, int alpha, int beta, bool use_mpc, double mpct_in){
    if (!global_searching)
        return -inf;
    if (depth <= simple_mid_threshold)
        return nega_alpha(b, skipped, depth, alpha, beta);
    search_statistics.nodes_increment();
    #if USE_MID_SC
        if (stability_cut(b, &alpha, &beta))
            return alpha;
    #endif
    #if USE_MID_MPC
        if (mpc_min_depth <= depth && depth <= mpc_max_depth){
            if (mpc_higher(b, skipped, depth, beta, mpct_in))
                return beta;
            if (mpc_lower(b, skipped, depth, alpha, mpct_in))
                return alpha;
        }
    #endif
    unsigned long long legal = b->mobility_ull();
    if (legal == 0){
        if (skipped)
            return end_evaluate(b);
        b->p = 1 - b->p;
        int res = -nega_alpha_ordering_nomemo(b, true, depth, -beta, -alpha, use_mpc, mpct_in);
        b->p = 1 - b->p;
        return res;
    }
    vector<board> nb;
    mobility mob;
    int canput = 0;
    for (const int &cell: vacant_lst){
        if (1 & (legal >> cell)){
            calc_flip(&mob, b, cell);
            nb.emplace_back(b->move_copy(&mob));
            move_ordering(&nb[canput]);
            ++canput;
        }
    }
    if (canput >= 2)
        sort(nb.begin(), nb.end());
    int g, v = -inf;
    for (board &nnb: nb){
        g = -nega_alpha_ordering_nomemo(&nnb, false, depth - 1, -beta, -alpha, use_mpc, mpct_in);
        if (beta <= g)
            return g;
        alpha = max(alpha, g);
        v = max(v, g);
    }
    return v;
}

inline bool mpc_higher(board *b, bool skipped, int depth, int beta, double t){
    int bound = beta + ceil(t * mpcsd[b->phase()][depth - mpc_min_depth]);
    if (bound > hw2)
        bound = hw2; //return false;
    return nega_alpha_ordering_nomemo(b, skipped, mpcd[depth], bound - search_epsilon, bound, true, t) >= bound;
}

inline bool mpc_lower(board *b, bool skipped, int depth, int alpha, double t){
    int bound = alpha - ceil(t * mpcsd[b->phase()][depth - mpc_min_depth]);
    if (bound < -hw2)
        bound = -hw2; //return false;
    return nega_alpha_ordering_nomemo(b, skipped, mpcd[depth], bound, bound + search_epsilon, true, t) <= bound;
}

int nega_alpha(board *b, bool skipped, int depth, int alpha, int beta){
    if (!global_searching)
        return -inf;
    search_statistics.nodes_increment();
    if (depth == 0)
        return mid_evaluate(b);
    #if USE_MID_SC
        if (stability_cut(b, &alpha, &beta))
            return alpha;
    #endif
    int g, v = -inf;
    unsigned long long legal = b->mobility_ull();
    if (legal == 0){
        if (skipped)
            return end_evaluate(b);
        b->p = 1 - b->p;
        int res = -nega_alpha(b, true, depth, -beta, -alpha);
        b->p = 1 - b->p;
        return res;
    }
    mobility mob;
    for (const int &cell: vacant_lst){
        if (1 & (legal >> cell)){
            calc_flip(&mob, b, cell);
            b->move(&mob);
            g = -nega_alpha(b, false, depth - 1, -beta, -alpha);
            b->undo(&mob);
            alpha = max(alpha, g);
            if (beta <= alpha)
                return alpha;
            v = max(v, g);
        }
    }
    return v;
}

int nega_alpha_ordering(board *b, bool skipped, const int depth, int alpha, int beta, bool use_multi_thread, bool use_mpc, double mpct_in){
    if (!global_searching)
        return -inf;
    if (depth <= simple_mid_threshold)
        return nega_alpha(b, skipped, depth, alpha, beta);
    search_statistics.nodes_increment();
    #if USE_MID_SC
        if (stability_cut(b, &alpha, &beta))
            return alpha;
    #endif
    int hash = (int)(b->hash() & search_hash_mask);
    int l, u;
    transpose_table.get_now(b, b->hash() & search_hash_mask, &l, &u);
    #if USE_MID_TC
        if (u == l)
            return u;
        if (l >= beta)
            return l;
        if (alpha >= u)
            return u;
    #endif
    alpha = max(alpha, l);
    beta = min(beta, u);
    #if USE_MID_MPC
        if (mpc_min_depth <= depth && depth <= mpc_max_depth && use_mpc){
            if (mpc_higher(b, skipped, depth, beta, mpct_in))
                return beta;
            if (mpc_lower(b, skipped, depth, alpha, mpct_in))
                return alpha;
        }
    #endif
    unsigned long long legal = b->mobility_ull();
    if (legal == 0){
        if (skipped)
            return end_evaluate(b);
        b->p = 1 - b->p;
        int res = -nega_alpha_ordering(b, true, depth, -beta, -alpha, use_multi_thread, use_mpc, mpct_in);
        b->p = 1 - b->p;
        return res;
    }
    vector<board> nb;
    mobility mob;
    int canput = 0;
    for (const int &cell: vacant_lst){
        if (1 & (legal >> cell)){
            calc_flip(&mob, b, cell);
            nb.emplace_back(b->move_copy(&mob));
            move_ordering(&nb[canput]);
            //move_ordering_eval(&(nb[canput]));
            ++canput;
        }
    }
    if (canput >= 2)
        sort(nb.begin(), nb.end());
    int first_alpha = alpha, g, v = -inf;
    #if USE_MULTI_THREAD
        if (use_multi_thread){
            int i;
            g = -nega_alpha_ordering(&nb[0], false, depth - 1, -beta, -alpha, true, use_mpc, mpct_in);
            alpha = max(alpha, g);
            if (beta <= alpha){
                if (l < alpha)
                    transpose_table.reg(b, hash, alpha, u);
                return alpha;
            }
            v = max(v, g);
            vector<future<int>> future_tasks;
            for (i = 1; i < canput; ++i)
                future_tasks.emplace_back(thread_pool.push(bind(&nega_alpha_ordering, &nb[i], false, depth - 1, -beta, -alpha, false, use_mpc, mpct_in)));
            for (i = 1; i < canput; ++i){
                g = -future_tasks[i - 1].get();
                alpha = max(alpha, g);
                v = max(v, g);
            }
            if (beta <= alpha){
                if (l < alpha)
                    transpose_table.reg(b, hash, alpha, u);
                return alpha;
            }
        } else{
            for (int i = 0; i < canput; ++i){
                g = -nega_alpha_ordering(&nb[i], false, depth - 1, -beta, -alpha, false, use_mpc, mpct_in);
                alpha = max(alpha, g);
                if (beta <= alpha){
                    if (l < g)
                        transpose_table.reg(b, hash, g, u);
                    return alpha;
                }
                v = max(v, g);
            }
        }
    #else
        for (board &nnb: nb){
            g = -nega_alpha_ordering(&nnb, false, depth - 1, -beta, -alpha, false, use_mpc, mpct_in);
            alpha = max(alpha, g);
            if (beta <= alpha){
                if (l < alpha)
                    transpose_table.reg(b, hash, alpha, u);
                return alpha;
            }
            v = max(v, g);
        }
    #endif
    if (v <= first_alpha)
        transpose_table.reg(b, hash, l, v);
    else
        transpose_table.reg(b, hash, v, v);
    return v;
}

int nega_scout_nomemo(board *b, bool skipped, const int depth, int alpha, int beta, bool use_mpc, double mpct_in){
    if (!global_searching)
        return -inf;
    if (depth <= simple_mid_threshold)
        return nega_alpha(b, skipped, depth, alpha, beta);
    search_statistics.nodes_increment();
    #if USE_MID_SC
        if (stability_cut(b, &alpha, &beta))
            return alpha;
    #endif
    #if USE_MID_MPC
        if (mpc_min_depth <= depth && depth <= mpc_max_depth && use_mpc){
            if (mpc_higher(b, skipped, depth, beta, mpct_in))
                return beta;
            if (mpc_lower(b, skipped, depth, alpha, mpct_in))
                return alpha;
        }
    #endif
    unsigned long long legal = b->mobility_ull();
    if (legal == 0){
        if (skipped)
            return end_evaluate(b);
        b->p = 1 - b->p;
        int res = -nega_scout_nomemo(b, true, depth, -beta, -alpha, use_mpc, mpct_in);
        b->p = 1 - b->p;
        return res;
    }
    vector<board> nb;
    mobility mob;
    int canput = 0;
    for (const int &cell: vacant_lst){
        if (1 & (legal >> cell)){
            calc_flip(&mob, b, cell);
            nb.emplace_back(b->move_copy(&mob));
            move_ordering(&nb[canput]);
            //move_ordering_eval(&(nb[canput]));
            ++canput;
        }
    }
    if (canput >= 2)
        sort(nb.begin(), nb.end());
    int g = alpha, v = -inf;
    #if USE_MULTI_THREAD && false
        int i;
        g = -nega_scout_nomemo(&nb[0], false, depth - 1, -beta, -alpha, use_mpc, mpct_in);
        alpha = max(alpha, g);
        if (beta <= alpha)
            return alpha;
        v = max(v, g);
        int first_alpha = alpha;
        vector<future<int>> future_tasks;
        vector<bool> re_search;
        for (i = 1; i < canput; ++i)
            future_tasks.emplace_back(thread_pool.push(bind(&nega_alpha_ordering, &nb[i], false, depth - 1, -alpha - search_epsilon, -alpha, false, use_mpc, mpct_in)));
        for (i = 1; i < canput; ++i){
            g = -future_tasks[i - 1].get();
            alpha = max(alpha, g);
            v = max(v, g);
            re_search.emplace_back(first_alpha < g);
        }
        for (i = 1; i < canput; ++i){
            if (re_search[i - 1]){
                g = -nega_scout_nomemo(&nb[i], false, depth - 1, -beta, -alpha, use_mpc, mpct_in);
                if (beta <= g){
                    if (l < g)
                        transpose_table.reg(b, hash, g, u);
                    return g;
                }
                alpha = max(alpha, g);
                v = max(v, g);
            }
        }
    #else
        for (int i = 0; i < canput; ++i){
            if (i > 0){
                g = -nega_alpha_ordering_nomemo(&nb[i], false, depth - 1, -alpha - search_epsilon, -alpha, use_mpc, mpct_in);
                if (beta < g)
                    return g;
            }
            if (alpha <= g){
                alpha = g;
                g = -nega_scout_nomemo(&nb[i], false, depth - 1, -beta, -alpha, use_mpc, mpct_in);
                alpha = max(alpha, g);
                if (beta <= alpha)
                    return alpha;
                v = max(v, g);
            }
        }
    #endif
    return v;
}

int mtd(board *b, bool skipped, int depth, int l, int u, bool use_mpc, double use_mpct){
    int g, beta;
    g = nega_alpha(b, skipped, 5, l, u);
    while (u - l > 0){
        beta = max(l + search_epsilon, g);
        g = nega_alpha_ordering(b, skipped, depth, beta - search_epsilon, beta, true, use_mpc, use_mpct);
        if (g < beta)
            u = g;
        else
            l = g;
    }
    return g;
    //return nega_scout_nomemo(b, skipped, depth, l, u, use_mpc, use_mpct);
}

inline search_result midsearch(board b, long long strt, int max_depth, bool use_mpc, double use_mpct){
    vector<pair<int, board>> nb;
    mobility mob;
    int i;
    unsigned long long legal = b.mobility_ull();
    for (const int &cell: vacant_lst){
        if (1 & (legal >> cell)){
            calc_flip(&mob, &b, cell);
            nb.emplace_back(make_pair(cell, b.move_copy(&mob)));
        }
    }
    int canput = nb.size();
    cerr << "canput: " << canput << endl;
    int res_depth;
    int policy = -1;
    int tmp_policy;
    int alpha, beta, g, value = -inf, former_value = -inf;
    search_statistics.searched_nodes = 0;
    transpose_table.hash_get = 0;
    transpose_table.hash_reg = 0;
    transpose_table.init_now();
    transpose_table.init_prev();
    for (int depth = min(16, max(0, max_depth - 5)); depth <= min(hw2 - b.n, max_depth); ++depth){
        alpha = -hw2;
        beta = hw2;
        transpose_table.init_now();
        for (i = 0; i < canput; ++i)
            move_ordering(&nb[i].second);
        if (canput >= 2)
            sort(nb.begin(), nb.end(), move_ordering_sort);
        g = -mtd(&nb[0].second, false, depth, -beta, -alpha, use_mpc, use_mpct);
        transpose_table.reg(&nb[0].second, (int)(nb[0].second.hash() & search_hash_mask), -g, -g);
        alpha = max(alpha, g);
        tmp_policy = nb[0].first;
        for (i = 1; i < canput; ++i){
            g = -nega_alpha_ordering(&nb[i].second, false, depth, -alpha - search_epsilon, -alpha, true, use_mpc, use_mpct);
            if (alpha < g){
                g = -mtd(&nb[i].second, false, depth, -beta, -g, use_mpc, use_mpct);
                transpose_table.reg(&nb[i].second, (int)(nb[i].second.hash() & search_hash_mask), -g, -g);
                if (alpha < g){
                    alpha = g;
                    tmp_policy = nb[i].first;
                }
            } else{
                transpose_table.reg(&nb[i].second, (int)(nb[i].second.hash() & search_hash_mask), -inf, -g);
            }
        }
        swap(transpose_table.now, transpose_table.prev);
        if (global_searching){
            policy = tmp_policy;
            if (value != -inf)
                former_value = value;
            else
                former_value = alpha;
            value = alpha;
            res_depth = depth;
            #if STATISTICS_MODE
                cerr << "depth: " << depth << " time: " << tim() - strt << " policy: " << policy << " value: " << alpha << " nodes: " << search_statistics.searched_nodes << " nps: " << (long long)search_statistics.searched_nodes * 1000 / max(1LL, tim() - strt) << " get: " << transpose_table.hash_get << " reg: " << transpose_table.hash_reg << endl;
            #else
                cerr << "depth: " << depth << " time: " << tim() - strt << " policy: " << policy << " value: " << alpha << endl;
            #endif
        } else 
            break;
    }
    search_result res;
    res.policy = policy;
    res.value = (value + former_value) / 2;
    res.depth = res_depth;
    res.nps = search_statistics.searched_nodes * 1000 / max(1LL, tim() - strt);
    return res;
}

inline search_result midsearch_value(board b, long long strt, int max_depth, bool use_mpc, double use_mpct){
    //int value = mtd(&b, false, max_depth, -hw2, hw2, use_mpc, use_mpct);
    //int value = nega_alpha_ordering_nomemo(&b, false, max_depth, -hw2, hw2, use_mpc, use_mpct);
    int value = nega_scout_nomemo(&b, false, max_depth, -hw2, hw2, use_mpc, use_mpct);
    search_result res;
    res.policy = -1;
    res.value = value;
    //cerr << res.value << endl;
    res.depth = max_depth;
    res.nps = 0;
    return res;
}

inline search_result midsearch_value_book(board b, long long strt, int max_depth, bool use_mpc, double use_mpct){
    transpose_table.init_now();
    transpose_table.init_prev();
    int value = mtd(&b, false, max_depth - 1, -hw2, hw2, use_mpc, use_mpct);
    //int value = nega_alpha_ordering_nomemo(&b, false, max_depth, -hw2, hw2, use_mpc, use_mpct);
    swap(transpose_table.now, transpose_table.prev);
    transpose_table.init_now();
    value += mtd(&b, false, max_depth, -hw2, hw2, use_mpc, use_mpct);
    search_result res;
    res.policy = -1;
    res.value = value / 2;
    //cerr << res.value << endl;
    res.depth = max_depth;
    res.nps = 0;
    return res;
}

inline search_result midsearch_value_nomemo(board b, long long strt, int max_depth, bool use_mpc, double use_mpct){
    //int value = nega_alpha_ordering_nomemo(&b, false, max_depth, -hw2, hw2, use_mpc, use_mpct);
    int value = nega_scout_nomemo(&b, false, max_depth, -hw2, hw2, use_mpc, use_mpct);
    search_result res;
    res.policy = -1;
    res.value = value;
    res.depth = max_depth;
    res.nps = 0;
    return res;
}