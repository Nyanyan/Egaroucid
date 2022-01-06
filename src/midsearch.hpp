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
#if USE_MULTI_THREAD
    #include "thread_pool.hpp"
#endif

using namespace std;

int nega_alpha(board *b, bool skipped, int depth, int alpha, int beta);
inline bool mpc_higher(board *b, bool skipped, int depth, int beta, double t);
inline bool mpc_lower(board *b, bool skipped, int depth, int alpha, double t);

int nega_alpha_ordering_nomemo(board *b, bool skipped, int depth, int alpha, int beta, double mpct_in){
    if (depth <= simple_mid_threshold)
        return nega_alpha(b, skipped, depth, alpha, beta);
    ++searched_nodes;
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
    vector<board> nb;
    int canput = 0;
    for (const int &cell: vacant_lst){
        if (b->legal(cell)){
            nb.emplace_back(b->move(cell));
            //move_ordering(&(nb[canput]));
            move_ordering_eval(&(nb[canput]));
            ++canput;
        }
    }
    if (canput == 0){
        if (skipped)
            return end_evaluate(b);
        board rb;
        for (int i = 0; i < b_idx_num; ++i)
            rb.b[i] = b->b[i];
        rb.p = 1 - b->p;
        rb.n = b->n;
        return -nega_alpha_ordering_nomemo(&rb, true, depth, -beta, -alpha, mpct_in);
    }
    if (canput >= 2)
        sort(nb.begin(), nb.end());
    int g, v = -inf;
    for (board &nnb: nb){
        g = -nega_alpha_ordering_nomemo(&nnb, false, depth - 1, -beta, -alpha, mpct_in);
        if (beta <= g)
            return g;
        alpha = max(alpha, g);
        v = max(v, g);
    }
    return v;
}

inline bool mpc_higher(board *b, bool skipped, int depth, int beta, double t){
    int bound = beta + ceil(t * mpcsd[calc_phase_idx(b)][depth - mpc_min_depth]);
    if (bound > hw2)
        return false;
    return nega_alpha_ordering_nomemo(b, skipped, mpcd[depth], bound - search_epsilon, bound, t) >= bound;
}

inline bool mpc_lower(board *b, bool skipped, int depth, int alpha, double t){
    int bound = alpha - floor(t * mpcsd[calc_phase_idx(b)][depth - mpc_min_depth]);
    if (bound < -hw2)
        return false;
    return nega_alpha_ordering_nomemo(b, skipped, mpcd[depth], bound, bound + search_epsilon, t) <= bound;
}

int nega_alpha(board *b, bool skipped, int depth, int alpha, int beta){
    ++searched_nodes;
    if (depth == 0){
        if (b->n < hw2)
            return mid_evaluate(b);
        else
            return end_evaluate(b);
    }
    #if USE_MID_SC
        if (stability_cut(b, &alpha, &beta))
            return alpha;
    #endif
    board nb;
    bool passed = true;
    int g, v = -inf;
    #if USE_MID_SMOOTH
        if (depth == 1){
            int nv = mid_evaluate(b);
            for (const int &cell: vacant_lst){
                if (b->legal(cell)){
                    passed = false;
                    b->move(cell, &nb);
                    g = (-nega_alpha(&nb, false, depth - 1, -beta, -alpha) + nv) / 2;
                    if (beta <= g)
                        return g;
                    alpha = max(alpha, g);
                    v = max(v, g);
                }
            }
            if (passed){
                if (skipped)
                    return end_evaluate(b);
                board rb;
                for (int i = 0; i < b_idx_num; ++i)
                    rb.b[i] = b->b[i];
                rb.p = 1 - b->p;
                rb.n = b->n;
                return -nega_alpha(&rb, true, depth, -beta, -alpha);
            }
            return v;
        }
    #endif
    for (const int &cell: vacant_lst){
        if (b->legal(cell)){
            passed = false;
            b->move(cell, &nb);
            //if (depth == 1)
            //    g = (-nega_alpha(&nb, false, depth - 1, -beta, -alpha) + nv) / 2;
            //else
            g = -nega_alpha(&nb, false, depth - 1, -beta, -alpha);
            if (beta <= g)
                return g;
            alpha = max(alpha, g);
            v = max(v, g);
        }
    }
    if (passed){
        if (skipped)
            return end_evaluate(b);
        board rb;
        for (int i = 0; i < b_idx_num; ++i)
            rb.b[i] = b->b[i];
        rb.p = 1 - b->p;
        rb.n = b->n;
        return -nega_alpha(&rb, true, depth, -beta, -alpha);
    }
    return v;
}

int nega_alpha_ordering(board *b, bool skipped, const int depth, int alpha, int beta, bool use_multi_thread, bool use_mpc, double mpct_in){
    if (depth <= simple_mid_threshold)
        return nega_alpha(b, skipped, depth, alpha, beta);
    ++searched_nodes;
    #if USE_MID_SC
        if (stability_cut(b, &alpha, &beta))
            return alpha;
    #endif
    int hash = (int)(b->hash() & search_hash_mask);
    int l, u;
    transpose_table.get_now(b, b->hash() & search_hash_mask, &l, &u);
    #if USE_MID_TC
        if (l >= beta)
            return l;
        if (alpha >= u)
            return u;
        if (u == l)
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
    vector<board> nb;
    int canput = 0;
    for (const int &cell: vacant_lst){
        if (b->legal(cell)){
            nb.emplace_back(b->move(cell));
            move_ordering(&nb[canput]);
            //move_ordering_eval(&(nb[canput]));
            ++canput;
        }
    }
    if (canput == 0){
        if (skipped)
            return end_evaluate(b);
        board rb;
        for (int i = 0; i < b_idx_num; ++i)
            rb.b[i] = b->b[i];
        rb.p = 1 - b->p;
        rb.n = b->n;
        return -nega_alpha_ordering(&rb, true, depth, -beta, -alpha, use_multi_thread, use_mpc, mpct_in);
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
                if (l < g)
                    transpose_table.reg(b, hash, g, u);
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

int nega_scout(board *b, bool skipped, const int depth, int alpha, int beta, bool use_mpc, double mpct_in){
    if (depth <= simple_mid_threshold)
        return nega_alpha(b, skipped, depth, alpha, beta);
    ++searched_nodes;
    #if USE_MID_SC
        if (stability_cut(b, &alpha, &beta))
            return alpha;
    #endif
    int hash = (int)(b->hash() & search_hash_mask);
    int l, u;
    transpose_table.get_now(b, hash, &l, &u);
    #if USE_MID_TC
        if (l >= beta)
            return l;
        if (alpha >= u)
            return u;
        if (u == l)
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
    vector<board> nb;
    int canput = 0;
    for (const int &cell: vacant_lst){
        if (b->legal(cell)){
            nb.emplace_back(b->move(cell));
            move_ordering(&nb[canput]);
            //move_ordering_eval(&(nb[canput]));
            ++canput;
        }
    }
    if (canput == 0){
        if (skipped)
            return end_evaluate(b);
        board rb;
        for (int i = 0; i < b_idx_num; ++i)
            rb.b[i] = b->b[i];
        rb.p = 1 - b->p;
        rb.n = b->n;
        return -nega_scout(&rb, true, depth, -beta, -alpha, use_mpc, mpct_in);
    }
    if (canput >= 2)
        sort(nb.begin(), nb.end());
    int g = alpha, v = -inf;
    #if USE_MULTI_THREAD
        int i;
        g = -nega_scout(&nb[0], false, depth - 1, -beta, -alpha, use_mpc, mpct_in);
        alpha = max(alpha, g);
        if (beta <= alpha){
            if (l < g)
                transpose_table.reg(b, hash, g, u);
            return alpha;
        }
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
                g = -nega_scout(&nb[i], false, depth - 1, -beta, -alpha, use_mpc, mpct_in);
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
        int first_alpha = alpha;
        for (int i = 0; i < canput; ++i){
            if (i > 0){
                g = -nega_alpha_ordering(&nb[i], false, depth - 1, -alpha - search_epsilon, -alpha, true, use_mpc, mpct_in);
                if (beta < g){
                    if (l < g)
                        transpose_table.reg(b, hash, g, u);
                    return g;
                }
                v = max(v, g);
            }
            if (alpha <= g){
                g = -nega_scout(&nb[i], false, depth - 1, -beta, -g, use_mpc, mpct_in);
                if (beta <= g){
                    if (l < g)
                        transpose_table.reg(b, hash, g, u);
                    return g;
                }
                alpha = max(alpha, g);
                v = max(v, g);
            }
        }
    #endif
    if (v <= first_alpha)
        transpose_table.reg(b, hash, l, v);
    else
        transpose_table.reg(b, hash, v, v);
    return v;
}

int mtd(board *b, bool skipped, int depth, int l, int u, bool use_mpc, double use_mpct){
    int g = mid_evaluate(b), beta;
    while (u - l > 0){
        beta = max(l + search_epsilon, g);
        g = nega_alpha_ordering(b, skipped, depth, beta - search_epsilon, beta, true, use_mpc, use_mpct);
        if (g < beta)
            u = g;
        else
            l = g;
    }
    return l;
    //return nega_scout(b, skipped, depth, l, u, use_mpc, use_mpct);
}

inline search_result midsearch(board b, long long strt, int max_depth){
    vector<board> nb;
    int i;
    for (const int &cell: vacant_lst){
        if (b.legal(cell)){
            cerr << cell << " ";
            nb.push_back(b.move(cell));
        }
    }
    cerr << endl;
    int canput = nb.size();
    cerr << "canput: " << canput << endl;
    int res_depth;
    int policy = -1;
    int tmp_policy;
    int alpha, beta, g, value;
    searched_nodes = 0;
    transpose_table.hash_get = 0;
    transpose_table.hash_reg = 0;
    transpose_table.init_now();
    transpose_table.init_prev();
    int order_l, order_u;
    //int depth = min(hw2 - b.n - 1, max_depth - 1);
    bool use_mpc = max_depth >= 11 ? true : false;
    double use_mpct = 1.7;
    if (max_depth >= 13)
        use_mpct = 1.5;
    if (max_depth >= 15)
        use_mpct = 1.2;
    if (max_depth >= 17)
        use_mpct = 0.8;
    if (max_depth >= 19)
        use_mpct = 0.6;
    if (max_depth >= 21)
        use_mpct = 0.5;
    if (max_depth >= 23)
        use_mpct = 0.4;
    for (int depth = min(11, max(0, max_depth - 5)); depth <= min(hw2 - b.n, max_depth - 1); ++depth){
        alpha = -hw2;
        beta = hw2;
        transpose_table.init_now();
        for (i = 0; i < canput; ++i){
            //move_ordering_eval(&nb[i]);
            
            transpose_table.get_prev(&nb[i], nb[i].hash() & search_hash_mask, &order_l, &order_u);
            nb[i].v = -max(order_l, order_u);
            if (order_l != -inf && order_u != -inf)
                nb[i].v += 100000;
            if (order_l == -inf && order_u == -inf)
                nb[i].v = -mid_evaluate(&nb[i]);
            else
                nb[i].v += cache_hit;
            
        }
        if (canput >= 2)
            sort(nb.begin(), nb.end());
        g = -mtd(&nb[0], false, depth, -beta, -alpha, use_mpc, use_mpct);
        transpose_table.reg(&nb[0], (int)(nb[0].hash() & search_hash_mask), -g, -g);
        alpha = max(alpha, g);
        tmp_policy = nb[0].policy;
        for (i = 1; i < canput; ++i){
            g = -nega_alpha_ordering(&nb[i], false, depth, -alpha - search_epsilon, -alpha, true, use_mpc, use_mpct);
            if (alpha < g){
                g = -mtd(&nb[i], false, depth, -beta, -g, use_mpc, use_mpct);
                transpose_table.reg(&nb[i], (int)(nb[i].hash() & search_hash_mask), -g, -g);
                if (alpha < g){
                    alpha = g;
                    tmp_policy = nb[i].policy;
                }
            } else{
                transpose_table.reg(&nb[i], (int)(nb[i].hash() & search_hash_mask), -inf, -g);
            }
        }
        swap(transpose_table.now, transpose_table.prev);
        policy = tmp_policy;
        value = alpha;
        res_depth = depth + 1;
        cerr << "depth: " << depth + 1 << " time: " << tim() - strt << " policy: " << policy << " value: " << alpha << " nodes: " << searched_nodes << " nps: " << (long long)searched_nodes * 1000 / max(1LL, tim() - strt) << " get: " << transpose_table.hash_get << " reg: " << transpose_table.hash_reg << endl;
    }
    search_result res;
    res.policy = policy;
    res.value = value;
    res.depth = res_depth;
    res.nps = searched_nodes * 1000 / max(1LL, tim() - strt);
    return res;
}