#pragma once
#include <iostream>
#include <algorithm>
#include <vector>
#include "setting.hpp"
#include "common.hpp"
#include "board.hpp"
#include "evaluate.hpp"
#include "search_common.hpp"
#if USE_MULTI_THREAD
    #include "multi_threading.hpp"
#endif

using namespace std;

int nega_alpha(board *b, bool skipped, int depth, int alpha, int beta);
int nega_alpha_ordering(board *b, bool skipped, int depth, int alpha, int beta, bool use_multi_thread);

inline bool mpc_higher(board *b, bool skipped, int depth, int beta){
    int bound = beta + mpctsd[calc_phase_idx(b)][depth];
    if (bound > sc_w)
        return false;
    return nega_alpha_ordering(b, skipped, mpcd[depth], bound - search_epsilon, bound, false) >= bound;
}

inline bool mpc_lower(board *b, bool skipped, int depth, int alpha){
    int bound = alpha - mpctsd[calc_phase_idx(b)][depth];
    if (bound < -sc_w)
        return false;
    return nega_alpha_ordering(b, skipped, mpcd[depth], bound, bound + search_epsilon, false) <= bound;
}

int nega_alpha(board *b, bool skipped, int depth, int alpha, int beta){
    ++searched_nodes;
    if (depth == 0){
        if (b->n < hw2)
            return mid_evaluate(b);
        else
            return end_evaluate(b);
    }
    board nb;
    bool passed = true;
    int g, v = -inf;
    for (const int &cell: vacant_lst){
        if (b->legal(cell)){
            passed = false;
            b->move(cell, &nb);
            #if USE_MID_SC
                if (stability_cut(&nb, &alpha, &beta))
                    return alpha;
            #endif
            g = -nega_alpha(&nb, false, depth - 1, -beta, -alpha);
            alpha = max(alpha, g);
            if (beta <= alpha)
                return alpha;
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

int nega_alpha_ordering(board *b, bool skipped, const int depth, int alpha, int beta, bool use_multi_thread){
    if (depth <= simple_mid_threshold)
        return nega_alpha(b, skipped, depth, alpha, beta);
    ++searched_nodes;
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
        if (mpc_min_depth <= depth && depth <= mpc_max_depth){
            if (mpc_higher(b, skipped, depth, beta))
                return beta;
            if (mpc_lower(b, skipped, depth, alpha))
                return alpha;
        }
    #endif
    vector<board> nb;
    int canput = 0;
    for (const int &cell: vacant_lst){
        if (b->legal(cell)){
            nb.emplace_back(b->move(cell));
            #if USE_MID_SC
                if (stability_cut(&nb[canput], &alpha, &beta))
                    return alpha;
            #endif
            move_ordering(&(nb[canput]));
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
        return -nega_alpha_ordering(&rb, true, depth, -beta, -alpha, use_multi_thread);
    }
    if (canput >= 2)
        sort(nb.begin(), nb.end());
    int first_alpha = alpha, g, v = -inf;
    #if USE_MULTI_THREAD
        if (use_multi_thread){
            int i;
            g = -nega_alpha_ordering(&nb[0], false, depth - 1, -beta, -alpha, true);
            alpha = max(alpha, g);
            if (beta <= alpha){
                if (l < g)
                    transpose_table.reg(b, hash, g, u);
                return alpha;
            }
            v = max(v, g);
            int task_ids[canput];
            for (i = 1; i < canput; ++i)
                task_ids[i] = thread_pool.push(bind(&nega_alpha_ordering, &nb[i], false, depth - 1, -beta, -alpha, false));
            for (i = 1; i < canput; ++i){
                g = -thread_pool.get(task_ids[i]);
                alpha = max(alpha, g);
                v = max(v, g);
            }
            if (beta <= alpha){
                if (l < g)
                    transpose_table.reg(b, hash, g, u);
                return alpha;
            }
        } else{
            for (int i = 0; i < canput; ++i){
                g = -nega_alpha_ordering(&nb[i], false, depth - 1, -beta, -alpha, false);
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
        for (int i = 0; i < canput; ++i){
            g = -nega_alpha_ordering(&nb[i], false, depth - 1, -beta, -alpha, false);
            alpha = max(alpha, g);
            if (beta <= alpha){
                if (l < g)
                    transpose_table.reg(b, hash, g, u);
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

int nega_scout(board *b, bool skipped, const int depth, int alpha, int beta){
    if (depth <= simple_mid_threshold)
        return nega_alpha(b, skipped, depth, alpha, beta);
    ++searched_nodes;
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
        if (mpc_min_depth <= depth && depth <= mpc_max_depth){
            if (mpc_higher(b, skipped, depth, beta))
                return beta;
            if (mpc_lower(b, skipped, depth, alpha))
                return alpha;
        }
    #endif
    vector<board> nb;
    int canput = 0;
    for (const int &cell: vacant_lst){
        if (b->legal(cell)){
            nb.emplace_back(b->move(cell));
            #if USE_MID_SC
                if (stability_cut(&nb[canput], &alpha, &beta))
                    return alpha;
            #endif
            move_ordering(&(nb[canput]));
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
        return -nega_scout(&rb, true, depth, -beta, -alpha);
    }
    if (canput >= 2)
        sort(nb.begin(), nb.end());
    int g = alpha, v = -inf;
    #if USE_MULTI_THREAD
        int i;
        g = -nega_scout(&nb[0], false, depth - 1, -beta, -alpha);
        alpha = max(alpha, g);
        if (beta <= alpha){
            if (l < g)
                transpose_table.reg(b, hash, g, u);
            return alpha;
        }
        v = max(v, g);
        int first_alpha = alpha;
        int task_ids[canput];
        for (i = 1; i < canput; ++i)
            task_ids[i] = thread_pool.push(bind(&nega_alpha_ordering, &nb[i], false, depth - 1, -alpha - search_epsilon, -alpha, false));
        bool re_search[canput];
        for (i = 1; i < canput; ++i){
            g = -thread_pool.get(task_ids[i]);
            alpha = max(alpha, g);
            v = max(v, g);
            re_search[i] = first_alpha < g;
        }
        for (i = 1; i < canput; ++i){
            if (re_search[i]){
                g = -nega_scout(&nb[i], false, depth - 1, -beta, -alpha);
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
                g = -nega_alpha_ordering(&nb[i], false, depth - 1, -alpha - search_epsilon, -alpha, true);
                if (beta <= g){
                    if (l < g)
                        transpose_table.reg(b, hash, g, u);
                    return g;
                }
                v = max(v, g);
            }
            if (alpha <= g){
                g = -nega_scout(&nb[i], false, depth - 1, -beta, -g);
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

int mtd(board *b, bool skipped, int depth, int l, int u){
    int g = mid_evaluate(b), beta;
    while (u - l > mtd_threshold){
        beta = g;
        g = nega_alpha_ordering(b, skipped, depth, beta - search_epsilon, beta, true);
        if (g < beta)
            u = g;
        else
            l = g;
        g = (l + u) / 2;
    }
    return nega_scout(b, skipped, depth, l, u);
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
    for (int depth = min(5, max(1, max_depth - 5)); depth < min(hw2 - b.n, max_depth); ++depth){
        alpha = -sc_w;
        beta = sc_w;
        transpose_table.init_now();
        for (i = 0; i < canput; ++i){
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
        g = -mtd(&nb[0], false, depth, -beta, -alpha);
        if (g == inf)
            break;
        transpose_table.reg(&nb[0], (int)(nb[0].hash() & search_hash_mask), g, g);
        alpha = max(alpha, g);
        tmp_policy = nb[0].policy;
        for (i = 1; i < canput; ++i){
            g = -nega_alpha_ordering(&nb[i], false, depth, -alpha - search_epsilon, -alpha, true);
            if (alpha < g){
                alpha = g;
                g = -mtd(&nb[i], false, depth, -beta, -alpha);
                transpose_table.reg(&nb[i], (int)(nb[i].hash() & search_hash_mask), g, g);
                if (alpha < g){
                    alpha = g;
                    tmp_policy = nb[i].policy;
                }
            } else{
                transpose_table.reg(&nb[i], (int)(nb[i].hash() & search_hash_mask), -inf, g);
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