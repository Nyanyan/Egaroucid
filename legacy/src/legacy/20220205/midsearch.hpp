#pragma once
#include <iostream>
#include <algorithm>
#include <vector>
#include "setting.hpp"
#include "common.hpp"
#include "Board.hpp"
#include "evaluate.hpp"
#include "search.hpp"
#include "transpose_table.hpp"
#include "level.hpp"
#if USE_MULTI_THREAD
    #include "thread_pool.hpp"
#endif

using namespace std;

int nega_alpha(Board *b, bool skipped, int depth, int alpha, int beta, unsigned long long *n_nodes);
inline bool mpc_higher(Board *b, bool skipped, int depth, int beta, double t, unsigned long long *n_nodes);
inline bool mpc_lower(Board *b, bool skipped, int depth, int alpha, double t, unsigned long long *n_nodes);

int nega_alpha_ordering_nomemo(Board *b, bool skipped, int depth, int alpha, int beta, bool use_mpc, double mpct_in, unsigned long long *n_nodes){
    if (!global_searching)
        return -inf;
    if (depth <= simple_mid_threshold)
        return nega_alpha(b, skipped, depth, alpha, beta, n_nodes);
    ++(*n_nodes);
    #if USE_MID_SC
        if (stability_cut(b, &alpha, &beta))
            return alpha;
    #endif
    #if USE_MID_MPC
        if (mpc_min_depth <= depth && depth <= mpc_max_depth){
            if (mpc_higher(b, skipped, depth, beta, mpct_in, n_nodes))
                return beta;
            if (mpc_lower(b, skipped, depth, alpha, mpct_in, n_nodes))
                return alpha;
        }
    #endif
    unsigned long long legal = b->mobility_ull();
    if (legal == 0){
        if (skipped)
            return end_evaluate(b);
        b->p = 1 - b->p;
        int res = -nega_alpha_ordering_nomemo(b, true, depth, -beta, -alpha, use_mpc, mpct_in, n_nodes);
        b->p = 1 - b->p;
        return res;
    }
    const int canput = pop_count_ull(legal);
    Board *nb = new Board[canput];
    Mobility mob;
    int idx = 0;
    int hash = b->hash() & search_hash_mask;
    for (const int &cell: vacant_lst){
        if (1 & (legal >> cell)){
            calc_flip(&mob, b, cell);
            b->move_copy(&mob, &nb[idx]);
            nb[idx].v = move_ordering(b, &nb[idx], hash, cell);
            ++idx;
        }
    }
    if (canput >= 2)
        sort(nb, nb + canput);
    int g, v = -inf;
    for (idx = 0; idx < canput; ++idx){
        g = -nega_alpha_ordering_nomemo(&nb[idx], false, depth - 1, -beta, -alpha, use_mpc, mpct_in, n_nodes);
        if (beta <= g)
            return g;
        alpha = max(alpha, g);
        v = max(v, g);
    }
    return v;
}

inline bool mpc_higher(Board *b, bool skipped, int depth, int beta, double t, unsigned long long *n_nodes){
    int bound = beta + ceil(t * mpcsd[b->phase()][depth - mpc_min_depth]);
    if (bound > hw2)
        bound = hw2; //return false;
    return nega_alpha_ordering_nomemo(b, skipped, mpcd[depth], bound - search_epsilon, bound, true, t, n_nodes) >= bound;
}

inline bool mpc_lower(Board *b, bool skipped, int depth, int alpha, double t, unsigned long long *n_nodes){
    int bound = alpha - ceil(t * mpcsd[b->phase()][depth - mpc_min_depth]);
    if (bound < -hw2)
        bound = -hw2; //return false;
    return nega_alpha_ordering_nomemo(b, skipped, mpcd[depth], bound, bound + search_epsilon, true, t, n_nodes) <= bound;
}

int nega_alpha(Board *b, bool skipped, int depth, int alpha, int beta, unsigned long long *n_nodes){
    if (!global_searching)
        return -inf;
    ++(*n_nodes);
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
        int res = -nega_alpha(b, true, depth, -beta, -alpha, n_nodes);
        b->p = 1 - b->p;
        return res;
    }
    Mobility mob;
    for (const int &cell: vacant_lst){
        if (1 & (legal >> cell)){
            calc_flip(&mob, b, cell);
            b->move(&mob);
            g = -nega_alpha(b, false, depth - 1, -beta, -alpha, n_nodes);
            b->undo(&mob);
            alpha = max(alpha, g);
            if (beta <= alpha)
                return alpha;
            v = max(v, g);
        }
    }
    return v;
}

int nega_alpha_ordering(Board *b, Transpose_table *tt, bool skipped, const int depth, int alpha, int beta, bool use_mpc, double mpct_in, unsigned long long *n_nodes);

inline Ybwc ybwc_midsearch(Board b, Transpose_table *tt, bool skipped, const int depth, int alpha, int beta, bool use_mpc, double mpct_in){
    Ybwc res;
    res.n_nodes = 0;
    tt->copy_prev(&res.tt);
    tt->copy_now(&res.tt);
    res.value = nega_alpha_ordering(&b, &res.tt, skipped, depth, alpha, beta, use_mpc, mpct_in, &res.n_nodes);
    return res;
}

int nega_alpha_ordering(Board *b, Transpose_table *tt, bool skipped, const int depth, int alpha, int beta, bool use_mpc, double mpct_in, unsigned long long *n_nodes){
    if (!global_searching)
        return -inf;
    if (depth <= simple_mid_threshold)
        return nega_alpha(b, skipped, depth, alpha, beta, n_nodes);
    ++(*n_nodes);
    #if USE_MID_SC
        if (stability_cut(b, &alpha, &beta))
            return alpha;
    #endif
    int hash = b->hash() & search_hash_mask;
    int l, u;
    tt->get_now(b, hash, &l, &u);
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
            if (mpc_higher(b, skipped, depth, beta, mpct_in, n_nodes)){
                if (l < beta)
                    tt->reg(b, hash, beta, u);
                return beta;
            }
            if (mpc_lower(b, skipped, depth, alpha, mpct_in, n_nodes)){
                if (alpha < u)
                    tt->reg(b, hash, l, alpha);
                return alpha;
            }
        }
    #endif
    unsigned long long legal = b->mobility_ull();
    if (legal == 0){
        if (skipped)
            return end_evaluate(b);
        b->p = 1 - b->p;
        int res = -nega_alpha_ordering(b, tt, true, depth, -beta, -alpha, use_mpc, mpct_in, n_nodes);
        b->p = 1 - b->p;
        return res;
    }
    const int canput = pop_count_ull(legal);
    Board *nb = new Board[canput];
    Mobility mob;
    int idx = 0;
    for (const int &cell: vacant_lst){
        if (1 & (legal >> cell)){
            calc_flip(&mob, b, cell);
            b->move_copy(&mob, &nb[idx]);
            nb[idx].v = move_ordering(b, &nb[idx], hash, cell);
            ++idx;
        }
    }
    if (canput >= 2)
        sort(nb, nb + canput);
    int first_alpha = alpha, g, v = -inf;
    #if USE_MULTI_THREAD
        int i;
        const int first_threshold = canput / mid_first_threshold_div + 1;
        for (i = 0; i < first_threshold; ++i){
            g = -nega_alpha_ordering(&nb[i], tt, false, depth - 1, -beta, -alpha, use_mpc, mpct_in, n_nodes);
            alpha = max(alpha, g);
            if (beta <= alpha){
                if (l < alpha)
                    tt->reg(b, hash, alpha, u);
                delete[] nb;
                return alpha;
            }
            v = max(v, g);
        }
        vector<future<Ybwc>> future_tasks;
        Ybwc task;
        int done_tasks = first_threshold;
        int next_done_tasks, additional_done_tasks;
        while (done_tasks < canput){
            next_done_tasks = canput;
            future_tasks.clear();
            for (i = done_tasks; i < canput; ++i){
                if (thread_pool.n_idle() == 0){
                    next_done_tasks = i;
                    break;
                }
                future_tasks.emplace_back(thread_pool.push(bind(&ybwc_midsearch, nb[i], tt, false, depth - 1, -beta, -alpha, use_mpc, mpct_in)));
            }
            additional_done_tasks = 0;
            if (next_done_tasks < canput){
                g = -nega_alpha_ordering(&nb[next_done_tasks], tt, false, depth - 1, -beta, -alpha,  use_mpc, mpct_in, n_nodes);
                alpha = max(alpha, g);
                v = max(v, g);
                additional_done_tasks = 1;
            }
            for (i = done_tasks; i < next_done_tasks; ++i){
                task = future_tasks[i - done_tasks].get();
                *n_nodes += task.n_nodes;
                tt->merge(&task.tt);
                g = -task.value;
                alpha = max(alpha, g);
                v = max(v, g);
            }
            if (beta <= alpha){
                #if USE_END_TC
                    if (l < alpha)
                        tt->reg(b, hash, alpha, u);
                #endif
                delete[] nb;
                return alpha;
            }
            done_tasks = next_done_tasks + additional_done_tasks;
        }
        delete[] nb;
    #else
        for (idx = 0; idx < canput; ++idx){
            g = -nega_alpha_ordering(&nb[idx], tt, false, depth - 1, -beta, -alpha, use_mpc, mpct_in, n_nodes);
            alpha = max(alpha, g);
            if (beta <= alpha){
                if (l < alpha)
                    tt->reg(b, hash, alpha, u);
                delete[] nb;
                return alpha;
            }
            v = max(v, g);
        }
        delete[] nb;
    #endif
    if (v <= first_alpha)
        tt->reg(b, hash, l, v);
    else
        tt->reg(b, hash, v, v);
    return v;
}

int nega_scout_nomemo(Board *b, bool skipped, const int depth, int alpha, int beta, bool use_mpc, double mpct_in, unsigned long long *n_nodes){
    if (!global_searching)
        return -inf;
    if (depth <= simple_mid_threshold)
        return nega_alpha(b, skipped, depth, alpha, beta, n_nodes);
    ++(*n_nodes);
    #if USE_MID_SC
        if (stability_cut(b, &alpha, &beta))
            return alpha;
    #endif
    #if USE_MID_MPC
        if (mpc_min_depth <= depth && depth <= mpc_max_depth && use_mpc){
            if (mpc_higher(b, skipped, depth, beta, mpct_in, n_nodes))
                return beta;
            if (mpc_lower(b, skipped, depth, alpha, mpct_in, n_nodes))
                return alpha;
        }
    #endif
    unsigned long long legal = b->mobility_ull();
    if (legal == 0){
        if (skipped)
            return end_evaluate(b);
        b->p = 1 - b->p;
        int res = -nega_scout_nomemo(b, true, depth, -beta, -alpha, use_mpc, mpct_in, n_nodes);
        b->p = 1 - b->p;
        return res;
    }
    vector<Board> nb;
    Mobility mob;
    int canput = 0;
    for (const int &cell: vacant_lst){
        if (1 & (legal >> cell)){
            calc_flip(&mob, b, cell);
            nb.emplace_back(b->move_copy(&mob));
            move_ordering_eval(&(nb[canput]));
            ++canput;
        }
    }
    if (canput >= 2)
        sort(nb.begin(), nb.end());
    int g = alpha, v = -inf;
    for (int i = 0; i < canput; ++i){
        if (i > 0){
            g = -nega_alpha_ordering_nomemo(&nb[i], false, depth - 1, -alpha - search_epsilon, -alpha, use_mpc, mpct_in, n_nodes);
            if (beta < g)
                return g;
            v = max(v, g);
        }
        if (alpha <= g || i == 0){
            alpha = g;
            g = -nega_scout_nomemo(&nb[i], false, depth - 1, -beta, -alpha, use_mpc, mpct_in, n_nodes);
            alpha = max(alpha, g);
            if (beta <= alpha)
                return alpha;
            v = max(v, g);
        }
    }
    return v;
}

int mtd(Board *b, bool skipped, int depth, int l, int u, bool use_mpc, double use_mpct, unsigned long long *n_nodes){
    int g, beta;
    g = nega_alpha(b, skipped, 5, l, u, n_nodes);
    while (u - l > 0){
        beta = max(l + search_epsilon, g);
        g = nega_alpha_ordering(b, &transpose_table, skipped, depth, beta - search_epsilon, beta, use_mpc, use_mpct, n_nodes);
        if (g < beta)
            u = g;
        else
            l = g;
    }
    return g;
}

inline search_result midsearch(Board b, long long strt, int max_depth, bool use_mpc, double use_mpct){
    vector<Board> nb;
    Mobility mob;
    int i = 0;
    int hash = b.hash() & search_hash_mask;
    unsigned long long legal = b.mobility_ull();
    for (const int &cell: vacant_lst){
        if (1 & (legal >> cell)){
            calc_flip(&mob, &b, cell);
            nb.emplace_back(b.move_copy(&mob));
            nb[i].v = move_ordering(&b, &nb[i], hash, cell);
            ++i;
        }
    }
    int canput = (int)nb.size();
    //cerr << "canput: " << canput << endl;
    int res_depth = -1;
    int policy = -1;
    int tmp_policy;
    int alpha, beta, g, value = -inf, former_value = -inf;
    unsigned long long searched_nodes = 0;
    transpose_table.init_now();
    transpose_table.init_prev();
    for (int depth = min(16, max(0, max_depth - 5)); depth <= min(hw2 - b.n, max_depth); ++depth){
        alpha = -hw2;
        beta = hw2;
        transpose_table.init_now();
        for (i = 0; i < canput; ++i)
            nb[i].v = move_ordering(&b, &nb[i], hash, nb[i].policy);
        if (canput >= 2)
            sort(nb.begin(), nb.end());
        for (i = 0; i < canput; ++i){
            g = -mtd(&nb[i], false, depth, -beta, -alpha, use_mpc, use_mpct, &searched_nodes);
            if (alpha < g || i == 0){
                transpose_table.reg(&nb[i], nb[i].hash() & search_hash_mask, g, g);
                alpha = g;
                tmp_policy = nb[i].policy;
            } else
                transpose_table.reg(&nb[i], nb[i].hash() & search_hash_mask, -inf, g);
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
            cerr << "depth: " << depth << " time: " << tim() - strt << " policy: " << policy << " value: " << alpha << " nodes: " << searched_nodes << " nps: " << (long long)searched_nodes * 1000 / max(1LL, tim() - strt) << endl;
        } else 
            break;
    }
    search_result res;
    res.policy = policy;
    res.value = (value + former_value) / 2;
    res.depth = res_depth;
    res.nps = searched_nodes * 1000 / max(1LL, tim() - strt);
    transpose_table.init_now();
    transpose_table.init_prev();
    search_completed = false;
    return res;
}

inline search_result midsearch_value_nomemo(Board b, long long strt, int max_depth, bool use_mpc, double use_mpct){
    unsigned long long searched_nodes = 0;
    int value = nega_scout_nomemo(&b, false, max_depth, -hw2, hw2, use_mpc, use_mpct, &searched_nodes);
    search_result res;
    res.policy = -1;
    res.value = value;
    //cerr << res.value << endl;
    res.depth = max_depth;
    res.nps = searched_nodes * 1000 / max(1LL, tim() - strt);
    return res;
}

inline search_result midsearch_value_memo(Board b, long long strt, int max_depth, bool use_mpc, double use_mpct){
    unsigned long long searched_nodes = 0;
    int value = -inf, former_value = -inf, g;
    for (int depth = min(16, max(0, max_depth - 5)); depth <= min(hw2 - b.n, max_depth); ++depth){
        transpose_table.init_now();
        g = mtd(&b, false, depth, -hw2, hw2, use_mpc, use_mpct, &searched_nodes);
        former_value = value;
        value = g;
        swap(transpose_table.now, transpose_table.prev);
        cerr << "midsearch depth " << depth << " value " << g << " nodes " << searched_nodes << " time " << tim() - strt << " nps " << searched_nodes * 1000 / max(1LL, tim() - strt) << endl;
    }
    if (former_value != -inf)
        value = (value + former_value) / 2;
    search_result res;
    res.policy = -1;
    res.value = value;
    //cerr << res.value << endl;
    res.depth = max_depth;
    res.nps = searched_nodes * 1000 / max(1LL, tim() - strt);
    return res;
}
