#pragma once
#include <iostream>
#include "setting.hpp"
#include "common.hpp"
#include "board.hpp"
#include "evaluate.hpp"
#include "search_common.hpp"
#include "midsearch.hpp"

using namespace std;

int nega_alpha_ordering_nompc(board *b, bool skipped, int depth, int alpha, int beta){
    if (depth <= simple_end_threshold)
        return nega_alpha(b, skipped, depth, alpha, beta);
    ++searched_nodes;
    #if USE_END_SC
        if (stability_cut(b, &alpha, &beta))
            return alpha;
    #endif
    vector<board> nb;
    int canput = 0;
    for (const int &cell: vacant_lst){
        if (b->legal(cell)){
            nb.emplace_back(b->move(cell));
            //move_ordering(&(nb[canput]));
            nb[canput].v = -calc_canput_exact(&(nb[canput]));
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
        return -nega_alpha_ordering_nompc(&rb, true, depth, -beta, -alpha);
    }
    if (canput >= 2)
        sort(nb.begin(), nb.end());
    int g, v = -inf;
    for (board &nnb: nb){
        g = -nega_alpha_ordering_nompc(&nnb, false, depth - 1, -beta, -alpha);
        if (beta <= g)
            return g;
        alpha = max(alpha, g);
        v = max(v, g);
    }
    return v;
}

inline bool mpc_higher_final(board *b, bool skipped, int depth, int beta){
    int bound = beta + mpctsd_final[depth];
    if (bound >= sc_w)
        return false;
    return nega_alpha_ordering_nompc(b, skipped, mpcd[depth], bound - search_epsilon, bound) >= bound;
}

inline bool mpc_lower_final(board *b, bool skipped, int depth, int alpha){
    int bound = alpha - mpctsd_final[depth];
    if (bound <= -sc_w)
        return false;
    return nega_alpha_ordering_nompc(b, skipped, mpcd[depth], bound, bound + search_epsilon) <= bound;
}

inline int last1(board *b, bool skipped, int p0){
    ++searched_nodes;
    int i, before_score = 0;
    for (i = 0; i < hw; ++i)
        before_score += count_black_arr[b->b[i]];
    if (b->p)
        before_score = -before_score;
    int score = before_score + 1 + (
        move_arr[b->p][b->b[place_included[p0][0]]][local_place[place_included[p0][0]][p0]][0] + move_arr[b->p][b->b[place_included[p0][0]]][local_place[place_included[p0][0]][p0]][1] + 
        move_arr[b->p][b->b[place_included[p0][1]]][local_place[place_included[p0][1]][p0]][0] + move_arr[b->p][b->b[place_included[p0][1]]][local_place[place_included[p0][1]][p0]][1] + 
        move_arr[b->p][b->b[place_included[p0][2]]][local_place[place_included[p0][2]][p0]][0] + move_arr[b->p][b->b[place_included[p0][2]]][local_place[place_included[p0][2]][p0]][1]) * 2;
    if (place_included[p0][3] != -1)
        score += (move_arr[b->p][b->b[place_included[p0][3]]][local_place[place_included[p0][3]][p0]][0] + move_arr[b->p][b->b[place_included[p0][3]]][local_place[place_included[p0][3]][p0]][1]) * 2;
    if (score == before_score + 1){
        if (skipped)
            return end_evaluate(b);
        board rb;
        for (i = 0; i < b_idx_num; ++i)
            rb.b[i] = b->b[i];
        rb.p = 1 - b->p;
        rb.n = b->n;
        return -last1(&rb, true, p0);
    }
    return score * step;
}

inline int last2(board *b, bool skipped, int alpha, int beta, int p0, int p1){
    ++searched_nodes;
    board nb;
    bool passed = true;
    int v = -inf, g;
    if (b->legal(p0)){
        passed = false;
        nb = b->move(p0);
        g = -last1(&nb, false, p1);
        alpha = max(alpha, g);
        if (beta <= alpha)
            return alpha;
        v = max(v, g);
    }
    if (b->legal(p1)){
        passed = false;
        nb = b->move(p1);
        g = -last1(&nb, false, p0);
        alpha = max(alpha, g);
        if (beta <= alpha)
            return alpha;
        v = max(v, g);
    }
    if (passed){
        if (skipped)
            return end_evaluate(b);
        board rb;
        for (int i = 0; i < b_idx_num; ++i)
            rb.b[i] = b->b[i];
        rb.p = 1 - b->p;
        rb.n = b->n;
        return -last2(&rb, true, -beta, -alpha, p0, p1);
    }
    return v;
}

inline int last3(board *b, bool skipped, int alpha, int beta, int p0, int p1, int p2){
    ++searched_nodes;
    board nb;
    bool passed = true;
    int v = -inf, g;
    if (b->legal(p0)){
        passed = false;
        nb = b->move(p0);
        g = -last2(&nb, false, -beta, -alpha, p1, p2);
        alpha = max(alpha, g);
        if (beta <= alpha)
            return alpha;
        v = max(v, g);
    }
    if (b->legal(p1)){
        passed = false;
        nb = b->move(p1);
        g = -last2(&nb, false, -beta, -alpha, p0, p2);
        alpha = max(alpha, g);
        if (beta <= alpha)
            return alpha;
        v = max(v, g);
    }
    if (b->legal(p2)){
        passed = false;
        nb = b->move(p2);
        g = -last2(&nb, false, -beta, -alpha, p0, p1);
        alpha = max(alpha, g);
        if (beta <= alpha)
            return alpha;
        v = max(v, g);
    }
    if (passed){
        if (skipped)
            return end_evaluate(b);
        board rb;
        for (int i = 0; i < b_idx_num; ++i)
            rb.b[i] = b->b[i];
        rb.p = 1 - b->p;
        rb.n = b->n;
        return -last3(&rb, true, -beta, -alpha, p0, p1, p2);
    }
    return v;
}

inline void pick_vacant(board *b, int cells[]){
    int idx = 0;
    for (const int &cell: vacant_lst){
        if (pop_digit[b->b[cell / hw]][cell % hw] == 2)
            cells[idx++] = cell;
    }
}

int nega_alpha_final(board *b, bool skipped, int depth, int alpha, int beta){
    if (b->n == hw2 - 3){
        int cells[3];
        pick_vacant(b, cells);
        return last3(b, skipped, alpha, beta, cells[0], cells[1], cells[2]);
    }
    ++searched_nodes;
    #if USE_END_SC
        if (stability_cut(b, &alpha, &beta))
            return alpha;
    #endif
    board nb;
    bool passed = true;
    int g, v = -inf;
    for (const int &cell: vacant_lst){
        if (b->legal(cell)){
            passed = false;
            nb = b->move(cell);
            g = -nega_alpha_final(&nb, false, depth - 1, -beta, -alpha);
            alpha = max(alpha, g);
            if (beta <= alpha)
                return alpha;
            v = max(v, g);
        }
    }
    if (passed){
        if (skipped)
            return end_evaluate(b);
        for (int i = 0; i < b_idx_num; ++i)
            nb.b[i] = b->b[i];
        nb.p = 1 - b->p;
        nb.n = b->n;
        return -nega_alpha_final(&nb, true, depth, -beta, -alpha);
    }
    return v;
}

int nega_alpha_ordering_final(board *b, bool skipped, int depth, int alpha, int beta){
    if (depth <= simple_end_threshold)
        return nega_alpha_final(b, skipped, depth, alpha, beta);
    ++searched_nodes;
    int hash = (int)(b->hash() & search_hash_mask);
    int l, u;
    transpose_table.get_now(b, hash, &l, &u);
    alpha = max(alpha, l);
    beta = min(beta, u);
    if (alpha >= beta)
        return alpha;
    #if USE_END_SC
        if (stability_cut(b, &alpha, &beta))
            return alpha;
    #endif
    #if USE_END_MPC
        if (mpc_min_depth_final <= depth && depth <= mpc_max_depth_final){
            if (mpc_higher_final(b, skipped, depth, beta))
                return beta;
            if (mpc_lower_final(b, skipped, depth, alpha))
                return alpha;
        }
    #endif
    vector<board> nb;
    int canput = 0;
    for (const int &cell: vacant_lst){
        if (b->legal(cell)){
            nb.push_back(b->move(cell));
            //move_ordering(&(nb[canput]));
            nb[canput].v = -calc_canput_exact(&(nb[canput]));
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
        int res = -nega_alpha_ordering_final(&rb, true, depth, -beta, -alpha);
        if (res >= beta)
            transpose_table.reg(b, hash, res, u);
        else if (res <= alpha)
            transpose_table.reg(b, hash, l, res);
        else
            transpose_table.reg(b, hash, res, res);
        return res;
    }
    if (canput >= 2)
        sort(nb.begin(), nb.end());
    int g, v = -inf, first_alpha = alpha;
    for (board &nnb: nb){
        g = -nega_alpha_ordering_final(&nnb, false, depth - 1, -beta, -alpha);
        alpha = max(alpha, g);
        if (beta <= alpha){
            if (l < g)
                transpose_table.reg(b, hash, alpha, u);
            return alpha;
        }
        v = max(v, g);
    }
    if (v <= first_alpha)
        transpose_table.reg(b, hash, l, v);
    else
        transpose_table.reg(b, hash, v, v);
    return v;
}

int nega_scout_final(board *b, bool skipped, int depth, int alpha, int beta){
    if (depth <= simple_end_threshold)
        return nega_alpha_final(b, skipped, depth, alpha, beta);
    ++searched_nodes;
    int hash = (int)(b->hash() & search_hash_mask);
    int l, u;
    transpose_table.get_now(b, hash, &l, &u);
    alpha = max(alpha, l);
    beta = min(beta, u);
    if (alpha >= beta)
        return alpha;
    #if USE_END_SC
        if (stability_cut(b, &alpha, &beta))
            return alpha;
    #endif
    #if USE_END_MPC
        if (mpc_min_depth_final <= depth && depth <= mpc_max_depth_final){
            if (mpc_higher_final(b, skipped, depth, beta))
                return beta;
            if (mpc_lower_final(b, skipped, depth, alpha))
                return alpha;
        }
    #endif
    #if USE_END_OO
        bool odd_vacant[hw2];
        pick_vacant_odd(b->b, odd_vacant);
    #endif
    vector<board> nb;
    int canput = 0;
    for (const int &cell: vacant_lst){
        if (b->legal(cell)){
            nb.push_back(b->move(cell));
            move_ordering(&(nb[canput]));
            nb[canput].v -= canput_bonus * calc_canput_exact(&(nb[canput]));
            #if USE_END_OO
                if (odd_vacant[cell])
                    nb[canput].v += odd_vacant_bonus;
            #endif
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
        int res = -nega_scout_final(&rb, true, depth, -beta, -alpha);
        if (res >= beta)
            transpose_table.reg(b, hash, res, u);
        else if (res <= alpha)
            transpose_table.reg(b, hash, l, res);
        else
            transpose_table.reg(b, hash, res, res);
        return res;
    }
    if (canput >= 2)
        sort(nb.begin(), nb.end());
    int g = alpha, v = -inf, first_alpha = alpha;
    for (board &nnb: nb){
        if (&nnb - &nb[0]){
            g = -nega_alpha_ordering_final(&nnb, false, depth - 1, -alpha - step, -alpha);
            if (beta <= g){
                if (l < g)
                    transpose_table.reg(b, hash, g, u);
                return g;
            }
            v = max(v, g);
        }
        if (alpha <= g){
            g = -nega_scout_final(&nnb, false, depth - 1, -beta, -g);
            if (beta <= g){
                if (l < g)
                    transpose_table.reg(b, hash, g, u);
                return g;
            }
            alpha = max(alpha, g);
            v = max(v, g);
        }
    }
    if (v <= first_alpha)
        transpose_table.reg(b, hash, l, v);
    else
        transpose_table.reg(b, hash, v, v);
    return v;
}

int mtd_final(board *b, bool skipped, int depth, int l, int u){
    int g = mid_evaluate(b), beta;
    while (u - l > mtd_threshold_final){
        beta = g;
        g = nega_alpha_ordering_final(b, skipped, depth, beta - search_epsilon, beta);
        if (g < beta)
            u = g;
        else
            l = g;
        g = (l + u) / 2 / step * step;
    }
    return nega_scout_final(b, skipped, depth, l, u);
}

inline search_result endsearch(board b, long long strt){
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
    int policy = -1;
    int tmp_policy;
    int alpha, beta, g, value;
    searched_nodes = 0;
    transpose_table.hash_get = 0;
    transpose_table.hash_reg = 0;
    int order_l, order_u;
    int max_depth = hw2 - b.n;
    int pre_search_depth = min(17, max_depth - simple_end_threshold);
    transpose_table.init_now();
    transpose_table.init_prev();
    if (pre_search_depth > 0)
        midsearch(b, strt, pre_search_depth);
    cerr << "start final search depth " << max_depth << endl;
    alpha = -sc_w;
    beta = sc_w;
    transpose_table.init_now();
    for (i = 0; i < canput; ++i){
        transpose_table.get_prev(&nb[i], nb[i].hash() & search_hash_mask, &order_l, &order_u);
        nb[i].v = -max(order_l, order_u);
        if (order_l != -inf && order_u != -inf)
            nb[i].v += cache_both;
        if (order_l == -inf && order_u == -inf)
            nb[i].v = -mid_evaluate(&nb[i]);
        else
            nb[i].v += cache_hit;
    }
    if (canput >= 2)
        sort(nb.begin(), nb.end());
    alpha = -nega_scout_final(&nb[0], false, max_depth, -beta, -alpha);
    tmp_policy = nb[0].policy;
    for (i = 1; i < canput; ++i){
        g = -nega_alpha_ordering_final(&nb[i], false, max_depth, -alpha - step, -alpha);
        if (alpha < g){
            g = -nega_scout_final(&nb[i], false, max_depth, -beta, -g);
            if (alpha <= g){
                alpha = g;
                tmp_policy = nb[i].policy;
            }
        }
    }
    swap(transpose_table.now, transpose_table.prev);
    policy = tmp_policy;
    value = alpha;
    cerr << "final depth: " << max_depth << " time: " << tim() - strt << " policy: " << policy << " value: " << alpha << " nodes: " << searched_nodes << " nps: " << (long long)searched_nodes * 1000 / max(1LL, tim() - strt) << " get: " << transpose_table.hash_get << " reg: " << transpose_table.hash_reg << endl;
    search_result res;
    res.policy = policy;
    res.value = value;
    res.depth = max_depth;
    res.nps = searched_nodes * 1000 / max(1LL, tim() - strt);
    return res;
}