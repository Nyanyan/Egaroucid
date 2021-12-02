#pragma once
#include <iostream>
#include <algorithm>
#include <vector>
#include "common.hpp"
#include "board.hpp"
#include "evaluate.hpp"
#include "search_common.hpp"

using namespace std;

vector<int> vacant_lst;

int nega_alpha(board *b, bool skipped, int depth, int alpha, int beta);

inline bool mpc_higher(board *b, bool skipped, int depth, int beta){
    int bound = beta + mpctsd[(b->n - 4) / 10][depth];
    return nega_alpha(b, skipped, mpcd[depth], bound - search_epsilon, bound) >= bound;
}

inline bool mpc_lower(board *b, bool skipped, int depth, int alpha){
    int bound = alpha - mpctsd[(b->n - 4) / 10][depth];
    return nega_alpha(b, skipped, mpcd[depth], bound, bound + search_epsilon) <= bound;
}

inline void move_ordering(board *b){
    int l, u;
    transpose_table.get_prev(b->b, b->hash() & search_hash_mask, &l, &u);
    b->v = -max(l, u);
    if (u != -inf && l != -inf)
        b->v += cache_both;
    if (u != -inf || l != -inf)
        b->v += cache_hit;
    else
        b->v = -mid_evaluate(b);
}

int nega_alpha(board *b, bool skipped, int depth, int alpha, int beta){
    ++searched_nodes;
    if (depth == 0){
        if (b->n < hw2)
            return mid_evaluate(b);
        else
            return end_evaluate(b);
    }
    if (mpc_min_depth <= depth && depth <= mpc_max_depth){
        if (mpc_higher(b, skipped, depth, beta + search_epsilon))
            return beta + search_epsilon;
        if (mpc_lower(b, skipped, depth, alpha - search_epsilon))
            return alpha - search_epsilon;
    }
    board nb;
    bool passed = true;
    int g, v = -inf;
    bool legal;
    for (int &cell: vacant_lst){
        legal = legal_arr[b->p][b->b[place_included[cell][0]]][local_place[place_included[cell][0]][cell]] || 
                legal_arr[b->p][b->b[place_included[cell][1]]][local_place[place_included[cell][1]][cell]] || 
                legal_arr[b->p][b->b[place_included[cell][2]]][local_place[place_included[cell][2]][cell]];
        if (place_included[cell][3] != -1)
            legal |= legal_arr[b->p][b->b[place_included[cell][3]]][local_place[place_included[cell][3]][cell]];
        if (legal){
            passed = false;
            nb = b->move(cell);
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

int nega_alpha_ordering(board *b, long long strt, bool skipped, int depth, int alpha, int beta){
    ++searched_nodes;
    if (mpc_min_depth <= depth && depth <= mpc_max_depth){
        if (mpc_higher(b, skipped, depth, beta + search_epsilon))
            return beta + search_epsilon;
        if (mpc_lower(b, skipped, depth, alpha - search_epsilon))
            return alpha - search_epsilon;
    }
    if (depth <= 3)
        return nega_alpha(b, skipped, depth, alpha, beta);
    int hash = (int)(b->hash() & search_hash_mask);
    int l, u;
    transpose_table.get_now(b->b, hash, &l, &u);
    if (l != -inf){
        if (l == u)
            return l;
        alpha = max(alpha, l);
        if (alpha >= beta)
            return alpha;
    }
    if (u != -inf){
        beta = min(beta, u);
        if (alpha >= beta)
            return alpha;
    }
    vector<board> nb;
    int canput = 0;
    bool legal;
    for (int &cell: vacant_lst){
        legal = legal_arr[b->p][b->b[place_included[cell][0]]][local_place[place_included[cell][0]][cell]] || 
                legal_arr[b->p][b->b[place_included[cell][1]]][local_place[place_included[cell][1]][cell]] || 
                legal_arr[b->p][b->b[place_included[cell][2]]][local_place[place_included[cell][2]][cell]];
        if (place_included[cell][3] != -1)
            legal |= legal_arr[b->p][b->b[place_included[cell][3]]][local_place[place_included[cell][3]][cell]];
        if (legal){
            nb.emplace_back(b->move(cell));
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
        return -nega_alpha_ordering(&rb, strt, true, depth, -beta, -alpha);
    }
    if (canput >= 2)
        sort(nb.begin(), nb.end());
    int first_alpha = alpha, g, v = -inf;
    for (board &nnb: nb){
        g = -nega_alpha_ordering(&nnb, strt, false, depth - 1, -beta, -alpha);
        if (beta <= g){
            if (l < g)
                transpose_table.reg(b->b, hash, g, u);
            return g;
        }
        alpha = max(alpha, g);
        v = max(v, g);
    }
    if (v <= first_alpha)
        transpose_table.reg(b->b, hash, l, v);
    else
        transpose_table.reg(b->b, hash, v, v);
    return v;
}

int nega_scout(board *b, long long strt, bool skipped, int depth, int alpha, int beta){
    ++searched_nodes;
    if (depth <= 3)
        return nega_alpha(b, skipped, depth, alpha, beta);
    int hash = (int)(b->hash() & search_hash_mask);
    int l, u;
    transpose_table.get_now(b->b, hash, &l, &u);
    if (l != -inf){
        if (l == u)
            return l;
        alpha = max(alpha, l);
        if (alpha >= beta)
            return alpha;
    }
    if (u != -inf){
        beta = min(beta, u);
        if (alpha >= beta)
            return alpha;
    }
    vector<board> nb;
    int canput = 0;
    bool legal;
    for (int &cell: vacant_lst){
        legal = legal_arr[b->p][b->b[place_included[cell][0]]][local_place[place_included[cell][0]][cell]] || 
                legal_arr[b->p][b->b[place_included[cell][1]]][local_place[place_included[cell][1]][cell]] || 
                legal_arr[b->p][b->b[place_included[cell][2]]][local_place[place_included[cell][2]][cell]];
        if (place_included[cell][3] != -1)
            legal |= legal_arr[b->p][b->b[place_included[cell][3]]][local_place[place_included[cell][3]][cell]];
        if (legal){
            nb.emplace_back(b->move(cell));
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
        return -nega_scout(&rb, strt, true, depth, -beta, -alpha);
    }
    if (canput >= 2)
        sort(nb.begin(), nb.end());
    int g = alpha, v = -inf, first_alpha = alpha;
    for (board &nnb: nb){
        if (&nnb - &nb[0]){
            g = -nega_alpha_ordering(&nnb, strt, false, depth - 1, -alpha - search_epsilon, -alpha);
            if (beta <= g){
                if (l < g)
                    transpose_table.reg(b->b, hash, g, u);
                return g;
            }
            v = max(v, g);
        }
        if (alpha <= g){
            g = -nega_scout(&nnb, strt, false, depth - 1, -beta, -g);
            if (beta <= g){
                if (l < g)
                    transpose_table.reg(b->b, hash, g, u);
                return g;
            }
            alpha = max(alpha, g);
            v = max(v, g);
        }
    }
    if (v <= first_alpha)
        transpose_table.reg(b->b, hash, l, v);
    else
        transpose_table.reg(b->b, hash, v, v);
    return v;
}

int mtd(board *b, long long strt, bool skipped, int depth, int l, int u){
    int g = mid_evaluate(b), beta;
    while (u - l > mtd_threshold){
        beta = g;
        g = nega_alpha_ordering(b, strt, skipped, depth, beta - search_epsilon, beta);
        if (g < beta)
            u = g;
        else
            l = g;
        g = (l + u) / 2;
    }
    return nega_scout(b, strt, skipped, depth, l, u);
}