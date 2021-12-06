#pragma once
#include <iostream>
#include <thread>
#include <future>
#include <functional>
#include "setting.hpp"
#include "common.hpp"
#include "board.hpp"
#include "evaluate.hpp"
#include "search_common.hpp"
#include "midsearch.hpp"
#if USE_MULTI_THREAD
    #include "multi_threading.hpp"
#endif

using namespace std;

int nega_alpha_ordering_nompc(board *b, bool skipped, int depth, int alpha, int beta){
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
            #if USE_END_SC
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
    int before_score = (b->p ? -1 : 1) * (
        count_black_arr[b->b[0]] + count_black_arr[b->b[1]] + count_black_arr[b->b[2]] + count_black_arr[b->b[3]] + 
        count_black_arr[b->b[4]] + count_black_arr[b->b[5]] + count_black_arr[b->b[6]] + count_black_arr[b->b[7]]);
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
        for (int i = 0; i < b_idx_num; ++i)
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
        b->move(p0, &nb);
        g = -last1(&nb, false, p1);
        alpha = max(alpha, g);
        if (beta <= alpha)
            return alpha;
        v = max(v, g);
    }
    if (b->legal(p1)){
        passed = false;
        b->move(p1, &nb);
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
    #if 0
        int p0_parity = (b->parity & cell_div4[p0]);
        int p1_parity = (b->parity & cell_div4[p1]);
        int p2_parity = (b->parity & cell_div4[p2]);
        if (!p0_parity && p1_parity && p2_parity){
            int tmp = p0;
            p0 = p1;
            p1 = p2;
            p2 = tmp;
        } else if (p0_parity && !p1_parity && p2_parity){
            swap(p1, p2);
        } else if (!p0_parity && !p1_parity && p2_parity){
            int tmp = p0;
            p0 = p2;
            p2 = p1;
            p1 = tmp;
        } else if (!p0_parity && p1_parity && !p2_parity){
            swap(p0, p1);
        }
    #endif
    board nb;
    bool passed = true;
    int v = -inf, g;
    if (b->legal(p0)){
        passed = false;
        b->move(p0, &nb);
        g = -last2(&nb, false, -beta, -alpha, p1, p2);
        alpha = max(alpha, g);
        if (beta <= alpha)
            return alpha;
        v = max(v, g);
    }
    if (b->legal(p1)){
        passed = false;
        b->move(p1, &nb);
        g = -last2(&nb, false, -beta, -alpha, p0, p2);
        alpha = max(alpha, g);
        if (beta <= alpha)
            return alpha;
        v = max(v, g);
    }
    if (b->legal(p2)){
        passed = false;
        b->move(p2, &nb);
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

inline int last4(board *b, bool skipped, int alpha, int beta, int p0, int p1, int p2, int p3){
    ++searched_nodes;
    board nb;
    #if 0
        int p0_parity = (b->parity & cell_div4[p0]);
        int p1_parity = (b->parity & cell_div4[p1]);
        int p2_parity = (b->parity & cell_div4[p2]);
        int p3_parity = (b->parity & cell_div4[p3]);
        if (!p0_parity && p1_parity && p2_parity && p3_parity){
            int tmp = p0;
            p0 = p1;
            p1 = p2;
            p2 = p3;
            p3 = tmp;
        } else if (p0_parity && !p1_parity && p2_parity && p3_parity){
            int tmp = p1;
            p1 = p2;
            p2 = p3;
            p3 = tmp;
        } else if (p0_parity && p1_parity && !p2_parity && p3_parity){
            swap(p2, p3);
        } else if (!p0_parity && !p1_parity && p2_parity && p3_parity){
            swap(p0, p2);
            swap(p1, p3);
        } else if (!p0_parity && p1_parity && !p2_parity && p3_parity){
            int tmp = p0;
            p0 = p1;
            p1 = p3;
            p3 = p2;
            p2 = tmp;
        } else if (!p0_parity && p1_parity && p2_parity && !p3_parity){
            int tmp = p0;
            p0 = p1;
            p1 = p2;
            p2 = tmp;
        } else if (p0_parity && !p1_parity && !p2_parity && p3_parity){
            int tmp = p1;
            p1 = p3;
            p3 = p2;
            p2 = tmp;
        } else if (p0_parity && !p1_parity && p2_parity && !p3_parity){
            swap(p1, p2);
        } else if (!p0_parity && !p1_parity && !p2_parity && p3_parity){
            int tmp = p0;
            p0 = p3;
            p3 = p2;
            p2 = p1;
            p1 = tmp;
        } else if (!p0_parity && !p1_parity && p2_parity && !p3_parity){
            int tmp = p0;
            p0 = p2;
            p2 = p1;
            p1 = tmp;
        } else if (!p0_parity && p1_parity && !p2_parity && !p3_parity){
            swap(p0, p1);
        }
    #endif
    bool passed = true;
    int v = -inf, g;
    if (b->legal(p0)){
        passed = false;
        b->move(p0, &nb);
        g = -last3(&nb, false, -beta, -alpha, p1, p2, p3);
        alpha = max(alpha, g);
        if (beta <= alpha)
            return alpha;
        v = max(v, g);
    }
    if (b->legal(p1)){
        passed = false;
        b->move(p1, &nb);
        g = -last3(&nb, false, -beta, -alpha, p0, p2, p3);
        alpha = max(alpha, g);
        if (beta <= alpha)
            return alpha;
        v = max(v, g);
    }
    if (b->legal(p2)){
        passed = false;
        b->move(p2, &nb);
        g = -last3(&nb, false, -beta, -alpha, p0, p1, p3);
        alpha = max(alpha, g);
        if (beta <= alpha)
            return alpha;
        v = max(v, g);
    }
    if (b->legal(p3)){
        passed = false;
        b->move(p3, &nb);
        g = -last3(&nb, false, -beta, -alpha, p0, p1, p2);
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
        return -last4(&rb, true, -beta, -alpha, p0, p1, p2, p3);
    }
    return v;
}

inline void pick_vacant(board *b, int cells[]){
    int idx = 0;
    for (const int &cell: vacant_lst){
        if (pop_digit[b->b[cell / hw]][cell % hw] == vacant)
            cells[idx++] = cell;
    }
}

int nega_alpha_final(board *b, bool skipped, const int depth, int alpha, int beta){
    if (b->n == hw2 - 4){
        int cells[5];
        pick_vacant(b, cells);
        return last4(b, skipped, alpha, beta, cells[0], cells[1], cells[2], cells[3]);
    }
    ++searched_nodes;
    board nb;
    bool passed = true;
    int g, v = -inf;
    #if USE_END_PO
        for (const int &cell: vacant_lst){
            if ((b->parity & cell_div4[cell]) && b->legal(cell)){
                passed = false;
                b->move(cell, &nb);
                //#if USE_END_SC
                //    if (stability_cut(&nb, &alpha, &beta))
                //        return alpha;
                //#endif
                g = -nega_alpha_final(&nb, false, depth - 1, -beta, -alpha);
                alpha = max(alpha, g);
                if (beta <= alpha)
                    return alpha;
                v = max(v, g);
            }
        }
        for (const int &cell: vacant_lst){
            if (!(b->parity & cell_div4[cell]) && b->legal(cell)){
                passed = false;
                b->move(cell, &nb);
                //#if USE_END_SC
                //    if (stability_cut(&nb, &alpha, &beta))
                //        return alpha;
                //#endif
                g = -nega_alpha_final(&nb, false, depth - 1, -beta, -alpha);
                alpha = max(alpha, g);
                if (beta <= alpha)
                    return alpha;
                v = max(v, g);
            }
        }
    #else
        for (const int &cell: vacant_lst){
            if (b->legal(cell)){
                passed = false;
                b->move(cell, &nb);
                #if USE_END_SC
                    if (stability_cut(&nb, &alpha, &beta))
                        return alpha;
                #endif
                g = -nega_alpha_final(&nb, false, depth - 1, -beta, -alpha);
                alpha = max(alpha, g);
                if (beta <= alpha)
                    return alpha;
                v = max(v, g);
            }
        }
    #endif
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

int nega_alpha_ordering_final(board *b, bool skipped, const int depth, int alpha, int beta, bool use_multi_thread){
    if (depth <= simple_end_threshold)
        return nega_alpha_final(b, skipped, depth, alpha, beta);
    ++searched_nodes;
    int hash = (int)(b->hash() & search_hash_mask);
    int l, u;
    transpose_table.get_now(b, b->hash() & search_hash_mask, &l, &u);
    #if USE_END_TC
        if (l >= beta)
            return l;
        if (alpha >= u)
            return u;
        if (u == l)
            return u;
    #endif
    alpha = max(alpha, l);
    beta = min(beta, u);
    #if USE_END_MPC
        if (mpc_min_depth_final <= depth && depth <= mpc_max_depth_final){
            if (mpc_higher_final(b, skipped, depth, beta))
                return beta;
            if (mpc_lower_final(b, skipped, depth, alpha))
                return alpha;
        }
    #endif
    board nb[depth];
    int canput = 0;
    for (const int &cell: vacant_lst){
        if (b->legal(cell)){
            b->move(cell, &nb[canput]);
            #if USE_END_SC
                if (stability_cut(&nb[canput], &alpha, &beta))
                    return alpha;
            #endif
            //move_ordering(&nb[canput]);
            nb[canput].v = -canput_bonus * calc_canput_exact(&nb[canput]);
            #if USE_END_PO
                if (depth <= po_max_depth && b->parity & cell_div4[cell])
                    nb[canput].v += parity_vacant_bonus;
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
        int res = -nega_alpha_ordering_final(&rb, true, depth, -beta, -alpha, use_multi_thread);
        if (res >= beta)
            transpose_table.reg(b, hash, res, u);
        else if (res <= alpha)
            transpose_table.reg(b, hash, l, res);
        else
            transpose_table.reg(b, hash, res, res);
        return res;
    }
    if (canput >= 2)
        sort(nb, nb + canput);
    int g, v = -inf, first_alpha = alpha;
    #if USE_MULTI_THREAD
        if (use_multi_thread){
            int i;
            for (i = 0; i < min(canput, multi_thread_single_num); ++i){
                g = -nega_alpha_ordering_final(&nb[i], false, depth - 1, -beta, -alpha, true);
                alpha = max(alpha, g);
                if (beta <= alpha){
                    if (l < g)
                        transpose_table.reg(b, hash, g, u);
                    return alpha;
                }
                v = max(v, g);
            }
            int task_ids[depth];
            for (i = multi_thread_single_num; i < canput; ++i)
                task_ids[i] = thread_pool.push(bind(&nega_alpha_ordering_final, &nb[i], false, depth - 1, -beta, -alpha, false));
            int done_tasks = multi_thread_single_num;
            bool done[depth];
            for (i = multi_thread_single_num; i < canput; ++i)
                done[i] = false;
            while (done_tasks < canput){
                for (i = multi_thread_single_num; i < canput; ++i){
                    if (!done[i]){
                        done[i] = thread_pool.get(task_ids[i], &g);
                        g = -g;
                        if (done[i]){
                            ++done_tasks;
                            alpha = max(alpha, g);
                            v = max(v, g);
                        }
                    }
                }
            }
            if (beta <= alpha){
                if (l < g)
                    transpose_table.reg(b, hash, g, u);
                return alpha;
            }
        } else{
            for (int i = 0; i < canput; ++i){
                g = -nega_alpha_ordering_final(&nb[i], false, depth - 1, -beta, -alpha, false);
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
            g = -nega_alpha_ordering_final(&nb[i], false, depth - 1, -beta, -alpha, false);
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

int nega_scout_final(board *b, bool skipped, int depth, int alpha, int beta){
    if (depth <= simple_end_threshold)
        return nega_alpha_final(b, skipped, depth, alpha, beta);
    if (beta - alpha <= step)
        return nega_alpha_ordering_final(b, skipped, depth, alpha, beta, true);
    ++searched_nodes;
    int hash = (int)(b->hash() & search_hash_mask);
    int l, u;
    transpose_table.get_now(b, b->hash() & search_hash_mask, &l, &u);
    #if USE_END_TC
        if (l >= beta)
            return l;
        if (alpha >= u)
            return u;
        if (u == l)
            return u;
    #endif
    alpha = max(alpha, l);
    beta = min(beta, u);
    int first_alpha = alpha;
    #if USE_END_MPC
        if (mpc_min_depth_final <= depth && depth <= mpc_max_depth_final){
            if (mpc_higher_final(b, skipped, depth, beta))
                return beta;
            if (mpc_lower_final(b, skipped, depth, alpha))
                return alpha;
        }
    #endif
    board nb[depth];
    int canput = 0;
    for (const int &cell: vacant_lst){
        if (b->legal(cell)){
            b->move(cell, &nb[canput]);
            #if USE_END_SC
                if (stability_cut(&nb[canput], &alpha, &beta))
                    return alpha;
            #endif
            move_ordering(&nb[canput]);
            nb[canput].v -= canput_bonus * calc_canput_exact(&nb[canput]);
            #if USE_END_PO
                if (depth <= po_max_depth && b->parity & cell_div4[cell])
                    nb[canput].v += parity_vacant_bonus;
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
        sort(nb, nb + canput);
    int g = alpha, v = -inf;
    for (int i = 0; i < canput; ++i){
        if (i > 0){
            g = -nega_alpha_ordering_final(&nb[i], false, depth - 1, -alpha - step, -alpha, true);
            if (beta <= g){
                if (l < g)
                    transpose_table.reg(b, hash, g, u);
                return g;
            }
            v = max(v, g);
        }
        if (alpha <= g){
            g = -nega_scout_final(&nb[i], false, depth - 1, -beta, -g);
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
    int max_depth = hw2 - b.n - 1;
    int pre_search_depth = min(17, max_depth - simple_end_threshold - 2);
    transpose_table.init_now();
    transpose_table.init_prev();
    if (pre_search_depth > 0)
        midsearch(b, strt, pre_search_depth);
    cerr << "start final search depth " << max_depth + 1 << endl;
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
    //alpha = -mtd_final(&nb[0], false, max_depth, -beta, -alpha);
    tmp_policy = nb[0].policy;
    for (i = 1; i < canput; ++i){
        g = -nega_alpha_ordering_final(&nb[i], false, max_depth, -alpha - step, -alpha, true);
        if (alpha < g){
            g = -nega_scout_final(&nb[i], false, max_depth, -beta, -g);
            //g = -mtd_final(&nb[i], false, max_depth, -beta, -g);
            if (alpha <= g){
                alpha = g;
                tmp_policy = nb[i].policy;
            }
        }
    }
    swap(transpose_table.now, transpose_table.prev);
    policy = tmp_policy;
    value = alpha;
    cerr << "final depth: " << max_depth + 1 << " time: " << tim() - strt << " policy: " << policy << " value: " << alpha << " nodes: " << searched_nodes << " nps: " << (long long)searched_nodes * 1000 / max(1LL, tim() - strt) << " get: " << transpose_table.hash_get << " reg: " << transpose_table.hash_reg << endl;
    search_result res;
    res.policy = policy;
    res.value = value;
    res.depth = max_depth;
    res.nps = searched_nodes * 1000 / max(1LL, tim() - strt);
    return res;
}