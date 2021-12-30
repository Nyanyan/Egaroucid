#pragma once
#include <iostream>
#include <functional>
#include "setting.hpp"
#include "common.hpp"
#include "board.hpp"
#include "evaluate.hpp"
#include "search.hpp"
#include "transpose_table.hpp"
#include "midsearch.hpp"
#if USE_MULTI_THREAD
    #include "thread_pool.hpp"
#endif

using namespace std;

int nega_alpha_ordering_final_mpc(board *b, bool skipped, int depth, int alpha, int beta, double use_mpct){
    if (depth <= simple_mid_threshold)
        return nega_alpha(b, skipped, depth, alpha, beta);
    ++searched_nodes;
    #if USE_END_SC
        if (stability_cut(b, &alpha, &beta))
            return alpha;
    #endif
    if (mpc_min_depth <= depth && depth <= mpc_max_depth){
        if (mpc_higher(b, skipped, depth, beta, use_mpct))
            return beta;
        if (mpc_lower(b, skipped, depth, alpha, use_mpct))
            return alpha;
    }
    vector<board> nb;
    int canput = 0;
    for (const int &cell: vacant_lst){
        if (b->legal(cell)){
            nb.push_back(b->move(cell));
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
        return -nega_alpha_ordering_final_mpc(&rb, true, depth, -beta, -alpha, use_mpct);
    }
    if (canput >= 2)
        sort(nb.begin(), nb.end());
    int g, v = -inf;
    for (board &nnb: nb){
        g = -nega_alpha_ordering_final_mpc(&nnb, false, depth - 1, -beta, -alpha, use_mpct);
        alpha = max(alpha, g);
        if (beta <= alpha)
            return alpha;
        v = max(v, g);
    }
    return v;
}

inline bool mpc_higher_final(board *b, bool skipped, int depth, int beta, double t){
    int bound = beta + t * mpcsd_final[depth - mpc_min_depth_final];
    if (bound > sc_w)
        return false;
    return nega_alpha_ordering_final_mpc(b, skipped, mpcd[depth], bound - step, bound, t) >= bound;
}

inline bool mpc_lower_final(board *b, bool skipped, int depth, int alpha, double t){
    int bound = alpha - t * mpcsd_final[depth - mpc_min_depth_final];
    if (bound < -sc_w)
        return false;
    return nega_alpha_ordering_final_mpc(b, skipped, mpcd[depth], bound, bound + step, t) <= bound;
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
    #if USE_END_PO
        int p0_parity = (b->parity & cell_div4[p0]);
        int p1_parity = (b->parity & cell_div4[p1]);
        if (!p0_parity && p1_parity)
            swap(p0, p1);
    #endif
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
    #if USE_END_PO
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
    #if USE_END_PO
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
    board nb;
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

inline int last5(board *b, bool skipped, int alpha, int beta, int p0, int p1, int p2, int p3, int p4){
    ++searched_nodes;
    #if USE_END_SC
        if (stability_cut(b, &alpha, &beta))
            return alpha;
    #endif
    #if USE_END_PO
        int p0_parity = (b->parity & cell_div4[p0]);
        int p1_parity = (b->parity & cell_div4[p1]);
        int p2_parity = (b->parity & cell_div4[p2]);
        int p3_parity = (b->parity & cell_div4[p3]);
        int p4_parity = (b->parity & cell_div4[p4]);
        board nb;
        bool passed = true;
        int v = -inf, g;
        if (p0_parity && b->legal(p0)){
            passed = false;
            b->move(p0, &nb);
            g = -last4(&nb, false, -beta, -alpha, p1, p2, p3, p4);
            alpha = max(alpha, g);
            if (beta <= alpha)
                return alpha;
            v = max(v, g);
        }
        if (p1_parity && b->legal(p1)){
            passed = false;
            b->move(p1, &nb);
            g = -last4(&nb, false, -beta, -alpha, p0, p2, p3, p4);
            alpha = max(alpha, g);
            if (beta <= alpha)
                return alpha;
            v = max(v, g);
        }
        if (p2_parity && b->legal(p2)){
            passed = false;
            b->move(p2, &nb);
            g = -last4(&nb, false, -beta, -alpha, p0, p1, p3, p4);
            alpha = max(alpha, g);
            if (beta <= alpha)
                return alpha;
            v = max(v, g);
        }
        if (p3_parity && b->legal(p3)){
            passed = false;
            b->move(p3, &nb);
            g = -last4(&nb, false, -beta, -alpha, p0, p1, p2, p4);
            alpha = max(alpha, g);
            if (beta <= alpha)
                return alpha;
            v = max(v, g);
        }
        if (p4_parity && b->legal(p4)){
            passed = false;
            b->move(p4, &nb);
            g = -last4(&nb, false, -beta, -alpha, p0, p1, p2, p3);
            alpha = max(alpha, g);
            if (beta <= alpha)
                return alpha;
            v = max(v, g);
        }
        if (p0_parity == 0 && b->legal(p0)){
            passed = false;
            b->move(p0, &nb);
            g = -last4(&nb, false, -beta, -alpha, p1, p2, p3, p4);
            alpha = max(alpha, g);
            if (beta <= alpha)
                return alpha;
            v = max(v, g);
        }
        if (p1_parity == 0 && b->legal(p1)){
            passed = false;
            b->move(p1, &nb);
            g = -last4(&nb, false, -beta, -alpha, p0, p2, p3, p4);
            alpha = max(alpha, g);
            if (beta <= alpha)
                return alpha;
            v = max(v, g);
        }
        if (p2_parity == 0 && b->legal(p2)){
            passed = false;
            b->move(p2, &nb);
            g = -last4(&nb, false, -beta, -alpha, p0, p1, p3, p4);
            alpha = max(alpha, g);
            if (beta <= alpha)
                return alpha;
            v = max(v, g);
        }
        if (p3_parity == 0 && b->legal(p3)){
            passed = false;
            b->move(p3, &nb);
            g = -last4(&nb, false, -beta, -alpha, p0, p1, p2, p4);
            alpha = max(alpha, g);
            if (beta <= alpha)
                return alpha;
            v = max(v, g);
        }
        if (p4_parity == 0 && b->legal(p4)){
            passed = false;
            b->move(p4, &nb);
            g = -last4(&nb, false, -beta, -alpha, p0, p1, p2, p3);
            alpha = max(alpha, g);
            if (beta <= alpha)
                return alpha;
            v = max(v, g);
        }
    #else
        board nb;
        bool passed = true;
        int v = -inf, g;
        if (b->legal(p0)){
            passed = false;
            b->move(p0, &nb);
            g = -last4(&nb, false, -beta, -alpha, p1, p2, p3, p4);
            alpha = max(alpha, g);
            if (beta <= alpha)
                return alpha;
            v = max(v, g);
        }
        if (b->legal(p1)){
            passed = false;
            b->move(p1, &nb);
            g = -last4(&nb, false, -beta, -alpha, p0, p2, p3, p4);
            alpha = max(alpha, g);
            if (beta <= alpha)
                return alpha;
            v = max(v, g);
        }
        if (b->legal(p2)){
            passed = false;
            b->move(p2, &nb);
            g = -last4(&nb, false, -beta, -alpha, p0, p1, p3, p4);
            alpha = max(alpha, g);
            if (beta <= alpha)
                return alpha;
            v = max(v, g);
        }
        if (b->legal(p3)){
            passed = false;
            b->move(p3, &nb);
            g = -last4(&nb, false, -beta, -alpha, p0, p1, p2, p4);
            alpha = max(alpha, g);
            if (beta <= alpha)
                return alpha;
            v = max(v, g);
        }
        if (b->legal(p4)){
            passed = false;
            b->move(p4, &nb);
            g = -last4(&nb, false, -beta, -alpha, p0, p1, p2, p3);
            alpha = max(alpha, g);
            if (beta <= alpha)
                return alpha;
            v = max(v, g);
        }
    #endif
    if (passed){
        if (skipped)
            return end_evaluate(b);
        board rb;
        for (int i = 0; i < b_idx_num; ++i)
            rb.b[i] = b->b[i];
        rb.p = 1 - b->p;
        rb.n = b->n;
        return -last5(&rb, true, -beta, -alpha, p0, p1, p2, p3, p4);
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


int nega_alpha_final(board *b, bool skipped, const int depth, int alpha, int beta, int worker_id){
    #if USE_MULTI_THREAD
        if (worker_id != -1){
            if (thread_pool.stop[worker_id])
                return -inf;
        }
    #endif
    if (depth == 5){
        int cells[5];
        pick_vacant(b, cells);
        return last5(b, skipped, alpha, beta, cells[0], cells[1], cells[2], cells[3], cells[4]);
    }
    ++searched_nodes;
    #if USE_END_SC
        if (stability_cut(b, &alpha, &beta))
            return alpha;
    #endif
    board nb;
    bool passed = true;
    int g, v = -inf;
    #if USE_END_PO
        if (0 < b->parity && b->parity < 15){
            for (const int &cell: vacant_lst){
                if ((b->parity & cell_div4[cell]) && b->legal(cell)){
                    passed = false;
                    b->move(cell, &nb);
                    g = -nega_alpha_final(&nb, false, depth - 1, -beta, -alpha, worker_id);
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
                    g = -nega_alpha_final(&nb, false, depth - 1, -beta, -alpha, worker_id);
                    alpha = max(alpha, g);
                    if (beta <= alpha)
                        return alpha;
                    v = max(v, g);
                }
            }
        } else{
            for (const int &cell: vacant_lst){
                if (b->legal(cell)){
                    passed = false;
                    b->move(cell, &nb);
                    g = -nega_alpha_final(&nb, false, depth - 1, -beta, -alpha, worker_id);
                    alpha = max(alpha, g);
                    if (beta <= alpha)
                        return alpha;
                    v = max(v, g);
                }
            }
        }
    #else
        for (const int &cell: vacant_lst){
            if (b->legal(cell)){
                passed = false;
                b->move(cell, &nb);
                g = -nega_alpha_final(&nb, false, depth - 1, -beta, -alpha, worker_id);
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
        return -nega_alpha_final(&nb, true, depth, -beta, -alpha, worker_id);
    }
    return v;
}

int nega_alpha_ordering_final(board *b, bool skipped, const int depth, int alpha, int beta, int use_multi_thread, int worker_id, bool use_mpc, double mpct_in){
    #if USE_MULTI_THREAD
        if (worker_id != -1){
            if (thread_pool.stop[worker_id])
                return -inf;
        }
    #endif
    if (depth <= simple_end_threshold)
        return nega_alpha_final(b, skipped, depth, alpha, beta, worker_id);
    ++searched_nodes;
    #if USE_END_SC
        if (stability_cut(b, &alpha, &beta))
            return alpha;
    #endif
    int hash = (int)(b->hash() & search_hash_mask);
    int l, u;
    transpose_table.get_now(b, hash, &l, &u);
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
        if (mpc_min_depth_final <= depth && depth <= mpc_max_depth_final && use_mpc){
            if (mpc_higher_final(b, skipped, depth, beta, mpct_in))
                return beta;
            if (mpc_lower_final(b, skipped, depth, alpha, mpct_in))
                return alpha;
        }
    #endif
    vector<board> nb;
    int canput = 0;
    for (const int &cell: vacant_lst){
        if (b->legal(cell)){
            nb.emplace_back(b->move(cell));
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
        int res = -nega_alpha_ordering_final(&rb, true, depth, -beta, -alpha, use_multi_thread, worker_id, use_mpc, mpct_in);
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
    int g, v = -inf;
    #if USE_MULTI_THREAD
        if (use_multi_thread > 0){
            int i, j;
            for (i = 0; i < min(canput, ybwc_end_first_num); ++i){
                g = -nega_alpha_ordering_final(&nb[i], false, depth - 1, -beta, -alpha, use_multi_thread, -1);
                alpha = max(alpha, g);
                if (beta <= alpha){
                    if (l < g)
                        transpose_table.reg(b, hash, g, u);
                    return alpha;
                }
                v = max(v, g);
            }
            int task_ids[canput];
            for (i = ybwc_end_first_num; i < canput; ++i){
                task_ids[i] = thread_pool.get_worker_id();
                thread_pool.push_id(bind(&nega_alpha_ordering_final, &nb[i], false, depth - 1, -beta, -alpha, use_multi_thread - 1, task_ids[i]), task_ids[i]);
            }
            bool got[canput];
            for (i = ybwc_end_first_num; i < canput; ++i)
                got[i] = false;
            int n_got = ybwc_end_first_num;
            while (n_got < canput){
                for (i = ybwc_end_first_num; i < canput; ++i){
                    if (!got[i]){
                        if (thread_pool.get_check(task_ids[i], &g)){
                            g *= -1;
                            alpha = max(alpha, g);
                            v = max(v, g);
                            got[i] = true;
                            ++n_got;
                            #if USE_YBWC_END_EARLY_CUT
                                if (alpha >= beta){
                                    for (j = ybwc_end_first_num; j < canput; ++j){
                                        if (!got[j])
                                            thread_pool.stop[task_ids[j]] = true;
                                    }
                                    for (j = ybwc_end_first_num; j < canput; ++j){
                                        if (!got[j])
                                            thread_pool.get(task_ids[j]);
                                    }
                                    if (l < alpha)
                                        transpose_table.reg(b, hash, g, u);
                                    return alpha;
                                }
                            #endif
                        }
                    }
                }
            }
            if (beta <= alpha){
                if (l < alpha)
                    transpose_table.reg(b, hash, alpha, u);
                return alpha;
            }
        } else{
            for (int i = 0; i < canput; ++i){
                g = -nega_alpha_ordering_final(&nb[i], false, depth - 1, -beta, -alpha, use_multi_thread, -1);
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
            g = -nega_alpha_ordering_final(&nnb, false, depth - 1, -beta, -alpha, 0, -1, use_mpc, mpct_in);
            alpha = max(alpha, g);
            if (beta <= alpha){
                if (l < g)
                    transpose_table.reg(b, hash, g, u);
                return alpha;
            }
            v = max(v, g);
        }
    #endif
    if (v <= alpha)
        transpose_table.reg(b, hash, l, v);
    else
        transpose_table.reg(b, hash, v, v);
    return v;
}

int nega_scout_final(board *b, bool skipped, const int depth, int alpha, int beta, bool use_mpc, double mpct_in){
    if (depth <= simple_end_threshold)
        return nega_alpha_final(b, skipped, depth, alpha, beta, -1);
    if (beta - alpha <= step)
        return nega_alpha_ordering_final(b, skipped, depth, alpha, beta, multi_thread_depth, -1, use_mpc, mpct_in);
    ++searched_nodes;
    #if USE_END_SC
        if (stability_cut(b, &alpha, &beta))
            return alpha;
    #endif
    int hash = (int)(b->hash() & search_hash_mask);
    int l, u;
    transpose_table.get_now(b, hash, &l, &u);
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
        if (mpc_min_depth_final <= depth && depth <= mpc_max_depth_final && use_mpc){
            if (mpc_higher_final(b, skipped, depth, beta, mpct_in))
                return beta;
            if (mpc_lower_final(b, skipped, depth, alpha, mpct_in))
                return alpha;
        }
    #endif
    vector<board> nb;
    int canput = 0;
    for (const int &cell: vacant_lst){
        if (b->legal(cell)){
            nb.push_back(b->move(cell));
            move_ordering_eval(&nb[canput]);
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
        int res = -nega_scout_final(&rb, true, depth, -beta, -alpha, use_mpc, mpct_in);
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
    int g = alpha, v = -inf;
    #if USE_MULTI_THREAD
        int i;
        for (i = 0; i < min(canput, ybwc_end_first_num); ++i){
            g = -nega_scout_final(&nb[i], false, depth - 1, -beta, -alpha);
            alpha = max(alpha, g);
            if (beta <= alpha){
                if (l < g)
                    transpose_table.reg(b, hash, g, u);
                return alpha;
            }
            v = max(v, g);
        }
        int first_alpha = alpha;
        int task_ids[canput];
        for (i = ybwc_end_first_num; i < canput; ++i){
            task_ids[i] = thread_pool.get_worker_id();
            thread_pool.push_id(bind(&nega_alpha_ordering_final, &nb[i], false, depth - 1, -alpha - step, -alpha, multi_thread_depth, task_ids[i]), task_ids[i]);
        }
        bool re_search[canput];
        bool got[canput];
        for (i = ybwc_end_first_num; i < canput; ++i)
            got[i] = false;
        int n_got = ybwc_end_first_num;
        while (n_got < canput){
            for (i = ybwc_end_first_num; i < canput; ++i){
                if (!got[i]){
                    if (thread_pool.get_check(task_ids[i], &g)){
                        g *= -1;
                        alpha = max(alpha, g);
                        v = max(v, g);
                        got[i] = true;
                        ++n_got;
                        re_search[i] = first_alpha < g;
                    }
                }
            }
        }
        for (i = ybwc_end_first_num; i < canput; ++i){
            if (re_search[i]){
                g = -nega_scout_final(&nb[i], false, depth - 1, -beta, -alpha);
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
                g = -nega_alpha_ordering_final(&nb[i], false, depth - 1, -alpha - step, -alpha, 0, -1, use_mpc, mpct_in);
                if (beta <= g){
                    if (l < g)
                        transpose_table.reg(b, hash, g, u);
                    return g;
                }
                v = max(v, g);
            }
            if (alpha < g || i == 0){
                g = -nega_scout_final(&nb[i], false, depth - 1, -beta, -g, use_mpc, mpct_in);
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
    if (v <= alpha)
        transpose_table.reg(b, hash, l, v);
    else
        transpose_table.reg(b, hash, v, v);
    return v;
}

inline search_result endsearch(board b, long long strt){
    vector<board> nb;
    vector<int> prev_vals;
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
    int max_depth = hw2 - b.n - 1;
    bool use_mpc = max_depth >= 21 ? true : false;
    double use_mpct = 1.7;
    if (max_depth >= 23)
        use_mpct = 1.1;
    if (max_depth >= 25)
        use_mpct = 0.9;
    if (max_depth >= 27)
        use_mpct = 0.6;
    if (max_depth >= 29)
        use_mpct = 0.4;
    if (max_depth >= 31)
        use_mpct = 0.3;
    if (max_depth >= 33)
        use_mpct = 0.2;
    int pre_search_depth = min(12, max_depth - simple_end_threshold);
    if (pre_search_depth > 0)
        midsearch(b, strt, pre_search_depth);
    transpose_table.init_now();
    cerr << "start final search depth " << max_depth + 1 << endl;
    alpha = -sc_w;
    beta = sc_w;
    for (i = 0; i < canput; ++i){
        //nb[i].v = -nega_scout(&nb[i], false, 10, -sc_w, sc_w);
        //move_ordering_eval(&nb[i]);
        move_ordering(&nb[i]);
    }
    if (canput >= 2)
        sort(nb.begin(), nb.end());
    //for (i = 0; i < canput; ++i)
    //    prev_vals.push_back(nega_scout(&nb[i], false, min(8, max_depth / 2), -sc_w, sc_w));
    transpose_table.init_now();
    long long final_strt = tim();
    searched_nodes = 0;
    if (nb[0].n < hw2 - 5){
        alpha = -nega_scout_final(&nb[0], false, max_depth, -beta, -alpha, use_mpc, use_mpct);
        //alpha = -mtd_final(&nb[0], false, max_depth, -beta, -alpha, prev_vals[0]);
        tmp_policy = nb[0].policy;
        for (i = 1; i < canput; ++i){
            g = -nega_alpha_ordering_final(&nb[i], false, max_depth, -alpha - step, -alpha, multi_thread_depth, -1, use_mpc, use_mpct);
            if (alpha < g){
                g = -nega_scout_final(&nb[i], false, max_depth, -beta, -g, use_mpc, use_mpct);
                //g = -mtd_final(&nb[i], false, max_depth, -beta, -g, prev_vals[i]);
                if (alpha < g){
                    alpha = g;
                    tmp_policy = nb[i].policy;
                }
            }
        }
    } else{
        int cells[5];
        for (i = 0; i < canput; ++i){
            pick_vacant(&nb[i], cells);
            if (nb[i].n == hw2 - 5)
                g = -last5(&nb[i], false, -beta, -alpha, cells[0], cells[1], cells[2], cells[3], cells[4]);
            else if (nb[i].n == hw2 - 4)
                g = -last4(&nb[i], false, -beta, -alpha, cells[0], cells[1], cells[2], cells[3]);
            else if (nb[i].n == hw2 - 3)
                g = -last3(&nb[i], false, -beta, -alpha, cells[0], cells[1], cells[2]);
            else if (nb[i].n == hw2 - 2)
                g = -last2(&nb[i], false, -beta, -alpha, cells[0], cells[1]);
            else if (nb[i].n == hw2 - 1)
                g = -last1(&nb[i], false, cells[0]);
            else
                g = -end_evaluate(&nb[i]);
            if (alpha < g){
                alpha = g;
                tmp_policy = nb[i].policy;
            }
        }
    }
    swap(transpose_table.now, transpose_table.prev);
    policy = tmp_policy;
    value = alpha;
    cerr << "final depth: " << max_depth + 1 << " time: " << tim() - strt << " policy: " << policy << " value: " << alpha << " nodes: " << searched_nodes << " nps: " << (long long)searched_nodes * 1000 / max(1LL, tim() - final_strt) << " get: " << transpose_table.hash_get << " reg: " << transpose_table.hash_reg << endl;
    search_result res;
    res.policy = policy;
    res.value = value;
    res.depth = max_depth;
    res.nps = searched_nodes * 1000 / max(1LL, tim() - strt);
    #if USE_MULTI_THREAD
        cerr << "worker_size " << thread_pool.get_worker_size() << endl;
    #endif
    return res;
}