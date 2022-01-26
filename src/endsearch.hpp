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
#include "level.hpp"
#if USE_MULTI_THREAD
    #include "thread_pool.hpp"
#endif

using namespace std;

int nega_alpha_ordering_final_nomemo(board *b, bool skipped, int depth, int alpha, int beta, bool use_mpc, double use_mpct){
    if (!global_searching)
        return -inf;
    if (depth <= simple_mid_threshold)
        return nega_alpha(b, skipped, depth, alpha, beta);
    search_statistics.nodes_increment();
    #if USE_END_SC
        if (stability_cut(b, &alpha, &beta))
            return alpha;
    #endif
    #if USE_END_MPC
        if (mpc_min_depth <= depth && depth <= mpc_max_depth && use_mpc){
            if (mpc_higher(b, skipped, depth, beta, use_mpct))
                return beta;
            if (mpc_lower(b, skipped, depth, alpha, use_mpct))
                return alpha;
        }
    #endif
    vector<board> nb;
    int canput = 0;
    for (const int &cell: vacant_lst){
        if (b->legal(cell)){
            nb.push_back(b->move(cell));
            move_ordering(&nb[canput]);
            //nb[canput].v = -canput_bonus * calc_canput_exact(&nb[canput]);
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
        return -nega_alpha_ordering_final_nomemo(&rb, true, depth, -beta, -alpha, use_mpc, use_mpct);
    }
    if (canput >= 2)
        sort(nb.begin(), nb.end());
    int g, v = -inf;
    for (board &nnb: nb){
        g = -nega_alpha_ordering_final_nomemo(&nnb, false, depth - 1, -beta, -alpha, use_mpc, use_mpct);
        alpha = max(alpha, g);
        if (beta <= alpha)
            return alpha;
        v = max(v, g);
    }
    return v;
}

inline bool mpc_higher_final(board *b, bool skipped, int depth, int beta, double t){
    if (b->n + mpcd[depth] >= hw2 - 5)
        return false;
    int bound = beta + ceil(t * mpcsd_final[depth - mpc_min_depth_final]);
    if (bound > hw2)
        bound = hw2; //return false;
    return nega_alpha_ordering_final_nomemo(b, skipped, mpcd[depth], bound - search_epsilon, bound, true, t) >= bound;
}

inline bool mpc_lower_final(board *b, bool skipped, int depth, int alpha, double t){
    if (b->n + mpcd[depth] >= hw2 - 5)
        return false;
    int bound = alpha - ceil(t * mpcsd_final[depth - mpc_min_depth_final]);
    if (bound < -hw2)
        bound = -hw2; //return false;
    return nega_alpha_ordering_final_nomemo(b, skipped, mpcd[depth], bound, bound + search_epsilon, true, t) <= bound;
}

inline int last1(board *b, int p0){
    search_statistics.nodes_increment();
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
        score = before_score - 1 - (
            move_arr[1 - b->p][b->b[place_included[p0][0]]][local_place[place_included[p0][0]][p0]][0] + move_arr[1 - b->p][b->b[place_included[p0][0]]][local_place[place_included[p0][0]][p0]][1] + 
            move_arr[1 - b->p][b->b[place_included[p0][1]]][local_place[place_included[p0][1]][p0]][0] + move_arr[1 - b->p][b->b[place_included[p0][1]]][local_place[place_included[p0][1]][p0]][1] + 
            move_arr[1 - b->p][b->b[place_included[p0][2]]][local_place[place_included[p0][2]][p0]][0] + move_arr[1 - b->p][b->b[place_included[p0][2]]][local_place[place_included[p0][2]][p0]][1]) * 2;
        if (place_included[p0][3] != -1)
            score += (move_arr[1 - b->p][b->b[place_included[p0][3]]][local_place[place_included[p0][3]][p0]][0] + move_arr[1 - b->p][b->b[place_included[p0][3]]][local_place[place_included[p0][3]][p0]][1]) * 2;
        if (score == before_score - 1){
            if (before_score > 0)
                score = before_score + 1;
            else
                score = before_score - 1;
        }
    }
    return score;
}

inline int last2(board *b, int alpha, int beta, int p0, int p1){
    search_statistics.nodes_increment();
    #if USE_END_PO
        int p0_parity = (b->parity & cell_div4[p0]);
        int p1_parity = (b->parity & cell_div4[p1]);
        if (!p0_parity && p1_parity)
            swap(p0, p1);
    #endif
    board nb;
    int v = -inf, g;
    if (b->legal(p0)){
        b->move(p0, &nb);
        g = -last1(&nb, p1);
        alpha = max(alpha, g);
        if (beta <= alpha)
            return alpha;
        v = max(v, g);
    }
    if (b->legal(p1)){
        b->move(p1, &nb);
        g = -last1(&nb, p0);
        alpha = max(alpha, g);
        if (beta <= alpha)
            return alpha;
        v = max(v, g);
    }
    if (v == -inf){
        b->p = 1 - b->p;
        if (b->legal(p0)){
            b->move(p0, &nb);
            g = last1(&nb, p1);
            alpha = max(alpha, g);
            if (beta <= alpha){
                b->p = 1 - b->p;
                return alpha;
            }
            v = max(v, g);
        }
        if (b->legal(p1)){
            b->move(p1, &nb);
            g = last1(&nb, p0);
            alpha = max(alpha, g);
            if (beta <= alpha){
                b->p = 1 - b->p;
                return alpha;
            }
            v = max(v, g);
        }
        b->p = 1 - b->p;
        if (v == -inf)
            v = end_evaluate(b);
    }
    return v;
}

inline int last3(board *b, bool skipped, int alpha, int beta, int p0, int p1, int p2){
    search_statistics.nodes_increment();
    #if USE_END_PO
        int p0_parity = (b->parity & cell_div4[p0]);
        int p1_parity = (b->parity & cell_div4[p1]);
        int p2_parity = (b->parity & cell_div4[p2]);
        if (p0_parity == 0 && p1_parity && p2_parity){
            int tmp = p0;
            p0 = p1;
            p1 = p2;
            p2 = tmp;
        } else if (p0_parity && p1_parity == 0 && p2_parity){
            swap(p1, p2);
        } else if (p0_parity == 0 && p1_parity == 0 && p2_parity){
            int tmp = p0;
            p0 = p2;
            p2 = p1;
            p1 = tmp;
        } else if (p0_parity == 0 && p1_parity && p2_parity == 0){
            swap(p0, p1);
        }
    #endif
    board nb;
    int v = -inf, g;
    if (b->legal(p0)){
        b->move(p0, &nb);
        g = -last2(&nb, -beta, -alpha, p1, p2);
        alpha = max(alpha, g);
        if (beta <= alpha)
            return alpha;
        v = max(v, g);
    }
    if (b->legal(p1)){
        b->move(p1, &nb);
        g = -last2(&nb, -beta, -alpha, p0, p2);
        alpha = max(alpha, g);
        if (beta <= alpha)
            return alpha;
        v = max(v, g);
    }
    if (b->legal(p2)){
        b->move(p2, &nb);
        g = -last2(&nb, -beta, -alpha, p0, p1);
        alpha = max(alpha, g);
        if (beta <= alpha)
            return alpha;
        v = max(v, g);
    }
    if (v == -inf){
        if (skipped)
            v = end_evaluate(b);
        else{
            b->p = 1 - b->p;
            v = -last3(b, true, -beta, -alpha, p0, p1, p2);
            b->p = 1 - b->p;
        }
    }
    return v;
}

inline int last4(board *b, bool skipped, int alpha, int beta, int p0, int p1, int p2, int p3){
    search_statistics.nodes_increment();
    #if USE_END_SC
        if (stability_cut(b, &alpha, &beta))
            return alpha;
    #endif
    #if USE_END_PO
        if (!skipped){
            int p0_parity = (b->parity & cell_div4[p0]);
            int p1_parity = (b->parity & cell_div4[p1]);
            int p2_parity = (b->parity & cell_div4[p2]);
            int p3_parity = (b->parity & cell_div4[p3]);
            if (p0_parity == 0 && p1_parity && p2_parity && p3_parity){
                int tmp = p0;
                p0 = p1;
                p1 = p2;
                p2 = p3;
                p3 = tmp;
            } else if (p0_parity && p1_parity == 0 && p2_parity && p3_parity){
                int tmp = p1;
                p1 = p2;
                p2 = p3;
                p3 = tmp;
            } else if (p0_parity && p1_parity && p2_parity == 0 && p3_parity){
                swap(p2, p3);
            } else if (p0_parity == 0 && p1_parity == 0 && p2_parity && p3_parity){
                swap(p0, p2);
                swap(p1, p3);
            } else if (p0_parity == 0 && p1_parity && p2_parity == 0 && p3_parity){
                int tmp = p0;
                p0 = p1;
                p1 = p3;
                p3 = p2;
                p2 = tmp;
            } else if (p0_parity == 0 && p1_parity && p2_parity && p3_parity == 0){
                int tmp = p0;
                p0 = p1;
                p1 = p2;
                p2 = tmp;
            } else if (p0_parity && p1_parity == 0 && p2_parity == 0 && p3_parity){
                int tmp = p1;
                p1 = p3;
                p3 = p2;
                p2 = tmp;
            } else if (p0_parity && p1_parity == 0 && p2_parity && p3_parity == 0){
                swap(p1, p2);
            } else if (p0_parity == 0 && p1_parity == 0 && p2_parity == 0 && p3_parity){
                int tmp = p0;
                p0 = p3;
                p3 = p2;
                p2 = p1;
                p1 = tmp;
            } else if (p0_parity == 0 && p1_parity == 0 && p2_parity && p3_parity == 0){
                int tmp = p0;
                p0 = p2;
                p2 = p1;
                p1 = tmp;
            } else if (p0_parity == 0 && p1_parity && p2_parity == 0 && p3_parity == 0){
                swap(p0, p1);
            }
        }
    #endif
    board nb;
    int v = -inf, g;
    if (b->legal(p0)){
        b->move(p0, &nb);
        g = -last3(&nb, false, -beta, -alpha, p1, p2, p3);
        alpha = max(alpha, g);
        if (beta <= alpha)
            return alpha;
        v = max(v, g);
    }
    if (b->legal(p1)){
        b->move(p1, &nb);
        g = -last3(&nb, false, -beta, -alpha, p0, p2, p3);
        alpha = max(alpha, g);
        if (beta <= alpha)
            return alpha;
        v = max(v, g);
    }
    if (b->legal(p2)){
        b->move(p2, &nb);
        g = -last3(&nb, false, -beta, -alpha, p0, p1, p3);
        alpha = max(alpha, g);
        if (beta <= alpha)
            return alpha;
        v = max(v, g);
    }
    if (b->legal(p3)){
        b->move(p3, &nb);
        g = -last3(&nb, false, -beta, -alpha, p0, p1, p2);
        alpha = max(alpha, g);
        if (beta <= alpha)
            return alpha;
        v = max(v, g);
    }
    if (v == -inf){
        if (skipped)
            v = end_evaluate(b);
        else{
            b->p = 1 - b->p;
            v = -last4(b, true, -beta, -alpha, p0, p1, p2, p3);
            b->p = 1 - b->p;
        }
    }
    return v;
}

inline int last5(board *b, bool skipped, int alpha, int beta, int p0, int p1, int p2, int p3, int p4){
    search_statistics.nodes_increment();
    #if USE_END_SC
        if (stability_cut(b, &alpha, &beta))
            return alpha;
    #endif
    #if USE_END_PO
        board nb;
        int v = -inf, g;
        if (!skipped){
            int p0_parity = (b->parity & cell_div4[p0]);
            int p1_parity = (b->parity & cell_div4[p1]);
            int p2_parity = (b->parity & cell_div4[p2]);
            int p3_parity = (b->parity & cell_div4[p3]);
            int p4_parity = (b->parity & cell_div4[p4]);
            if (p0_parity && b->legal(p0)){
                b->move(p0, &nb);
                g = -last4(&nb, false, -beta, -alpha, p1, p2, p3, p4);
                alpha = max(alpha, g);
                if (beta <= alpha)
                    return alpha;
                v = max(v, g);
            }
            if (p1_parity && b->legal(p1)){
                b->move(p1, &nb);
                g = -last4(&nb, false, -beta, -alpha, p0, p2, p3, p4);
                alpha = max(alpha, g);
                if (beta <= alpha)
                    return alpha;
                v = max(v, g);
            }
            if (p2_parity && b->legal(p2)){
                b->move(p2, &nb);
                g = -last4(&nb, false, -beta, -alpha, p0, p1, p3, p4);
                alpha = max(alpha, g);
                if (beta <= alpha)
                    return alpha;
                v = max(v, g);
            }
            if (p3_parity && b->legal(p3)){
                b->move(p3, &nb);
                g = -last4(&nb, false, -beta, -alpha, p0, p1, p2, p4);
                alpha = max(alpha, g);
                if (beta <= alpha)
                    return alpha;
                v = max(v, g);
            }
            if (p4_parity && b->legal(p4)){
                b->move(p4, &nb);
                g = -last4(&nb, false, -beta, -alpha, p0, p1, p2, p3);
                alpha = max(alpha, g);
                if (beta <= alpha)
                    return alpha;
                v = max(v, g);
            }
            if (p0_parity == 0 && b->legal(p0)){
                b->move(p0, &nb);
                g = -last4(&nb, false, -beta, -alpha, p1, p2, p3, p4);
                alpha = max(alpha, g);
                if (beta <= alpha)
                    return alpha;
                v = max(v, g);
            }
            if (p1_parity == 0 && b->legal(p1)){
                b->move(p1, &nb);
                g = -last4(&nb, false, -beta, -alpha, p0, p2, p3, p4);
                alpha = max(alpha, g);
                if (beta <= alpha)
                    return alpha;
                v = max(v, g);
            }
            if (p2_parity == 0 && b->legal(p2)){
                b->move(p2, &nb);
                g = -last4(&nb, false, -beta, -alpha, p0, p1, p3, p4);
                alpha = max(alpha, g);
                if (beta <= alpha)
                    return alpha;
                v = max(v, g);
            }
            if (p3_parity == 0 && b->legal(p3)){
                b->move(p3, &nb);
                g = -last4(&nb, false, -beta, -alpha, p0, p1, p2, p4);
                alpha = max(alpha, g);
                if (beta <= alpha)
                    return alpha;
                v = max(v, g);
            }
            if (p4_parity == 0 && b->legal(p4)){
                b->move(p4, &nb);
                g = -last4(&nb, false, -beta, -alpha, p0, p1, p2, p3);
                alpha = max(alpha, g);
                if (beta <= alpha)
                    return alpha;
                v = max(v, g);
            }
        } else{
            if (b->legal(p0)){
                b->move(p0, &nb);
                g = -last4(&nb, false, -beta, -alpha, p1, p2, p3, p4);
                alpha = max(alpha, g);
                if (beta <= alpha)
                    return alpha;
                v = max(v, g);
            }
            if (b->legal(p1)){
                b->move(p1, &nb);
                g = -last4(&nb, false, -beta, -alpha, p0, p2, p3, p4);
                alpha = max(alpha, g);
                if (beta <= alpha)
                    return alpha;
                v = max(v, g);
            }
            if (b->legal(p2)){
                b->move(p2, &nb);
                g = -last4(&nb, false, -beta, -alpha, p0, p1, p3, p4);
                alpha = max(alpha, g);
                if (beta <= alpha)
                    return alpha;
                v = max(v, g);
            }
            if (b->legal(p3)){
                b->move(p3, &nb);
                g = -last4(&nb, false, -beta, -alpha, p0, p1, p2, p4);
                alpha = max(alpha, g);
                if (beta <= alpha)
                    return alpha;
                v = max(v, g);
            }
            if (b->legal(p4)){
                b->move(p4, &nb);
                g = -last4(&nb, false, -beta, -alpha, p0, p1, p2, p3);
                alpha = max(alpha, g);
                if (beta <= alpha)
                    return alpha;
                v = max(v, g);
            }
        }
    #else
        board nb;
        int v = -inf, g;
        if (b->legal(p0)){
            b->move(p0, &nb);
            g = -last4(&nb, false, -beta, -alpha, p1, p2, p3, p4);
            alpha = max(alpha, g);
            if (beta <= alpha)
                return alpha;
            v = max(v, g);
        }
        if (b->legal(p1)){
            b->move(p1, &nb);
            g = -last4(&nb, false, -beta, -alpha, p0, p2, p3, p4);
            alpha = max(alpha, g);
            if (beta <= alpha)
                return alpha;
            v = max(v, g);
        }
        if (b->legal(p2)){
            b->move(p2, &nb);
            g = -last4(&nb, false, -beta, -alpha, p0, p1, p3, p4);
            alpha = max(alpha, g);
            if (beta <= alpha)
                return alpha;
            v = max(v, g);
        }
        if (b->legal(p3)){
            b->move(p3, &nb);
            g = -last4(&nb, false, -beta, -alpha, p0, p1, p2, p4);
            alpha = max(alpha, g);
            if (beta <= alpha)
                return alpha;
            v = max(v, g);
        }
        if (b->legal(p4)){
            b->move(p4, &nb);
            g = -last4(&nb, false, -beta, -alpha, p0, p1, p2, p3);
            alpha = max(alpha, g);
            if (beta <= alpha)
                return alpha;
            v = max(v, g);
        }
    #endif
    if (v == -inf){
        if (skipped)
            v = end_evaluate(b);
        else{
            b->p = 1 - b->p;
            v = -last5(b, true, -beta, -alpha, p0, p1, p2, p3, p4);
            b->p = 1 - b->p;
        }
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
    if (!global_searching)
        return -inf;
    if (depth == 5){
        int cells[5];
        pick_vacant(b, cells);
        return last5(b, skipped, alpha, beta, cells[0], cells[1], cells[2], cells[3], cells[4]);
    }
    search_statistics.nodes_increment();
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
                    g = -nega_alpha_final(&nb, false, depth - 1, -beta, -alpha);
                    alpha = max(alpha, g);
                    if (beta <= alpha)
                        return alpha;
                    v = max(v, g);
                }
            }
            for (const int &cell: vacant_lst){
                if ((b->parity & cell_div4[cell]) == 0 && b->legal(cell)){
                    passed = false;
                    b->move(cell, &nb);
                    g = -nega_alpha_final(&nb, false, depth - 1, -beta, -alpha);
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
                    g = -nega_alpha_final(&nb, false, depth - 1, -beta, -alpha);
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

int nega_alpha_ordering_final(board *b, bool skipped, const int depth, int alpha, int beta, bool use_multi_thread, bool use_mpc, double mpct_in){
    if (!global_searching)
        return -inf;
    if (depth <= simple_end_threshold)
        return nega_alpha_final(b, skipped, depth, alpha, beta);
    search_statistics.nodes_increment();
    #if USE_END_SC
        if (stability_cut(b, &alpha, &beta))
            return alpha;
    #endif
    int hash = (int)(b->hash() & search_hash_mask);
    int l, u;
    transpose_table.get_now(b, hash, &l, &u);
    #if USE_END_TC
        if (u == l)
            return u;
        if (l >= beta)
            return l;
        if (alpha >= u)
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
            move_ordering(&nb[canput]);
            nb[canput].v -= canput_bonus * calc_canput_exact(&nb[canput]);
            #if USE_END_PO
                if (depth <= po_max_depth && (b->parity & cell_div4[cell]))
                    nb[canput].v += parity_vacant_bonus;
            #endif
            ++canput;
        }
    }
    if (canput == 0){
        if (skipped)
            return end_evaluate(b);
        b->p = 1 - b->p;
        int res = -nega_alpha_ordering_final(b, true, depth, -beta, -alpha, use_multi_thread, use_mpc, mpct_in);
        b->p = 1 - b->p;
        return res;
    }
    if (canput >= 2)
        sort(nb.begin(), nb.end());
    int g, v = -inf;
    #if USE_MULTI_THREAD
        if (use_multi_thread){
            int i, first_threshold = canput / 4;
            for (i = 0; i <= first_threshold; ++i){
                g = -nega_alpha_ordering_final(&nb[i], false, depth - 1, -beta, -alpha, true, use_mpc, mpct_in);
                alpha = max(alpha, g);
                if (beta <= alpha){
                    if (l < alpha)
                        transpose_table.reg(b, hash, alpha, u);
                    return alpha;
                }
                v = max(v, g);
            }
            vector<future<int>> future_tasks;
            for (i = first_threshold + 1; i < canput; ++i)
                future_tasks.emplace_back(thread_pool.push(bind(&nega_alpha_ordering_final, &nb[i], false, depth - 1, -beta, -alpha, false, use_mpc, mpct_in)));
            for (i = first_threshold + 1; i < canput; ++i){
                g = -future_tasks[i - first_threshold - 1].get();
                alpha = max(alpha, g);
                v = max(v, g);
            }
            if (beta <= alpha){
                if (l < alpha)
                    transpose_table.reg(b, hash, alpha, u);
                return alpha;
            }
            /*
            int done_tasks = 1;
            while (done_tasks < canput){
                for (i = done_tasks; i < canput; ++i){
                    if (thread_pool.n_idle() == 0)
                        break;
                    future_tasks.emplace_back(thread_pool.push(bind(&nega_alpha_ordering_final, &nb[i], false, depth - 1, -beta, -alpha, false, use_mpc, mpct_in)));
                }
                for (i = done_tasks - 1; i < (int)future_tasks.size(); ++i){
                    g = -future_tasks[i].get();
                    alpha = max(alpha, g);
                    v = max(v, g);
                }
                if (beta <= alpha){
                    if (l < alpha)
                        transpose_table.reg(b, hash, alpha, u);
                    return alpha;
                }
                done_tasks = (int)future_tasks.size() + 1;
                if (done_tasks < canput){
                    g = -nega_alpha_ordering_final(&nb[done_tasks], false, depth - 1, -beta, -alpha, false, use_mpc, mpct_in);
                    alpha = max(alpha, g);
                    if (beta <= alpha){
                        if (l < alpha)
                            transpose_table.reg(b, hash, alpha, u);
                        return alpha;
                    }
                    v = max(v, g);
                    ++done_tasks;
                }
            }
            */
        } else{
            for (board &nnb: nb){
                g = -nega_alpha_ordering_final(&nnb, false, depth - 1, -beta, -alpha, false, use_mpc, mpct_in);
                alpha = max(alpha, g);
                if (beta <= alpha){
                    if (l < alpha)
                        transpose_table.reg(b, hash, alpha, u);
                    return alpha;
                }
                v = max(v, g);
            }
        }
    #else
        for (board &nnb: nb){
            g = -nega_alpha_ordering_final(&nnb, false, depth - 1, -beta, -alpha, false, use_mpc, mpct_in);
            alpha = max(alpha, g);
            if (beta <= alpha){
                if (l < alpha)
                    transpose_table.reg(b, hash, alpha, u);
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

int nega_scout_final_nomemo(board *b, bool skipped, const int depth, int alpha, int beta, bool use_mpc, double mpct_in){
    if (!global_searching)
        return -inf;
    if (depth <= simple_end_threshold)
        return nega_alpha_final(b, skipped, depth, alpha, beta);
    search_statistics.nodes_increment();
    #if USE_END_SC
        if (stability_cut(b, &alpha, &beta))
            return alpha;
    #endif
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
            //nb[canput].v -= canput_bonus * calc_canput_exact(&nb[canput]);
            #if USE_END_PO
                if (depth <= po_max_depth && (b->parity & cell_div4[cell]))
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
        return -nega_scout_final_nomemo(&rb, true, depth, -beta, -alpha, use_mpc, mpct_in);
    }
    if (canput >= 2)
        sort(nb.begin(), nb.end());
    int g = alpha + 1, v = -inf;
    #if USE_MULTI_THREAD && false
        int i;
        g = -nega_scout_final(&nb[0], false, depth - 1, -beta, -alpha, use_mpc, mpct_in);
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
            future_tasks.emplace_back(thread_pool.push(bind(&nega_alpha_ordering_final, &nb[i], false, depth - 1, -alpha - search_epsilon, -alpha, false, use_mpc, mpct_in)));
        for (i = 1; i < canput; ++i){
            g = -future_tasks[i - 1].get();
            alpha = max(alpha, g);
            v = max(v, g);
            re_search.emplace_back(first_alpha < g);
        }
        for (i = 1; i < canput; ++i){
            if (re_search[i - 1]){
                g = -nega_scout_final(&nb[i], false, depth - 1, -beta, -alpha, use_mpc, mpct_in);
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
        g = -nega_scout_final_nomemo(&nb[0], false, depth - 1, -beta, -alpha, use_mpc, mpct_in);
        alpha = max(alpha, g);
        if (beta <= alpha)
            return alpha;
        v = max(v, g);
        for (int i = 1; i < canput; ++i){
            g = -nega_alpha_ordering_final_nomemo(&nb[i], false, depth - 1, -alpha - search_epsilon, -alpha, use_mpc, mpct_in);
            if (beta <= g)
                return g;
            if (alpha <= g){
                alpha = g;
                g = -nega_scout_final_nomemo(&nb[i], false, depth - 1, -beta, -alpha, use_mpc, mpct_in);
                alpha = max(alpha, g);
                if (beta <= alpha)
                    return alpha;
                v = max(v, g);
            }
        }
    #endif
    return v;
}

int mtd_final(board *b, bool skipped, int depth, int l, int u, bool use_mpc, double use_mpct, int g, bool use_multi_thread){
    int beta;
    l /= 2;
    u /= 2;
    g = max(l + search_epsilon, min(u, g / 2));
    #if USE_MULTI_THREAD && false
        int i, start_beta;
        vector<int> result(n_threads);
        //cerr << l << " " << g << " " << u << endl;
        while (u - l > 0){
            vector<future<int>> future_tasks;
            start_beta = max(l + search_epsilon, g - (int)n_threads / 2);
            for (i = 0; i < (int)n_threads; ++i)
                future_tasks.emplace_back(thread_pool.push(bind(&nega_alpha_ordering_final, b, skipped, depth, start_beta + i - search_epsilon, start_beta + i, true, use_mpc, use_mpct)));
            for (i = 0; i < (int)n_threads; ++i)
                result[i] = future_tasks[i].get();
            while (u - l > 0){
                beta = max(l + search_epsilon, g);
                g = result[beta - start_beta];
                if (g < beta)
                    u = g;
                else
                    l = g;
                //cerr << l << " " << g << " " << u << endl;
            }
        }
        //cerr << g << endl;
    #else
        cerr << l << " " << g << " " << u << endl;
        while (u - l > 0){
            beta = max(l + search_epsilon, g);
            g = nega_alpha_ordering_final(b, skipped, depth, beta * 2 - search_epsilon, beta * 2, use_multi_thread, use_mpc, use_mpct) / 2;
            if (g < beta)
                u = g;
            else
                l = g;
            cerr << l << " " << g << " " << u << endl;
        }
        cerr << g << endl;
    #endif
    return g * 2;
}

inline search_result endsearch(board b, long long strt, bool use_mpc, double use_mpct){
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
    //cerr << "canput: " << canput << endl;
    int policy = -1;
    int tmp_policy;
    int alpha, beta, g, value;
    search_statistics.searched_nodes = 0;
    transpose_table.hash_get = 0;
    transpose_table.hash_reg = 0;
    int max_depth = hw2 - b.n;
    alpha = -hw2;
    beta = hw2;
    int pre_search_depth = max(1, min(30, max_depth - simple_end_threshold + simple_mid_threshold + 3));
    cerr << "pre search depth " << pre_search_depth << endl;
    double pre_search_mpcd = 0.8;
    transpose_table.init_now();
    for (i = 0; i < canput; ++i)
        nb[i].v = -mtd(&nb[i], false, pre_search_depth - 1, -hw2, hw2, true, pre_search_mpcd);
    swap(transpose_table.now, transpose_table.prev);
    transpose_table.init_now();
    for (i = 0; i < canput; ++i){
        nb[i].v += -mtd(&nb[i], false, pre_search_depth, -hw2, hw2, true, pre_search_mpcd);
        nb[i].v /= 2;
    }
    swap(transpose_table.now, transpose_table.prev);
    if (canput >= 2)
        sort(nb.begin(), nb.end());
    transpose_table.init_now();
    long long final_strt = tim();
    search_statistics.searched_nodes = 0;
    cerr << "pre search depth " << pre_search_depth << " time " << tim() - strt << " policy " << nb[0].policy << " value " << nb[0].v << endl;
    if (nb[0].n < hw2 - 5){
        //alpha = -nega_scout_final(&nb[0], false, max_depth, -beta, -alpha, use_mpc, use_mpct);
        //alpha = -mtd_final(&nb[0], false, max_depth - 1, -beta, -alpha, use_mpc, use_mpct, -nb[0].v, true);
        //tmp_policy = nb[0].policy;
        for (i = 0; i < canput; ++i){
            //g = -nega_alpha_ordering_final(&nb[i], false, max_depth - 1, -alpha - search_epsilon, -alpha, true, use_mpc, use_mpct);
            //cerr << g << endl;
            //if (alpha < g){
            //g = -nega_scout_final(&nb[i], false, max_depth, -beta, -g, use_mpc, use_mpct);
            g = -mtd_final(&nb[i], false, max_depth - 1, -beta, -alpha, use_mpc, use_mpct, -nb[i].v, true);
            if (alpha < g || i == 0){
                alpha = g;
                tmp_policy = nb[i].policy;
            }
            //}
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
                g = -last2(&nb[i], -beta, -alpha, cells[0], cells[1]);
            else if (nb[i].n == hw2 - 1)
                g = -last1(&nb[i], cells[0]);
            else
                g = -end_evaluate(&nb[i]);
            cerr << g << endl;
            if (alpha < g || i == 0){
                alpha = g;
                tmp_policy = nb[i].policy;
            }
        }
    }
    swap(transpose_table.now, transpose_table.prev);
    if (global_searching){
        policy = tmp_policy;
        value = alpha;
        #if STATISTICS_MODE
            cerr << "final depth: " << max_depth << " time: " << tim() - strt << " policy: " << policy << " value: " << alpha << " nodes: " << search_statistics.searched_nodes << " nps: " << (long long)search_statistics.searched_nodes * 1000 / max(1LL, tim() - final_strt) << " get: " << transpose_table.hash_get << " reg: " << transpose_table.hash_reg << endl;
        #else
            cerr << "final depth: " << max_depth << " time: " << tim() - strt << " policy: " << policy << " value: " << alpha << endl;
        #endif
    } else {
        value = -inf;
        for (int i = 0; i < (int)nb.size(); ++i){
            if (nb[i].v > value){
                value = nb[i].v;
                policy = nb[i].policy;
            }
        }
    }
    search_result res;
    res.policy = policy;
    res.value = value;
    res.depth = max_depth;
    res.nps = search_statistics.searched_nodes * 1000 / max(1LL, tim() - strt);
    return res;
}

inline search_result endsearch_value(board b, long long strt, bool use_mpc, double use_mpct){
    int max_depth = hw2 - b.n;
    search_result res;
    res.policy = -1;
    if (b.n < hw2 - 5)
        res.value = nega_scout_final_nomemo(&b, false, max_depth, -hw2, hw2, use_mpc, use_mpct);
    else{
        int cells[5];
        pick_vacant(&b, cells);
        if (b.n == hw2 - 5)
            res.value = last5(&b, false, -hw2, hw2, cells[0], cells[1], cells[2], cells[3], cells[4]);
        else if (b.n == hw2 - 4)
            res.value = last4(&b, false, -hw2, hw2, cells[0], cells[1], cells[2], cells[3]);
        else if (b.n == hw2 - 3)
            res.value = last3(&b, false, -hw2, hw2, cells[0], cells[1], cells[2]);
        else if (b.n == hw2 - 2)
            res.value = last2(&b, -hw2, hw2, cells[0], cells[1]);
        else if (b.n == hw2 - 1)
            res.value = last1(&b, cells[0]);
        else
            res.value = end_evaluate(&b);
    }
    res.depth = max_depth;
    res.nps = 0;
    //cerr << res.value << endl;
    return res;
}
