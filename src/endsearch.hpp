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

int nega_alpha_ordering_final_nomemo(board *b, bool skipped, int depth, int alpha, int beta, bool use_mpc, double use_mpct, int *n_nodes){
    ++(*n_nodes);
    if (!global_searching)
        return -inf;
    if (depth <= simple_mid_threshold)
        return nega_alpha(b, skipped, depth, alpha, beta, n_nodes);
    #if USE_END_SC
        if (stability_cut(b, &alpha, &beta))
            return alpha;
    #endif
    #if USE_END_MPC
        if (mpc_min_depth <= depth && depth <= mpc_max_depth && use_mpc){
            if (mpc_higher(b, skipped, depth, beta, use_mpct, n_nodes))
                return beta;
            if (mpc_lower(b, skipped, depth, alpha, use_mpct, n_nodes))
                return alpha;
        }
    #endif
    unsigned long long legal = b->mobility_ull();
    if (legal == 0){
        if (skipped)
            return end_evaluate(b);
        b->p = 1 - b->p;
        int res = -nega_alpha_ordering_final_nomemo(b, true, depth, -beta, -alpha, use_mpc, use_mpct, n_nodes);
        b->p = 1 - b->p;
        return res;
    }
    vector<board> nb;
    mobility mob;
    int canput = 0;
    int hash = b->hash() & search_hash_mask;
    for (const int &cell: vacant_lst){
        if (1 & (legal >> cell)){
            calc_flip(&mob, b, cell);
            nb.emplace_back(b->move_copy(&mob));
            nb[canput].v = move_ordering(b, hash, cell);
            //nb[canput].v -= canput_bonus * calc_canput_exact(&nb[canput]);
            #if USE_END_PO && false
                if (depth <= po_max_depth && b->parity & cell_div4[cell])
                    nb[canput].v += parity_vacant_bonus;
            #endif
            ++canput;
        }
    }
    if (canput >= 2)
        sort(nb.begin(), nb.end());
    int g, v = -inf;
    for (board &nnb: nb){
        g = -nega_alpha_ordering_final_nomemo(&nnb, false, depth - 1, -beta, -alpha, use_mpc, use_mpct, n_nodes);
        alpha = max(alpha, g);
        if (beta <= alpha)
            return alpha;
        v = max(v, g);
    }
    return v;
}

inline bool mpc_higher_final(board *b, bool skipped, int depth, int beta, double t, int *n_nodes){
    if (b->n + mpcd[depth] >= hw2 - 5)
        return false;
    int bound = beta + ceil(t * mpcsd_final[depth - mpc_min_depth_final]);
    if (bound > hw2)
        bound = hw2; //return false;
    return nega_alpha_ordering_final_nomemo(b, skipped, mpcd[depth], bound - search_epsilon, bound, true, t, n_nodes) >= bound;
}

inline bool mpc_lower_final(board *b, bool skipped, int depth, int alpha, double t, int *n_nodes){
    if (b->n + mpcd[depth] >= hw2 - 5)
        return false;
    int bound = alpha - ceil(t * mpcsd_final[depth - mpc_min_depth_final]);
    if (bound < -hw2)
        bound = -hw2; //return false;
    return nega_alpha_ordering_final_nomemo(b, skipped, mpcd[depth], bound, bound + search_epsilon, true, t, n_nodes) <= bound;
}

inline int last1(board *b, int p0, int *n_nodes){
    ++(*n_nodes);
    unsigned long long legal = b->mobility_ull();
    mobility mob;
    int score;
    if (legal == 0){
        ++(*n_nodes);
        b->p = 1 - b->p;
        legal = b->mobility_ull();
        if (legal == 0)
            score = -end_evaluate(b);
        else{
            calc_flip(&mob, b, p0);
            score = hw2 - 2 * (b->raw_count() + pop_count_ull(mob.flip) + 1);
        }
        b->p = 1 - b->p;
    } else{
        calc_flip(&mob, b, p0);
        score = 2 * (b->raw_count() + pop_count_ull(mob.flip) + 1) - hw2;
    }
    return score;
}

inline int last2(board *b, bool skipped, int alpha, int beta, int p0, int p1, int *n_nodes){
    ++(*n_nodes);
    #if USE_END_PO & false
        int p0_parity = (b->parity & cell_div4[p0]);
        int p1_parity = (b->parity & cell_div4[p1]);
        if (!p0_parity && p1_parity)
            swap(p0, p1);
    #endif
    unsigned long long legal = b->mobility_ull();
    int v = -inf, g;
    if (legal == 0){
        if (skipped)
            v = end_evaluate(b);
        else{
            b->p = 1 - b->p;
            v = -last2(b, true, -beta, -alpha, p0, p1, n_nodes);
            b->p = 1 - b->p;
        }
        return v;
    }
    mobility mob;
    if (1 & (legal >> p0)){
        calc_flip(&mob, b, p0);
        b->move(&mob);
        g = -last1(b, p1, n_nodes);
        b->undo(&mob);
        alpha = max(alpha, g);
        if (beta <= alpha)
            return alpha;
        v = max(v, g);
    }
    if (1 & (legal >> p1)){
        calc_flip(&mob, b, p1);
        b->move(&mob);
        g = -last1(b, p0, n_nodes);
        b->undo(&mob);
        alpha = max(alpha, g);
        if (beta <= alpha)
            return alpha;
        v = max(v, g);
    }
    return v;
}

inline int last3(board *b, bool skipped, int alpha, int beta, int p0, int p1, int p2, int *n_nodes){
    ++(*n_nodes);
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
    unsigned long long legal = b->mobility_ull();
    int v = -inf, g;
    if (legal == 0){
        if (skipped)
            v = end_evaluate(b);
        else{
            b->p = 1 - b->p;
            v = -last3(b, true, -beta, -alpha, p0, p1, p2, n_nodes);
            b->p = 1 - b->p;
        }
        return v;
    }
    mobility mob;
    if (1 & (legal >> p0)){
        calc_flip(&mob, b, p0);
        b->move(&mob);
        g = -last2(b, false, -beta, -alpha, p1, p2, n_nodes);
        b->undo(&mob);
        alpha = max(alpha, g);
        if (beta <= alpha)
            return alpha;
        v = max(v, g);
    }
    if (1 & (legal >> p1)){
        calc_flip(&mob, b, p1);
        b->move(&mob);
        g = -last2(b, false, -beta, -alpha, p0, p2, n_nodes);
        b->undo(&mob);
        alpha = max(alpha, g);
        if (beta <= alpha)
            return alpha;
        v = max(v, g);
    }
    if (1 & (legal >> p2)){
        calc_flip(&mob, b, p2);
        b->move(&mob);
        g = -last2(b, false, -beta, -alpha, p0, p1, n_nodes);
        b->undo(&mob);
        alpha = max(alpha, g);
        if (beta <= alpha)
            return alpha;
        v = max(v, g);
    }
    return v;
}

inline int last4(board *b, bool skipped, int alpha, int beta, int p0, int p1, int p2, int p3, int *n_nodes){
    ++(*n_nodes);
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
    unsigned long long legal = b->mobility_ull();
    int v = -inf, g;
    if (legal == 0){
        if (skipped)
            v = end_evaluate(b);
        else{
            b->p = 1 - b->p;
            v = -last4(b, true, -beta, -alpha, p0, p1, p2, p3, n_nodes);
            b->p = 1 - b->p;
        }
        return v;
    }
    mobility mob;
    if (1 & (legal >> p0)){
        calc_flip(&mob, b, p0);
        b->move(&mob);
        g = -last3(b, false, -beta, -alpha, p1, p2, p3, n_nodes);
        b->undo(&mob);
        alpha = max(alpha, g);
        if (beta <= alpha)
            return alpha;
        v = max(v, g);
    }
    if (1 & (legal >> p1)){
        calc_flip(&mob, b, p1);
        b->move(&mob);
        g = -last3(b, false, -beta, -alpha, p0, p2, p3, n_nodes);
        b->undo(&mob);
        alpha = max(alpha, g);
        if (beta <= alpha)
            return alpha;
        v = max(v, g);
    }
    if (1 & (legal >> p2)){
        calc_flip(&mob, b, p2);
        b->move(&mob);
        g = -last3(b, false, -beta, -alpha, p0, p1, p3, n_nodes);
        b->undo(&mob);
        alpha = max(alpha, g);
        if (beta <= alpha)
            return alpha;
        v = max(v, g);
    }
    if (1 & (legal >> p3)){
        calc_flip(&mob, b, p3);
        b->move(&mob);
        g = -last3(b, false, -beta, -alpha, p0, p1, p2, n_nodes);
        b->undo(&mob);
        alpha = max(alpha, g);
        if (beta <= alpha)
            return alpha;
        v = max(v, g);
    }
    return v;
}

inline int last5(board *b, bool skipped, int alpha, int beta, int p0, int p1, int p2, int p3, int p4, int *n_nodes){
    ++(*n_nodes);
    unsigned long long legal = b->mobility_ull();
    int v = -inf, g;
    if (legal == 0){
        if (skipped)
            v = end_evaluate(b);
        else{
            b->p = 1 - b->p;
            v = -last5(b, true, -beta, -alpha, p0, p1, p2, p3, p4, n_nodes);
            b->p = 1 - b->p;
        }
        return v;
    }
    mobility mob;
    #if USE_END_PO
        if (!skipped){
            int p0_parity = (b->parity & cell_div4[p0]);
            int p1_parity = (b->parity & cell_div4[p1]);
            int p2_parity = (b->parity & cell_div4[p2]);
            int p3_parity = (b->parity & cell_div4[p3]);
            int p4_parity = (b->parity & cell_div4[p4]);
            if (p0_parity && (1 & (legal >> p0))){
                calc_flip(&mob, b, p0);
                b->move(&mob);
                g = -last4(b, false, -beta, -alpha, p1, p2, p3, p4, n_nodes);
                b->undo(&mob);
                alpha = max(alpha, g);
                if (beta <= alpha)
                    return alpha;
                v = max(v, g);
            }
            if (p1_parity && (1 & (legal >> p1))){
                calc_flip(&mob, b, p1);
                b->move(&mob);
                g = -last4(b, false, -beta, -alpha, p0, p2, p3, p4, n_nodes);
                b->undo(&mob);
                alpha = max(alpha, g);
                if (beta <= alpha)
                    return alpha;
                v = max(v, g);
            }
            if (p2_parity && (1 & (legal >> p2))){
                calc_flip(&mob, b, p2);
                b->move(&mob);
                g = -last4(b, false, -beta, -alpha, p0, p1, p3, p4, n_nodes);
                b->undo(&mob);
                alpha = max(alpha, g);
                if (beta <= alpha)
                    return alpha;
                v = max(v, g);
            }
            if (p3_parity && (1 & (legal >> p3))){
                calc_flip(&mob, b, p3);
                b->move(&mob);
                g = -last4(b, false, -beta, -alpha, p0, p1, p2, p4, n_nodes);
                b->undo(&mob);
                alpha = max(alpha, g);
                if (beta <= alpha)
                    return alpha;
                v = max(v, g);
            }
            if (p4_parity && (1 & (legal >> p4))){
                calc_flip(&mob, b, p4);
                b->move(&mob);
                g = -last4(b, false, -beta, -alpha, p0, p1, p2, p3, n_nodes);
                b->undo(&mob);
                alpha = max(alpha, g);
                if (beta <= alpha)
                    return alpha;
                v = max(v, g);
            }
            if (p0_parity == 0 && (1 & (legal >> p0))){
                calc_flip(&mob, b, p0);
                b->move(&mob);
                g = -last4(b, false, -beta, -alpha, p1, p2, p3, p4, n_nodes);
                b->undo(&mob);
                alpha = max(alpha, g);
                if (beta <= alpha)
                    return alpha;
                v = max(v, g);
            }
            if (p1_parity == 0 && (1 & (legal >> p1))){
                calc_flip(&mob, b, p1);
                b->move(&mob);
                g = -last4(b, false, -beta, -alpha, p0, p2, p3, p4, n_nodes);
                b->undo(&mob);
                alpha = max(alpha, g);
                if (beta <= alpha)
                    return alpha;
                v = max(v, g);
            }
            if (p2_parity == 0 && (1 & (legal >> p2))){
                calc_flip(&mob, b, p2);
                b->move(&mob);
                g = -last4(b, false, -beta, -alpha, p0, p1, p3, p4, n_nodes);
                b->undo(&mob);
                alpha = max(alpha, g);
                if (beta <= alpha)
                    return alpha;
                v = max(v, g);
            }
            if (p3_parity == 0 && (1 & (legal >> p3))){
                calc_flip(&mob, b, p3);
                b->move(&mob);
                g = -last4(b, false, -beta, -alpha, p0, p1, p2, p4, n_nodes);
                b->undo(&mob);
                alpha = max(alpha, g);
                if (beta <= alpha)
                    return alpha;
                v = max(v, g);
            }
            if (p4_parity == 0 && (1 & (legal >> p4))){
                calc_flip(&mob, b, p4);
                b->move(&mob);
                g = -last4(b, false, -beta, -alpha, p0, p1, p2, p3, n_nodes);
                b->undo(&mob);
                alpha = max(alpha, g);
                if (beta <= alpha)
                    return alpha;
                v = max(v, g);
            }
        } else{
            if (1 & (legal >> p0)){
                calc_flip(&mob, b, p0);
                b->move(&mob);
                g = -last4(b, false, -beta, -alpha, p1, p2, p3, p4, n_nodes);
                b->undo(&mob);
                alpha = max(alpha, g);
                if (beta <= alpha)
                    return alpha;
                v = max(v, g);
            }
            if (1 & (legal >> p1)){
                calc_flip(&mob, b, p1);
                b->move(&mob);
                g = -last4(b, false, -beta, -alpha, p0, p2, p3, p4, n_nodes);
                b->undo(&mob);
                alpha = max(alpha, g);
                if (beta <= alpha)
                    return alpha;
                v = max(v, g);
            }
            if (1 & (legal >> p2)){
                calc_flip(&mob, b, p2);
                b->move(&mob);
                g = -last4(b, false, -beta, -alpha, p0, p1, p3, p4, n_nodes);
                b->undo(&mob);
                alpha = max(alpha, g);
                if (beta <= alpha)
                    return alpha;
                v = max(v, g);
            }
            if (1 & (legal >> p3)){
                calc_flip(&mob, b, p3);
                b->move(&mob);
                g = -last4(b, false, -beta, -alpha, p0, p1, p2, p4, n_nodes);
                b->undo(&mob);
                alpha = max(alpha, g);
                if (beta <= alpha)
                    return alpha;
                v = max(v, g);
            }
            if (1 & (legal >> p4)){
                calc_flip(&mob, b, p4);
                b->move(&mob);
                g = -last4(b, false, -beta, -alpha, p0, p1, p2, p3, n_nodes);
                b->undo(&mob);
                alpha = max(alpha, g);
                if (beta <= alpha)
                    return alpha;
                v = max(v, g);
            }
        }
    #else
        if (1 & (legal >> p0)){
            calc_flip(&mob, b, p0);
            b->move(&mob);
            g = -last4(b, false, -beta, -alpha, p1, p2, p3, p4, n_nodes);
            b->undo(&mob);
            alpha = max(alpha, g);
            if (beta <= alpha)
                return alpha;
            v = max(v, g);
        }
        if (1 & (legal >> p1)){
            calc_flip(&mob, b, p1);
            b->move(&mob);
            g = -last4(b, false, -beta, -alpha, p0, p2, p3, p4, n_nodes);
            b->undo(&mob);
            alpha = max(alpha, g);
            if (beta <= alpha)
                return alpha;
            v = max(v, g);
        }
        if (1 & (legal >> p2)){
            calc_flip(&mob, b, p2);
            b->move(&mob);
            g = -last4(b, false, -beta, -alpha, p0, p1, p3, p4, n_nodes);
            b->undo(&mob);
            alpha = max(alpha, g);
            if (beta <= alpha)
                return alpha;
            v = max(v, g);
        }
        if (1 & (legal >> p3)){
            calc_flip(&mob, b, p3);
            b->move(&mob);
            g = -last4(b, false, -beta, -alpha, p0, p1, p2, p4, n_nodes);
            b->undo(&mob);
            alpha = max(alpha, g);
            if (beta <= alpha)
                return alpha;
            v = max(v, g);
        }
        if (1 & (legal >> p4)){
            calc_flip(&mob, b, p4);
            b->move(&mob);
            g = -last4(b, false, -beta, -alpha, p0, p1, p2, p3, n_nodes);
            b->undo(&mob);
            alpha = max(alpha, g);
            if (beta <= alpha)
                return alpha;
            v = max(v, g);
        }
    #endif
    return v;
}

inline void pick_vacant(board *b, int cells[]){
    int idx = 0;
    unsigned long long empties = ~(b->b | b->w);
    for (const int &cell: vacant_lst){
        if (1 & (empties >> cell))
            cells[idx++] = cell;
    }
}


int nega_alpha_final(board *b, bool skipped, const int depth, int alpha, int beta, int *n_nodes){
    ++(*n_nodes);
    if (!global_searching)
        return -inf;
    if (depth == 5){
        int cells[5];
        pick_vacant(b, cells);
        return last5(b, skipped, alpha, beta, cells[0], cells[1], cells[2], cells[3], cells[4], n_nodes);
    }
    #if USE_END_SC
        if (stability_cut(b, &alpha, &beta))
            return alpha;
    #endif
    unsigned long long legal = b->mobility_ull();
    if (legal == 0){
        if (skipped)
            return end_evaluate(b);
        b->p = 1 - b->p;
        int res = -nega_alpha_final(b, true, depth, -beta, -alpha, n_nodes);
        b->p = 1 - b->p;
        return res;
    }
    int g, v = -inf;
    mobility mob;
    #if USE_END_PO
        if (0 < b->parity && b->parity < 15){
            for (const int &cell: vacant_lst){
                if ((b->parity & cell_div4[cell]) && (1 & (legal >> cell))){
                    calc_flip(&mob, b, cell);
                    b->move(&mob);
                    g = -nega_alpha_final(b, false, depth - 1, -beta, -alpha, n_nodes);
                    b->undo(&mob);
                    alpha = max(alpha, g);
                    if (beta <= alpha)
                        return alpha;
                    v = max(v, g);
                }
            }
            for (const int &cell: vacant_lst){
                if ((b->parity & cell_div4[cell]) == 0 && (1 & (legal >> cell))){
                    calc_flip(&mob, b, cell);
                    b->move(&mob);
                    g = -nega_alpha_final(b, false, depth - 1, -beta, -alpha, n_nodes);
                    b->undo(&mob);
                    alpha = max(alpha, g);
                    if (beta <= alpha)
                        return alpha;
                    v = max(v, g);
                }
            }
        } else{
            for (const int &cell: vacant_lst){
                if (1 & (legal >> cell)){
                    calc_flip(&mob, b, cell);
                    b->move(&mob);
                    g = -nega_alpha_final(b, false, depth - 1, -beta, -alpha, n_nodes);
                    b->undo(&mob);
                    alpha = max(alpha, g);
                    if (beta <= alpha)
                        return alpha;
                    v = max(v, g);
                }
            }
        }
    #else
        for (const int &cell: vacant_lst){
            if (1 & (legal >> cell)){
                calc_flip(&mob, b, cell);
                b->move(&mob);
                g = -nega_alpha_final(b, false, depth - 1, -beta, -alpha, n_nodes);
                b->undo(&mob);
                alpha = max(alpha, g);
                if (beta <= alpha)
                    return alpha;
                v = max(v, g);
            }
        }
    #endif
    return v;
}

int nega_alpha_ordering_final(board *b, bool skipped, const int depth, int alpha, int beta, bool use_multi_thread, bool use_mpc, double mpct_in, int *n_nodes){
    ++(*n_nodes);
    if (!global_searching)
        return -inf;
    if (depth <= simple_end_threshold)
        return nega_alpha_final(b, skipped, depth, alpha, beta, n_nodes);
    #if USE_END_SC
        if (stability_cut(b, &alpha, &beta))
            return alpha;
    #endif
    int hash = b->hash() & search_hash_mask;
    #if USE_END_TC
        int l, u;
        transpose_table.get_now(b, hash, &l, &u);
        if (u == l)
            return u;
        if (l >= beta)
            return l;
        if (alpha >= u)
            return u;
        alpha = max(alpha, l);
        beta = min(beta, u);
    #endif
    #if USE_END_MPC
        if (mpc_min_depth_final <= depth && depth <= mpc_max_depth_final && use_mpc){
            if (mpc_higher_final(b, skipped, depth, beta, mpct_in, n_nodes))
                return beta;
            if (mpc_lower_final(b, skipped, depth, alpha, mpct_in, n_nodes))
                return alpha;
        }
    #endif
    unsigned long long legal = b->mobility_ull();
    if (legal == 0){
        if (skipped)
            return end_evaluate(b);
        b->p = 1 - b->p;
        int res = -nega_alpha_ordering_final(b, true, depth, -beta, -alpha, use_multi_thread, use_mpc, mpct_in, n_nodes);
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
            if (depth >= 18){
                nb[canput].v = move_ordering(b, hash, cell);
                nb[canput].v -= canput_bonus * calc_canput_exact(&nb[canput]);
            } else
                nb[canput].v = -canput_bonus * calc_canput_exact(&nb[canput]);
            #if USE_END_PO && false
                if (depth <= po_max_depth && (b->parity & cell_div4[cell]))
                    nb[canput].v += parity_vacant_bonus;
            #endif
            ++canput;
        }
    }
    if (canput >= 2)
        sort(nb.begin(), nb.end());
    int g, v = -inf;
    #if USE_MULTI_THREAD
        if (use_multi_thread){
            int i;
            const int first_threshold = canput / 6 + 1;
            for (i = 0; i < first_threshold; ++i){
                g = -nega_alpha_ordering_final(&nb[i], false, depth - 1, -beta, -alpha, true, use_mpc, mpct_in, n_nodes);
                alpha = max(alpha, g);
                if (beta <= alpha){
                    if (l < alpha)
                        transpose_table.reg(b, hash, alpha, u);
                    return alpha;
                }
                v = max(v, g);
            }
            vector<future<int>> future_tasks;
            vector<int> n_n_nodes;
            /*
            for (i = first_threshold; i < canput; ++i){
                n_n_nodes[i - first_threshold] = 0;
                future_tasks.emplace_back(thread_pool.push(bind(&nega_alpha_ordering_final, &nb[i], false, depth - 1, -beta, -alpha, false, use_mpc, mpct_in, &n_n_nodes[i - first_threshold])));
            }
            for (i = first_threshold; i < canput; ++i){
                g = -future_tasks[i - first_threshold].get();
                alpha = max(alpha, g);
                v = max(v, g);
                *n_nodes += n_n_nodes[i - first_threshold];
            }
            if (beta <= alpha){
                if (l < alpha)
                    transpose_table.reg(b, hash, alpha, u);
                return alpha;
            }
            */
            int done_tasks = first_threshold;
            int next_done_tasks = -1;
            for (i = 0; i < canput; ++i)
                n_n_nodes.emplace_back(0);
            while (done_tasks < canput){
                future_tasks.clear();
                next_done_tasks = -1;
                for (i = done_tasks; i < canput; ++i){
                    if (thread_pool.n_idle() == 0){
                        next_done_tasks = i;
                        break;
                    }
                    n_n_nodes[i] = 0;
                    future_tasks.emplace_back(thread_pool.push(bind(&nega_alpha_ordering_final, &nb[i], false, depth - 1, -beta, -alpha, false, use_mpc, mpct_in, &n_n_nodes[i])));
                }
                if (next_done_tasks == -1)
                    next_done_tasks = canput;
                for (i = done_tasks; i < next_done_tasks; ++i){
                    g = -future_tasks[i - done_tasks].get();
                    alpha = max(alpha, g);
                    v = max(v, g);
                    *n_nodes += n_n_nodes[i];
                }
                if (beta <= alpha){
                    #if USE_END_TC
                        if (l < alpha)
                            transpose_table.reg(b, hash, alpha, u);
                    #endif
                    return alpha;
                }
                done_tasks = next_done_tasks;
                if (done_tasks + first_threshold < canput){
                    g = -nega_alpha_ordering_final(&nb[done_tasks + first_threshold], false, depth - 1, -beta, -alpha, false, use_mpc, mpct_in, n_nodes);
                    alpha = max(alpha, g);
                    if (beta <= alpha){
                        #if USE_END_TC
                            if (l < alpha)
                                transpose_table.reg(b, hash, alpha, u);
                        #endif
                        return alpha;
                    }
                    v = max(v, g);
                    ++done_tasks;
                }
            }
        } else{
            for (board &nnb: nb){
                g = -nega_alpha_ordering_final(&nnb, false, depth - 1, -beta, -alpha, false, use_mpc, mpct_in, n_nodes);
                alpha = max(alpha, g);
                if (beta <= alpha){
                    #if USE_END_TC
                        if (l < alpha)
                            transpose_table.reg(b, hash, alpha, u);
                    #endif
                    return alpha;
                }
                v = max(v, g);
            }
        }
    #else
        for (board &nnb: nb){
            g = -nega_alpha_ordering_final(&nnb, false, depth - 1, -beta, -alpha, false, use_mpc, mpct_in, n_nodes);
            alpha = max(alpha, g);
            if (beta <= alpha){
                #if USE_END_TC
                    if (l < alpha)
                        transpose_table.reg(b, hash, alpha, u);
                #endif
                return alpha;
            }
            v = max(v, g);
        }
    #endif
    #if USE_END_TC
        if (v <= alpha)
            transpose_table.reg(b, hash, l, v);
        else
            transpose_table.reg(b, hash, v, v);
    #endif
    return v;
}

int nega_scout_final_nomemo(board *b, bool skipped, const int depth, int alpha, int beta, bool use_mpc, double mpct_in, int *n_nodes){
    ++(*n_nodes);
    if (!global_searching)
        return -inf;
    if (depth <= simple_end_threshold)
        return nega_alpha_final(b, skipped, depth, alpha, beta, n_nodes);
    #if USE_END_SC
        if (stability_cut(b, &alpha, &beta))
            return alpha;
    #endif
    #if USE_END_MPC
        if (mpc_min_depth_final <= depth && depth <= mpc_max_depth_final && use_mpc){
            if (mpc_higher_final(b, skipped, depth, beta, mpct_in, n_nodes))
                return beta;
            if (mpc_lower_final(b, skipped, depth, alpha, mpct_in, n_nodes))
                return alpha;
        }
    #endif
    unsigned long long legal = b->mobility_ull();
    if (legal == 0){
        if (skipped)
            return end_evaluate(b);
        b->p = 1 - b->p;
        int res = -nega_scout_final_nomemo(b, true, depth, -beta, -alpha, use_mpc, mpct_in, n_nodes);
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
            move_ordering_eval(&nb[canput]);
            //nb[canput].v -= canput_bonus * calc_canput_exact(&nb[canput]);
            #if USE_END_PO
                if (depth <= po_max_depth && (b->parity & cell_div4[cell]))
                    nb[canput].v += parity_vacant_bonus;
            #endif
            ++canput;
        }
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
        g = -nega_scout_final_nomemo(&nb[0], false, depth - 1, -beta, -alpha, use_mpc, mpct_in, n_nodes);
        alpha = max(alpha, g);
        if (beta <= alpha)
            return alpha;
        v = max(v, g);
        for (int i = 1; i < canput; ++i){
            g = -nega_alpha_ordering_final_nomemo(&nb[i], false, depth - 1, -alpha - search_epsilon, -alpha, use_mpc, mpct_in, n_nodes);
            if (beta <= g)
                return g;
            if (alpha <= g){
                alpha = g;
                g = -nega_scout_final_nomemo(&nb[i], false, depth - 1, -beta, -alpha, use_mpc, mpct_in, n_nodes);
                alpha = max(alpha, g);
                if (beta <= alpha)
                    return alpha;
                v = max(v, g);
            }
        }
    #endif
    return v;
}

int mtd_final(board *b, bool skipped, int depth, int l, int u, bool use_mpc, double use_mpct, int g, bool use_multi_thread, int *n_nodes){
    int beta;
    l /= 2;
    u /= 2;
    g = max(l, min(u, g / 2));
    //cerr << l << " " << g << " " << u << endl;
    while (u - l > 0){
        beta = max(l + search_epsilon, g);
        g = nega_alpha_ordering_final(b, skipped, depth, beta * 2 - search_epsilon, beta * 2, use_multi_thread, use_mpc, use_mpct, n_nodes) / 2;
        if (g < beta)
            u = g;
        else
            l = g;
        //cerr << l << " " << g << " " << u << endl;
    }
    //cerr << g << endl;
    return g * 2;
}

inline search_result endsearch(board b, long long strt, bool use_mpc, double use_mpct){
    unsigned long long legal = b.mobility_ull();
    vector<pair<int, board>> nb;
    mobility mob;
    vector<int> prev_vals;
    int i;
    for (const int &cell: vacant_lst){
        if (1 & (legal >> cell)){
            calc_flip(&mob, &b, cell);
            nb.emplace_back(make_pair(cell, b.move_copy(&mob)));
            //cerr << cell << " ";
        }
    }
    //cerr << endl;
    int canput = nb.size();
    //cerr << "canput: " << canput << endl;
    int policy = -1;
    int tmp_policy;
    int alpha, beta, g, value;
    int searched_nodes = 0;
    transpose_table.hash_get = 0;
    transpose_table.hash_reg = 0;
    int max_depth = hw2 - b.n;
    alpha = -hw2 - 1;
    beta = hw2;
    int pre_search_depth = max(1, min(30, max_depth - simple_end_threshold + simple_mid_threshold + 3));
    cerr << "pre search depth " << pre_search_depth << endl;
    double pre_search_mpcd = 0.8;
    transpose_table.init_now();
    for (i = 0; i < canput; ++i)
        nb[i].second.v = -mtd(&nb[i].second, false, pre_search_depth - 1, -hw2, hw2, true, pre_search_mpcd, &searched_nodes);
    swap(transpose_table.now, transpose_table.prev);
    transpose_table.init_now();
    for (i = 0; i < canput; ++i){
        nb[i].second.v += -mtd(&nb[i].second, false, pre_search_depth, -hw2, hw2, true, pre_search_mpcd, &searched_nodes);
        nb[i].second.v /= 2;
    }
    swap(transpose_table.now, transpose_table.prev);
    if (canput >= 2)
        sort(nb.begin(), nb.end(), move_ordering_sort);
    cerr << "pre search depth " << pre_search_depth << " time " << tim() - strt << " policy " << nb[0].first << " value " << nb[0].second.v << " nodes " << searched_nodes << " nps " << (long long)searched_nodes * 1000 / max(1LL, tim() - strt) << endl;
    transpose_table.init_now();
    long long final_strt = tim();
    searched_nodes = 0;
    if (nb[0].second.n < hw2 - 5){
        for (i = 0; i < canput; ++i){
            g = -mtd_final(&nb[i].second, false, max_depth - 1, -beta, -alpha, use_mpc, use_mpct, -nb[i].second.v, true, &searched_nodes);
            //cerr << "result " << nb[i].first << " " << g << " " << nb[i].second.v << endl;
            if (alpha < g || i == 0){
                alpha = g;
                tmp_policy = nb[i].first;
            }
        }
    } else{
        int cells[5];
        for (i = 0; i < canput; ++i){
            pick_vacant(&nb[i].second, cells);
            if (nb[i].second.n == hw2 - 5)
                g = -last5(&nb[i].second, false, -beta, -alpha, cells[0], cells[1], cells[2], cells[3], cells[4], &searched_nodes);
            else if (nb[i].second.n == hw2 - 4)
                g = -last4(&nb[i].second, false, -beta, -alpha, cells[0], cells[1], cells[2], cells[3], &searched_nodes);
            else if (nb[i].second.n == hw2 - 3)
                g = -last3(&nb[i].second, false, -beta, -alpha, cells[0], cells[1], cells[2], &searched_nodes);
            else if (nb[i].second.n == hw2 - 2)
                g = -last2(&nb[i].second, false, -beta, -alpha, cells[0], cells[1], &searched_nodes);
            else if (nb[i].second.n == hw2 - 1)
                g = -last1(&nb[i].second, cells[0], &searched_nodes);
            else
                g = -end_evaluate(&nb[i].second);
            if (alpha < g || i == 0){
                alpha = g;
                tmp_policy = nb[i].first;
            }
        }
    }
    swap(transpose_table.now, transpose_table.prev);
    if (global_searching){
        policy = tmp_policy;
        value = alpha;
        cerr << "final depth: " << max_depth << " time: " << tim() - strt << " policy: " << policy << " value: " << alpha << " nodes: " << searched_nodes << " nps: " << (long long)searched_nodes * 1000 / max(1LL, tim() - final_strt) << endl;
    } else {
        value = -inf;
        for (int i = 0; i < (int)nb.size(); ++i){
            if (nb[i].second.v > value){
                value = nb[i].second.v;
                policy = nb[i].first;
            }
        }
    }
    search_result res;
    res.policy = policy;
    res.value = value;
    res.depth = max_depth;
    res.nps = searched_nodes * 1000 / max(1LL, tim() - strt);
    return res;
}

inline search_result endsearch_value(board b, long long strt, bool use_mpc, double use_mpct){
    int max_depth = hw2 - b.n;
    int searched_nodes = 0;
    search_result res;
    res.policy = -1;
    if (b.n < hw2 - 5)
        res.value = nega_scout_final_nomemo(&b, false, max_depth, -hw2, hw2, use_mpc, use_mpct, &searched_nodes);
    else{
        int cells[5];
        pick_vacant(&b, cells);
        if (b.n == hw2 - 5)
            res.value = last5(&b, false, -hw2, hw2, cells[0], cells[1], cells[2], cells[3], cells[4], &searched_nodes);
        else if (b.n == hw2 - 4)
            res.value = last4(&b, false, -hw2, hw2, cells[0], cells[1], cells[2], cells[3], &searched_nodes);
        else if (b.n == hw2 - 3)
            res.value = last3(&b, false, -hw2, hw2, cells[0], cells[1], cells[2], &searched_nodes);
        else if (b.n == hw2 - 2)
            res.value = last2(&b, false, -hw2, hw2, cells[0], cells[1], &searched_nodes);
        else if (b.n == hw2 - 1)
            res.value = last1(&b, cells[0], &searched_nodes);
        else
            res.value = end_evaluate(&b);
    }
    res.depth = max_depth;
    res.nps = searched_nodes * 1000 / max(1LL, tim() - strt);
    //cerr << res.value << endl;
    return res;
}
