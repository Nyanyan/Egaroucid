#pragma once
#include <iostream>
#include <functional>
#include <queue>
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

inline bool mpc_higher_final2(board *b, bool skipped, int depth, int beta, double t, unsigned long long *n_nodes, const vector<int> &vacant_lst);
inline bool mpc_lower_final2(board *b, bool skipped, int depth, int alpha, double t, unsigned long long *n_nodes, const vector<int> &vacant_lst);

inline bool mpc_higher_final(board *b, int depth, int beta, double t, int val){
    int bound = beta + ceil(t * mpcsd_final[(depth - mpc_min_depth_final) / 5][(val + hw2) / 6]);
    return val >= bound;
}

inline bool mpc_lower_final(board *b, int depth, int alpha, double t, int val){
    int bound = alpha - ceil(t * mpcsd_final[(depth - mpc_min_depth_final) / 5][(val + hw2) / 6]);
    return val <= bound;
}

inline bool mpc_higher_final(board *b, int depth, int beta, double t, unsigned long long *n_nodes){
    return mpc_higher_final(b, depth, beta, t, mid_evaluate(b));
}

inline bool mpc_lower_final(board *b, int depth, int alpha, double t, unsigned long long *n_nodes){
    return mpc_lower_final(b, depth, alpha, t, mid_evaluate(b));
}

int nega_alpha_final_nomemo(board *b, bool skipped, int depth, int alpha, int beta, bool use_mpc, double use_mpct, unsigned long long *n_nodes, const vector<int> &vacant_lst){
    if (!global_searching)
        return -inf;
    ++(*n_nodes);
    if (depth == 0)
        return mid_evaluate(b);
    #if USE_END_SC
        int stab_res = stability_cut(b, &alpha, &beta);
        if (stab_res != -inf)
            return stab_res;
    #endif
    #if USE_END_MPC
        if (mpc_min_depth_final <= depth && depth <= mpc_max_depth_final && use_mpc){
            if (mpc_higher_final(b, depth, beta, use_mpct, n_nodes))
                return beta;
            if (mpc_lower_final(b, depth, alpha, use_mpct, n_nodes))
                return alpha;
        }
    #endif
    int g, v = -inf;
    unsigned long long legal = b->mobility_ull();
    if (legal == 0){
        if (skipped)
            return end_evaluate(b);
        b->p = 1 - b->p;
        int res = -nega_alpha_final_nomemo(b, true, depth, -beta, -alpha, use_mpc, use_mpct, n_nodes, vacant_lst);
        b->p = 1 - b->p;
        return res;
    }
    mobility mob;
    for (const int &cell: vacant_lst){
        if (1 & (legal >> cell)){
            calc_flip(&mob, b, cell);
            b->move(&mob);
            g = -nega_alpha_final_nomemo(b, false, depth - 1, -beta, -alpha, use_mpc, use_mpct, n_nodes, vacant_lst);
            b->undo(&mob);
            alpha = max(alpha, g);
            if (beta <= alpha)
                return alpha;
            v = max(v, g);
        }
    }
    return v;
}

int nega_alpha_ordering_final_nomemo(board *b, bool skipped, int depth, int alpha, int beta, bool use_mpc, double use_mpct, unsigned long long *n_nodes, const vector<int> &vacant_lst){
    ++(*n_nodes);
    if (!global_searching)
        return -inf;
    if (depth <= simple_end_mpc_threshold)
        return nega_alpha_final_nomemo(b, skipped, depth, alpha, beta, use_mpc, use_mpct, n_nodes, vacant_lst);
    #if USE_END_SC
        int stab_res = stability_cut(b, &alpha, &beta);
        if (stab_res != -inf)
            return stab_res;
    #endif
    #if USE_END_MPC
        if (mpc_min_depth_final <= depth && depth <= mpc_max_depth_final && use_mpc){
            if (mpc_higher_final(b, depth, beta, use_mpct, n_nodes))
                return beta;
            if (mpc_lower_final(b, depth, alpha, use_mpct, n_nodes))
                return alpha;
        }
    #endif
    unsigned long long legal = b->mobility_ull();
    if (legal == 0){
        if (skipped)
            return end_evaluate(b);
        b->p = 1 - b->p;
        int res = -nega_alpha_ordering_final_nomemo(b, true, depth, -beta, -alpha, use_mpc, use_mpct, n_nodes, vacant_lst);
        b->p = 1 - b->p;
        return res;
    }
    const int canput = pop_count_ull(legal);
    board *nb = new board[canput];
    mobility mob;
    //int hash = b->hash() & search_hash_mask;
    int idx = 0;
    for (const int &cell: vacant_lst){
        if (1 & (legal >> cell)){
            calc_flip(&mob, b, cell);
            b->move_copy(&mob, &nb[idx]);
            nb[idx].v = -canput_bonus * pop_count_ull(nb[idx].mobility_ull());
            //nb[idx].v = move_ordering(b, &nb[idx], hash, cell);
            #if USE_END_PO
                if (depth <= po_max_depth && b->parity & cell_div4[cell])
                    nb[idx].v += parity_vacant_bonus;
            #endif
            ++idx;
        }
    }
    if (canput >= 2)
        sort(nb, nb + canput);
    int g, v = -inf;
    #if USE_MULTI_THREAD
        int i;
        const int first_threshold = canput / end_first_threshold_div + 1;
        for (i = 0; i < first_threshold; ++i){
            g = -nega_alpha_ordering_final_nomemo(&nb[i], false, depth - 1, -beta, -alpha, use_mpc, use_mpct, n_nodes, vacant_lst);
            alpha = max(alpha, g);
            if (beta <= alpha){
                delete[] nb;
                return alpha;
            }
            v = max(v, g);
        }
        vector<future<int>> future_tasks;
        unsigned long long *n_n_nodes = new unsigned long long[canput - first_threshold];
        int done_tasks = first_threshold;
        for (i = first_threshold; i < canput; ++i)
            n_n_nodes[i - first_threshold] = 0;
        int next_done_tasks, additional_done_tasks;
        while (done_tasks < canput){
            next_done_tasks = canput;
            future_tasks.clear();
            for (i = done_tasks; i < canput; ++i){
                if (thread_pool.n_idle() == 0){
                    next_done_tasks = i;
                    break;
                }
                future_tasks.emplace_back(thread_pool.push(bind(&nega_alpha_ordering_final_nomemo, &nb[i], false, depth - 1, -beta, -alpha, use_mpc, use_mpct, &n_n_nodes[i - first_threshold], vacant_lst)));
            }
            additional_done_tasks = 0;
            if (next_done_tasks < canput){
                g = -nega_alpha_ordering_final_nomemo(&nb[next_done_tasks], false, depth - 1, -beta, -alpha,  use_mpc, use_mpct, n_nodes, vacant_lst);
                alpha = max(alpha, g);
                v = max(v, g);
                additional_done_tasks = 1;
            }
            for (i = done_tasks; i < next_done_tasks; ++i){
                g = -future_tasks[i - done_tasks].get();
                alpha = max(alpha, g);
                v = max(v, g);
                *n_nodes += n_n_nodes[i - first_threshold];
            }
            if (beta <= alpha){
                delete[] nb;
                delete[] n_n_nodes;
                return alpha;
            }
            done_tasks = next_done_tasks + additional_done_tasks;
        }
        delete[] nb;
        delete[] n_n_nodes;
    #else
        for (idx = 0; idx < canput; ++idx){
            g = -nega_alpha_ordering_final_nomemo(&nb[idx], false, depth - 1, -beta, -alpha, use_mpc, use_mpct, n_nodes, vacant_lst);
            alpha = max(alpha, g);
            if (beta <= alpha)
                return alpha;
            v = max(v, g);
        }
    #endif
    return v;
}

inline bool mpc_higher_final2(board *b, bool skipped, int depth, int beta, double t, unsigned long long *n_nodes, const vector<int> &vacant_lst){
    int bound = beta + ceil(t * mpcsd[b->phase()][depth - mpc_min_depth]);
    if (bound > hw2)
        bound = hw2; //return false;
    return nega_alpha_ordering_final_nomemo(b, skipped, mpcd[depth], bound - search_epsilon, bound, true, t, n_nodes, vacant_lst) >= bound;
}

inline bool mpc_lower_final2(board *b, bool skipped, int depth, int alpha, double t, unsigned long long *n_nodes, const vector<int> &vacant_lst){
    int bound = alpha - ceil(t * mpcsd[b->phase()][depth - mpc_min_depth]);
    if (bound < -hw2)
        bound = -hw2; //return false;
    return nega_alpha_ordering_final_nomemo(b, skipped, mpcd[depth], bound, bound + search_epsilon, true, t, n_nodes, vacant_lst) <= bound;
}

inline int last1(board *b, int alpha, int beta, int p0, unsigned long long *n_nodes){
    ++(*n_nodes);
    int score = hw2 - 2 * b->count_opponent();
    int n_flip;
    if (b->p == black)
        n_flip = count_last_flip(b->b, b->w, p0);
    else
        n_flip = count_last_flip(b->w, b->b, p0);
    if (n_flip == 0){
        ++(*n_nodes);
        if (score <= 0){
            score -= 2;
            if (score >= alpha){
                if (b->p == white)
                    n_flip = count_last_flip(b->b, b->w, p0);
                else
                    n_flip = count_last_flip(b->w, b->b, p0);
                score -= 2 * n_flip;
            }
        } else{
            if (score >= alpha){
                if (b->p == white)
                    n_flip = count_last_flip(b->b, b->w, p0);
                else
                    n_flip = count_last_flip(b->w, b->b, p0);
                if (n_flip)
                    score -= 2 * n_flip + 2;
            }
        }
        
    } else
        score += 2 * n_flip;
    return score;
}

inline int last2(board *b, bool skipped, int alpha, int beta, int p0, int p1, unsigned long long *n_nodes){
    ++(*n_nodes);
    #if USE_END_PO & false
        int p0_parity = (b->parity & cell_div4[p0]);
        int p1_parity = (b->parity & cell_div4[p1]);
        if (!p0_parity && p1_parity)
            swap(p0, p1);
    #endif
    int v = -inf, g;
    mobility mob;
    calc_flip(&mob, b, p0);
    if (mob.flip){
        b->move(&mob);
        g = -last1(b, -beta, -alpha, p1, n_nodes);
        b->undo(&mob);
        alpha = max(alpha, g);
        if (beta <= alpha)
            return alpha;
        v = max(v, g);
    }
    calc_flip(&mob, b, p1);
    if (mob.flip){
        b->move(&mob);
        g = -last1(b, -beta, -alpha, p0, n_nodes);
        b->undo(&mob);
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
            v = -last2(b, true, -beta, -alpha, p0, p1, n_nodes);
            b->p = 1 - b->p;
        }
        return v;
    }
    return v;
}

inline int last3(board *b, bool skipped, int alpha, int beta, int p0, int p1, int p2, unsigned long long *n_nodes){
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

inline int last4(board *b, bool skipped, int alpha, int beta, int p0, int p1, int p2, int p3, unsigned long long *n_nodes){
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

inline int last5(board *b, bool skipped, int alpha, int beta, int p0, int p1, int p2, int p3, int p4, unsigned long long *n_nodes){
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

inline void pick_vacant(board *b, int cells[], const vector<int> &vacant_lst){
    int idx = 0;
    unsigned long long empties = ~(b->b | b->w);
    for (const int &cell: vacant_lst){
        if (1 & (empties >> cell))
            cells[idx++] = cell;
    }
}


int nega_alpha_final(board *b, bool skipped, const int depth, int alpha, int beta, unsigned long long *n_nodes, const vector<int> &vacant_lst, const bool *searching){
    if (!global_searching)
        return -inf;
    if (!(*searching))
        return -inf;
    if (depth == 5){
        int cells[5];
        pick_vacant(b, cells, vacant_lst);
        return last5(b, skipped, alpha, beta, cells[0], cells[1], cells[2], cells[3], cells[4], n_nodes);
    }
    ++(*n_nodes);
    #if USE_END_SC
        int stab_res = stability_cut(b, &alpha, &beta);
        if (stab_res != -inf)
            return stab_res;
    #endif
    unsigned long long legal = b->mobility_ull();
    if (legal == 0){
        if (skipped)
            return end_evaluate(b);
        b->p = 1 - b->p;
        int res = -nega_alpha_final(b, true, depth, -beta, -alpha, n_nodes, vacant_lst, searching);
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
                    g = -nega_alpha_final(b, false, depth - 1, -beta, -alpha, n_nodes, vacant_lst, searching);
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
                    g = -nega_alpha_final(b, false, depth - 1, -beta, -alpha, n_nodes, vacant_lst, searching);
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
                    g = -nega_alpha_final(b, false, depth - 1, -beta, -alpha, n_nodes, vacant_lst, searching);
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
                g = -nega_alpha_final(b, false, depth - 1, -beta, -alpha, n_nodes, vacant_lst, searching);
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

int nega_alpha_ordering_simple_final(board *b, bool skipped, const int depth, int alpha, int beta, bool use_mpc, double mpct_in, unsigned long long *n_nodes, const vector<int> &vacant_lst, const bool *searching){
    if (!global_searching)
        return -inf;
    if (!(*searching))
        return -inf;
    //if (depth == 5){
    //    int cells[5];
    //    pick_vacant(b, cells, vacant_lst);
    //    return last5(b, skipped, alpha, beta, cells[0], cells[1], cells[2], cells[3], cells[4], n_nodes);
    //}
    if (depth <= simple_end_threshold)
        return nega_alpha_final(b, skipped, depth, alpha, beta, n_nodes, vacant_lst, searching);
    ++(*n_nodes);
    #if USE_END_SC && false
        int stab_res = stability_cut(b, &alpha, &beta);
        if (stab_res != -inf)
            return stab_res;
    #endif
    #if USE_END_TC
        int hash = b->hash() & search_hash_mask;
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
            if (mpc_higher_final(b, depth, beta, mpct_in, n_nodes)){
                #if USE_END_TC
                    if (*searching && l < beta)
                        transpose_table.reg(b, hash, beta, u);
                #endif
                return beta;
            }
            if (mpc_lower_final(b, depth, alpha, mpct_in, n_nodes)){
                #if USE_END_TC
                    if (*searching && alpha < u)
                        transpose_table.reg(b, hash, l, alpha);
                #endif
                return alpha;
            }
        }
    #endif
    unsigned long long legal = b->mobility_ull();
    if (legal == 0){
        if (skipped)
            return end_evaluate(b);
        b->p = 1 - b->p;
        int res = -nega_alpha_ordering_simple_final(b, true, depth, -beta, -alpha, use_mpc, mpct_in, n_nodes, vacant_lst, searching);
        b->p = 1 - b->p;
        #if USE_END_TC
            if (*searching){
                if (res >= beta){
                    if (l < alpha)
                        transpose_table.reg(b, hash, res, u);
                } else if (res <= alpha){
                    if (res < u)
                        transpose_table.reg(b, hash, l, res);
                } else
                    transpose_table.reg(b, hash, res, res);
            }
        #endif
        return res;
    }
    const int canput = pop_count_ull(legal);
    board *nb = new board[canput];
    mobility mob;
    int idx = 0;
    for (const int &cell: vacant_lst){
        if (1 & (legal >> cell)){
            calc_flip(&mob, b, cell);
            b->move_copy(&mob, &nb[idx]);
            nb[idx].v = -canput_bonus * pop_count_ull(nb[idx].mobility_ull());
            #if USE_END_PO
                if (depth <= po_max_depth && (b->parity & cell_div4[cell]))
                    nb[idx].v += parity_vacant_bonus;
            #endif
            ++idx;
        }
    }
    if (canput >= 2)
        sort(nb, nb + canput);
    int g, v = -inf;
    for (idx = 0; idx < canput; ++idx){
        g = -nega_alpha_ordering_simple_final(&nb[idx], false, depth - 1, -beta, -alpha, use_mpc, mpct_in, n_nodes, vacant_lst, searching);
        alpha = max(alpha, g);
        if (beta <= alpha){
            #if USE_END_TC
                if (*searching && l < alpha)
                    transpose_table.reg(b, hash, alpha, u);
            #endif
            delete[] nb;
            return alpha;
        }
        v = max(v, g);
    }
    delete[] nb;
    #if USE_END_TC
        if (*searching){
            if (v <= alpha){
                if (v < u)
                    transpose_table.reg(b, hash, l, v);
            } else
                transpose_table.reg(b, hash, v, v);
        }
    #endif
    return v;
}

int nega_alpha_ordering_final(board *b, bool skipped, const int depth, int alpha, int beta, bool use_mpc, double mpct_in, unsigned long long *n_nodes, const vector<int> &vacant_lst, const bool *searching);

ybwc_result ybwc_final(int id, board b, bool skipped, const int depth, int alpha, int beta, bool use_mpc, double mpct_in, const vector<int> vacant_lst, const bool *searching){
    ybwc_result res;
    res.n_nodes = 0;
    res.value = nega_alpha_ordering_final(&b, skipped, depth, alpha, beta, use_mpc, mpct_in, &res.n_nodes, vacant_lst, searching);
    return res;
}

int nega_alpha_ordering_final(board *b, bool skipped, const int depth, int alpha, int beta, bool use_mpc, double mpct_in, unsigned long long *n_nodes, const vector<int> &vacant_lst, const bool *searching){
    if (!global_searching)
        return -inf;
    if (!(*searching))
        return -inf;
    if (depth <= simple_end_threshold2)
        return nega_alpha_ordering_simple_final(b, skipped, depth, alpha, beta, use_mpc, mpct_in, n_nodes, vacant_lst, searching);
    ++(*n_nodes);
    #if USE_END_SC && false
        int stab_res = stability_cut(b, &alpha, &beta);
        if (stab_res != -inf)
            return stab_res;
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
    unsigned long long legal = b->mobility_ull();
    if (legal == 0){
        if (skipped)
            return end_evaluate(b);
        b->p = 1 - b->p;
        int res = -nega_alpha_ordering_final(b, true, depth, -beta, -alpha, use_mpc, mpct_in, n_nodes, vacant_lst, searching);
        b->p = 1 - b->p;
        #if USE_END_TC
            if (*searching){
                if (res >= beta){
                    if (l < alpha)
                        transpose_table.reg(b, hash, res, u);
                } else if (res <= alpha){
                    if (res < u)
                        transpose_table.reg(b, hash, l, res);
                } else
                    transpose_table.reg(b, hash, res, res);
            }
        #endif
        return res;
    }
    const int canput_all = pop_count_ull(legal);
    board *nb = new board[canput_all];
    mobility mob;
    int idx = 0, n_val;
    for (const int &cell: vacant_lst){
        if (1 & (legal >> cell)){
            calc_flip(&mob, b, cell);
            b->move_copy(&mob, &nb[idx]);
            #if USE_END_MPC
                if (mpc_min_depth_final <= depth - 1 && depth - 1 <= mpc_max_depth_final && use_mpc){
                    n_val = -mid_evaluate(&nb[idx]);
                    if (mpc_higher_final(&nb[idx], depth - 1, beta, mpct_in, n_val)){
                        #if USE_END_TC
                            if (*searching && l < beta)
                                transpose_table.reg(b, hash, beta, u);
                        #endif
                        delete[] nb;
                        return beta;
                    } else if (!mpc_lower_final(&nb[idx], depth - 1, alpha, mpct_in, n_val)){
                        nb[idx].v = move_ordering(b, &nb[idx], hash, cell, n_val);
                        ++idx;
                    }
                } else{
                    nb[idx].v = move_ordering(b, &nb[idx], hash, cell);
                    ++idx;
                }
            #else
                nb[idx].v = move_ordering(b, &nb[idx], hash, cell);
                ++idx;
            #endif
        }
    }
    const int canput = idx;
    if (canput == 0){
        #if USE_END_TC
            if (alpha < u)
                transpose_table.reg(b, hash, l, alpha);
        #endif
        delete[] nb;
        return alpha;
    }
    if (canput >= 2)
        sort(nb, nb + canput);
    int g, v = -inf, i;
    #if USE_MULTI_THREAD
        const int first_threshold = canput / end_first_threshold_div + 1;
        for (i = 0; i < first_threshold; ++i){
            g = -inf;
            if (g == -inf)
                g = -nega_alpha_ordering_final(&nb[i], false, depth - 1, -beta, -alpha, use_mpc, mpct_in, n_nodes, vacant_lst, searching);
            alpha = max(alpha, g);
            if (beta <= alpha){
                #if USE_END_TC
                    if (*searching && l < alpha)
                        transpose_table.reg(b, hash, alpha, u);
                #endif
                delete[] nb;
                return alpha;
            }
            v = max(v, g);
        }
        const int n_parallel_tasks = canput - first_threshold;
        ybwc_result ybwc_res;
        vector<int> n_vacant_lst;
        for (const int &cell: vacant_lst)
            n_vacant_lst.emplace_back(cell);
        int n_task_doing = 0;
        vector<future<ybwc_result>> future_tasks;
        int *task_doing = new int[n_parallel_tasks];
        bool *task_done = new bool[n_parallel_tasks];
        bool all_done = false; //, doing_done;
        bool n_searching = true;
        for (i = 0; i < n_parallel_tasks; ++i){
            task_doing[i] = -1;
            task_done[i] = false;
        }
        while (!all_done){
            if (alpha < beta){
                for (i = 0; i < n_parallel_tasks && thread_pool.n_idle() && n_task_doing < n_parallel_tasks - 1; ++i){
                    if (!task_done[i] && task_doing[i] == -1){
                        task_doing[i] = (int)future_tasks.size();
                        ++n_task_doing;
                        future_tasks.emplace_back(thread_pool.push(ybwc_final, nb[i + first_threshold], false, depth - 1, -beta, -alpha, use_mpc, mpct_in, n_vacant_lst, &n_searching));
                    }
                }
                for (i = 0; i < n_parallel_tasks; ++i){
                    if (!task_done[i] && task_doing[i] == -1){
                        ++n_task_doing;
                        g = -nega_alpha_ordering_final(&nb[i + first_threshold], false, depth - 1, -beta, -alpha,  use_mpc, mpct_in, n_nodes, vacant_lst, searching);
                        task_done[i] = true;
                        alpha = max(alpha, g);
                        v = max(v, g);
                        break;
                    }
                }
            }
            for (i = 0; i < n_parallel_tasks; ++i){
                if (!task_done[i] && task_doing[i] != -1){
                    //if (future_tasks[task_doing[i]].wait_for(chrono::seconds(0)) == future_status::ready){
                    ybwc_res = future_tasks[task_doing[i]].get();
                    task_done[i] = true;
                    if (n_searching){
                        if (alpha < -ybwc_res.value){
                            alpha = -ybwc_res.value;
                            if (beta <= alpha)
                                n_searching = false;
                        }
                        v = max(v, -ybwc_res.value);
                    }
                    *n_nodes += ybwc_res.n_nodes;
                    //}
                }
            }
            //if (doing_done){
            if (beta <= alpha){
                #if USE_END_TC
                    if (*searching && l < alpha)
                        transpose_table.reg(b, hash, alpha, u);
                #endif
                delete[] nb;
                delete[] task_doing;
                delete[] task_done;
                return alpha;
            }
            //}
            all_done = true;
            //doing_done = true;
            for (i = 0; i < n_parallel_tasks; ++i){
                all_done &= task_done[i];
                //doing_done &= task_done[i] || task_doing[i] == -1;
            }
        }
        delete[] nb;
        delete[] task_doing;
        delete[] task_done;
    #else
        for (i = 0; i < canput; ++i){
            g = -nega_alpha_ordering_final(&nb[i], false, depth - 1, -beta, -alpha, use_mpc, mpct_in, n_nodes, vacant_lst, searching);
            alpha = max(alpha, g);
            if (beta <= alpha){
                #if USE_END_TC
                    if (l < alpha)
                        transpose_table.reg(b, hash, alpha, u);
                #endif
                delete[] nb;
                return alpha;
            }
            v = max(v, g);
        }
        delete[] nb;
    #endif
    #if USE_END_TC
        if (*searching){
            if (v <= alpha){
                if (v < u)
                    transpose_table.reg(b, hash, l, v);
            } else
                transpose_table.reg(b, hash, v, v);
        }
    #endif
    return v;
}

int nega_scout_final(board *b, bool skipped, const int depth, int alpha, int beta, bool use_mpc, double mpct_in, unsigned long long *n_nodes, const vector<int> &vacant_lst, const bool *searching){
    if (!global_searching)
        return -inf;
    if (depth <= simple_end_threshold)
        return nega_alpha_final(b, skipped, depth, alpha, beta, n_nodes, vacant_lst, searching);
    ++(*n_nodes);
    #if USE_END_SC && false
        int stab_res = stability_cut(b, &alpha, &beta);
        if (stab_res != -inf)
            return stab_res;
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
    unsigned long long legal = b->mobility_ull();
    if (legal == 0){
        if (skipped)
            return end_evaluate(b);
        b->p = 1 - b->p;
        int res = -nega_scout_final(b, true, depth, -beta, -alpha, use_mpc, mpct_in, n_nodes, vacant_lst, searching);
        b->p = 1 - b->p;
        #if USE_END_TC
            if (res >= beta){
                if (l < alpha)
                    transpose_table.reg(b, hash, res, u);
            } else if (res <= alpha){
                if (res < u)
                    transpose_table.reg(b, hash, l, res);
            } else
                transpose_table.reg(b, hash, res, res);
        #endif
        return res;
    }
    const int canput_all = pop_count_ull(legal);
    board *nb = new board[canput_all];
    mobility mob;
    int idx = 0, n_val;
    for (const int &cell: vacant_lst){
        if (1 & (legal >> cell)){
            calc_flip(&mob, b, cell);
            b->move_copy(&mob, &nb[idx]);
            #if USE_END_MPC
                if (mpc_min_depth_final <= depth - 1 && depth - 1 <= mpc_max_depth_final && use_mpc){
                    n_val = -mid_evaluate(&nb[idx]);
                    if (mpc_higher_final(&nb[idx], depth - 1, beta, mpct_in, n_val)){
                        #if USE_END_TC
                            if (l < beta)
                                transpose_table.reg(b, hash, beta, u);
                        #endif
                        delete[] nb;
                        return beta;
                    } else if (!mpc_lower_final(&nb[idx], depth - 1, alpha, mpct_in, n_val)){
                        nb[idx].v = move_ordering(b, &nb[idx], hash, cell, n_val);
                        ++idx;
                    }
                } else{
                    nb[idx].v = move_ordering(b, &nb[idx], hash, cell);
                    ++idx;
                }
            #else
                nb[idx].v = move_ordering(b, &nb[idx], hash, cell);
                ++idx;
            #endif
        }
    }
    const int canput = idx;
    if (canput == 0){
        #if USE_END_TC
            if (alpha < u)
                transpose_table.reg(b, hash, l, alpha);
        #endif
        delete[] nb;
        return alpha;
    }
    if (canput >= 2)
        sort(nb, nb + canput);
    int g, v = -inf;
    #if USE_MULTI_THREAD
        int i;
        const int first_threshold = canput / end_first_threshold_div + 1;
        for (i = 0; i < first_threshold; ++i){
            g = -nega_scout_final(&nb[i], false, depth - 1, -beta, -alpha, use_mpc, mpct_in, n_nodes, vacant_lst, searching);
            alpha = max(alpha, g);
            if (beta <= alpha){
                #if USE_END_TC
                    if (l < alpha)
                        transpose_table.reg(b, hash, alpha, u);
                #endif
                delete[] nb;
                return alpha;
            }
            v = max(v, g);
        }
        const int n_parallel_tasks = canput - first_threshold;
        ybwc_result ybwc_res;
        vector<int> n_vacant_lst;
        for (const int &cell: vacant_lst)
            n_vacant_lst.emplace_back(cell);
        int n_task_doing = 0;
        vector<future<ybwc_result>> future_tasks;
        int *task_doing = new int[n_parallel_tasks];
        bool *task_done = new bool[n_parallel_tasks];
        bool *re_search = new bool[n_parallel_tasks];
        bool all_done = false; //, doing_done;
        bool n_searching = true;
        int before_alpha;
        for (i = 0; i < n_parallel_tasks; ++i){
            task_doing[i] = -1;
            task_done[i] = false;
            re_search[i] = false;
        }
        while (!all_done){
            before_alpha = alpha;
            for (i = 0; i < n_parallel_tasks && thread_pool.n_idle() && n_task_doing < n_parallel_tasks - 1; ++i){
                if (!task_done[i] && task_doing[i] == -1){
                    task_doing[i] = (int)future_tasks.size();
                    ++n_task_doing;
                    future_tasks.emplace_back(thread_pool.push(ybwc_final, nb[i + first_threshold], false, depth - 1, -before_alpha - search_epsilon, -before_alpha, use_mpc, mpct_in, n_vacant_lst, &n_searching));
                }
            }
            for (i = 0; i < n_parallel_tasks; ++i){
                if (!task_done[i] && task_doing[i] == -1){
                    ++n_task_doing;
                    g = -nega_alpha_ordering_final(&nb[i + first_threshold], false, depth - 1, -before_alpha - search_epsilon, -before_alpha,  use_mpc, mpct_in, n_nodes, vacant_lst, searching);
                    task_done[i] = true;
                    alpha = max(alpha, g);
                    v = max(v, g);
                    break;
                }
            }
            for (i = 0; i < n_parallel_tasks; ++i){
                if (!task_done[i] && task_doing[i] != -1){
                    //if (future_tasks[task_doing[i]].wait_for(chrono::seconds(0)) == future_status::ready){
                    ybwc_res = future_tasks[task_doing[i]].get();
                    task_done[i] = true;
                    if (n_searching){
                        if (before_alpha < -ybwc_res.value){
                            alpha = max(alpha, -ybwc_res.value);
                            re_search[i] = true;
                            //if (beta <= alpha)
                            //    n_searching = true;
                        }
                        v = max(v, -ybwc_res.value);
                    }
                    *n_nodes += ybwc_res.n_nodes;
                    //}
                }
            }
            if (beta <= alpha){
                #if USE_END_TC
                    if (l < alpha)
                        transpose_table.reg(b, hash, alpha, u);
                #endif
                delete[] nb;
                delete[] task_doing;
                delete[] task_done;
                delete[] re_search;
                return alpha;
            }
            for (i = 0; i < n_parallel_tasks; ++i){
                if (re_search[i]){
                    g = -nega_scout_final(&nb[i], false, depth - 1, -beta, -alpha, use_mpc, mpct_in, n_nodes, vacant_lst, searching);
                    alpha = max(alpha, g);
                    if (beta <= alpha){
                        #if USE_END_TC
                            if (l < alpha)
                                transpose_table.reg(b, hash, alpha, u);
                        #endif
                        delete[] nb;
                        delete[] task_doing;
                        delete[] task_done;
                        delete[] re_search;
                        return alpha;
                    }
                    v = max(v, g);
                }
            }
            all_done = true;
            //doing_done = true;
            for (i = 0; i < n_parallel_tasks; ++i){
                all_done &= task_done[i];
                //doing_done &= task_done[i] || task_doing[i] == -1;
            }
        }
        delete[] nb;
        delete[] task_doing;
        delete[] task_done;
        delete[] re_search;
    #else
        for (idx = 0; idx < canput; ++idx){
            g = -nega_alpha_ordering_final(&nb[idx], false, depth - 1, -beta, -alpha, use_mpc, mpct_in, n_nodes, vacant_lst, searching);
            alpha = max(alpha, g);
            if (beta <= alpha){
                #if USE_END_TC
                    if (l < alpha)
                        transpose_table.reg(b, hash, alpha, u);
                #endif
                delete[] nb;
                return alpha;
            }
            v = max(v, g);
        }
        delete[] nb;
    #endif
    #if USE_END_TC
        if (v <= alpha){
            if (v < u)
                transpose_table.reg(b, hash, l, v);
        } else
            transpose_table.reg(b, hash, v, v);
    #endif
    return v;
}

int nega_scout_final_nomemo(board *b, bool skipped, const int depth, int alpha, int beta, bool use_mpc, double mpct_in, unsigned long long *n_nodes, const vector<int> &vacant_lst){
    if (!global_searching)
        return -inf;
    if (depth <= simple_end_threshold){
        bool searching = true;
        return nega_alpha_final(b, skipped, depth, alpha, beta, n_nodes, vacant_lst, &searching);
    }
    ++(*n_nodes);
    #if USE_END_SC
        int stab_res = stability_cut(b, &alpha, &beta);
        if (stab_res != -inf)
            return stab_res;
    #endif
    #if USE_END_MPC
        if (mpc_min_depth_final <= depth && depth <= mpc_max_depth_final && use_mpc){
            if (mpc_higher_final(b, depth, beta, mpct_in, n_nodes))
                return beta;
            if (mpc_lower_final(b, depth, alpha, mpct_in, n_nodes))
                return alpha;
        }
    #endif
    unsigned long long legal = b->mobility_ull();
    if (legal == 0){
        if (skipped)
            return end_evaluate(b);
        b->p = 1 - b->p;
        int res = -nega_scout_final_nomemo(b, true, depth, -beta, -alpha, use_mpc, mpct_in, n_nodes, vacant_lst);
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
        g = -nega_scout_final(&nb[0], false, depth - 1, -beta, -alpha, use_mpc, mpct_in, vacant_lst);
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
            future_tasks.emplace_back(thread_pool.push(bind(&nega_alpha_ordering_final, &nb[i], false, depth - 1, -alpha - search_epsilon, -alpha, use_mpc, mpct_in, vacant_lst)));
        for (i = 1; i < canput; ++i){
            g = -future_tasks[i - 1].get();
            alpha = max(alpha, g);
            v = max(v, g);
            re_search.emplace_back(first_alpha < g);
        }
        for (i = 1; i < canput; ++i){
            if (re_search[i - 1]){
                g = -nega_scout_final(&nb[i], false, depth - 1, -beta, -alpha, use_mpc, mpct_in, vacant_lst);
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
        g = -nega_scout_final_nomemo(&nb[0], false, depth - 1, -beta, -alpha, use_mpc, mpct_in, n_nodes, vacant_lst);
        alpha = max(alpha, g);
        if (beta <= alpha)
            return alpha;
        v = max(v, g);
        for (int i = 1; i < canput; ++i){
            g = -nega_alpha_ordering_final_nomemo(&nb[i], false, depth - 1, -alpha - search_epsilon, -alpha, use_mpc, mpct_in, n_nodes, vacant_lst);
            if (beta <= g)
                return g;
            if (alpha < g){
                alpha = g;
                g = -nega_scout_final_nomemo(&nb[i], false, depth - 1, -beta, -alpha, use_mpc, mpct_in, n_nodes, vacant_lst);
                alpha = max(alpha, g);
                if (beta <= alpha)
                    return alpha;
                v = max(v, g);
            }
        }
    #endif
    return v;
}

int mtd_final(board *b, bool skipped, int depth, int l, int u, bool use_mpc, double use_mpct, int g, unsigned long long *n_nodes, const vector<int> &vacant_lst){
    int beta;
    l /= 2;
    u /= 2;
    g = max(l, min(u, g / 2));
    //cerr << l * 2 << " " << g * 2 << " " << u * 2 << endl;
    bool searching = true;
    while (u - l > 0){
        beta = max(l + search_epsilon, g);
        g = nega_alpha_ordering_final(b, skipped, depth, beta * 2 - 2, beta * 2, use_mpc, use_mpct, n_nodes, vacant_lst, &searching) / 2;
        if (g < beta)
            u = g;
        else
            l = g;
        //cerr << l * 2 << " " << g * 2 << " " << u * 2 << endl;
    }
    //cerr << g * 2 << endl;
    return g * 2;
}

inline search_result endsearch(board b, long long strt, bool use_mpc, double use_mpct, const vector<int> vacant_lst){
    unsigned long long legal = b.mobility_ull();
    vector<pair<int, board>> nb;
    mobility mob;
    vector<int> prev_vals;
    int i;
    for (const int &cell: vacant_lst){
        if (1 & (legal >> cell)){
            calc_flip(&mob, &b, cell);
            nb.emplace_back(make_pair(cell, b.move_copy(&mob)));
            nb[nb.size() - 1].second.v = -mid_evaluate(&nb[nb.size() - 1].second);
        }
    }
    int canput = nb.size();
    if (canput == 0){
        search_result empty_res;
        empty_res.policy = -1;
        empty_res.value = -100;
        empty_res.depth = -1;
        empty_res.nps = -1;
        return empty_res;
    }
    if (canput >= 2)
        sort(nb.begin(), nb.end(), move_ordering_sort);
    int policy = -1;
    int tmp_policy = -1;
    int alpha, beta, g, value;
    unsigned long long searched_nodes = 0;
    int max_depth = hw2 - b.n;
    long long final_strt = tim();
    searched_nodes = 0;
    bool searching = true;
    if (nb[0].second.n < hw2 - 5){
        /*
        if (search_completed){
            int l, u;
            for (i = 0; i < canput; ++i){
                transpose_table.get_prev(&nb[i].second, nb[i].second.hash() & search_hash_mask, &l, &u);
                if (l != -inf)
                    nb[i].second.v = -l;
                else if (u != inf)
                    nb[i].second.v = -u;
                else
                    nb[i].second.v = -hw2 - 1;
            }
            if (canput >= 2)
                sort(nb.begin(), nb.end(), move_ordering_sort);
            cerr << "already completely searched policy " << nb[0].first << " value " << nb[0].second.v << endl;
        }
        */
        if (true || !search_completed || nb[0].second.v == -hw2 - 1){
            if (canput >= 2)
                sort(nb.begin(), nb.end(), move_ordering_sort);
            swap(transpose_table.now, transpose_table.prev);
            transpose_table.init_now();
            vector<double> pre_search_mpcts;
            pre_search_mpcts.emplace_back(0.5);
            if (use_mpct > 1.6 || !use_mpc)
                pre_search_mpcts.emplace_back(1.0);
            if (use_mpct > 2.0 || !use_mpc)
                pre_search_mpcts.emplace_back(1.5);
            for (double pre_search_mpct: pre_search_mpcts){
                alpha = -hw2;
                beta = hw2;
                for (i = 0; i < canput; ++i){
                    //nb[i].second.v = -mtd_final(&nb[i].second, false, max_depth - 1, -beta, min(hw2, -alpha + 6), true, pre_search_mpct, -nb[i].second.v, &searched_nodes);
                    nb[i].second.v = -nega_scout_final(&nb[i].second, false, max_depth - 1, -beta, min(hw2, -alpha + 6), true, pre_search_mpct, &searched_nodes, vacant_lst, &searching) / 2 * 2;
                    alpha = max(alpha, nb[i].second.v);
                }
                if (canput >= 2)
                    sort(nb.begin(), nb.end(), move_ordering_sort);
                swap(transpose_table.now, transpose_table.prev);
                transpose_table.init_now();
                cerr << "pre search mpct " << pre_search_mpct << " time " << tim() - strt << " policy " << nb[0].first << " value " << nb[0].second.v << " nodes " << searched_nodes << " nps " << searched_nodes * 1000 / max(1LL, tim() - strt) << endl;
            }
        }
        alpha = -hw2;
        beta = hw2;
        final_strt = tim();
        searched_nodes = 0;
        unsigned long long pre_nodes;
        for (i = 0; i < canput; ++i){
            pre_nodes = searched_nodes;
            if (use_mpc)
                g = -nega_scout_final(&nb[i].second, false, max_depth - 1, -beta, -alpha, use_mpc, use_mpct, &searched_nodes, vacant_lst, &searching) / 2 * 2;
            else
                g = -mtd_final(&nb[i].second, false, max_depth - 1, -beta, -alpha, use_mpc, use_mpct, -nb[i].second.v, &searched_nodes, vacant_lst);
            cerr << "policy " << nb[i].first << " value " << g << " expected " << nb[i].second.v << " alpha " << alpha << " nodes " << searched_nodes - pre_nodes << " nps: " << (long long)searched_nodes * 1000 / max(1LL, tim() - final_strt) << endl;
            if (alpha < g || i == 0){
                alpha = g;
                tmp_policy = nb[i].first;
            }
        }
    } else{
        int cells[5];
        alpha = -hw2;
        beta = hw2;
        for (i = 0; i < canput; ++i){
            pick_vacant(&nb[i].second, cells, vacant_lst);
            if (nb[i].second.n == hw2 - 5)
                g = -last5(&nb[i].second, false, -beta, -alpha, cells[0], cells[1], cells[2], cells[3], cells[4], &searched_nodes);
            else if (nb[i].second.n == hw2 - 4)
                g = -last4(&nb[i].second, false, -beta, -alpha, cells[0], cells[1], cells[2], cells[3], &searched_nodes);
            else if (nb[i].second.n == hw2 - 3)
                g = -last3(&nb[i].second, false, -beta, -alpha, cells[0], cells[1], cells[2], &searched_nodes);
            else if (nb[i].second.n == hw2 - 2)
                g = -last2(&nb[i].second, false, -beta, -alpha, cells[0], cells[1], &searched_nodes);
            else if (nb[i].second.n == hw2 - 1)
                g = -last1(&nb[i].second, -beta, -alpha, cells[0], &searched_nodes);
            else
                g = -end_evaluate(&nb[i].second);
            if (alpha < g || i == 0){
                alpha = g;
                tmp_policy = nb[i].first;
            }
        }
    }
    if (global_searching){
        policy = tmp_policy;
        value = alpha;
        transpose_table.reg(&b, b.hash() & search_hash_mask, value, value);
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
    search_completed = !use_mpc;
    return res;
}

inline search_result endsearch_value_nomemo(board b, long long strt, bool use_mpc, double use_mpct, const vector<int> vacant_lst){
    int max_depth = hw2 - b.n;
    unsigned long long searched_nodes = 0;
    search_result res;
    res.policy = -1;
    if (b.n < hw2 - 5)
        res.value = nega_scout_final_nomemo(&b, false, max_depth, -hw2, hw2, use_mpc, use_mpct, &searched_nodes, vacant_lst);
    else{
        int cells[5];
        pick_vacant(&b, cells, vacant_lst);
        if (b.n == hw2 - 5)
            res.value = last5(&b, false, -hw2, hw2, cells[0], cells[1], cells[2], cells[3], cells[4], &searched_nodes);
        else if (b.n == hw2 - 4)
            res.value = last4(&b, false, -hw2, hw2, cells[0], cells[1], cells[2], cells[3], &searched_nodes);
        else if (b.n == hw2 - 3)
            res.value = last3(&b, false, -hw2, hw2, cells[0], cells[1], cells[2], &searched_nodes);
        else if (b.n == hw2 - 2)
            res.value = last2(&b, false, -hw2, hw2, cells[0], cells[1], &searched_nodes);
        else if (b.n == hw2 - 1)
            res.value = last1(&b, -hw2, hw2, cells[0], &searched_nodes);
        else
            res.value = end_evaluate(&b);
    }
    res.depth = max_depth;
    res.nps = searched_nodes * 1000 / max(1LL, tim() - strt);
    //cerr << res.value << endl;
    return res;
}

inline search_result endsearch_value_memo(board b, long long strt, bool use_mpc, double use_mpct, int pre_calc_value, const vector<int> vacant_lst){
    int max_depth = hw2 - b.n;
    unsigned long long searched_nodes = 0;
    search_result res;
    res.policy = -1;
    bool searching = true;
    if (b.n < hw2 - 5){
        if (use_mpct)
            res.value = nega_scout_final(&b, false, max_depth, -hw2, hw2, use_mpc, use_mpct, &searched_nodes, vacant_lst, &searching);
        else
            res.value = mtd_final(&b, false, max_depth, -hw2, hw2, use_mpc, use_mpct, pre_calc_value, &searched_nodes, vacant_lst);
    } else{
        int cells[5];
        pick_vacant(&b, cells, vacant_lst);
        if (b.n == hw2 - 5)
            res.value = last5(&b, false, -hw2, hw2, cells[0], cells[1], cells[2], cells[3], cells[4], &searched_nodes);
        else if (b.n == hw2 - 4)
            res.value = last4(&b, false, -hw2, hw2, cells[0], cells[1], cells[2], cells[3], &searched_nodes);
        else if (b.n == hw2 - 3)
            res.value = last3(&b, false, -hw2, hw2, cells[0], cells[1], cells[2], &searched_nodes);
        else if (b.n == hw2 - 2)
            res.value = last2(&b, false, -hw2, hw2, cells[0], cells[1], &searched_nodes);
        else if (b.n == hw2 - 1)
            res.value = last1(&b, -hw2, hw2, cells[0], &searched_nodes);
        else
            res.value = end_evaluate(&b);
    }
    //cerr << "endsearch depth " << max_depth << " value " << res.value << " nodes " << searched_nodes << " time " << tim() - strt << " nps " << searched_nodes * 1000 / max(1LL, tim() - strt) << endl;
    res.depth = max_depth;
    res.nps = searched_nodes * 1000 / max(1LL, tim() - strt);
    //cerr << res.value << endl;
    return res;
}

inline search_result endsearch_value_analyze_memo(board b, long long strt, bool use_mpc, double use_mpct, const vector<int> vacant_lst){
    int max_depth = hw2 - b.n;
    unsigned long long searched_nodes = 0;
    int pre_calc_value;
    transpose_table.init_now();
    vector<double> pre_search_mpcts;
    bool searching = true;
    pre_search_mpcts.emplace_back(0.35);
    if (use_mpct > 1.0 || !use_mpc)
        pre_search_mpcts.emplace_back(0.5);
    if (use_mpct > 1.6 || !use_mpc)
        pre_search_mpcts.emplace_back(0.9);
    if (use_mpct > 2.0 || !use_mpc)
        pre_search_mpcts.emplace_back(1.2);
    for (double pre_search_mpct: pre_search_mpcts){
        pre_calc_value = -nega_scout_final(&b, false, max_depth, -hw2, hw2, true, pre_search_mpct, &searched_nodes, vacant_lst, &searching) / 2 * 2;
        cerr << "endsearch presearch mpct " << pre_search_mpct << " value " << pre_calc_value << " nodes " << searched_nodes << " time " << tim() - strt << " nps " << searched_nodes * 1000 / max(1LL, tim() - strt) << endl;
        swap(transpose_table.now, transpose_table.prev);
        transpose_table.init_now();
        searched_nodes = 0;
    }
    search_result res;
    res.policy = -1;
    if (b.n < hw2 - 5){
        if (use_mpct)
            res.value = nega_scout_final(&b, false, max_depth, -hw2, hw2, use_mpc, use_mpct, &searched_nodes, vacant_lst, &searching) / 2 * 2;
        else
            res.value = mtd_final(&b, false, max_depth, -hw2, hw2, use_mpc, use_mpct, pre_calc_value, &searched_nodes, vacant_lst);
    } else{
        int cells[5];
        pick_vacant(&b, cells, vacant_lst);
        if (b.n == hw2 - 5)
            res.value = last5(&b, false, -hw2, hw2, cells[0], cells[1], cells[2], cells[3], cells[4], &searched_nodes);
        else if (b.n == hw2 - 4)
            res.value = last4(&b, false, -hw2, hw2, cells[0], cells[1], cells[2], cells[3], &searched_nodes);
        else if (b.n == hw2 - 3)
            res.value = last3(&b, false, -hw2, hw2, cells[0], cells[1], cells[2], &searched_nodes);
        else if (b.n == hw2 - 2)
            res.value = last2(&b, false, -hw2, hw2, cells[0], cells[1], &searched_nodes);
        else if (b.n == hw2 - 1)
            res.value = last1(&b, -hw2, hw2, cells[0], &searched_nodes);
        else
            res.value = end_evaluate(&b);
    }
    cerr << "endsearch depth " << max_depth << " value " << res.value << " nodes " << searched_nodes << " time " << tim() - strt << " nps " << searched_nodes * 1000 / max(1LL, tim() - strt) << endl;
    res.depth = max_depth;
    res.nps = searched_nodes * 1000 / max(1LL, tim() - strt);
    //cerr << res.value << endl;
    return res;
}