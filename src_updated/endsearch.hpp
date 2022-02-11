#pragma once
#include <iostream>
#include <vector>
#include <functional>
#include "setting.hpp"
#include "common.hpp"
#include "board.hpp"
#include "evaluate.hpp"
#include "search.hpp"
#include "transpose_table.hpp"
#if USE_MULTI_THREAD
    #include "thread_pool.hpp"
    #include "ybwc.hpp"
#endif

using namespace std;

inline bool mpc_end_higher(Search *search, int beta, int val){
    int bound = beta + ceil(search->mpct * mpcsd_final[(HW2 - search->board.n - END_MPC_MIN_DEPTH) / 5][(val + HW2) / 6]);
    return val >= bound;
}

inline bool mpc_end_lower(Search *search, int alpha, int val){
    int bound = alpha - ceil(search->mpct * mpcsd_final[(HW2 - search->board.n - END_MPC_MIN_DEPTH) / 5][(val + HW2) / 6]);
    return val <= bound;
}

inline int last1(Search *search, int alpha, int beta, int p0){
    ++search->n_nodes;
    int score = HW2 - 2 * search->board.count_opponent();
    int n_flip;
    if (search->board.p == BLACK)
        n_flip = count_last_flip(search->board.b, search->board.w, p0);
    else
        n_flip = count_last_flip(search->board.w, search->board.b, p0);
    if (n_flip == 0){
        ++search->n_nodes;
        if (score <= 0){
            score -= 2;
            if (score >= alpha){
                if (search->board.p == WHITE)
                    n_flip = count_last_flip(search->board.b, search->board.w, p0);
                else
                    n_flip = count_last_flip(search->board.w, search->board.b, p0);
                score -= 2 * n_flip;
            }
        } else{
            if (score >= alpha){
                if (search->board.p == WHITE)
                    n_flip = count_last_flip(search->board.b, search->board.w, p0);
                else
                    n_flip = count_last_flip(search->board.w, search->board.b, p0);
                if (n_flip)
                    score -= 2 * n_flip + 2;
            }
        }
        
    } else
        score += 2 * n_flip;
    return score;
}

inline int last2(Search *search, int alpha, int beta, int p0, int p1){
    ++search->n_nodes;
    #if USE_END_PO & false
        int p0_parity = (search->board.parity & cell_div4[p0]);
        int p1_parity = (search->board.parity & cell_div4[p1]);
        if (!p0_parity && p1_parity)
            swap(p0, p1);
    #endif
    int v = -INF, g;
    Mobility mob;
    calc_flip(&mob, &search->board, p0);
    if (mob.flip){
        search->board.move(&mob);
            g = -last1(search, -beta, -alpha, p1);
        search->board.undo(&mob);
        alpha = max(alpha, g);
        if (beta <= alpha)
            return alpha;
        v = max(v, g);
    }
    calc_flip(&mob, &search->board, p1);
    if (mob.flip){
        search->board.move(&mob);
            g = -last1(search, -beta, -alpha, p0);
        search->board.undo(&mob);
        alpha = max(alpha, g);
        if (beta <= alpha)
            return alpha;
        v = max(v, g);
    }
    if (v == -INF){
        if (search->skipped)
            v = end_evaluate(&search->board);
        else{
            search->pass();
                v = -last2(search, -beta, -alpha, p0, p1);
            search->undo_pass();
        }
    }
    return v;
}

inline int last3(Search *search, int alpha, int beta, int p0, int p1, int p2){
    ++search->n_nodes;
    #if USE_END_PO
        if (!search->skipped){
            int p0_parity = (search->board.parity & cell_div4[p0]);
            int p1_parity = (search->board.parity & cell_div4[p1]);
            int p2_parity = (search->board.parity & cell_div4[p2]);
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
        }
    #endif
    unsigned long long legal = search->board.mobility_ull();
    int v = -INF, g;
    if (legal == 0){
        if (search->skipped)
            v = end_evaluate(&search->board);
        else{
            search->pass();
                v = -last3(search, -beta, -alpha, p0, p1, p2);
            search->undo_pass();
        }
        return v;
    }
    search->skipped = false;
    Mobility mob;
    if (1 & (legal >> p0)){
        calc_flip(&mob, &search->board, p0);
        search->board.move(&mob);
            g = -last2(search, -beta, -alpha, p1, p2);
        search->board.undo(&mob);
        alpha = max(alpha, g);
        if (beta <= alpha)
            return alpha;
        v = max(v, g);
    }
    if (1 & (legal >> p1)){
        calc_flip(&mob, &search->board, p1);
        search->board.move(&mob);
            g = -last2(search, -beta, -alpha, p0, p2);
        search->board.undo(&mob);
        alpha = max(alpha, g);
        if (beta <= alpha)
            return alpha;
        v = max(v, g);
    }
    if (1 & (legal >> p2)){
        calc_flip(&mob, &search->board, p2);
        search->board.move(&mob);
            g = -last2(search, -beta, -alpha, p0, p1);
        search->board.undo(&mob);
        alpha = max(alpha, g);
        if (beta <= alpha)
            return alpha;
        v = max(v, g);
    }
    return v;
}

inline int last4(Search *search, int alpha, int beta, int p0, int p1, int p2, int p3){
    ++search->n_nodes;
    #if USE_END_PO
        if (!search->skipped){
            int p0_parity = (search->board.parity & cell_div4[p0]);
            int p1_parity = (search->board.parity & cell_div4[p1]);
            int p2_parity = (search->board.parity & cell_div4[p2]);
            int p3_parity = (search->board.parity & cell_div4[p3]);
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
    unsigned long long legal = search->board.mobility_ull();
    int v = -INF, g;
    if (legal == 0){
        if (search->skipped)
            v = end_evaluate(&search->board);
        else{
            search->pass();
                v = -last4(search, -beta, -alpha, p0, p1, p2, p3);
            search->undo_pass();
        }
        return v;
    }
    search->skipped = false;
    Mobility mob;
    if (1 & (legal >> p0)){
        calc_flip(&mob, &search->board, p0);
        search->board.move(&mob);
            g = -last3(search, -beta, -alpha, p1, p2, p3);
        search->board.undo(&mob);
        alpha = max(alpha, g);
        if (beta <= alpha)
            return alpha;
        v = max(v, g);
    }
    if (1 & (legal >> p1)){
        calc_flip(&mob, &search->board, p1);
        search->board.move(&mob);
            g = -last3(search, -beta, -alpha, p0, p2, p3);
        search->board.undo(&mob);
        alpha = max(alpha, g);
        if (beta <= alpha)
            return alpha;
        v = max(v, g);
    }
    if (1 & (legal >> p2)){
        calc_flip(&mob, &search->board, p2);
        search->board.move(&mob);
            g = -last3(search, -beta, -alpha, p0, p1, p3);
        search->board.undo(&mob);
        alpha = max(alpha, g);
        if (beta <= alpha)
            return alpha;
        v = max(v, g);
    }
    if (1 & (legal >> p3)){
        calc_flip(&mob, &search->board, p3);
        search->board.move(&mob);
            g = -last3(search, -beta, -alpha, p0, p1, p2);
        search->board.undo(&mob);
        alpha = max(alpha, g);
        if (beta <= alpha)
            return alpha;
        v = max(v, g);
    }
    return v;
}

inline void pick_vacant(Board *b, int cells[], const vector<int> &vacant_list){
    int idx = 0;
    unsigned long long empties = ~(b->b | b->w);
    for (const int &cell: vacant_list){
        if (1 & (empties >> cell))
            cells[idx++] = cell;
    }
}


int nega_alpha_end_fast(Search *search, int alpha, int beta){
    if (!global_searching)
        return SCORE_UNDEFINED;
    if (search->board.n == 60){
        int cells[4];
        pick_vacant(&search->board, cells, search->vacant_list);
        return last4(search, alpha, beta, cells[0], cells[1], cells[2], cells[3]);
    }
    ++search->n_nodes;
    #if USE_END_SC
        int stab_res = stability_cut(search, &alpha, &beta);
        if (stab_res != SCORE_UNDEFINED)
            return stab_res;
    #endif
    unsigned long long legal = search->board.mobility_ull();
    int g, v = -INF;
    if (legal == 0){
        if (search->skipped)
            return end_evaluate(&search->board);
        search->pass();
            v = -nega_alpha_end_fast(search, -beta, -alpha);
        search->undo_pass();
        return v;
    }
    search->skipped = false;
    Mobility mob;
    #if USE_END_PO
        if (0 < search->board.parity && search->board.parity < 15){
            for (const int &cell: search->vacant_list){
                if ((search->board.parity & cell_div4[cell]) && (1 & (legal >> cell))){
                    calc_flip(&mob, &search->board, cell);
                    search->board.move(&mob);
                        g = -nega_alpha_end_fast(search, -beta, -alpha);
                    search->board.undo(&mob);
                    alpha = max(alpha, g);
                    if (beta <= alpha)
                        return alpha;
                    v = max(v, g);
                }
            }
            for (const int &cell: search->vacant_list){
                if ((search->board.parity & cell_div4[cell]) == 0 && (1 & (legal >> cell))){
                    calc_flip(&mob, &search->board, cell);
                    search->board.move(&mob);
                        g = -nega_alpha_end_fast(search, -beta, -alpha);
                    search->board.undo(&mob);
                    alpha = max(alpha, g);
                    if (beta <= alpha)
                        return alpha;
                    v = max(v, g);
                }
            }
        } else{
            for (const int &cell: search->vacant_list){
                if (1 & (legal >> cell)){
                    calc_flip(&mob, &search->board, cell);
                    search->board.move(&mob);
                        g = -nega_alpha_end_fast(search, -beta, -alpha);
                    search->board.undo(&mob);
                    alpha = max(alpha, g);
                    if (beta <= alpha)
                        return alpha;
                    v = max(v, g);
                }
            }
        }
    #else
        for (const int &cell: search->vacant_list){
            if (1 & (legal >> cell)){
                calc_flip(&mob, &search->board, cell);
                search->board.move(&mob);
                    g = -nega_alpha_end_fast(search, -beta, -alpha);
                search->board.undo(&mob);
                alpha = max(alpha, g);
                if (beta <= alpha)
                    return alpha;
                v = max(v, g);
            }
        }
    #endif
    return v;
}

int nega_alpha_end(Search *search, int alpha, int beta){
    if (!global_searching)
        return SCORE_UNDEFINED;
    if (search->board.n >= HW2 - END_FAST_DEPTH){
        cout_log();
        return nega_alpha_end_fast(search, alpha, beta);
    }
    ++search->n_nodes;
    #if USE_END_SC
        int stab_res = stability_cut(search, &alpha, &beta);
        if (stab_res != SCORE_UNDEFINED)
            return stab_res;
    #endif
    #if USE_END_TC
        int l, u, hash_code = search->board.hash() & TRANSPOSE_TABLE_MASK;
        parent_transpose_table.get(&search->board, hash_code, &l, &u);
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
        int depth = HW2 - search->board.n;
        if (END_MPC_MIN_DEPTH <= depth && depth <= END_MPC_MAX_DEPTH && search->use_mpc){
            int val = mid_evaluate(&search->board);
            if (mpc_end_higher(search, beta, val)){
                #if USE_END_TC && false
                    if (l < beta)
                        parent_transpose_table.reg(&search->board, hash_code, beta, u);
                #endif
                return beta;
            }
            if (mpc_end_lower(search, alpha, val)){
                #if USE_END_TC && false
                    if (alpha < u)
                        parent_transpose_table.reg(&search->board, hash_code, l, alpha);
                #endif
                return alpha;
            }
        }
    #endif
    unsigned long long legal = search->board.mobility_ull();
    int g, v = -INF;
    if (legal == 0){
        if (search->skipped)
            return end_evaluate(&search->board);
        search->pass();
            v = -nega_alpha_end(search, -beta, -alpha);
        search->undo_pass();
        return v;
    }
    search->skipped = false;
    vector<Mobility> move_list;
    for (const int &cell: search->vacant_list){
        if (1 & (legal >> cell))
            move_list.emplace_back(calc_flip(&search->board, cell));
    }
    move_ordering_fast_first(search, move_list);
    #if USE_MULTI_THREAD
        const int canput = pop_count_ull(legal);
        int pv_idx = 0, split_count = 0;
        vector<future<pair<int, unsigned long long>>> parallel_tasks;
        for (const Mobility &mob: move_list){
            search->board.move(&mob);
                if (ybwc_split_end(search, -beta, -alpha, mob.pos, pv_idx++, canput, split_count, parallel_tasks)){
                    search->board.undo(&mob);
                    ++split_count;
                } else{
                    g = -nega_alpha_end(search, -beta, -alpha);
                    search->board.undo(&mob);
                    alpha = max(alpha, g);
                    if (v < g){
                        v = g;
                        child_transpose_table.reg(&search->board, hash_code, mob.pos, g);
                    }
                    if (beta <= alpha)
                        break;
                }
        }
        if (split_count){
            g = ybwc_wait(search, parallel_tasks);
            alpha = max(alpha, g);
            v = max(v, g);
        }
    #else
        for (const Mobility &mob: move_list){
            search->board.move(&mob);
                g = -nega_alpha_end(search, -beta, -alpha);
            search->board.undo(&mob);
            alpha = max(alpha, g);
            if (v < g){
                v = g;
                child_transpose_table.reg(&search->board, hash_code, mob.pos, g);
            }
            if (beta <= alpha)
                break;
        }
    #endif
    #if USE_END_TC
        if (beta <= v)
            parent_transpose_table.reg(&search->board, hash_code, v, u);
        else if (v <= alpha)
            parent_transpose_table.reg(&search->board, hash_code, l, v);
        else
            parent_transpose_table.reg(&search->board, hash_code, v, v);
    #endif
    return v;
}

int nega_scout_end(Search *search, int alpha, int beta){
    if (!global_searching)
        return -INF;
    if (search->board.n >= HW2 - END_FAST_DEPTH)
        return nega_alpha_end_fast(search, alpha, beta);
    ++(search->n_nodes);
    #if USE_END_SC
        int stab_res = stability_cut(search, &alpha, &beta);
        if (stab_res != SCORE_UNDEFINED)
            return stab_res;
    #endif
    int hash_code = search->board.hash() & TRANSPOSE_TABLE_MASK;
    #if USE_END_TC
        int l, u;
        parent_transpose_table.get(&search->board, hash_code, &l, &u);
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
        int depth = HW2 - search->board.n;
        if (END_MPC_MIN_DEPTH <= depth && depth <= END_MPC_MAX_DEPTH && search->use_mpc){
            int val = mid_evaluate(&search->board);
            if (mpc_end_higher(search, beta, val)){
                #if USE_END_TC && false
                    if (l < beta)
                        parent_transpose_table.reg(&search->board, hash_code, beta, u);
                #endif
                return beta;
            }
            if (mpc_end_lower(search, alpha, val)){
                #if USE_END_TC && false
                    if (alpha < u)
                        parent_transpose_table.reg(&search->board, hash_code, l, alpha);
                #endif
                return alpha;
            }
        }
    #endif
    unsigned long long legal = search->board.mobility_ull();
    int g, v = -INF;
    if (legal == 0){
        if (search->skipped)
            return end_evaluate(&search->board);
        search->pass();
            v = -nega_scout_end(search, -beta, -alpha);
        search->undo_pass();
        return v;
    }
    search->skipped = false;
    vector<Mobility> move_list;
    for (const int &cell: search->vacant_list){
        if (1 & (legal >> cell))
            move_list.emplace_back(calc_flip(&search->board, cell));
    }
    move_ordering_fast_first(search, move_list);
    for (const Mobility &mob: move_list){
        search->board.move(&mob);
            if (v == -INF)
                g = -nega_scout_end(search, -beta, -alpha);
            else{
                g = -nega_alpha_end(search, -alpha - 1, -alpha);
                if (alpha < g)
                    g = -nega_scout_end(search, -beta, -g);
            }
        search->board.undo(&mob);
        alpha = max(alpha, g);
        if (v < g){
            v = g;
            child_transpose_table.reg(&search->board, hash_code, mob.pos, g);
        }
        if (beta <= alpha)
            break;
    }
    #if USE_END_TC
        if (beta <= v)
            parent_transpose_table.reg(&search->board, hash_code, v, u);
        else if (v <= alpha)
            parent_transpose_table.reg(&search->board, hash_code, l, v);
        else
            parent_transpose_table.reg(&search->board, hash_code, v, v);
    #endif
    return v;
}