#pragma once
#include <iostream>
#include <vector>
#include <functional>
#include "setting.hpp"
#include "common.hpp"
#include "board.hpp"
#include "evaluate.hpp"
#include "search.hpp"
#include "move_ordering.hpp"
#include "transpose_table.hpp"
#if USE_MULTI_THREAD
    #include "thread_pool.hpp"
    #include "ybwc.hpp"
#endif

using namespace std;

int nega_alpha_eval1(Search *search, int alpha, int beta, bool skipped);
int nega_alpha(Search *search, int alpha, int beta, int depth, bool skipped);
int nega_alpha_end_nomemo(Search *search, int alpha, int beta, int depth, bool skipped, uint64_t legal);

inline bool mpc_end_higher(Search *search, int beta, uint64_t legal){
    const int depth = HW2 - search->board.n;
    int bound = beta + ceil(search->mpct * mpcsd_final[depth - END_MPC_MIN_DEPTH]);
    bool res;
    switch(mpcd_final[depth]){
        case 0:
            res = mid_evaluate(&search->board) >= bound;
            break;
        case 1:
            res = nega_alpha_eval1(search, bound - 1, bound, false) >= bound;
            break;
        default:
            if (mpcd_final[depth] <= MID_FAST_DEPTH)
                res = nega_alpha(search, bound - 1, bound, mpcd_final[depth], false) >= bound;
            else{
                double mpct = search->mpct;
                search->mpct = 1.18;
                    res = nega_alpha_end_nomemo(search, bound - 1, bound, mpcd_final[depth], false, legal) >= bound;
                search->mpct = mpct;
            }
            break;
    }
    return res;
}

inline bool mpc_end_lower(Search *search, int alpha, uint64_t legal){
    const int depth = HW2 - search->board.n;
    int bound = alpha - ceil(search->mpct * mpcsd_final[depth - END_MPC_MIN_DEPTH]);
    bool res;
    switch(mpcd_final[depth]){
        case 0:
            res = mid_evaluate(&search->board) <= bound;
            break;
        case 1:
            res = nega_alpha_eval1(search, bound, bound + 1, false) <= bound;
            break;
        default:
            if (mpcd_final[depth] <= MID_FAST_DEPTH)
                res = nega_alpha(search, bound, bound + 1, mpcd_final[depth], false) <= bound;
            else{
                double mpct = search->mpct;
                search->mpct = 1.18;
                    res = nega_alpha_end_nomemo(search, bound, bound + 1, mpcd_final[depth], false, legal) <= bound;
                search->mpct = mpct;
            }
            break;
    }
    return res;
}

int nega_alpha_end_nomemo(Search *search, int alpha, int beta, int depth, bool skipped, uint64_t legal){
    if (!global_searching)
        return SCORE_UNDEFINED;
    if (depth <= MID_FAST_DEPTH)
        return nega_alpha(search, alpha, beta, depth, skipped);
    if (depth == 1)
        return nega_alpha_eval1(search, alpha, beta, skipped);
    ++(search->n_nodes);
    #if USE_END_SC
        int stab_res = stability_cut(search, &alpha, &beta);
        if (stab_res != SCORE_UNDEFINED)
            return stab_res;
    #endif
    if (legal == LEGAL_UNDEFINED)
        legal = search->board.get_legal();
    int g, v = -INF;
    if (legal == 0ULL){
        if (skipped)
            return end_evaluate(&search->board);
        search->board.pass();
            v = -nega_alpha_end_nomemo(search, -beta, -alpha, depth, true, LEGAL_UNDEFINED);
        search->board.pass();
        return v;
    }
    const int canput = pop_count_ull(legal);
    vector<Flip> move_list(canput);
    int idx = 0;
    for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal))
        calc_flip(&move_list[idx++], &search->board, cell);
    move_ordering(search, move_list, depth, alpha, beta, false);
    for (const Flip &flip: move_list){
        search->board.move(&flip);
            g = -nega_alpha_end_nomemo(search, -beta, -alpha, depth - 1, false, flip.n_legal);
        search->board.undo(&flip);
        alpha = max(alpha, g);
        v = max(v, g);
        if (beta <= alpha)
            break;
    }
    return v;
}

inline int last1(Search *search, int alpha, int beta, int p0){
    ++search->n_nodes;
    int score = HW2 - 2 * search->board.count_opponent();
    int n_flip;
    n_flip = count_last_flip(search->board.player, search->board.opponent, p0);
    if (n_flip == 0){
        ++search->n_nodes;
        if (score <= 0){
            score -= 2;
            if (score >= alpha){
                n_flip = count_last_flip(search->board.opponent, search->board.player, p0);
                score -= 2 * n_flip;
            }
        } else{
            if (score >= alpha){
                n_flip = count_last_flip(search->board.opponent, search->board.player, p0);
                if (n_flip)
                    score -= 2 * n_flip + 2;
            }
        }
        
    } else
        score += 2 * n_flip;
    return score;
}

inline int last2(Search *search, int alpha, int beta, int p0, int p1, bool skipped){
    ++search->n_nodes;
    #if USE_END_PO & false
        int p0_parity = (search->board.parity & cell_div4[p0]);
        int p1_parity = (search->board.parity & cell_div4[p1]);
        if (!p0_parity && p1_parity)
            swap(p0, p1);
    #endif
    int v = -INF, g;
    Flip flip;
    calc_flip(&flip, &search->board, p0);
    if (flip.flip){
        search->board.move(&flip);
            g = -last1(search, -beta, -alpha, p1);
        search->board.undo(&flip);
        alpha = max(alpha, g);
        if (beta <= alpha)
            return alpha;
        v = max(v, g);
    }
    calc_flip(&flip, &search->board, p1);
    if (flip.flip){
        search->board.move(&flip);
            g = -last1(search, -beta, -alpha, p0);
        search->board.undo(&flip);
        alpha = max(alpha, g);
        if (beta <= alpha)
            return alpha;
        v = max(v, g);
    }
    if (v == -INF){
        if (skipped)
            v = end_evaluate(&search->board);
        else{
            search->board.pass();
                v = -last2(search, -beta, -alpha, p0, p1, true);
            search->board.pass();
        }
    }
    return v;
}

inline int last3(Search *search, int alpha, int beta, int p0, int p1, int p2, bool skipped){
    ++search->n_nodes;
    #if USE_END_PO
        if (!skipped){
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
    uint64_t legal = search->board.get_legal();
    int v = -INF, g;
    if (legal == 0ULL){
        if (skipped)
            v = end_evaluate(&search->board);
        else{
            search->board.pass();
                v = -last3(search, -beta, -alpha, p0, p1, p2, true);
            search->board.pass();
        }
        return v;
    }
    Flip flip;
    if (1 & (legal >> p0)){
        calc_flip(&flip, &search->board, p0);
        search->board.move(&flip);
            g = -last2(search, -beta, -alpha, p1, p2, false);
        search->board.undo(&flip);
        alpha = max(alpha, g);
        if (beta <= alpha)
            return alpha;
        v = max(v, g);
    }
    if (1 & (legal >> p1)){
        calc_flip(&flip, &search->board, p1);
        search->board.move(&flip);
            g = -last2(search, -beta, -alpha, p0, p2, false);
        search->board.undo(&flip);
        alpha = max(alpha, g);
        if (beta <= alpha)
            return alpha;
        v = max(v, g);
    }
    if (1 & (legal >> p2)){
        calc_flip(&flip, &search->board, p2);
        search->board.move(&flip);
            g = -last2(search, -beta, -alpha, p0, p1, false);
        search->board.undo(&flip);
        alpha = max(alpha, g);
        if (beta <= alpha)
            return alpha;
        v = max(v, g);
    }
    return v;
}

inline int last4(Search *search, int alpha, int beta, int p0, int p1, int p2, int p3, bool skipped){
    ++search->n_nodes;
    #if USE_END_PO
        if (!skipped){
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
    uint64_t legal = search->board.get_legal();
    int v = -INF, g;
    if (legal == 0ULL){
        if (skipped)
            v = end_evaluate(&search->board);
        else{
            search->board.pass();
                v = -last4(search, -beta, -alpha, p0, p1, p2, p3, true);
            search->board.pass();
        }
        return v;
    }
    Flip flip;
    if (1 & (legal >> p0)){
        calc_flip(&flip, &search->board, p0);
        search->board.move(&flip);
            g = -last3(search, -beta, -alpha, p1, p2, p3, false);
        search->board.undo(&flip);
        alpha = max(alpha, g);
        if (beta <= alpha)
            return alpha;
        v = max(v, g);
    }
    if (1 & (legal >> p1)){
        calc_flip(&flip, &search->board, p1);
        search->board.move(&flip);
            g = -last3(search, -beta, -alpha, p0, p2, p3, false);
        search->board.undo(&flip);
        alpha = max(alpha, g);
        if (beta <= alpha)
            return alpha;
        v = max(v, g);
    }
    if (1 & (legal >> p2)){
        calc_flip(&flip, &search->board, p2);
        search->board.move(&flip);
            g = -last3(search, -beta, -alpha, p0, p1, p3, false);
        search->board.undo(&flip);
        alpha = max(alpha, g);
        if (beta <= alpha)
            return alpha;
        v = max(v, g);
    }
    if (1 & (legal >> p3)){
        calc_flip(&flip, &search->board, p3);
        search->board.move(&flip);
            g = -last3(search, -beta, -alpha, p0, p1, p2, false);
        search->board.undo(&flip);
        alpha = max(alpha, g);
        if (beta <= alpha)
            return alpha;
        v = max(v, g);
    }
    return v;
}

inline void pick_vacant(Search *search, uint_fast8_t cells[]){
    int idx = 0;
    uint64_t empties = ~(search->board.player | search->board.opponent);
    for (uint_fast8_t cell = first_bit(&empties); empties; cell = next_bit(&empties))
        cells[idx++] = cell;
}

int nega_alpha_end_fast(Search *search, int alpha, int beta, bool skipped){
    if (!global_searching)
        return SCORE_UNDEFINED;
    if (search->board.n == 60){
        uint_fast8_t cells[4];
        pick_vacant(search, cells);
        return last4(search, alpha, beta, cells[0], cells[1], cells[2], cells[3], skipped);
    }
    ++search->n_nodes;
    #if USE_END_SC
        int stab_res = stability_cut(search, &alpha, &beta);
        if (stab_res != SCORE_UNDEFINED)
            return stab_res;
    #endif
    uint64_t legal = search->board.get_legal();
    int g, v = -INF;
    if (legal == 0ULL){
        if (skipped)
            return end_evaluate(&search->board);
        search->board.pass();
            v = -nega_alpha_end_fast(search, -beta, -alpha, true);
        search->board.pass();
        return v;
    }
    Flip flip;
    #if USE_END_PO
        if (0 < search->board.parity && search->board.parity < 15){
            uint64_t legal_copy = legal;
            for (uint_fast8_t cell = first_odd_bit(&legal, search->board.parity); legal; cell = next_odd_bit(&legal, search->board.parity)){
                calc_flip(&flip, &search->board, cell);
                search->board.move(&flip);
                    g = -nega_alpha_end_fast(search, -beta, -alpha, false);
                search->board.undo(&flip);
                alpha = max(alpha, g);
                if (beta <= alpha)
                    return alpha;
                v = max(v, g);
            }
            for (uint_fast8_t cell = first_even_bit(&legal_copy, search->board.parity); legal_copy; cell = next_even_bit(&legal_copy, search->board.parity)){
                calc_flip(&flip, &search->board, cell);
                search->board.move(&flip);
                    g = -nega_alpha_end_fast(search, -beta, -alpha, false);
                search->board.undo(&flip);
                alpha = max(alpha, g);
                if (beta <= alpha)
                    return alpha;
                v = max(v, g);
            }
        } else{
            for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)){
                calc_flip(&flip, &search->board, cell);
                search->board.move(&flip);
                    g = -nega_alpha_end_fast(search, -beta, -alpha, false);
                search->board.undo(&flip);
                alpha = max(alpha, g);
                if (beta <= alpha)
                    return alpha;
                v = max(v, g);
            }
        }
    #else
        for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)){
            calc_flip(&flip, &search->board, cell);
            search->board.move(&flip);
                g = -nega_alpha_end_fast(search, -beta, -alpha, false);
            search->board.undo(&flip);
            alpha = max(alpha, g);
            if (beta <= alpha)
                return alpha;
            v = max(v, g);
        }
    #endif
    return v;
}

int nega_alpha_end(Search *search, int alpha, int beta, bool skipped, uint64_t legal, const bool *searching){
    if (!global_searching || !(*searching))
        return SCORE_UNDEFINED;
    if (search->board.n >= HW2 - END_FAST_DEPTH)
        return nega_alpha_end_fast(search, alpha, beta, skipped);
    ++search->n_nodes;
    #if USE_END_SC
        int stab_res = stability_cut(search, &alpha, &beta);
        if (stab_res != SCORE_UNDEFINED)
            return stab_res;
    #endif
    uint32_t hash_code = search->board.hash() & TRANSPOSE_TABLE_MASK;
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
            if (mpc_end_higher(search, beta, legal))
                return beta;
            if (mpc_end_lower(search, alpha, legal))
                return alpha;
        }
    #endif
    if (legal == LEGAL_UNDEFINED)
        legal = search->board.get_legal();
    int g, v = -INF;
    if (legal == 0ULL){
        if (skipped)
            return end_evaluate(&search->board);
        search->board.pass();
            v = -nega_alpha_end(search, -beta, -alpha, true, LEGAL_UNDEFINED, searching);
        search->board.pass();
        return v;
    }
    const int canput = pop_count_ull(legal);
    vector<Flip> move_list(canput);
    int idx = 0;
    for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal))
        calc_flip(&move_list[idx++], &search->board, cell);
    move_ordering_fast_first(search, move_list);
    int best_move = -1;
    for (const Flip &flip: move_list){
        search->board.move(&flip);
            g = -nega_alpha_end(search, -beta, -alpha, false, flip.n_legal, searching);
        search->board.undo(&flip);
        alpha = max(alpha, g);
        if (v < g){
            v = g;
            best_move = flip.pos;
        }
        if (beta <= alpha)
            break;
    }
    child_transpose_table.reg(&search->board, hash_code, best_move, v);
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
