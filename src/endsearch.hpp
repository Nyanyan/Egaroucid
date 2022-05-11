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
#include "probcut.hpp"
#include "transpose_table.hpp"
#if USE_MULTI_THREAD
    #include "thread_pool.hpp"
    #include "ybwc.hpp"
#endif
#include "util.hpp"

using namespace std;
/*
int nega_alpha_end_nomemo(Search *search, int alpha, int beta, int depth, bool skipped, uint64_t legal){
    if (!global_searching)
        return SCORE_UNDEFINED;
    if (depth <= MID_FAST_DEPTH)
        return nega_alpha(search, alpha, beta, depth, skipped);
    //if (depth == 1)
    //    return nega_alpha_eval1(search, alpha, beta, skipped);
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
    #if USE_MID_MPC
        if (search->use_mpc){
            if (mpc(search, alpha, beta, depth, legal, false, &v))
                return v;
        }
    #endif
    const int canput = pop_count_ull(legal);
    vector<Flip> move_list(canput);
    int idx = 0;
    for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal))
        calc_flip(&move_list[idx++], &search->board, cell);
    move_ordering(search, move_list, depth, alpha, beta, false);
    for (const Flip &flip: move_list){
        eval_move(search, &flip);
        search->board.move(&flip);
            g = -nega_alpha_end_nomemo(search, -beta, -alpha, depth - 1, false, flip.n_legal);
        search->board.undo(&flip);
        eval_undo(search, &flip);
        alpha = max(alpha, g);
        v = max(v, g);
        if (beta <= alpha)
            break;
    }
    return v;
}
*/
/*
inline int last1(Search *search, int alpha, int beta, int p0){
    ++search->n_nodes;
    int score = HW2 - 2 * search->board.count_opponent();
    int n_flip = count_last_flip(search->board.player, search->board.opponent, p0);
    if (n_flip == 0){
        ++search->n_nodes;
        n_flip = count_last_flip(search->board.opponent, search->board.player, p0);
        if (n_flip == 0){
            if (score < 1)
                score -= 2;
        } else
            score -= 2 * n_flip + 2;
    } else
        score += 2 * n_flip;
    return score_to_value(score);
}
*/

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
            uint64_t legal_mask = 0ULL;
            if (search->board.parity & 1)
                legal_mask |= 0x000000000F0F0F0FULL;
            if (search->board.parity & 2)
                legal_mask |= 0x00000000F0F0F0F0ULL;
            if (search->board.parity & 4)
                legal_mask |= 0x0F0F0F0F00000000ULL;
            if (search->board.parity & 8)
                legal_mask |= 0xF0F0F0F000000000ULL;
            uint64_t legal_copy;
            uint_fast8_t cell;
            int i;
            for (i = 0; i < N_CELL_WEIGHT_MASK; ++i){
                legal_copy = legal & legal_mask & cell_weight_mask[i];
                if (legal_copy){
                    for (cell = first_bit(&legal_copy); legal_copy; cell = next_bit(&legal_copy)){
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
            }
            legal_mask = ~legal_mask;
            for (i = 0; i < N_CELL_WEIGHT_MASK; ++i){
                legal_copy = legal & legal_mask & cell_weight_mask[i];
                if (legal_copy){
                    for (cell = first_bit(&legal_copy); legal_copy; cell = next_bit(&legal_copy)){
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
            }
        } else{
            uint64_t legal_copy;
            uint_fast8_t cell;
            int i;
            for (i = 0; i < N_CELL_WEIGHT_MASK; ++i){
                legal_copy = legal & cell_weight_mask[i];
                if (legal_copy){
                    for (cell = first_bit(&legal_copy); legal_copy; cell = next_bit(&legal_copy)){
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
            }
        }
    #else
        uint64_t legal_copy;
        uint_fast8_t cell;
        int i;
        for (i = 0; i < N_CELL_WEIGHT_MASK; ++i){
            legal_copy = legal & cell_weight_mask[i];
            if (legal_copy){
                for (cell = first_bit(&legal_copy); legal_copy; cell = next_bit(&legal_copy)){
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
        }
    #endif
    return v;
}
/*
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
    const int canput = pop_count_ull(legal);
    vector<Flip> move_list(canput);
    int idx = 0;
    for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal))
        calc_flip(&move_list[idx++], &search->board, cell);
    move_ordering_fast_first(search, move_list);
    for (const Flip &flip: move_list){
        search->board.move(&flip);
            g = -nega_alpha_end_fast(search, -beta, -alpha, false);
        search->board.undo(&flip);
        alpha = max(alpha, g);
        if (beta <= alpha)
            return alpha;
        v = max(v, g);
    }
    return v;
}
*/

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
        if (beta <= l)
            return l;
        if (u <= alpha)
            return u;
        alpha = max(alpha, l);
        beta = min(beta, u);
    #endif
    if (legal == LEGAL_UNDEFINED)
        legal = search->board.get_legal();
    int g, v = -INF;
    if (legal == 0ULL){
        if (skipped)
            return end_evaluate(&search->board);
        //search->eval_feature_reversed ^= 1;
        search->board.pass();
            v = -nega_alpha_end(search, -beta, -alpha, true, LEGAL_UNDEFINED, searching);
        search->board.pass();
        //search->eval_feature_reversed ^= 1;
        return v;
    }
    #if USE_END_MPC && false
        if (search->use_mpc){
            if (mpc(search, alpha, beta, HW2 - search->board.n, legal, false, &v))
                return v;
        }
    #endif
    int best_move = child_transpose_table.get(&search->board, hash_code);
    int f_best_move = best_move;
    if (best_move != TRANSPOSE_TABLE_UNDEFINED){
        Flip flip_best;
        calc_flip(&flip_best, &search->board, best_move);
        //eval_move(search, &flip);
        search->board.move(&flip_best);
            g = -nega_alpha_end(search, -beta, -alpha, false, LEGAL_UNDEFINED, searching);
        search->board.undo(&flip_best);
        //eval_undo(search, &flip);
        alpha = max(alpha, g);
        v = g;
        legal ^= 1ULL << best_move;
    }
    if (alpha < beta && legal){
        const int canput = pop_count_ull(legal);
        vector<Flip> move_list(canput);
        int idx = 0;
        for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal))
            calc_flip(&move_list[idx++], &search->board, cell);
        move_ordering_fast_first(search, move_list);
        #if USE_MULTI_THREAD && false
            int pv_idx = 0, split_count = 0;
            if (best_move != TRANSPOSE_TABLE_UNDEFINED)
                pv_idx = 1;
            vector<future<pair<int, uint64_t>>> parallel_tasks;
            bool n_searching = true;
            int depth = HW2 - pop_count_ull(search->board.player | search->board.opponent);
            for (const Flip &flip: move_list){
                if (!(*searching))
                    break;
                search->board.move(&flip);
                    if (ybwc_split_without_move_end(search, &flip, -beta, -alpha, depth - 1, flip.n_legal, &n_searching, pv_idx++, canput, split_count, parallel_tasks, move_list[0].value)){
                        ++split_count;
                    } else{
                        g = -nega_alpha_end(search, -beta, -alpha, false, flip.n_legal, searching);
                        if (*searching){
                            alpha = max(alpha, g);
                            if (v < g){
                                v = g;
                                best_move = flip.pos;
                            }
                            if (beta <= alpha){
                                search->board.undo(&flip);
                                break;
                            }
                        }
                    }
                search->board.undo(&flip);
            }
            if (split_count){
                if (beta <= alpha || !(*searching)){
                    n_searching = false;
                    ybwc_wait_all(search, parallel_tasks);
                } else{
                    g = ybwc_wait_all(search, parallel_tasks);
                    alpha = max(alpha, g);
                    v = max(v, g);
                }
            }
        #else
            for (const Flip &flip: move_list){
                //eval_move(search, &flip);
                search->board.move(&flip);
                    g = -nega_alpha_end(search, -beta, -alpha, false, flip.n_legal, searching);
                search->board.undo(&flip);
                //eval_undo(search, &flip);
                alpha = max(alpha, g);
                if (v < g){
                    v = g;
                    best_move = flip.pos;
                }
                if (beta <= alpha)
                    break;
            }
        #endif
    }
    if (best_move != f_best_move)
        child_transpose_table.reg(&search->board, hash_code, best_move);
    #if USE_END_TC
        if (beta <= v && l < v)
            parent_transpose_table.reg(&search->board, hash_code, v, u);
        else if (v <= alpha && v < u)
            parent_transpose_table.reg(&search->board, hash_code, l, v);
        else if (alpha < v && v < beta)
            parent_transpose_table.reg(&search->board, hash_code, v, v);
    #endif
    return v;
}
