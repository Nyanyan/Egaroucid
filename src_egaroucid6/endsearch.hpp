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
#include "util.hpp"
#include "stability.hpp"

using namespace std;

inline int last1(Search *search, int alpha, int beta, uint_fast8_t p0){
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

inline int last2(Search *search, int alpha, int beta, uint_fast8_t p0, uint_fast8_t p1, bool skipped){
    ++search->n_nodes;
    #if USE_END_PO & false
        uint_fast8_t p0_parity = (search->parity & cell_div4[p0]);
        uint_fast8_t p1_parity = (search->parity & cell_div4[p1]);
        if (!p0_parity && p1_parity)
            swap(p0, p1);
    #endif
    int v, g;
    Flip flip;
    if (bit_around[p0] & search->board.opponent){
        calc_flip(&flip, &search->board, p0);
        if (flip.flip){
            search->move(&flip);
                g = -last1(search, -beta, -alpha, p1);
            search->undo(&flip);
            alpha = max(alpha, g);
            if (beta <= alpha)
                return alpha;
            v = g;
        } else
            v = -INF;
    } else
        v = -INF;
    if (bit_around[p1] & search->board.opponent){
        calc_flip(&flip, &search->board, p1);
        if (flip.flip){
            search->move(&flip);
                g = -last1(search, -beta, -alpha, p0);
            search->undo(&flip);
            alpha = max(alpha, g);
            if (beta <= alpha)
                return alpha;
            v = max(v, g);
        }
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

inline int last3(Search *search, int alpha, int beta, uint_fast8_t p0, uint_fast8_t p1, uint_fast8_t p2, bool skipped){
    ++search->n_nodes;
    #if USE_END_PO
        if (!skipped){
            const bool p0_parity = (search->parity & cell_div4[p0]) > 0;
            const bool p1_parity = (search->parity & cell_div4[p1]) > 0;
            const bool p2_parity = (search->parity & cell_div4[p2]) > 0;
            #if LAST_PO_OPTIMISE
                if (!p0_parity && p2_parity){
                    swap(p0, p2);
                } else if (!p0_parity && p1_parity && !p2_parity){
                    swap(p0, p1);
                } else if (p0_parity && !p1_parity && p2_parity){
                    swap(p1, p2);
                }
            #else
                if (!p0_parity && p1_parity && p2_parity){
                    swap(p0, p2);
                } else if (!p0_parity && !p1_parity && p2_parity){
                    swap(p0, p2);
                } else if (!p0_parity && p1_parity && !p2_parity){
                    swap(p0, p1);
                } else if (p0_parity && !p1_parity && p2_parity){
                    swap(p1, p2);
                }
            #endif
        }
    #endif
    int v = -INF, g;
    Flip flip;
    if (bit_around[p0] & search->board.opponent){
        calc_flip(&flip, &search->board, p0);
        if (flip.flip){
            search->move(&flip);
                g = -last2(search, -beta, -alpha, p1, p2, false);
            search->undo(&flip);
            alpha = max(alpha, g);
            if (beta <= alpha)
                return alpha;
            v = g;
        }
    }
    if (bit_around[p1] & search->board.opponent){
        calc_flip(&flip, &search->board, p1);
        if (flip.flip){
            search->move(&flip);
                g = -last2(search, -beta, -alpha, p0, p2, false);
            search->undo(&flip);
            alpha = max(alpha, g);
            if (beta <= alpha)
                return alpha;
            v = max(v, g);
        }
    }
    if (bit_around[p2] & search->board.opponent){
        calc_flip(&flip, &search->board, p2);
        if (flip.flip){
            search->move(&flip);
                g = -last2(search, -beta, -alpha, p0, p1, false);
            search->undo(&flip);
            alpha = max(alpha, g);
            if (beta <= alpha)
                return alpha;
            v = max(v, g);
        }
    }
    if (v == -INF){
        if (skipped)
            v = end_evaluate(&search->board);
        else{
            search->board.pass();
                v = -last3(search, -beta, -alpha, p0, p1, p2, true);
            search->board.pass();
        }
        return v;
    }
    return v;
}

inline int last4(Search *search, int alpha, int beta, uint_fast8_t p0, uint_fast8_t p1, uint_fast8_t p2, uint_fast8_t p3, bool skipped){
    ++search->n_nodes;
    #if USE_END_SC
        int stab_res = stability_cut(search, &alpha, &beta);
        if (stab_res != SCORE_UNDEFINED){
            return stab_res;
        }
    #endif
    #if USE_END_PO
        if (!skipped){
            const bool p0_parity = (search->parity & cell_div4[p0]) > 0;
            const bool p1_parity = (search->parity & cell_div4[p1]) > 0;
            const bool p2_parity = (search->parity & cell_div4[p2]) > 0;
            const bool p3_parity = (search->parity & cell_div4[p3]) > 0;
            #if LAST_PO_OPTIMISE
                if (!p0_parity && p3_parity){
                    swap(p0, p3);
                    if (!p1_parity && p2_parity)
                        swap(p1, p2);
                } else if (!p0_parity && p2_parity && !p3_parity){
                    swap(p0, p2);
                } else if (!p0_parity && p1_parity && !p2_parity && !p3_parity){
                    swap(p0, p1);
                } else if (p0_parity && !p1_parity && p3_parity){
                    swap(p1, p3);
                } else if (p0_parity && !p1_parity && p2_parity && !p3_parity){
                    swap(p1, p2);
                } else if (p0_parity && p1_parity && !p2_parity && p3_parity){
                    swap(p2, p3);
                }
            #else
                if (!p0_parity && p1_parity && p2_parity && p3_parity){
                    swap(p0, p3);
                } else if (!p0_parity && !p1_parity && p2_parity && p3_parity){
                    swap(p0, p2);
                    swap(p1, p3);
                } else if (!p0_parity && p1_parity && !p2_parity && p3_parity){
                    swap(p0, p3);
                } else if (!p0_parity && p1_parity && p2_parity && !p3_parity){
                    swap(p0, p2);
                } else if (!p0_parity && !p1_parity && !p2_parity && p3_parity){
                    swap(p0, p3);
                } else if (!p0_parity && !p1_parity && p2_parity && !p3_parity){
                    swap(p0, p2);
                } else if (!p0_parity && p1_parity && !p2_parity && !p3_parity){
                    swap(p0, p1);
                } else if (p0_parity && !p1_parity && p2_parity && p3_parity){
                    swap(p1, p3);
                } else if (p0_parity && !p1_parity && !p2_parity && p3_parity){
                    swap(p1, p3);
                } else if (p0_parity && !p1_parity && p2_parity && !p3_parity){
                    swap(p1, p2);
                } else if (p0_parity && p1_parity && !p2_parity && p3_parity){
                    swap(p2, p3);
                }
            #endif
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
        search->move(&flip);
            g = -last3(search, -beta, -alpha, p1, p2, p3, false);
        search->undo(&flip);
        alpha = max(alpha, g);
        if (beta <= alpha)
            return alpha;
        v = max(v, g);
    }
    if (1 & (legal >> p1)){
        calc_flip(&flip, &search->board, p1);
        search->move(&flip);
            g = -last3(search, -beta, -alpha, p0, p2, p3, false);
        search->undo(&flip);
        alpha = max(alpha, g);
        if (beta <= alpha)
            return alpha;
        v = max(v, g);
    }
    if (1 & (legal >> p2)){
        calc_flip(&flip, &search->board, p2);
        search->move(&flip);
            g = -last3(search, -beta, -alpha, p0, p1, p3, false);
        search->undo(&flip);
        alpha = max(alpha, g);
        if (beta <= alpha)
            return alpha;
        v = max(v, g);
    }
    if (1 & (legal >> p3)){
        calc_flip(&flip, &search->board, p3);
        search->move(&flip);
            g = -last3(search, -beta, -alpha, p0, p1, p2, false);
        search->undo(&flip);
        alpha = max(alpha, g);
        if (beta <= alpha)
            return alpha;
        v = max(v, g);
    }
    return v;
}

int nega_alpha_end_fast(Search *search, int alpha, int beta, bool skipped, bool stab_cut, const bool *searching){
    if (!global_searching || !(*searching))
        return SCORE_UNDEFINED;
    ++search->n_nodes;
    #if USE_END_SC
        if (stab_cut){
            int stab_res = stability_cut(search, &alpha, &beta);
            if (stab_res != SCORE_UNDEFINED){
                return stab_res;
            }
        }
    #endif
    uint64_t legal = search->board.get_legal();
    int g, v = -INF;
    if (legal == 0ULL){
        if (skipped)
            return end_evaluate(&search->board);
        search->board.pass();
            v = -nega_alpha_end_fast(search, -beta, -alpha, true, false, searching);
        search->board.pass();
        return v;
    }
    Flip flip;
    #if USE_END_PO
        int i;
        uint64_t legal_copy;
        uint_fast8_t cell;
        if (0 < search->parity && search->parity < 15){
            uint64_t legal_mask = 0ULL;
            if (search->parity & 1)
                legal_mask |= 0x000000000F0F0F0FULL;
            if (search->parity & 2)
                legal_mask |= 0x00000000F0F0F0F0ULL;
            if (search->parity & 4)
                legal_mask |= 0x0F0F0F0F00000000ULL;
            if (search->parity & 8)
                legal_mask |= 0xF0F0F0F000000000ULL;
            legal_copy = legal & legal_mask;
            if (legal_copy){
                if (search->n_discs == 59){
                    uint64_t empties;
                    uint_fast8_t p0, p1, p2, p3;
                    for (cell = first_bit(&legal_copy); legal_copy; cell = next_bit(&legal_copy)){
                        calc_flip(&flip, &search->board, cell);
                        search->move(&flip);
                            empties = ~(search->board.player | search->board.opponent);
                            p0 = first_bit(&empties);
                            p1 = next_bit(&empties);
                            p2 = next_bit(&empties);
                            p3 = next_bit(&empties);
                            g = -last4(search, -beta, -alpha, p0, p1, p2, p3, skipped);
                        search->undo(&flip);
                        alpha = max(alpha, g);
                        if (beta <= alpha)
                            return alpha;
                        v = max(v, g);
                    }
                } else{
                    for (cell = first_bit(&legal_copy); legal_copy; cell = next_bit(&legal_copy)){
                        calc_flip(&flip, &search->board, cell);
                        search->move(&flip);
                            g = -nega_alpha_end_fast(search, -beta, -alpha, false, true, searching);
                        search->undo(&flip);
                        alpha = max(alpha, g);
                        if (beta <= alpha)
                            return alpha;
                        v = max(v, g);
                    }
                }
            }
            legal_copy = legal & ~legal_mask;
            if (legal_copy){
                if (search->n_discs == 59){
                    uint64_t empties;
                    uint_fast8_t p0, p1, p2, p3;
                    for (cell = first_bit(&legal_copy); legal_copy; cell = next_bit(&legal_copy)){
                        calc_flip(&flip, &search->board, cell);
                        search->move(&flip);
                            empties = ~(search->board.player | search->board.opponent);
                            p0 = first_bit(&empties);
                            p1 = next_bit(&empties);
                            p2 = next_bit(&empties);
                            p3 = next_bit(&empties);
                            g = -last4(search, -beta, -alpha, p0, p1, p2, p3, skipped);
                        search->undo(&flip);
                        alpha = max(alpha, g);
                        if (beta <= alpha)
                            return alpha;
                        v = max(v, g);
                    }
                } else{
                    for (cell = first_bit(&legal_copy); legal_copy; cell = next_bit(&legal_copy)){
                        calc_flip(&flip, &search->board, cell);
                        search->move(&flip);
                            g = -nega_alpha_end_fast(search, -beta, -alpha, false, true, searching);
                        search->undo(&flip);
                        alpha = max(alpha, g);
                        if (beta <= alpha)
                            return alpha;
                        v = max(v, g);
                    }
                }
            }
        } else{
            if (search->n_discs == 59){
                uint64_t empties;
                    uint_fast8_t p0, p1, p2, p3;
                    for (cell = first_bit(&legal); legal; cell = next_bit(&legal)){
                        calc_flip(&flip, &search->board, cell);
                        search->move(&flip);
                            empties = ~(search->board.player | search->board.opponent);
                            p0 = first_bit(&empties);
                            p1 = next_bit(&empties);
                            p2 = next_bit(&empties);
                            p3 = next_bit(&empties);
                            g = -last4(search, -beta, -alpha, p0, p1, p2, p3, skipped);
                        search->undo(&flip);
                        alpha = max(alpha, g);
                        if (beta <= alpha)
                            return alpha;
                        v = max(v, g);
                    }
            } else{
                for (cell = first_bit(&legal); legal; cell = next_bit(&legal)){
                    calc_flip(&flip, &search->board, cell);
                    search->move(&flip);
                        g = -nega_alpha_end_fast(search, -beta, -alpha, false, true, searching);
                    search->undo(&flip);
                    alpha = max(alpha, g);
                    if (beta <= alpha)
                        return alpha;
                    v = max(v, g);
                }
            }
        }
    #else
        if (search->n_discs == 59){
            uint64_t empties;
                uint_fast8_t p0, p1, p2, p3;
                for (cell = first_bit(&legal); legal; cell = next_bit(&legal)){
                    calc_flip(&flip, &search->board, cell);
                    search->move(&flip);
                        empties = ~(search->board.player | search->board.opponent);
                        p0 = first_bit(&empties);
                        p1 = next_bit(&empties);
                        p2 = next_bit(&empties);
                        p3 = next_bit(&empties);
                        g = -last4(search, -beta, -alpha, p0, p1, p2, p3, skipped);
                    search->undo(&flip);
                    alpha = max(alpha, g);
                    if (beta <= alpha)
                        return alpha;
                    v = max(v, g);
                }
        } else{
            for (cell = first_bit(&legal); legal; cell = next_bit(&legal)){
                calc_flip(&flip, &search->board, cell);
                search->move(&flip);
                    g = -nega_alpha_end_fast(search, -beta, -alpha, false, true, searching);
                search->undo(&flip);
                alpha = max(alpha, g);
                if (beta <= alpha)
                    return alpha;
                v = max(v, g);
            }
        }
    #endif
    return v;
}

int nega_alpha_end(Search *search, int alpha, int beta, bool skipped, uint64_t legal, const bool *searching){
    if (!global_searching || !(*searching))
        return SCORE_UNDEFINED;
    if (search->n_discs >= HW2 - END_FAST_DEPTH)
        return nega_alpha_end_fast(search, alpha, beta, skipped, false, searching);
    ++search->n_nodes;
    uint32_t hash_code = search->board.hash() & TRANSPOSE_TABLE_MASK;
    int l = -INF, u = INF;
    if (search->n_discs <= HW2 - USE_TT_DEPTH_THRESHOLD){
        parent_transpose_table.get(&search->board, hash_code, &l, &u);
        if (u == l)
            return u;
        if (beta <= l)
            return l;
        if (u <= alpha)
            return u;
        alpha = max(alpha, l);
        beta = min(beta, u);
    }
    #if USE_END_SC
        int stab_res = stability_cut(search, &alpha, &beta);
        if (stab_res != SCORE_UNDEFINED)
            return stab_res;
    #endif
    int first_alpha = alpha;
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
    int best_move = TRANSPOSE_TABLE_UNDEFINED;
    if (search->n_discs <= HW2 - USE_TT_DEPTH_THRESHOLD)
        best_move = child_transpose_table.get(&search->board, hash_code);
    if (best_move != TRANSPOSE_TABLE_UNDEFINED){
        if (1 & (legal >> best_move)){
            Flip flip_best;
            calc_flip(&flip_best, &search->board, best_move);
            search->move(&flip_best);
                g = -nega_alpha_end(search, -beta, -alpha, false, LEGAL_UNDEFINED, searching);
            search->undo(&flip_best);
            if (*searching){
                alpha = max(alpha, g);
                v = g;
                legal ^= 1ULL << best_move;
            } else
                return SCORE_UNDEFINED;
        } else
            best_move = TRANSPOSE_TABLE_UNDEFINED;
    }
    if (alpha < beta && legal){
        const int canput = pop_count_ull(legal);
        vector<Flip_value> move_list(canput);
        int idx = 0;
        for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal))
            calc_flip(&move_list[idx++].flip, &search->board, cell);
        move_evaluate_fast_first(search, move_list);
        const int move_ordering_threshold = MOVE_ORDERING_THRESHOLD - (int)(best_move != TRANSPOSE_TABLE_UNDEFINED);
        for (int move_idx = 0; move_idx < canput; ++move_idx){
            if (move_idx < move_ordering_threshold)
                swap_next_best_move(move_list, move_idx, canput);
            search->move(&move_list[move_idx].flip);
                g = -nega_alpha_end(search, -beta, -alpha, false, move_list[move_idx].n_legal, searching);
            search->undo(&move_list[move_idx].flip);
            if (*searching){
                alpha = max(alpha, g);
                if (v < g){
                    v = g;
                    best_move = move_list[move_idx].flip.pos;
                    if (beta <= alpha){
                        register_tt(search, hash_code, first_alpha, v, best_move, l, u, alpha, beta, searching);
                        return alpha;
                    }
                }
            } else
                return SCORE_UNDEFINED;
        }
    }
    register_tt(search, hash_code, first_alpha, v, best_move, l, u, alpha, beta, searching);
    return v;
}
