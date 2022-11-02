/*
    Egaroucid Project

    @date 2021-2022
    @author Takuto Yamana (a.k.a Nyanyan)
    @license GPL-3.0 license
*/

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
#include "endsearch_common.hpp"
#include "ybwc.hpp"
#include "parallel.hpp"

using namespace std;

inline bool ybwc_split_end_nws(const Search *search, const Flip *flip, int alpha, uint64_t legal, const bool *searching, uint_fast8_t policy, const int canput, const int pv_idx, const int split_count, vector<future<Parallel_task>> &parallel_tasks);
inline void ybwc_get_end_tasks_nws(Search *search, vector<future<Parallel_task>> &parallel_tasks, int *v);
inline void ybwc_get_end_tasks(Search *search, vector<future<Parallel_task>> &parallel_tasks, int *v, int *best_move);
inline void ybwc_wait_all(Search *search, vector<future<Parallel_task>> &parallel_tasks);
inline void ybwc_wait_all_nws(Search *search, vector<future<Parallel_task>> &parallel_tasks, int *v, int alpha, bool *searching);
inline void ybwc_wait_all_nws(Search *search, vector<future<Parallel_task>> &parallel_tasks, int *v, int *best_move, int alpha, bool *searching);

inline int last2_nws(Search *search, int alpha, uint_fast8_t p0, uint_fast8_t p1, bool skipped){
    ++search->n_nodes;
    #if USE_END_PO & false
        uint_fast8_t p0_parity = (search->parity & cell_div4[p0]);
        uint_fast8_t p1_parity = (search->parity & cell_div4[p1]);
        if (!p0_parity && p1_parity)
            swap(p0, p1);
    #endif
    int v = -INF;
    Flip flip;
    if (bit_around[p0] & search->board.opponent){
        calc_flip(&flip, &search->board, p0);
        if (flip.flip){
            search->move(&flip);
                v = -last1(search, -alpha - 1, p1);
            search->undo(&flip);
            if (alpha < v)
                return v;
        }
    }
    if (bit_around[p1] & search->board.opponent){
        calc_flip(&flip, &search->board, p1);
        if (flip.flip){
            search->move(&flip);
                int g = -last1(search, -alpha - 1, p0);
            search->undo(&flip);
            if (v < g)
                return g;
            else
                return v;
        }
    }
    if (v == -INF){
        if (skipped)
            v = end_evaluate(&search->board);
        else{
            search->board.pass();
                v = -last2_nws(search, -alpha - 1, p0, p1, true);
            search->board.pass();
        }
    }
    return v;
}

inline int last3_nws(Search *search, int alpha, uint_fast8_t p0, uint_fast8_t p1, uint_fast8_t p2, bool skipped){
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
    int v = -INF;
    Flip flip;
    if (bit_around[p0] & search->board.opponent){
        calc_flip(&flip, &search->board, p0);
        if (flip.flip){
            search->move(&flip);
                v = -last2_nws(search, -alpha - 1, p1, p2, false);
            search->undo(&flip);
            if (alpha < v)
                return v;
        }
    }
    int g;
    if (bit_around[p1] & search->board.opponent){
        calc_flip(&flip, &search->board, p1);
        if (flip.flip){
            search->move(&flip);
                g = -last2_nws(search, -alpha - 1, p0, p2, false);
            search->undo(&flip);
            if (v < g){
                if (alpha < g)
                    return g;
                v = g;
            }
        }
    }
    if (bit_around[p2] & search->board.opponent){
        calc_flip(&flip, &search->board, p2);
        if (flip.flip){
            search->move(&flip);
                g = -last2_nws(search, -alpha - 1, p0, p1, false);
            search->undo(&flip);
            if (v < g)
                return g;
            else
                return v;
        }
    }
    if (v == -INF){
        if (skipped)
            v = end_evaluate(&search->board);
        else{
            search->board.pass();
                v = -last3_nws(search, -alpha - 1, p0, p1, p2, true);
            search->board.pass();
        }
    }
    return v;
}

inline int last4_nws(Search *search, int alpha, uint_fast8_t p0, uint_fast8_t p1, uint_fast8_t p2, uint_fast8_t p3, bool skipped){
    ++search->n_nodes;
    #if USE_END_SC
        int stab_res = stability_cut_nws(search, &alpha);
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
    int v = -INF;
    if (legal == 0ULL){
        if (skipped)
            v = end_evaluate(&search->board);
        else{
            search->board.pass();
                v = -last4_nws(search, -alpha - 1, p0, p1, p2, p3, true);
            search->board.pass();
        }
        return v;
    }
    Flip flip;
    if (1 & (legal >> p0)){
        calc_flip(&flip, &search->board, p0);
        search->move(&flip);
            v = -last3_nws(search, -alpha - 1, p1, p2, p3, false);
        search->undo(&flip);
        if (alpha < v)
            return v;
    }
    int g;
    if (1 & (legal >> p1)){
        calc_flip(&flip, &search->board, p1);
        search->move(&flip);
            g = -last3_nws(search, -alpha - 1, p0, p2, p3, false);
        search->undo(&flip);
        if (v < g){
            if (alpha < g)
                return g;
            v = g;
        }
    }
    if (1 & (legal >> p2)){
        calc_flip(&flip, &search->board, p2);
        search->move(&flip);
            g = -last3_nws(search, -alpha - 1, p0, p1, p3, false);
        search->undo(&flip);
        if (v < g){
            if (alpha < g)
                return g;
            v = g;
        }
    }
    if (1 & (legal >> p3)){
        calc_flip(&flip, &search->board, p3);
        search->move(&flip);
            g = -last3_nws(search, -alpha - 1, p0, p1, p2, false);
        search->undo(&flip);
        if (v < g)
            return g;
    }
    return v;
}

int nega_alpha_end_fast_nws(Search *search, int alpha, bool skipped, bool stab_cut, const bool *searching){
    if (!global_searching || !(*searching))
        return SCORE_UNDEFINED;
    ++search->n_nodes;
    #if USE_END_SC
        if (stab_cut){
            int stab_res = stability_cut_nws(search, &alpha);
            if (stab_res != SCORE_UNDEFINED){
                return stab_res;
            }
        }
    #endif
    uint64_t legal = search->board.get_legal();
    int v = -INF;
    if (legal == 0ULL){
        if (skipped)
            return end_evaluate(&search->board);
        search->board.pass();
            v = -nega_alpha_end_fast_nws(search, -alpha - 1, true, false, searching);
        search->board.pass();
        return v;
    }
    int g;
    Flip flip;
    uint_fast8_t cell;
    #if USE_END_PO
        uint64_t legal_copy;
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
            if (search->n_discs == 59){
                uint64_t empties;
                uint_fast8_t p0, p1, p2, p3;
                legal_copy = legal & legal_mask;
                for (cell = first_bit(&legal_copy); legal_copy; cell = next_bit(&legal_copy)){
                    calc_flip(&flip, &search->board, cell);
                    search->move(&flip);
                        empties = ~(search->board.player | search->board.opponent);
                        p0 = first_bit(&empties);
                        p1 = next_bit(&empties);
                        p2 = next_bit(&empties);
                        p3 = next_bit(&empties);
                        g = -last4_nws(search, -alpha - 1, p0, p1, p2, p3, false);
                    search->undo(&flip);
                    if (v < g){
                        if (alpha < g)
                            return g;
                        v = g;
                    }
                }
                legal_copy = legal & ~legal_mask;
                for (cell = first_bit(&legal_copy); legal_copy; cell = next_bit(&legal_copy)){
                    calc_flip(&flip, &search->board, cell);
                    search->move(&flip);
                        empties = ~(search->board.player | search->board.opponent);
                        p0 = first_bit(&empties);
                        p1 = next_bit(&empties);
                        p2 = next_bit(&empties);
                        p3 = next_bit(&empties);
                        g = -last4_nws(search, -alpha - 1, p0, p1, p2, p3, false);
                    search->undo(&flip);
                    if (v < g){
                        if (alpha < g)
                            return g;
                        v = g;
                    }
                }
            } else{
                legal_copy = legal & legal_mask;
                for (cell = first_bit(&legal_copy); legal_copy; cell = next_bit(&legal_copy)){
                    calc_flip(&flip, &search->board, cell);
                    search->move(&flip);
                        g = -nega_alpha_end_fast_nws(search, -alpha - 1, false, true, searching);
                    search->undo(&flip);
                    if (v < g){
                        if (alpha < g)
                            return g;
                        v = g;
                    }
                }
                legal_copy = legal & ~legal_mask;
                for (cell = first_bit(&legal_copy); legal_copy; cell = next_bit(&legal_copy)){
                    calc_flip(&flip, &search->board, cell);
                    search->move(&flip);
                        g = -nega_alpha_end_fast_nws(search, -alpha - 1, false, true, searching);
                    search->undo(&flip);
                    if (v < g){
                        if (alpha < g)
                            return g;
                        v = g;
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
                        g = -last4_nws(search, -alpha - 1, p0, p1, p2, p3, false);
                    search->undo(&flip);
                    if (v < g){
                        if (alpha < g)
                            return g;
                        v = g;
                    }
                }
            } else{
                for (cell = first_bit(&legal); legal; cell = next_bit(&legal)){
                    calc_flip(&flip, &search->board, cell);
                    search->move(&flip);
                        g = -nega_alpha_end_fast_nws(search, -alpha - 1, false, true, searching);
                    search->undo(&flip);
                    if (v < g){
                        if (alpha < g)
                            return g;
                        v = g;
                    }
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
                    g = -last4_nws(search, -alpha - 1, p0, p1, p2, p3, false);
                search->undo(&flip);
                if (v < g){
                    if (alpha < g)
                        return g;
                    v = g;
                }
            }
        } else{
            for (cell = first_bit(&legal); legal; cell = next_bit(&legal)){
                calc_flip(&flip, &search->board, cell);
                search->move(&flip);
                    g = -nega_alpha_end_fast_nws(search, -alpha - 1, false, true, searching);
                search->undo(&flip);
                if (v < g){
                    if (alpha < g)
                        return g;
                    v = g;
                }
            }
        }
    #endif
    return v;
}

int nega_alpha_end_nws(Search *search, int alpha, bool skipped, uint64_t legal, const bool *searching){
    if (!global_searching || !(*searching))
        return SCORE_UNDEFINED;
    if (search->n_discs >= HW2 - END_FAST_DEPTH)
        return nega_alpha_end_fast_nws(search, alpha, skipped, false, searching);
    ++search->n_nodes;
    uint32_t hash_code = search->board.hash();
    int l = -INF, u = INF;
    const bool use_tt = search->n_discs <= HW2 - USE_TT_DEPTH_THRESHOLD;
    if (use_tt){
        parent_transpose_table.get(&search->board, hash_code, &l, &u, NOMPC, HW2 - search->n_discs);
        if (u == l)
            return u;
        if (u <= alpha)
            return u;
        if (alpha < l)
            return l;
    }
    #if USE_END_SC
        int stab_res = stability_cut_nws(search, &alpha);
        if (stab_res != SCORE_UNDEFINED)
            return stab_res;
    #endif
    if (legal == LEGAL_UNDEFINED)
        legal = search->board.get_legal();
    int v = -INF;
    if (legal == 0ULL){
        if (skipped)
            return end_evaluate(&search->board);
        search->board.pass();
            v = -nega_alpha_end_nws(search, -alpha - 1, true, LEGAL_UNDEFINED, searching);
        search->board.pass();
        return v;
    }
    int best_move = TRANSPOSE_TABLE_UNDEFINED;
    if (use_tt){
        best_move = child_transpose_table.get(&search->board, hash_code);
        if (best_move != TRANSPOSE_TABLE_UNDEFINED){
            if (1 & (legal >> best_move)){
                Flip flip_best;
                calc_flip(&flip_best, &search->board, best_move);
                search->move(&flip_best);
                    v = -nega_alpha_end_nws(search, -alpha - 1, false, LEGAL_UNDEFINED, searching);
                search->undo(&flip_best);
                if (alpha < v)
                    return v;
                legal ^= 1ULL << best_move;
            } else
                best_move = TRANSPOSE_TABLE_UNDEFINED;
        }
    }
    if (legal){
        #if USE_ALL_NODE_PREDICTION
            const bool seems_to_be_all_node = is_like_all_node(search, alpha, HW2 - search->n_discs, LEGAL_UNDEFINED, true, searching);
        #else
            constexpr bool seems_to_be_all_node = false;
        #endif
        int g;
        const int canput = pop_count_ull(legal);
        vector<Flip_value> move_list(canput);
        int idx = 0;
        for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal))
            calc_flip(&move_list[idx++].flip, &search->board, cell);
        move_list_evaluate_end(search, move_list);
        if (search->use_multi_thread){
            int pv_idx = 0, split_count = 0;
            if (best_move != TRANSPOSE_TABLE_UNDEFINED)
                pv_idx = 1;
            vector<future<Parallel_task>> parallel_tasks;
            bool n_searching = true;
            for (int move_idx = 0; move_idx < canput; ++move_idx){
                swap_next_best_move(move_list, move_idx, canput);
                search->move(&move_list[move_idx].flip);
                    if (ybwc_split_end_nws(search, &move_list[move_idx].flip, -alpha - 1, move_list[move_idx].n_legal, &n_searching, move_list[move_idx].flip.pos, canput, pv_idx++, seems_to_be_all_node, split_count, parallel_tasks)){
                        ++split_count;
                    } else{
                        g = -nega_alpha_end_nws(search, -alpha - 1, false, move_list[move_idx].n_legal, searching);
                        if (v < g){
                            v = g;
                            if (alpha < v){
                                search->undo(&move_list[move_idx].flip);
                                break;
                            }
                        }
                        if (split_count){
                            ybwc_get_end_tasks(search, parallel_tasks, &v, &best_move);
                            if (alpha < v){
                                search->undo(&move_list[move_idx].flip);
                                break;
                            }
                        }
                    }
                search->undo(&move_list[move_idx].flip);
            }
            if (split_count){
                if (alpha < v || !(*searching)){
                    n_searching = false;
                    ybwc_wait_all(search, parallel_tasks);
                } else
                    ybwc_wait_all_nws(search, parallel_tasks, &v, &best_move, alpha, &n_searching);
            }
        } else{
            for (int move_idx = 0; move_idx < canput; ++move_idx){
                swap_next_best_move(move_list, move_idx, canput);
                search->move(&move_list[move_idx].flip);
                    g = -nega_alpha_end_nws(search, -alpha - 1, false, move_list[move_idx].n_legal, searching);
                search->undo(&move_list[move_idx].flip);
                if (v < g){
                    v = g;
                    if (alpha < v)
                        break;
                }
            }
        }
    }
    register_tt_nws_mpct(search, HW2 - search->n_discs, hash_code, alpha, v, best_move, l, u, searching, NOMPC);
    return v;
}
