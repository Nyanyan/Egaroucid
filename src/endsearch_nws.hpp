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

using namespace std;

inline int last1_nws(Search *search, int alpha, uint_fast8_t p0){
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

inline int last2_nws(Search *search, int alpha, uint_fast8_t p0, uint_fast8_t p1, bool skipped){
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
                g = -last1_nws(search, -alpha - 1, p1);
            search->undo(&flip);
            if (alpha < g)
                return g;
            v = g;
        } else
            v = -INF;
    } else
        v = -INF;
    if (bit_radiation[p0] & bit_radiation[p1]){
        if (bit_around[p1] & search->board.opponent){
            calc_flip(&flip, &search->board, p1);
            if (flip.flip){
                search->move(&flip);
                    g = -last1_nws(search, -alpha - 1, p0);
                search->undo(&flip);
                if (alpha < g)
                    return g;
                v = max(v, g);
            }
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
    int v = -INF, g;
    Flip flip;
    if (bit_around[p0] & search->board.opponent){
        calc_flip(&flip, &search->board, p0);
        if (flip.flip){
            search->move(&flip);
                g = -last2_nws(search, -alpha - 1, p1, p2, false);
            search->undo(&flip);
            if (alpha < g)
                return g;
            v = g;
        }
    }
    if (bit_radiation[p0] & bit_radiation[p1]){
        if (bit_around[p1] & search->board.opponent){
            calc_flip(&flip, &search->board, p1);
            if (flip.flip){
                search->move(&flip);
                    g = -last2_nws(search, -alpha - 1, p0, p2, false);
                search->undo(&flip);
                if (alpha < g)
                    return g;
                v = max(v, g);
            }
        }
    }
    if ((bit_radiation[p0] & bit_radiation[p2]) || (bit_radiation[p1] & bit_radiation[p2])){
        if (bit_around[p2] & search->board.opponent){
            calc_flip(&flip, &search->board, p2);
            if (flip.flip){
                search->move(&flip);
                    g = -last2_nws(search, -alpha - 1, p0, p1, false);
                search->undo(&flip);
                if (alpha < g)
                    return g;
                v = max(v, g);
            }
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
        return v;
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
    int v = -INF, g;
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
    uint64_t all_radiation = 0ULL;
    if (1 & (legal >> p0)){
        calc_flip(&flip, &search->board, p0);
        search->move(&flip);
            g = -last3_nws(search, -alpha - 1, p1, p2, p3, false);
        search->undo(&flip);
        if (alpha < g)
            return g;
        v = max(v, g);
    }
    all_radiation |= bit_radiation[p0];
    if ((1 & (legal >> p1)) && (all_radiation & bit_radiation[p1])){
        calc_flip(&flip, &search->board, p1);
        search->move(&flip);
            g = -last3_nws(search, -alpha - 1, p0, p2, p3, false);
        search->undo(&flip);
        if (alpha < g)
            return g;
        v = max(v, g);
    }
    all_radiation |= bit_radiation[p1];
    if ((1 & (legal >> p2)) && (all_radiation & bit_radiation[p2])){
        calc_flip(&flip, &search->board, p2);
        search->move(&flip);
            g = -last3_nws(search, -alpha - 1, p0, p1, p3, false);
        search->undo(&flip);
        if (alpha < g)
            return g;
        v = max(v, g);
    }
    all_radiation |= bit_radiation[p2];
    if ((1 & (legal >> p3)) && (all_radiation & bit_radiation[p3])){
        calc_flip(&flip, &search->board, p3);
        search->move(&flip);
            g = -last3_nws(search, -alpha - 1, p0, p1, p2, false);
        search->undo(&flip);
        if (alpha < g)
            return g;
        v = max(v, g);
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
    int g, v = -INF;
    if (legal == 0ULL){
        if (skipped)
            return end_evaluate(&search->board);
        search->board.pass();
            v = -nega_alpha_end_fast_nws(search, -alpha - 1, true, false, searching);
        search->board.pass();
        return v;
    }
    Flip flip;
    #if USE_END_PO
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
                            g = -last4_nws(search, -alpha - 1, p0, p1, p2, p3, false);
                        search->undo(&flip);
                        if (alpha < g)
                            return g;
                        v = max(v, g);
                    }
                } else{
                    for (cell = first_bit(&legal_copy); legal_copy; cell = next_bit(&legal_copy)){
                        calc_flip(&flip, &search->board, cell);
                        search->move(&flip);
                            g = -nega_alpha_end_fast_nws(search, -alpha - 1, false, true, searching);
                        search->undo(&flip);
                        if (alpha < g)
                            return g;
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
                            g = -last4_nws(search, -alpha - 1, p0, p1, p2, p3, false);
                        search->undo(&flip);
                        if (alpha < g)
                            return g;
                        v = max(v, g);
                    }
                } else{
                    for (cell = first_bit(&legal_copy); legal_copy; cell = next_bit(&legal_copy)){
                        calc_flip(&flip, &search->board, cell);
                        search->move(&flip);
                            g = -nega_alpha_end_fast_nws(search, -alpha - 1, false, true, searching);
                        search->undo(&flip);
                        if (alpha < g)
                            return g;
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
                            g = -last4_nws(search, -alpha - 1, p0, p1, p2, p3, false);
                        search->undo(&flip);
                        if (alpha < g)
                            return g;
                        v = max(v, g);
                    }
            } else{
                for (cell = first_bit(&legal); legal; cell = next_bit(&legal)){
                    calc_flip(&flip, &search->board, cell);
                    search->move(&flip);
                        g = -nega_alpha_end_fast_nws(search, -alpha - 1, false, true, searching);
                    search->undo(&flip);
                    if (alpha < g)
                        return g;
                    v = max(v, g);
                }
            }
        }
    #else
        if (search->n_discs == 59){
            uint64_t empties;
                uint_fast8_t p0, p1, p2, p3;
                for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)){
                    calc_flip(&flip, &search->board, cell);
                    search->move(&flip);
                        empties = ~(search->board.player | search->board.opponent);
                        p0 = first_bit(&empties);
                        p1 = next_bit(&empties);
                        p2 = next_bit(&empties);
                        p3 = next_bit(&empties);
                        g = -last4_nws(search, -alpha - 1, p0, p1, p2, p3, skipped);
                    search->undo(&flip);
                    if (alpha < g)
                        return g;
                    v = max(v, g);
                }
        } else{
            for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)){
                calc_flip(&flip, &search->board, cell);
                search->move(&flip);
                    g = -nega_alpha_end_fast_nws(search, -alpha - 1, false, true, searching);
                search->undo(&flip);
                if (alpha < g)
                    return g;
                v = max(v, g);
            }
        }
    #endif
    return v;
}
