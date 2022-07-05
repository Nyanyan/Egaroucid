#pragma once
#include <iostream>
#include <fstream>
#include "setting.hpp"
#include "common.hpp"
#include "board.hpp"
#include "search.hpp"
#include "util.hpp"

using namespace std;



uint64_t stability_edge_arr[N_8BIT][N_8BIT][2];

inline void probably_move_line(int p, int o, int place, int *np, int *no){
    int i, j;
    *np = p | (1 << place);
    for (i = place - 1; i > 0 && (1 & (o >> i)); --i);
    if (1 & (p >> i)){
        for (j = place - 1; j > i; --j)
            *np ^= 1 << j;
    }
    for (i = place + 1; i < HW_M1 && (1 & (o >> i)); ++i);
    if (1 & (p >> i)){
        for (j = place + 1; j < i; ++j)
            *np ^= 1 << j;
    }
    *no = o & ~(*np);
}

int calc_stability_line(int b, int w){
    int i, nb, nw, res = b | w;
    int empties = ~(b | w);
    for (i = 0; i < HW; ++i){
        if (1 & (empties >> i)){
            probably_move_line(b, w, i, &nb, &nw);
            res &= b | nw;
            res &= calc_stability_line(nb, nw);
            probably_move_line(w, b, i, &nw, &nb);
            res &= w | nb;
            res &= calc_stability_line(nb, nw);
        }
    }
    return res;
}

inline void stability_init() {
    int place, b, w, stab;
    for (b = 0; b < N_8BIT; ++b) {
        for (w = b; w < N_8BIT; ++w){
            if (b & w){
                stability_edge_arr[b][w][0] = 0;
                stability_edge_arr[b][w][1] = 0;
                stability_edge_arr[w][b][0] = 0;
                stability_edge_arr[w][b][1] = 0;
            } else{
                stab = calc_stability_line(b, w);
                stability_edge_arr[b][w][0] = 0;
                stability_edge_arr[b][w][1] = 0;
                for (place = 0; place < HW; ++place){
                    if (1 & (stab >> place)){
                        stability_edge_arr[b][w][0] |= 1ULL << place;
                        stability_edge_arr[b][w][1] |= 1ULL << (place * HW);
                    }
                }
                stability_edge_arr[w][b][0] = stability_edge_arr[b][w][0];
                stability_edge_arr[w][b][1] = stability_edge_arr[b][w][1];
            }
        }
    }
}

#if USE_SIMD && false
    inline int calc_stability_player(uint64_t player, uint64_t opponent){
        uint64_t full_h, full_v, full_d7, full_d9;
        uint64_t edge_stability = 0, player_stability = 0, n_stability;
        const uint64_t player_mask = player & 0b0000000001111110011111100111111001111110011111100111111000000000ULL;
        uint8_t pl, op;
        pl = player & 0b11111111U;
        op = opponent & 0b11111111U;
        edge_stability |= stability_edge_arr[pl][op][0];
        pl = (player >> 56) & 0b11111111U;
        op = (opponent >> 56) & 0b11111111U;
        edge_stability |= stability_edge_arr[pl][op][0] << 56;
        pl = join_v_line(player, 0);
        op = join_v_line(opponent, 0);
        edge_stability |= stability_edge_arr[pl][op][1];
        pl = join_v_line(player, 7);
        op = join_v_line(opponent, 7);
        edge_stability |= stability_edge_arr[pl][op][1] << 7;
        full_stability(player, opponent, &full_h, &full_v, &full_d7, &full_d9);
        u64_4 hvd7d9;
        const u64_4 shift(1, HW, HW_M1, HW_P1);
        u64_4 full(full_h, full_v, full_d7, full_d9);
        u64_4 stab;
        n_stability = (edge_stability & player) | (full_h & full_v & full_d7 & full_d9 & player_mask);
        while (n_stability & ~player_stability){
            player_stability |= n_stability;
            stab = player_stability;
            hvd7d9 = (stab >> shift) | (stab << shift) | full;
            n_stability = all_and(hvd7d9) & player_mask;
        }
        return pop_count_ull(player_stability);
    }

    inline void calc_stability(Board *b, int *stab0, int *stab1){
        uint64_t full_h, full_v, full_d7, full_d9;
        uint64_t edge_stability = 0, player_stability = 0, opponent_stability = 0, n_stability;
        const uint64_t player_mask = b->player & 0b0000000001111110011111100111111001111110011111100111111000000000ULL;
        const uint64_t opponent_mask = b->opponent & 0b0000000001111110011111100111111001111110011111100111111000000000ULL;
        uint8_t pl, op;
        pl = b->player & 0b11111111U;
        op = b->opponent & 0b11111111U;
        edge_stability |= stability_edge_arr[pl][op][0];
        pl = (b->player >> 56) & 0b11111111U;
        op = (b->opponent >> 56) & 0b11111111U;
        edge_stability |= stability_edge_arr[pl][op][0] << 56;
        pl = join_v_line(b->player, 0);
        op = join_v_line(b->opponent, 0);
        edge_stability |= stability_edge_arr[pl][op][1];
        pl = join_v_line(b->player, 7);
        op = join_v_line(b->opponent, 7);
        edge_stability |= stability_edge_arr[pl][op][1] << 7;
        b->full_stability(&full_h, &full_v, &full_d7, &full_d9);

        u64_4 hvd7d9;
        const u64_4 shift(1, HW, HW_M1, HW_P1);
        u64_4 full(full_h, full_v, full_d7, full_d9);
        u64_4 stab;

        n_stability = (edge_stability & b->player) | (full_h & full_v & full_d7 & full_d9 & player_mask);
        while (n_stability & ~player_stability){
            player_stability |= n_stability;
            stab = player_stability;
            hvd7d9 = (stab >> shift) | (stab << shift) | full;
            n_stability = all_and(hvd7d9) & player_mask;
        }

        n_stability = (edge_stability & b->opponent) | (full_h & full_v & full_d7 & full_d9 & opponent_mask);
        while (n_stability & ~opponent_stability){
            opponent_stability |= n_stability;
            stab = opponent_stability;
            hvd7d9 = (stab >> shift) | (stab << shift) | full;
            n_stability = all_and(hvd7d9) & opponent_mask;
        }

        *stab0 = pop_count_ull(player_stability);
        *stab1 = pop_count_ull(opponent_stability);
    }
#else
    inline int calc_stability_player(uint64_t player, uint64_t opponent){
        uint64_t full_h, full_v, full_d7, full_d9;
        uint64_t edge_stability = 0, player_stability = 0, n_stability;
        uint64_t h, v, d7, d9;
        const uint64_t player_mask = player & 0b0000000001111110011111100111111001111110011111100111111000000000ULL;
        uint8_t pl, op;
        pl = player & 0b11111111U;
        op = opponent & 0b11111111U;
        edge_stability |= stability_edge_arr[pl][op][0];
        pl = join_h_line(player, 7);
        op = join_h_line(opponent, 7);
        edge_stability |= stability_edge_arr[pl][op][0] << 56;
        pl = join_v_line(player, 0);
        op = join_v_line(opponent, 0);
        edge_stability |= stability_edge_arr[pl][op][1];
        pl = join_v_line(player, 7);
        op = join_v_line(opponent, 7);
        edge_stability |= stability_edge_arr[pl][op][1] << 7;
        full_stability(player, opponent, &full_h, &full_v, &full_d7, &full_d9);
        n_stability = (edge_stability & player) | (full_h & full_v & full_d7 & full_d9 & player_mask);
        while (n_stability & ~player_stability){
            player_stability |= n_stability;
            h = (player_stability >> 1) | (player_stability << 1) | full_h;
            v = (player_stability >> HW) | (player_stability << HW) | full_v;
            d7 = (player_stability >> HW_M1) | (player_stability << HW_M1) | full_d7;
            d9 = (player_stability >> HW_P1) | (player_stability << HW_P1) | full_d9;
            n_stability = h & v & d7 & d9 & player_mask;
        }
        return pop_count_ull(player_stability);
    }

    inline void calc_stability(Board *b, int *stab0, int *stab1){
        uint64_t full_h, full_v, full_d7, full_d9;
        uint64_t edge_stability = 0, player_stability = 0, opponent_stability = 0, n_stability;
        uint64_t h, v, d7, d9;
        const uint64_t player_mask = b->player & 0b0000000001111110011111100111111001111110011111100111111000000000ULL;
        const uint64_t opponent_mask = b->opponent & 0b0000000001111110011111100111111001111110011111100111111000000000ULL;
        uint8_t pl, op;
        pl = b->player & 0b11111111U;
        op = b->opponent & 0b11111111U;
        edge_stability |= stability_edge_arr[pl][op][0];
        pl = join_h_line(b->player, 7);
        op = join_h_line(b->opponent, 7);
        edge_stability |= stability_edge_arr[pl][op][0] << 56;
        pl = join_v_line(b->player, 0);
        op = join_v_line(b->opponent, 0);
        edge_stability |= stability_edge_arr[pl][op][1];
        pl = join_v_line(b->player, 7);
        op = join_v_line(b->opponent, 7);
        edge_stability |= stability_edge_arr[pl][op][1] << 7;
        b->full_stability(&full_h, &full_v, &full_d7, &full_d9);

        n_stability = (edge_stability & b->player) | (full_h & full_v & full_d7 & full_d9 & player_mask);
        while (n_stability & ~player_stability){
            player_stability |= n_stability;
            h = (player_stability >> 1) | (player_stability << 1) | full_h;
            v = (player_stability >> HW) | (player_stability << HW) | full_v;
            d7 = (player_stability >> HW_M1) | (player_stability << HW_M1) | full_d7;
            d9 = (player_stability >> HW_P1) | (player_stability << HW_P1) | full_d9;
            n_stability = h & v & d7 & d9 & player_mask;
        }

        n_stability = (edge_stability & b->opponent) | (full_h & full_v & full_d7 & full_d9 & opponent_mask);
        while (n_stability & ~opponent_stability){
            opponent_stability |= n_stability;
            h = (opponent_stability >> 1) | (opponent_stability << 1) | full_h;
            v = (opponent_stability >> HW) | (opponent_stability << HW) | full_v;
            d7 = (opponent_stability >> HW_M1) | (opponent_stability << HW_M1) | full_d7;
            d9 = (opponent_stability >> HW_P1) | (opponent_stability << HW_P1) | full_d9;
            n_stability = h & v & d7 & d9 & opponent_mask;
        }

        *stab0 = pop_count_ull(player_stability);
        *stab1 = pop_count_ull(opponent_stability);
    }

    inline void calc_stability(Board *b, uint64_t edge_stability, int *stab0, int *stab1){
        uint64_t full_h, full_v, full_d7, full_d9;
        uint64_t player_stability = 0, opponent_stability = 0, n_stability;
        uint64_t h, v, d7, d9;
        const uint64_t player_mask = b->player & 0b0000000001111110011111100111111001111110011111100111111000000000ULL;
        const uint64_t opponent_mask = b->opponent & 0b0000000001111110011111100111111001111110011111100111111000000000ULL;
        b->full_stability(&full_h, &full_v, &full_d7, &full_d9);

        n_stability = (edge_stability & b->player) | (full_h & full_v & full_d7 & full_d9 & player_mask);
        while (n_stability & ~player_stability){
            player_stability |= n_stability;
            h = (player_stability >> 1) | (player_stability << 1) | full_h;
            v = (player_stability >> HW) | (player_stability << HW) | full_v;
            d7 = (player_stability >> HW_M1) | (player_stability << HW_M1) | full_d7;
            d9 = (player_stability >> HW_P1) | (player_stability << HW_P1) | full_d9;
            n_stability = h & v & d7 & d9 & player_mask;
        }

        n_stability = (edge_stability & b->opponent) | (full_h & full_v & full_d7 & full_d9 & opponent_mask);
        while (n_stability & ~opponent_stability){
            opponent_stability |= n_stability;
            h = (opponent_stability >> 1) | (opponent_stability << 1) | full_h;
            v = (opponent_stability >> HW) | (opponent_stability << HW) | full_v;
            d7 = (opponent_stability >> HW_M1) | (opponent_stability << HW_M1) | full_d7;
            d9 = (opponent_stability >> HW_P1) | (opponent_stability << HW_P1) | full_d9;
            n_stability = h & v & d7 & d9 & opponent_mask;
        }

        *stab0 = pop_count_ull(player_stability);
        *stab1 = pop_count_ull(opponent_stability);
    }
#endif

inline int calc_stability_edge(Board *b){
    uint64_t edge_stability = 0;
    uint8_t pl, op;
    pl = b->player & 0b11111111U;
    op = b->opponent & 0b11111111U;
    edge_stability |= stability_edge_arr[pl][op][0] << 56;
    pl = join_h_line(b->player, 7);
    op = join_h_line(b->opponent, 7);
    edge_stability |= stability_edge_arr[pl][op][0];
    pl = join_v_line(b->player, 0);
    op = join_v_line(b->opponent, 0);
    edge_stability |= stability_edge_arr[pl][op][1] << 7;
    pl = join_v_line(b->player, 7);
    op = join_v_line(b->opponent, 7);
    edge_stability |= stability_edge_arr[pl][op][1];
    return pop_count_ull(edge_stability & b->player) - pop_count_ull(edge_stability & b->opponent);
}

inline void calc_stability_edge(Board *b, int *stab0, int *stab1){
    uint64_t edge_stability = 0;
    uint8_t pl, op;
    pl = b->player & 0b11111111U;
    op = b->opponent & 0b11111111U;
    edge_stability |= stability_edge_arr[pl][op][0] << 56;
    pl = join_h_line(b->player, 7);
    op = join_h_line(b->opponent, 7);
    edge_stability |= stability_edge_arr[pl][op][0];
    pl = join_v_line(b->player, 0);
    op = join_v_line(b->opponent, 0);
    edge_stability |= stability_edge_arr[pl][op][1] << 7;
    pl = join_v_line(b->player, 7);
    op = join_v_line(b->opponent, 7);
    edge_stability |= stability_edge_arr[pl][op][1];
    *stab0 = pop_count_ull(edge_stability & b->player);
    *stab1 = pop_count_ull(edge_stability & b->opponent);
}

inline void calc_stability_edge(Board *b, int *stab0, int *stab1, uint64_t *edge_stability){
    *edge_stability = 0;
    uint8_t pl, op;
    pl = b->player & 0b11111111U;
    op = b->opponent & 0b11111111U;
    *edge_stability |= stability_edge_arr[pl][op][0] << 56;
    pl = join_h_line(b->player, 7);
    op = join_h_line(b->opponent, 7);
    *edge_stability |= stability_edge_arr[pl][op][0];
    pl = join_v_line(b->player, 0);
    op = join_v_line(b->opponent, 0);
    *edge_stability |= stability_edge_arr[pl][op][1] << 7;
    pl = join_v_line(b->player, 7);
    op = join_v_line(b->opponent, 7);
    *edge_stability |= stability_edge_arr[pl][op][1];
    *stab0 = pop_count_ull(*edge_stability & b->player);
    *stab1 = pop_count_ull(*edge_stability & b->opponent);
}

inline int calc_stability_edge_player(uint64_t player, uint64_t opponent){
    uint64_t edge_stability = 0;
    uint8_t pl, op;
    pl = player & 0b11111111U;
    op = opponent & 0b11111111U;
    edge_stability |= stability_edge_arr[pl][op][0] << 56;
    pl = join_h_line(player, 7);
    op = join_h_line(opponent, 7);
    edge_stability |= stability_edge_arr[pl][op][0];
    pl = join_v_line(player, 0);
    op = join_v_line(opponent, 0);
    edge_stability |= stability_edge_arr[pl][op][1] << 7;
    pl = join_v_line(player, 7);
    op = join_v_line(opponent, 7);
    edge_stability |= stability_edge_arr[pl][op][1];
    return pop_count_ull(edge_stability & player);
}

inline int stability_cut(Search *search, int *alpha, int *beta){
    if (*alpha >= nws_stability_threshold[HW2 - search->board.n]){
        int stab_player, stab_opponent;
        calc_stability(&search->board, &stab_player, &stab_opponent);
        int n_alpha = 2 * stab_player - HW2;
        int n_beta = HW2 - 2 * stab_opponent;
        if (*beta <= n_alpha)
            return n_alpha;
        if (n_beta <= *alpha)
            return n_beta;
        if (n_beta <= n_alpha)
            return n_alpha;
        *alpha = max(*alpha, n_alpha);
        *beta = min(*beta, n_beta);
    }
    return SCORE_UNDEFINED;
}
/* // increase nodes
inline int stability_cut(Search *search, int *alpha, int *beta){
    if (*alpha >= nws_stability_threshold[HW2 - search->board.n]){
        int stab_player, stab_opponent, n_alpha, n_beta;
        uint64_t edge_stability;
        calc_stability_edge(&search->board, &stab_player, &stab_opponent, &edge_stability);
        n_alpha = 2 * stab_player - HW2;
        n_beta = HW2 - 2 * stab_opponent;
        if (*beta <= n_alpha)
            return n_alpha;
        if (n_beta <= *alpha)
            return n_beta;
        if (n_beta <= n_alpha)
            return n_alpha;
        calc_stability(&search->board, edge_stability, &stab_player, &stab_opponent);
        n_alpha = 2 * stab_player - HW2;
        n_beta = HW2 - 2 * stab_opponent;
        if (*beta <= n_alpha)
            return n_alpha;
        if (n_beta <= *alpha)
            return n_beta;
        if (n_beta <= n_alpha)
            return n_alpha;
        *alpha = max(*alpha, n_alpha);
        *beta = min(*beta, n_beta);
    }
    return SCORE_UNDEFINED;
}
*/
/*
inline int stability_cut(Search *search, Flip *flip, int *alpha, int *beta){
    int n_alpha = 2 * flip->stab0 - HW2;
    //int n_beta = HW2 - 2 * flip->stab1;
    if (*beta <= n_alpha)
        return n_alpha;
    //if (n_beta <= *alpha)
    //    return n_beta;
    //if (n_beta <= n_alpha)
    //    return n_alpha;
    *alpha = max(*alpha, n_alpha);
    //*beta = min(*beta, n_beta);
    return SCORE_UNDEFINED;
}
*/
inline int stability_cut_move(Search *search, Flip *flip, int *alpha, int *beta){
    if (flip->stab0 != STABILITY_UNDEFINED){
        int n_alpha = 2 * flip->stab0 - HW2;
        if (*beta <= n_alpha)
            return n_alpha;
        *alpha = max(*alpha, n_alpha);
    } else if (-(*beta) >= nws_stability_threshold[HW2 - search->board.n]){
        flip->stab0 = calc_stability_player(search->board.opponent, search->board.player);
        int n_alpha = 2 * flip->stab0 - HW2;
        if (*beta <= n_alpha)
            return n_alpha;
        *alpha = max(*alpha, n_alpha);
    }
    return SCORE_UNDEFINED;
}

inline void register_tt(Search *search, uint32_t hash_code, int first_alpha, int v, int best_move, int l, int u, int alpha, int beta){
    #if USE_END_TC
        if (search->board.n <= HW2 - USE_TT_DEPTH_THRESHOLD){
            if (first_alpha < v && best_move != TRANSPOSE_TABLE_UNDEFINED)
                child_transpose_table.reg(&search->board, hash_code, best_move);
            if (first_alpha < v && v < beta)
                parent_transpose_table.reg(&search->board, hash_code, v, v);
            else if (beta <= v && l < v)
                parent_transpose_table.reg(&search->board, hash_code, v, u);
            else if (v <= alpha && v < u)
                parent_transpose_table.reg(&search->board, hash_code, l, v);
        }
    #endif
}
