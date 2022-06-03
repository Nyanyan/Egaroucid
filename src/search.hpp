#pragma once
#include <iostream>
#include <fstream>
#include <algorithm>
#include "setting.hpp"
#include "common.hpp"
#include "board.hpp"
#include "evaluate.hpp"
#include "transpose_table.hpp"

using namespace std;

#ifndef N_SYMMETRY_PATTERNS
    #define N_SYMMETRY_PATTERNS 62
#endif
/*
#ifndef N_ADDITIONAL_SYMMETRY_PATTERNS
    #define N_ADDITIONAL_SYMMETRY_PATTERNS 12
#endif
*/
#define MID_FAST_DEPTH 1
#define END_FAST_DEPTH 7
#define MID_TO_END_DEPTH 13
#define CUDA_YBWC_SPLIT_MAX_DEPTH 10
#define USE_PARENT_TT_DEPTH_THRESHOLD 8

#define SCORE_UNDEFINED -INF

#define N_CELL_WEIGHT_MASK 5

constexpr int cell_weight[HW2] = {
    18,  4,  16, 12, 12, 16,  4, 18,
     4,  2,   6,  8,  8,  6,  2,  4,
    16,  6,  14, 10, 10, 14,  6, 16,
    12,  8,  10,  0,  0, 10,  8, 12,
    12,  8,  10,  0,  0, 10,  8, 12,
    16,  6,  14, 10, 10, 14,  6, 16,
     4,  2,   6,  8,  8,  6,  2,  4,
    18,  4,  16, 12, 12, 16,  4, 18
};

constexpr uint64_t cell_weight_mask[N_CELL_WEIGHT_MASK] = {
    0b10000001'00000000'00000000'00000000'00000000'00000000'00000000'10000001ULL, // corner
    0b00011000'00000000'00000000'10000001'10000001'00000000'00000000'00011000ULL, // B
    0b01100110'10000001'10000001'00000000'00000000'10000001'10000001'01100110ULL, // A & C
    0b00000000'01000010'00000000'00000000'00000000'00000000'01000010'00000000ULL, // X
    0b00000000'00111100'01111110'01111110'01111110'01111110'00111100'00000000ULL  // other
};

/*
from https://github.com/abulmo/edax-reversi/blob/1ae7c9fe5322ac01975f1b3196e788b0d25c1e10/src/search.c
modified by Nyanyan
*/
constexpr int nws_stability_threshold[61] = {
    99, 99, 99,  4,  6,  8, 10, 12,
    14, 16, 20, 22, 24, 26, 28, 30,
    32, 34, 36, 38, 40, 42, 44, 46,
    48, 48, 50, 50, 52, 52, 54, 54,
    56, 56, 58, 58, 60, 60, 62, 62,
    64, 64, 64, 64, 64, 64, 64, 64,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99
};
/*
end of modification
*/

struct Search_result{
    int policy;
    int value;
    int depth;
    int nps;
    uint64_t time;
    uint64_t nodes;
};

struct Search{
    Board board;
    bool use_mpc;
    double mpct;
    uint64_t n_nodes;
    //int eval_features[N_SYMMETRY_PATTERNS + N_ADDITIONAL_SYMMETRY_PATTERNS];
    int eval_features[N_SYMMETRY_PATTERNS];
    uint_fast8_t eval_feature_reversed;
};

inline void calc_stability(Board *b, int *stab0, int *stab1);
inline int calc_stability_player(uint64_t player, uint64_t opponent);

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

inline int stability_cut_move(Search *search, Flip *flip, int *alpha, int *beta){
    if (-(*beta) >= nws_stability_threshold[HW2 - search->board.n]){
        if (flip->stab0 == STABILITY_UNDEFINED)
            flip->stab0 = calc_stability_player(search->board.opponent, search->board.player);
        int n_alpha = 2 * flip->stab0 - HW2;
        if (*beta <= n_alpha)
            return n_alpha;
        *alpha = max(*alpha, n_alpha);
    }
    return SCORE_UNDEFINED;
}

inline void register_tt(Search *search, uint32_t hash_code, int first_alpha, int v, int best_move, int l, int u, int alpha, int beta){
    if (first_alpha < v && best_move != TRANSPOSE_TABLE_UNDEFINED)
        child_transpose_table.reg(&search->board, hash_code, best_move);
    #if USE_END_TC
        if (search->board.n <= HW2 - USE_PARENT_TT_DEPTH_THRESHOLD){
            if (first_alpha < v && v < beta)
                parent_transpose_table.reg(&search->board, hash_code, v, v);
            else if (beta <= v && l < v)
                parent_transpose_table.reg(&search->board, hash_code, v, u);
            else if (v <= alpha && v < u)
                parent_transpose_table.reg(&search->board, hash_code, l, v);
        }
    #endif
}