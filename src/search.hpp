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

#define MID_FAST_DEPTH 1
#define END_FAST_DEPTH 7
#define MID_TO_END_DEPTH 13
#define CUDA_YBWC_SPLIT_MAX_DEPTH 10

#define SCORE_UNDEFINED -INF

#define N_CELL_WEIGHT_MASK 6

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
    0b00100100'00000000'10000001'00000000'00000000'10000001'00000000'00100100ULL, // A
    0b01000010'10000001'00000000'00000000'00000000'00000000'10000001'01000010ULL, // C
    0b00011000'00000000'00000000'10000001'10000001'00000000'00000000'00011000ULL, // B
    0b00000000'01000010'00000000'00000000'00000000'00000000'01000010'00000000ULL, // X
    0b00000000'00111100'01111110'01111110'01111110'01111110'00111100'00000000ULL  // other
};

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
    int *eval_features;
    uint_fast8_t eval_feature_reversed;
    
    inline void init(){
        eval_features = (int*)malloc(sizeof(int) * N_SYMMETRY_PATTERNS);
    }

    inline void del(){
        free(eval_features);
    }
    //uint_fast8_t p;
};

inline void calc_stability(Board *b, int *stab0, int *stab1);

inline int stability_cut(Search *search, int *alpha, int *beta){
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
    return SCORE_UNDEFINED;
}
