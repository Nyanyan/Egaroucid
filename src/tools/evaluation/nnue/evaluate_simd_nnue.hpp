/*
    Egaroucid Project

    @file evaluate_simd.hpp
        NNUE Evaluation function with AVX2
    @date 2021-2024
    @author Takuto Yamana
    @license GPL-3.0 license
*/

#pragma once
#include <iostream>
#include <fstream>
#ifdef _MSC_VER
    #include <intrin.h>
#else
    #include <x86intrin.h>
#endif
#include "setting.hpp"
#include "common.hpp"
#include "board.hpp"
#include "search.hpp"
#include "util.hpp"
#include "evaluate_common.hpp"

#define EVAL_NNUE_N_INPUT 128
#define EVAL_NNUE_N_NODES_LAYER 16

// Input: 64 x 2 (0 or 1)
// LayerA: 16 nodes
// LayerB: 16 nodes
// Output: 1 nodes
__m256i eval_nnue_layer_A_bias;
__m256i eval_nnue_layer_A_weight[EVAL_NNUE_N_INPUT];
__m256i eval_nnue_layer_B_bias;
__m256i eval_nnue_layer_B_weight[EVAL_NNUE_N_NODES_LAYER];
int eval_nnue_layer_out_bias;
__m256i eval_nnue_layer_out_weight;

inline __m256i clipped_ReLU(__m256i a){
    a = _mm256_max_epi16(a, _mm256_set1_epi16(-127));
    a = _mm256_min_epi16(a, _mm256_set1_epi16(127));
    return a;
}

/*
    @brief midgame evaluation function

    @param search               search information
    @return evaluation value
*/
inline int mid_evaluate(Search *search){
    int16_t layer_B_in_arr[EVAL_NNUE_N_NODES_LAYER];
    _mm256_storeu_epi16(layer_B_in_arr, clipped_ReLU(search->eval.layer_A[search->eval.feature_idx]));
    __m256i layer_B_out = eval_nnue_layer_B_bias;
    // layer B
    for (int i = 0; i < EVAL_NNUE_N_NODES_LAYER; ++i){
        layer_B_out = _mm256_add_epi16(layer_B_out, _mm256_mullo_epi16(eval_nnue_layer_B_weight[i], _mm256_set1_epi16(layer_B_in_arr[i])));
    }
    layer_B_out = clipped_ReLU(layer_B_out);
    // output layer
    __m256i out = _mm256_mullo_epi16(layer_B_out, eval_nnue_layer_out_weight);
    int16_t out_arr[EVAL_NNUE_N_NODES_LAYER];
    _mm256_storeu_epi16(out_arr, out);
    int res = eval_nnue_layer_out_bias;
    for (int i = 0; i < EVAL_NNUE_N_NODES_LAYER; ++i){
        res += out_arr[i];
    }
    res += res >= 0 ? STEP_2 : -STEP_2;
    res /= STEP;
    res = std::clamp(res, -SCORE_MAX, SCORE_MAX);
    return res;
}


/*
    @brief calculate features for pattern evaluation

    @param search               search information
*/
inline void calc_eval_features(const Board *board, Eval_search *eval){
    eval->layer_A[0] = eval_nnue_layer_A_bias;
    uint64_t bits = board->player;
    for (uint_fast8_t cell = first_bit(&bits); bits; cell = next_bit(&bits)){
        eval->layer_A[0] = _mm256_add_epi16(eval->layer_A[0], eval_nnue_layer_A_weight[cell]);
    }
    bits = board->opponent;
    for (uint_fast8_t cell = first_bit(&bits); bits; cell = next_bit(&bits)){
        eval->layer_A[0] = _mm256_add_epi16(eval->layer_A[0], eval_nnue_layer_A_weight[cell + HW2]);
    }
    eval->feature_idx = 0;
}


/*
    @brief move evaluation features

        put cell        2 -> 1 (empty -> opponent)  sub
        player discs    0 -> 1 (player -> opponent) add
        opponent discs  1 -> 0 (player -> opponent) sub
        flipped discs   1 -> 1 (opponent -> opponent)
        empty cells     2 -> 2 (empty -> empty)
    
    @param eval                 evaluation features
    @param flip                 flip information
*/
inline void eval_move(Eval_search *eval, const Flip *flip, const Board *board){
    calc_eval_features(board, eval);
}

