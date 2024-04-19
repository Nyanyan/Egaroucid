/*
    Egaroucid Project

    @file evaluation_definition.hpp
        Evaluation Function Definition
    @date 2021-2024
    @author Takuto Yamana
    @license GPL-3.0 license
*/

#pragma once
#ifndef OPTIMIZER_INCLUDE
    #include "./../../engine/board.hpp"
#endif
#include "evaluation_definition_common.hpp"

#define EVAL_DEFINITION_NAME "20240419_1_cell_weight"
#define EVAL_DEFINITION_DESCRIPTION "cell weight evaluation"

/*
    @brief evaluation pattern definition
*/
// overall
#define ADJ_MAX_EVALUATE_IDX 10
#define ADJ_N_EVAL 1
#define ADJ_N_FEATURES 64

// phase
#define ADJ_N_PHASES 1
#define ADJ_N_PHASE_DISCS 60 // 60 / ADJ_N_PHASES

//#define ADJ_SCORE_MAX HW2

/*
    @brief value definition

    Raw score is STEP times larger than the real score.
*/
#define ADJ_STEP 256
#define ADJ_STEP_2 128

constexpr int adj_eval_sizes[ADJ_N_EVAL] = {
    10
};

constexpr int adj_feature_to_eval_idx[ADJ_N_FEATURES] = {
    0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0
};

constexpr int adj_pattern_to_feature[ADJ_N_FEATURES] = {
    0, 1, 2, 3, 3, 2, 1, 0, 
    1, 4, 5, 6, 6, 5, 4, 1, 
    2, 5, 7, 8, 8, 7, 5, 2, 
    3, 6, 8, 9, 9, 8, 6, 3, 
    3, 6, 8, 9, 9, 8, 6, 3, 
    2, 5, 7, 8, 8, 7, 5, 2, 
    1, 4, 5, 6, 6, 5, 4, 1, 
    0, 1, 2, 3, 3, 2, 1, 0
};

uint16_t adj_calc_rev_idx(int feature, int idx){
    uint16_t res = 0;
    res = idx;
    return res;
}

#ifndef OPTIMIZER_INCLUDE

void adj_calc_features(Board *board, uint16_t res[]){
    uint_fast8_t b_arr[HW2];
    board->translate_to_arr_player(b_arr);
    int idx = 0;
    for (int i = 0; i < ADJ_N_FEATURES; ++i)
        res[idx++] = adj_pattern_to_feature[i];
}

int calc_phase(Board *board, int16_t player){
    return (pop_count_ull(board->player | board->opponent) - 4) / ADJ_N_PHASE_DISCS;
}

void evaluation_definition_init(){
    mobility_init();
}

#endif
