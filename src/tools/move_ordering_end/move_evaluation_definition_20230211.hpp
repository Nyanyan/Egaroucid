#pragma once
#include "./../../engine/board.hpp"

/*
    @brief evaluation pattern definition
*/
// features
#define ADJ_N_FEATURES 20
#define ADJ_N_EVAL 20
#define ADJ_MAX_EVALUATE_IDX 8

constexpr int adj_eval_sizes[ADJ_N_EVAL] = {
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8
};

constexpr int adj_feature_to_eval_idx[ADJ_N_FEATURES] = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 
    10, 11, 12, 13, 14, 15, 16, 17, 18, 19
};

constexpr uint64_t cell_weight_mask[10] = {
    0x8100000000000081ULL, 0x4281000000008142ULL, 0x2400810000810024ULL, 0x1800008181000018ULL, 0x0042000000004200ULL, 
	0x0024420000422400ULL, 0x0018004242001800ULL, 0x0000240000240000ULL, 0x0000182424180000ULL, 0x0000001818000000ULL
};

uint16_t cell_type[HW2] = {
    0, 1, 2, 3, 3, 2, 1, 0, 
    1, 4, 5, 6, 6, 5, 4, 1, 
    2, 5, 7, 8, 8, 7, 5, 2, 
    3, 6, 8, 9, 9, 8, 6, 3, 
    3, 6, 8, 9, 9, 8, 6, 3, 
    2, 5, 7, 8, 8, 7, 5, 2, 
    1, 4, 5, 6, 6, 5, 4, 1, 
    0, 1, 2, 3, 3, 2, 1, 0
};

constexpr uint_fast8_t cell_div4[HW2] = {
    1, 1, 1, 1, 2, 2, 2, 2, 
    1, 1, 1, 1, 2, 2, 2, 2, 
    1, 1, 1, 1, 2, 2, 2, 2, 
    1, 1, 1, 1, 2, 2, 2, 2, 
    4, 4, 4, 4, 8, 8, 8, 8, 
    4, 4, 4, 4, 8, 8, 8, 8, 
    4, 4, 4, 4, 8, 8, 8, 8, 
    4, 4, 4, 4, 8, 8, 8, 8
};

void adj_calc_features(Board *board, uint16_t res[]){
    uint64_t empty = ~(board->player | board->opponent);
    uint_fast8_t parity = 1 & pop_count_ull(empty & 0x000000000F0F0F0FULL);
    parity |= (1 & pop_count_ull(empty & 0x00000000F0F0F0F0ULL)) << 1;
    parity |= (1 & pop_count_ull(empty & 0x0F0F0F0F00000000ULL)) << 2;
    parity |= (1 & pop_count_ull(empty & 0xF0F0F0F000000000ULL)) << 3;
    int idx = 0;
    for (int i = 0; i < 10; ++i)
        res[idx++] = pop_count_ull(board->player & cell_weight_mask[i]);
    for (int i = 0; i < 10; ++i)
        res[idx++] = pop_count_ull(board->opponent & cell_weight_mask[i]);
    //res[idx++] = (parity & cell_div4[cell]) > 0;
}

uint16_t adj_calc_rev_idx(int feature, int idx){
    return idx;
}

void evaluation_definition_init(){
    mobility_init();
}