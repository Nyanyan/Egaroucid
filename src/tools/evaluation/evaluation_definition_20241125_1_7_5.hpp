/*
    Egaroucid Project

    @file evaluation_definition.hpp
        Evaluation Function Definition
    @date 2021-2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once
#ifndef OPTIMIZER_INCLUDE
    #include "./../../engine/board.hpp"
#endif
#include "evaluation_definition_common.hpp"

#define EVAL_DEFINITION_NAME "20241125_1"
#define EVAL_DEFINITION_DESCRIPTION "pattern (7.5-) + n_discs_of_player"

/*
[15, 11, 9, 9, 9, 9, 11, 15]
[11, 14, 9, 7, 7, 9, 14, 11]
[9, 9, 10, 6, 6, 10, 9, 9]
[9, 7, 6, 7, 7, 6, 7, 9]
[9, 7, 6, 7, 7, 6, 7, 9]
[9, 9, 10, 6, 6, 10, 9, 9]
[11, 14, 9, 7, 7, 9, 14, 11]
[15, 11, 9, 9, 9, 9, 11, 15]
*/

/*
    @brief evaluation pattern definition
*/
// disc pattern
#define ADJ_N_PATTERNS 16
#define ADJ_N_SYMMETRY_PATTERNS 64
#define ADJ_MAX_PATTERN_CELLS 10

// additional features
#define ADJ_N_ADDITIONAL_EVALS 1
#define ADJ_MAX_STONE_NUM 65

// overall
#define ADJ_MAX_EVALUATE_IDX 59049
#define ADJ_N_EVAL (16 + 1)
#define ADJ_N_FEATURES (64 + 1)

// phase
#define ADJ_N_PHASES 60
#define ADJ_N_PHASE_DISCS 1 // 60 / ADJ_N_PHASES

//#define ADJ_SCORE_MAX HW2

/*
    @brief value definition

    Raw score is STEP times larger than the real score.
*/
#define ADJ_STEP 32
#define ADJ_STEP_2 16
#define ADJ_EVAL_PARAM_MAX 4091

/*
    @brief definition of patterns in evaluation function

    pattern -> coordinate

    @param n_cells              number of cells included in the pattern
    @param cells                coordinates of each cell
*/
struct Adj_Feature_to_coord{
    uint_fast8_t n_cells;
    uint_fast8_t cells[ADJ_MAX_PATTERN_CELLS];
};

constexpr Adj_Feature_to_coord adj_feature_to_coord[ADJ_N_SYMMETRY_PATTERNS] = {
    // 0 hv2
    {8, {COORD_A2, COORD_B2, COORD_C2, COORD_D2, COORD_E2, COORD_F2, COORD_G2, COORD_H2, COORD_NO, COORD_NO}},
    {8, {COORD_B1, COORD_B2, COORD_B3, COORD_B4, COORD_B5, COORD_B6, COORD_B7, COORD_B8, COORD_NO, COORD_NO}},
    {8, {COORD_A7, COORD_B7, COORD_C7, COORD_D7, COORD_E7, COORD_F7, COORD_G7, COORD_H7, COORD_NO, COORD_NO}},
    {8, {COORD_G1, COORD_G2, COORD_G3, COORD_G4, COORD_G5, COORD_G6, COORD_G7, COORD_G8, COORD_NO, COORD_NO}},
    
    // 1 d6 + 2C + X
    {9, {COORD_B1, COORD_C1, COORD_D2, COORD_E3, COORD_G2, COORD_F4, COORD_G5, COORD_H6, COORD_H7, COORD_NO}},
    {9, {COORD_H2, COORD_H3, COORD_G4, COORD_F5, COORD_G7, COORD_E6, COORD_D7, COORD_C8, COORD_B8, COORD_NO}},
    {9, {COORD_G8, COORD_F8, COORD_E7, COORD_D6, COORD_B7, COORD_C5, COORD_B4, COORD_A3, COORD_A2, COORD_NO}},
    {9, {COORD_A7, COORD_A6, COORD_B5, COORD_C4, COORD_B2, COORD_D3, COORD_E2, COORD_F1, COORD_G1, COORD_NO}},

    // 2 hv3
    {8, {COORD_A3, COORD_B3, COORD_C3, COORD_D3, COORD_E3, COORD_F3, COORD_G3, COORD_H3, COORD_NO, COORD_NO}},
    {8, {COORD_C1, COORD_C2, COORD_C3, COORD_C4, COORD_C5, COORD_C6, COORD_C7, COORD_C8, COORD_NO, COORD_NO}},
    {8, {COORD_A6, COORD_B6, COORD_C6, COORD_D6, COORD_E6, COORD_F6, COORD_G6, COORD_H6, COORD_NO, COORD_NO}},
    {8, {COORD_F1, COORD_F2, COORD_F3, COORD_F4, COORD_F5, COORD_F6, COORD_F7, COORD_F8, COORD_NO, COORD_NO}},

    // 3 d7 + 2 corner
    {9, {COORD_A1, COORD_B1, COORD_C2, COORD_D3, COORD_E4, COORD_F5, COORD_G6, COORD_H7, COORD_H8, COORD_NO}},
    {9, {COORD_H1, COORD_H2, COORD_G3, COORD_F4, COORD_E5, COORD_D6, COORD_C7, COORD_B8, COORD_A8, COORD_NO}},
    {9, {COORD_H8, COORD_G8, COORD_F7, COORD_E6, COORD_D5, COORD_C4, COORD_B3, COORD_A2, COORD_A1, COORD_NO}},
    {9, {COORD_A8, COORD_A7, COORD_B6, COORD_C5, COORD_D4, COORD_E3, COORD_F2, COORD_G1, COORD_H1, COORD_NO}},

    // 4 hv4
    {8, {COORD_A4, COORD_B4, COORD_C4, COORD_D4, COORD_E4, COORD_F4, COORD_G4, COORD_H4, COORD_NO, COORD_NO}},
    {8, {COORD_D1, COORD_D2, COORD_D3, COORD_D4, COORD_D5, COORD_D6, COORD_D7, COORD_D8, COORD_NO, COORD_NO}},
    {8, {COORD_A5, COORD_B5, COORD_C5, COORD_D5, COORD_E5, COORD_F5, COORD_G5, COORD_H5, COORD_NO, COORD_NO}},
    {8, {COORD_E1, COORD_E2, COORD_E3, COORD_E4, COORD_E5, COORD_E6, COORD_E7, COORD_E8, COORD_NO, COORD_NO}},

    // 5 corner9
    {9, {COORD_A1, COORD_B1, COORD_C1, COORD_A2, COORD_B2, COORD_C2, COORD_A3, COORD_B3, COORD_C3, COORD_NO}},
    {9, {COORD_H1, COORD_G1, COORD_F1, COORD_H2, COORD_G2, COORD_F2, COORD_H3, COORD_G3, COORD_F3, COORD_NO}},
    {9, {COORD_A8, COORD_B8, COORD_C8, COORD_A7, COORD_B7, COORD_C7, COORD_A6, COORD_B6, COORD_C6, COORD_NO}},
    {9, {COORD_H8, COORD_G8, COORD_F8, COORD_H7, COORD_G7, COORD_F7, COORD_H6, COORD_G6, COORD_F6, COORD_NO}},

    // 6 d5 + 2X
    {7, {COORD_B2, COORD_D1, COORD_E2, COORD_F3, COORD_G4, COORD_H5, COORD_G7, COORD_NO, COORD_NO, COORD_NO}},
    {7, {COORD_G2, COORD_H4, COORD_G5, COORD_F6, COORD_E7, COORD_D8, COORD_B7, COORD_NO, COORD_NO, COORD_NO}},
    {7, {COORD_G7, COORD_E8, COORD_D7, COORD_C6, COORD_B5, COORD_A4, COORD_B2, COORD_NO, COORD_NO, COORD_NO}},
    {7, {COORD_B7, COORD_A5, COORD_B4, COORD_C3, COORD_D2, COORD_E1, COORD_G2, COORD_NO, COORD_NO, COORD_NO}},

    // 7 d8 + 2C
    {10, {COORD_A1, COORD_B2, COORD_C3, COORD_D4, COORD_E5, COORD_F6, COORD_G7, COORD_H8, COORD_A2, COORD_B1}},
    {10, {COORD_H8, COORD_G7, COORD_F6, COORD_E5, COORD_D4, COORD_C3, COORD_B2, COORD_A1, COORD_H7, COORD_G8}},
    {10, {COORD_H1, COORD_G2, COORD_F3, COORD_E4, COORD_D5, COORD_C6, COORD_B7, COORD_A8, COORD_H2, COORD_G1}},
    {10, {COORD_A8, COORD_B7, COORD_C6, COORD_D5, COORD_E4, COORD_F3, COORD_G2, COORD_H1, COORD_A7, COORD_B8}},
 
    // 8 edge + 2x
    {10, {COORD_B2, COORD_A1, COORD_B1, COORD_C1, COORD_D1, COORD_E1, COORD_F1, COORD_G1, COORD_H1, COORD_G2}},
    {10, {COORD_B2, COORD_A1, COORD_A2, COORD_A3, COORD_A4, COORD_A5, COORD_A6, COORD_A7, COORD_A8, COORD_B7}},
    {10, {COORD_B7, COORD_A8, COORD_B8, COORD_C8, COORD_D8, COORD_E8, COORD_F8, COORD_G8, COORD_H8, COORD_G7}},
    {10, {COORD_G2, COORD_H1, COORD_H2, COORD_H3, COORD_H4, COORD_H5, COORD_H6, COORD_H7, COORD_H8, COORD_G7}},

    // 9 triangle
    {10, {COORD_A1, COORD_B1, COORD_C1, COORD_D1, COORD_A2, COORD_B2, COORD_C2, COORD_A3, COORD_B3, COORD_A4}},
    {10, {COORD_H1, COORD_G1, COORD_F1, COORD_E1, COORD_H2, COORD_G2, COORD_F2, COORD_H3, COORD_G3, COORD_H4}},
    {10, {COORD_A8, COORD_B8, COORD_C8, COORD_D8, COORD_A7, COORD_B7, COORD_C7, COORD_A6, COORD_B6, COORD_A5}},
    {10, {COORD_H8, COORD_G8, COORD_F8, COORD_E8, COORD_H7, COORD_G7, COORD_F7, COORD_H6, COORD_G6, COORD_H5}},

    // 10 corner + block
    {10, {COORD_A1, COORD_C1, COORD_D1, COORD_E1, COORD_F1, COORD_H1, COORD_C2, COORD_D2, COORD_E2, COORD_F2}},
    {10, {COORD_A1, COORD_A3, COORD_A4, COORD_A5, COORD_A6, COORD_A8, COORD_B3, COORD_B4, COORD_B5, COORD_B6}},
    {10, {COORD_A8, COORD_C8, COORD_D8, COORD_E8, COORD_F8, COORD_H8, COORD_C7, COORD_D7, COORD_E7, COORD_F7}},
    {10, {COORD_H1, COORD_H3, COORD_H4, COORD_H5, COORD_H6, COORD_H8, COORD_G3, COORD_G4, COORD_G5, COORD_G6}},

    // 11 cross
    {10, {COORD_A1, COORD_B2, COORD_C3, COORD_D4, COORD_B1, COORD_C2, COORD_D3, COORD_A2, COORD_B3, COORD_C4}},
    {10, {COORD_H1, COORD_G2, COORD_F3, COORD_E4, COORD_G1, COORD_F2, COORD_E3, COORD_H2, COORD_G3, COORD_F4}},
    {10, {COORD_A8, COORD_B7, COORD_C6, COORD_D5, COORD_B8, COORD_C7, COORD_D6, COORD_A7, COORD_B6, COORD_C5}},
    {10, {COORD_H8, COORD_G7, COORD_F6, COORD_E5, COORD_G8, COORD_F7, COORD_E6, COORD_H7, COORD_G6, COORD_F5}},

    // 12 edge + y
    {10, {COORD_C2, COORD_A1, COORD_B1, COORD_C1, COORD_D1, COORD_E1, COORD_F1, COORD_G1, COORD_H1, COORD_F2}},
    {10, {COORD_B3, COORD_A1, COORD_A2, COORD_A3, COORD_A4, COORD_A5, COORD_A6, COORD_A7, COORD_A8, COORD_B6}},
    {10, {COORD_C7, COORD_A8, COORD_B8, COORD_C8, COORD_D8, COORD_E8, COORD_F8, COORD_G8, COORD_H8, COORD_F7}},
    {10, {COORD_G3, COORD_H1, COORD_H2, COORD_H3, COORD_H4, COORD_H5, COORD_H6, COORD_H7, COORD_H8, COORD_G6}},

    // 13 narrow triangle
    {10, {COORD_A1, COORD_B1, COORD_C1, COORD_D1, COORD_E1, COORD_A2, COORD_B2, COORD_A3, COORD_A4, COORD_A5}},
    {10, {COORD_H1, COORD_G1, COORD_F1, COORD_E1, COORD_D1, COORD_H2, COORD_G2, COORD_H3, COORD_H4, COORD_H5}},
    {10, {COORD_A8, COORD_B8, COORD_C8, COORD_D8, COORD_E8, COORD_A7, COORD_B7, COORD_A6, COORD_A5, COORD_A4}},
    {10, {COORD_H8, COORD_G8, COORD_F8, COORD_E8, COORD_D8, COORD_H7, COORD_G7, COORD_H6, COORD_H5, COORD_H4}},

    // 14 fish
    {10, {COORD_A1, COORD_B1, COORD_A2, COORD_B2, COORD_C2, COORD_D2, COORD_B3, COORD_C3, COORD_B4, COORD_D4}},
    {10, {COORD_H1, COORD_G1, COORD_H2, COORD_G2, COORD_F2, COORD_E2, COORD_G3, COORD_F3, COORD_G4, COORD_E4}},
    {10, {COORD_A8, COORD_B8, COORD_A7, COORD_B7, COORD_C7, COORD_D7, COORD_B6, COORD_C6, COORD_B5, COORD_D5}},
    {10, {COORD_H8, COORD_G8, COORD_H7, COORD_G7, COORD_F7, COORD_E7, COORD_G6, COORD_F6, COORD_G5, COORD_E5}},

    // 15 anvil
    {10, {COORD_C6, COORD_D6, COORD_D7, COORD_D8, COORD_C8, COORD_F8, COORD_E8, COORD_E7, COORD_E6, COORD_F6}},
    {10, {COORD_C3, COORD_C4, COORD_B4, COORD_A4, COORD_A3, COORD_A6, COORD_A5, COORD_B5, COORD_C5, COORD_C6}},
    {10, {COORD_F3, COORD_E3, COORD_E2, COORD_E1, COORD_F1, COORD_C1, COORD_D1, COORD_D2, COORD_D3, COORD_C3}},
    {10, {COORD_F6, COORD_F5, COORD_G5, COORD_H5, COORD_H6, COORD_H3, COORD_H4, COORD_G4, COORD_F4, COORD_F3}}
};

constexpr int adj_pattern_n_cells[ADJ_N_PATTERNS] = {8, 9, 8, 9, 8, 9, 7, 10, 10, 10, 10, 10, 10, 10, 10, 10};

constexpr int adj_rev_patterns[ADJ_N_PATTERNS][ADJ_MAX_PATTERN_CELLS] = {
    {7, 6, 5, 4, 3, 2, 1, 0}, // 0 hv2
    {8, 7, 6, 5, 4, 3, 2, 1, 0}, // 1 d6 + 2C + x
    {7, 6, 5, 4, 3, 2, 1, 0}, // 2 hv3
    {8, 7, 6, 5, 4, 3, 2, 1, 0}, // 3 d7 + 2Corner
    {7, 6, 5, 4, 3, 2, 1, 0}, // 4 hv4
    {0, 3, 6, 1, 4, 7, 2, 5, 8}, // 5 corner9
    {6, 5, 4, 3, 2, 1, 0}, // 6 d5 + 2X
    {0, 1, 2, 3, 4, 5, 6, 7, 9, 8}, // 7 d8 + 2C    
    {9, 8, 7, 6, 5, 4, 3, 2, 1, 0}, // 8 edge + 2x
    {0, 4, 7, 9, 1, 5, 8, 2, 6, 3}, // 9 triangle
    {5, 4, 3, 2, 1, 0, 9, 8, 7, 6}, // 10 corner + block
    {0, 1, 2, 3, 7, 8, 9, 4, 5, 6}, // 11 cross
    {9, 8, 7, 6, 5, 4, 3, 2, 1, 0}, // 12 edge + y
    {0, 5, 7, 8, 9, 1, 6, 2, 3, 4}, // 13 narrow triangle
    {0, 2, 1, 3, 6, 8, 4, 7, 5, 9}, // 14 fish
    {9, 8, 7, 6, 5, 4, 3, 2, 1, 0}  // 15 anvil
};

constexpr int adj_eval_sizes[ADJ_N_EVAL] = {
    P38, P39, P38, P39, 
    P38, P39, P37, P310, 
    P310, P310, P310, P310, 
    P310, P310, P310, P310, 
    ADJ_MAX_STONE_NUM
};

constexpr int adj_feature_to_eval_idx[ADJ_N_FEATURES] = {
    0, 0, 0, 0, 
    1, 1, 1, 1, 
    2, 2, 2, 2, 
    3, 3, 3, 3, 
    4, 4, 4, 4, 
    5, 5, 5, 5, 
    6, 6, 6, 6, 
    7, 7, 7, 7, 
    8, 8, 8, 8, 
    9, 9, 9, 9, 
    10, 10, 10, 10, 
    11, 11, 11, 11, 
    12, 12, 12, 12, 
    13, 13, 13, 13, 
    14, 14, 14, 14, 
    15, 15, 15, 15, 
    16
};

int adj_pick_digit3(int num, int d, int n_digit){
    num /= adj_pow3[n_digit - 1 - d];
    return num % 3;
}

int adj_pick_digit2(int num, int d){
    return 1 & (num >> d);
}

uint16_t adj_calc_rev_idx(int feature, int idx){
    uint16_t res = 0;
    if (feature < ADJ_N_PATTERNS){
        for (int i = 0; i < adj_pattern_n_cells[feature]; ++i){
            res += adj_pick_digit3(idx, adj_rev_patterns[feature][i], adj_pattern_n_cells[feature]) * adj_pow3[adj_pattern_n_cells[feature] - 1 - i];
        }
    } else{
        res = idx;
    }
    return res;
}

#ifndef OPTIMIZER_INCLUDE

int adj_calc_num_feature(Board *board){
    return pop_count_ull(board->player); // board->opponent is unnecessary because of the 60 phases
}

inline int adj_pick_pattern(const uint_fast8_t b_arr[], int pattern_idx){
    int res = 0;
    for (int i = 0; i < adj_feature_to_coord[pattern_idx].n_cells; ++i){
        res *= 3;
        res += b_arr[adj_feature_to_coord[pattern_idx].cells[i]];
    }
    return res;
}

void adj_calc_features(Board *board, uint16_t res[]){
    uint_fast8_t b_arr[HW2];
    board->translate_to_arr_player(b_arr);
    int idx = 0;
    for (int i = 0; i < ADJ_N_SYMMETRY_PATTERNS; ++i)
        res[idx++] = adj_pick_pattern(b_arr, i);
    res[idx++] = adj_calc_num_feature(board);
}

int calc_phase(Board *board, int16_t player){
    return (pop_count_ull(board->player | board->opponent) - 4) / ADJ_N_PHASE_DISCS;
}

void evaluation_definition_init(){
    mobility_init();
}

#endif