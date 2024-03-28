/*
    Egaroucid Project

    @file evaluate_common.hpp
        Common things of evaluation function
    @date 2021-2024
    @author Takuto Yamana
    @license GPL-3.0 license
*/

#pragma once
#include "board.hpp"

/*
    @brief evaluation pattern definition
*/
// disc patterns
#define N_PATTERNS 20
#define MAX_PATTERN_CELLS 8
#define MAX_CELL_PATTERNS 16
#define MAX_EVALUATE_IDX 6561
#define N_SYMMETRY_PATTERNS 78
#if USE_SIMD_EVALUATION
    #define N_SIMD_EVAL_FEATURES 5 // 16 (elems per 256 bit vector) * N_SIMD_EVAL_FEATURES >= N_SYMMETRY_PATTERNS
#endif

// additional features
#define MAX_SURROUND 64
#define MAX_STONE_NUM 65

// evaluation phase definition
#define N_PHASES 60
#define PHASE_N_DISCS 1

// move ordering evaluation function
/*
#define MAX_EVALUATE_IDX_MO 19683
#define N_PATTERNS_MO_END 4
#define N_SYMMETRY_PATTERNS_MO_END 16
*/

/*
    @brief value definition

    Raw score is STEP times larger than the real score.
*/
#define STEP 32
#define STEP_2 16

/*
    @brief 3 ^ N definition
*/
#define PNO 0
#define P30 1
#define P31 3
#define P32 9
#define P33 27
#define P34 81
#define P35 243
#define P36 729
#define P37 2187
#define P38 6561
#define P39 19683
#define P310 59049

/*
    @brief coordinate definition
*/
#define COORD_A1 63
#define COORD_B1 62
#define COORD_C1 61
#define COORD_D1 60
#define COORD_E1 59
#define COORD_F1 58
#define COORD_G1 57
#define COORD_H1 56

#define COORD_A2 55
#define COORD_B2 54
#define COORD_C2 53
#define COORD_D2 52
#define COORD_E2 51
#define COORD_F2 50
#define COORD_G2 49
#define COORD_H2 48

#define COORD_A3 47
#define COORD_B3 46
#define COORD_C3 45
#define COORD_D3 44
#define COORD_E3 43
#define COORD_F3 42
#define COORD_G3 41
#define COORD_H3 40

#define COORD_A4 39
#define COORD_B4 38
#define COORD_C4 37
#define COORD_D4 36
#define COORD_E4 35
#define COORD_F4 34
#define COORD_G4 33
#define COORD_H4 32

#define COORD_A5 31
#define COORD_B5 30
#define COORD_C5 29
#define COORD_D5 28
#define COORD_E5 27
#define COORD_F5 26
#define COORD_G5 25
#define COORD_H5 24

#define COORD_A6 23
#define COORD_B6 22
#define COORD_C6 21
#define COORD_D6 20
#define COORD_E6 19
#define COORD_F6 18
#define COORD_G6 17
#define COORD_H6 16

#define COORD_A7 15
#define COORD_B7 14
#define COORD_C7 13
#define COORD_D7 12
#define COORD_E7 11
#define COORD_F7 10
#define COORD_G7 9
#define COORD_H7 8

#define COORD_A8 7
#define COORD_B8 6
#define COORD_C8 5
#define COORD_D8 4
#define COORD_E8 3
#define COORD_F8 2
#define COORD_G8 1
#define COORD_H8 0

#define COORD_NO 64

/*
    @brief definition of patterns in evaluation function

    pattern -> coordinate

    @param n_cells              number of cells included in the pattern
    @param cells                coordinates of each cell
*/
struct Feature_to_coord{
    uint_fast8_t n_cells;
    uint_fast8_t cells[MAX_PATTERN_CELLS];
};

/*
    @brief definition of patterns in evaluation function

    coordinate -> pattern

    @param feature              the index of feature
    @param x                    the offset value of the cell in this feature
*/
struct Coord_feature{
    uint_fast8_t feature;
    uint_fast16_t x;
};

/*
    @brief definition of patterns in evaluation function

    coordinate -> all patterns

    @param n_features           number of features the cell is used by
    @param features             information for each feature
*/
struct Coord_to_feature{
    uint_fast8_t n_features;
    Coord_feature features[MAX_CELL_PATTERNS];
};

#if USE_SIMD
    union Eval_features{
        __m256i f256[N_SIMD_EVAL_FEATURES];
        __m128i f128[N_SIMD_EVAL_FEATURES * 2];
    };

    struct Eval_search{
        Eval_features features[HW2 - 4];
        uint_fast8_t feature_idx;
    };
#else
    struct Eval_search{
        uint_fast16_t features[N_SYMMETRY_PATTERNS];
        bool reversed;
    };
#endif

/*
    @brief constants of 3 ^ N
*/
constexpr uint_fast16_t pow3[11] = {1, P31, P32, P33, P34, P35, P36, P37, P38, P39, P310};

/*
    @brief evaluation function for game over

    @param b                    board
    @return final score
*/
inline int end_evaluate(Board *b){
    return b->score_player();
}

/*
    @brief evaluation function for game over

    @param b                    board
    @param e                    number of empty squares
    @return final score
*/
inline int end_evaluate(Board *b, int e){
    int score = b->count_player() * 2 + e;
    score += (((score >> 6) & 1) + (((score + HW2_M1) >> 7) & 1) - 1) * e;
    return score - HW2;
}

/*
    @brief evaluation function for game over (odd empties)

    @param b                    board
    @param e                    number of empty squares
    @return final score
*/
inline int end_evaluate_odd(Board *b, int e){
    int score = b->count_player() * 2 + e;
    score += (((score >> 5) & 2) - 1) * e;
    return score - HW2;
}
