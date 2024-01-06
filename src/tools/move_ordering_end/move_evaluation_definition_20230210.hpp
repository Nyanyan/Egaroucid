#pragma once
#include "./../../engine/board.hpp"

/*
    @brief evaluation pattern definition
*/
// features
#define ADJ_N_NORMAL_FEATURES 5
#define ADJ_MO_MAX_MOBILITY 16
#define ADJ_MO_MAX_POT_MOBILITY 16

// disc pattern
#define ADJ_N_PATTERNS 4
#define ADJ_N_SYMMETRY_PATTERNS 16
#define ADJ_MAX_PATTERN_CELLS 10
#define ADJ_MAX_CELL_PATTERNS 5

// overall
#define ADJ_MAX_EVALUATE_IDX 59049
#define ADJ_N_EVAL (5 + 8)
#define ADJ_N_FEATURES (5 + 32)

#define ADJ_MO_SCORE_MAX 16384


/*
    @brief value definition

    Raw score is STEP times larger than the real score.
*/

#ifndef PNO
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
        @brief 4 ^ N definition
    */
    #define P40 1
    #define P41 4
    #define P42 16
    #define P43 64
    #define P44 256
    #define P45 1024
    #define P46 4096
    #define P47 16384
    #define P48 65536
#endif

#ifndef COORD_NO
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
#endif

constexpr int adj_pow3[11] = {P30, P31, P32, P33, P34, P35, P36, P37, P38, P39, P310};

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
    {8, {COORD_A2, COORD_B2, COORD_C2, COORD_D2, COORD_E2, COORD_F2, COORD_G2, COORD_H2, COORD_NO, COORD_NO}}, // 0
    {8, {COORD_B1, COORD_B2, COORD_B3, COORD_B4, COORD_B5, COORD_B6, COORD_B7, COORD_B8, COORD_NO, COORD_NO}}, // 1
    {8, {COORD_A7, COORD_B7, COORD_C7, COORD_D7, COORD_E7, COORD_F7, COORD_G7, COORD_H7, COORD_NO, COORD_NO}}, // 2
    {8, {COORD_G1, COORD_G2, COORD_G3, COORD_G4, COORD_G5, COORD_G6, COORD_G7, COORD_G8, COORD_NO, COORD_NO}}, // 3

    // 2 d7
    {7, {COORD_B1, COORD_C2, COORD_D3, COORD_E4, COORD_F5, COORD_G6, COORD_H7, COORD_NO, COORD_NO, COORD_NO}}, // 4
    {7, {COORD_G1, COORD_F2, COORD_E3, COORD_D4, COORD_C5, COORD_B6, COORD_A7, COORD_NO, COORD_NO, COORD_NO}}, // 5
    {7, {COORD_A2, COORD_B3, COORD_C4, COORD_D5, COORD_E6, COORD_F7, COORD_G8, COORD_NO, COORD_NO, COORD_NO}}, // 6
    {7, {COORD_H2, COORD_G3, COORD_F4, COORD_E5, COORD_D6, COORD_C7, COORD_B8, COORD_NO, COORD_NO, COORD_NO}}, // 7

    // 3 edge + 2x
    {10, {COORD_B2, COORD_A1, COORD_B1, COORD_C1, COORD_D1, COORD_E1, COORD_F1, COORD_G1, COORD_H1, COORD_G2}}, // 8
    {10, {COORD_B2, COORD_A1, COORD_A2, COORD_A3, COORD_A4, COORD_A5, COORD_A6, COORD_A7, COORD_A8, COORD_B7}}, // 9
    {10, {COORD_B7, COORD_A8, COORD_B8, COORD_C8, COORD_D8, COORD_E8, COORD_F8, COORD_G8, COORD_H8, COORD_G7}}, // 10
    {10, {COORD_G2, COORD_H1, COORD_H2, COORD_H3, COORD_H4, COORD_H5, COORD_H6, COORD_H7, COORD_H8, COORD_G7}}, // 11

    // 4 triangle
    {10, {COORD_A1, COORD_B1, COORD_C1, COORD_D1, COORD_A2, COORD_B2, COORD_C2, COORD_A3, COORD_B3, COORD_A4}}, // 12
    {10, {COORD_H1, COORD_G1, COORD_F1, COORD_E1, COORD_H2, COORD_G2, COORD_F2, COORD_H3, COORD_G3, COORD_H4}}, // 13
    {10, {COORD_A8, COORD_B8, COORD_C8, COORD_D8, COORD_A7, COORD_B7, COORD_C7, COORD_A6, COORD_B6, COORD_A5}}, // 14
    {10, {COORD_H8, COORD_G8, COORD_F8, COORD_E8, COORD_H7, COORD_G7, COORD_F7, COORD_H6, COORD_G6, COORD_H5}}  // 15
};

constexpr int adj_pattern_n_cells[ADJ_N_PATTERNS] = {8, 7, 10, 10};

constexpr int adj_rev_patterns[ADJ_N_PATTERNS][ADJ_MAX_PATTERN_CELLS] = {
    {7, 6, 5, 4, 3, 2, 1, 0}, // 0 hv2
    {6, 5, 4, 3, 2, 1, 0}, // 1 d7
    {9, 8, 7, 6, 5, 4, 3, 2, 1, 0}, // 2 edge + 2x
    {0, 4, 7, 9, 1, 5, 8, 2, 6, 3}  // 3 triangle
};


/*
    @brief definition of patterns in evaluation function

    coordinate -> pattern

    @param feature              the index of feature
    @param x                    the offset value of the cell in this feature
*/
struct Adj_Coord_feature{
    uint_fast8_t feature;
    uint_fast16_t x;
};

/*
    @brief definition of patterns in evaluation function

    coordinate -> all patterns

    @param n_features           number of features the cell is used by
    @param features             information for each feature
*/
struct Adj_Coord_to_feature{
    uint_fast8_t n_features;
    Adj_Coord_feature features[ADJ_MAX_CELL_PATTERNS];
};


constexpr Adj_Coord_to_feature adj_coord_to_feature[HW2] = {
    { 3, {{10, P31}, {11, P31}, {15, P39}, { 0, PNO}, { 0, PNO}}}, // COORD_H8
    { 4, {{ 3, P30}, { 6, P30}, {10, P32}, {15, P38}, { 0, PNO}}}, // COORD_G8
    { 2, {{10, P33}, {15, P37}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_F8
    { 2, {{10, P34}, {15, P36}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_E8
    { 2, {{10, P35}, {14, P36}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_D8
    { 2, {{10, P36}, {14, P37}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_C8
    { 4, {{ 1, P30}, { 7, P30}, {10, P37}, {14, P38}, { 0, PNO}}}, // COORD_B8
    { 3, {{ 9, P31}, {10, P38}, {14, P39}, { 0, PNO}, { 0, PNO}}}, // COORD_A8
    { 4, {{ 2, P30}, { 4, P30}, {11, P32}, {15, P35}, { 0, PNO}}}, // COORD_H7
    { 5, {{ 2, P31}, { 3, P31}, {10, P30}, {11, P30}, {15, P34}}}, // COORD_G7
    { 3, {{ 2, P32}, { 6, P31}, {15, P33}, { 0, PNO}, { 0, PNO}}}, // COORD_F7
    { 1, {{ 2, P33}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_E7
    { 1, {{ 2, P34}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_D7
    { 3, {{ 2, P35}, { 7, P31}, {14, P33}, { 0, PNO}, { 0, PNO}}}, // COORD_C7
    { 5, {{ 1, P31}, { 2, P36}, { 9, P30}, {10, P39}, {14, P34}}}, // COORD_B7
    { 4, {{ 2, P37}, { 5, P30}, { 9, P32}, {14, P35}, { 0, PNO}}}, // COORD_A7
    { 2, {{11, P33}, {15, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_H6
    { 3, {{ 3, P32}, { 4, P31}, {15, P31}, { 0, PNO}, { 0, PNO}}}, // COORD_G6
    { 0, {{ 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_F6
    { 1, {{ 6, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_E6
    { 1, {{ 7, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_D6
    { 0, {{ 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_C6
    { 3, {{ 1, P32}, { 5, P31}, {14, P31}, { 0, PNO}, { 0, PNO}}}, // COORD_B6
    { 2, {{ 9, P33}, {14, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_A6
    { 2, {{11, P34}, {15, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_H5
    { 1, {{ 3, P33}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_G5
    { 1, {{ 4, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_F5
    { 1, {{ 7, P33}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_E5
    { 1, {{ 6, P33}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_D5
    { 1, {{ 5, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_C5
    { 1, {{ 1, P33}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_B5
    { 2, {{ 9, P34}, {14, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_A5
    { 2, {{11, P35}, {13, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_H4
    { 1, {{ 3, P34}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_G4
    { 1, {{ 7, P34}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_F4
    { 1, {{ 4, P33}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_E4
    { 1, {{ 5, P33}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_D4
    { 1, {{ 6, P34}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_C4
    { 1, {{ 1, P34}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_B4
    { 2, {{ 9, P35}, {12, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_A4
    { 2, {{11, P36}, {13, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_H3
    { 3, {{ 3, P35}, { 7, P35}, {13, P31}, { 0, PNO}, { 0, PNO}}}, // COORD_G3
    { 0, {{ 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_F3
    { 1, {{ 5, P34}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_E3
    { 1, {{ 4, P34}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_D3
    { 0, {{ 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_C3
    { 3, {{ 1, P35}, { 6, P35}, {12, P31}, { 0, PNO}, { 0, PNO}}}, // COORD_B3
    { 2, {{ 9, P36}, {12, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_A3
    { 4, {{ 0, P30}, { 7, P36}, {11, P37}, {13, P35}, { 0, PNO}}}, // COORD_H2
    { 5, {{ 0, P31}, { 3, P36}, { 8, P30}, {11, P39}, {13, P34}}}, // COORD_G2
    { 3, {{ 0, P32}, { 5, P35}, {13, P33}, { 0, PNO}, { 0, PNO}}}, // COORD_F2
    { 1, {{ 0, P33}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_E2
    { 1, {{ 0, P34}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_D2
    { 3, {{ 0, P35}, { 4, P35}, {12, P33}, { 0, PNO}, { 0, PNO}}}, // COORD_C2
    { 5, {{ 0, P36}, { 1, P36}, { 8, P39}, { 9, P39}, {12, P34}}}, // COORD_B2
    { 4, {{ 0, P37}, { 6, P36}, { 9, P37}, {12, P35}, { 0, PNO}}}, // COORD_A2
    { 3, {{ 8, P31}, {11, P38}, {13, P39}, { 0, PNO}, { 0, PNO}}}, // COORD_H1
    { 4, {{ 3, P37}, { 5, P36}, { 8, P32}, {13, P38}, { 0, PNO}}}, // COORD_G1
    { 2, {{ 8, P33}, {13, P37}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_F1
    { 2, {{ 8, P34}, {13, P36}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_E1
    { 2, {{ 8, P35}, {12, P36}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_D1
    { 2, {{ 8, P36}, {12, P37}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_C1
    { 4, {{ 1, P37}, { 4, P36}, { 8, P37}, {12, P38}, { 0, PNO}}}, // COORD_B1
    { 3, {{ 8, P38}, { 9, P38}, {12, P39}, { 0, PNO}, { 0, PNO}}}  // COORD_A1
};

constexpr int adj_eval_sizes[ADJ_N_EVAL] = {
    10, 1, 1, ADJ_MO_MAX_MOBILITY, ADJ_MO_MAX_POT_MOBILITY, 
    P38, P37, P310, P310, 
    P38, P37, P310, P310
};

constexpr int adj_feature_to_eval_idx[ADJ_N_FEATURES] = {
    0, 1, 2, 3, 4, 
    5, 5, 5, 5, 
    6, 6, 6, 6, 
    7, 7, 7, 7, 
    8, 8, 8, 8, 
    9, 9, 9, 9, 
    10, 10, 10, 10, 
    11, 11, 11, 11, 
    12, 12, 12, 12
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

constexpr uint64_t parity_table[16] = {
    0x0000000000000000ULL, 0x000000000F0F0F0FULL, 0x00000000F0F0F0F0ULL, 0x00000000FFFFFFFFULL,
    0x0F0F0F0F00000000ULL, 0x0F0F0F0F0F0F0F0FULL, 0x0F0F0F0FF0F0F0F0ULL, 0x0F0F0F0FFFFFFFFFULL,
    0xF0F0F0F000000000ULL, 0xF0F0F0F00F0F0F0FULL, 0xF0F0F0F0F0F0F0F0ULL, 0xF0F0F0F0FFFFFFFFULL,
    0xFFFFFFFF00000000ULL, 0xFFFFFFFF0F0F0F0FULL, 0xFFFFFFFFF0F0F0F0ULL, 0xFFFFFFFFFFFFFFFFULL
};

inline int get_potential_mobility(uint64_t opponent, uint64_t empties){
    const u64_4 shift(1, HW, HW_M1, HW_P1);
    const u64_4 mask(0x7E7E7E7E7E7E7E7EULL, 0x00FFFFFFFFFFFF00ULL, 0x007E7E7E7E7E7E00ULL, 0x007E7E7E7E7E7E00ULL);
    u64_4 op(opponent);
    op = op & mask;
    return pop_count_ull(empties & all_or((op << shift) | (op >> shift)));
}

inline bool is_free_odd_empties(Board *board, uint_fast8_t pos){
    uint64_t masked_empties = ~(board->player | board->opponent) & parity_table[cell_div4[pos]];
    return get_potential_mobility(board->player, masked_empties) > 0;
}

inline int adj_pick_pattern(const uint_fast8_t b_arr[], int pattern_idx){
    int res = 0;
    for (int i = 0; i < adj_feature_to_coord[pattern_idx].n_cells; ++i){
        res *= 3;
        res += b_arr[adj_feature_to_coord[pattern_idx].cells[i]];
    }
    return res;
}


void adj_calc_features(Board *board, uint_fast8_t cell, uint16_t res[ADJ_N_FEATURES]){
    uint_fast8_t b_arr[HW2];
    uint64_t empty = ~(board->player | board->opponent);
    uint_fast8_t parity = 1 & pop_count_ull(empty & 0x000000000F0F0F0FULL);
    parity |= (1 & pop_count_ull(empty & 0x00000000F0F0F0F0ULL)) << 1;
    parity |= (1 & pop_count_ull(empty & 0x0F0F0F0F00000000ULL)) << 2;
    parity |= (1 & pop_count_ull(empty & 0xF0F0F0F000000000ULL)) << 3;
    Flip flip;
    int idx = 0;
    res[idx++] = cell_type[cell];
    res[idx++] = (parity & cell_div4[cell]) > 0;
    res[idx++] = is_free_odd_empties(board, cell);
    calc_flip(&flip, board, cell);
    board->move_board(&flip);
        res[idx++] = pop_count_ull(board->get_legal());
        res[idx++] = get_potential_mobility(board->player, ~(board->player | board->opponent));
        board->translate_to_arr_player(b_arr);
        for (int i = 0; i < ADJ_N_SYMMETRY_PATTERNS; ++i)
            res[idx++] = adj_pick_pattern(b_arr, i);
    board->undo_board(&flip);
    board->translate_to_arr_player(b_arr);
    for (int i = 0; i < ADJ_N_SYMMETRY_PATTERNS; ++i)
        res[idx++] = adj_pick_pattern(b_arr, i);
}

int adj_pick_digit3(int num, int d, int n_digit){
    num /= adj_pow3[n_digit - 1 - d];
    return num % 3;
}

uint16_t adj_calc_rev_idx(int feature, int idx){
    uint16_t res = 0;
    if (feature < ADJ_N_NORMAL_FEATURES){
        res = idx;
    } else{
        int f = (feature - ADJ_N_NORMAL_FEATURES) % 4;
        for (int i = 0; i < adj_pattern_n_cells[f]; ++i){
            res += adj_pick_digit3(idx, adj_rev_patterns[f][i], adj_pattern_n_cells[f]) * adj_pow3[adj_pattern_n_cells[f] - 1 - i];
        }
    }
    return res;
}

void evaluation_definition_init(){
    mobility_init();
}