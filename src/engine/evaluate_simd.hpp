/*
    Egaroucid Project

    @file evaluate_generic.hpp
        Evaluation function with AVX2
    @date 2021-2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once
#include <iostream>
#include <fstream>
#ifdef _MSC_VER
    #include <intrin.h>
#else
    #include <x86intrin.h>
#endif
#include <cstring>
#include "setting.hpp"
#include "common.hpp"
#include "board.hpp"
#include "search.hpp"
#include "util.hpp"
#include "evaluate_common.hpp"

/*
    @brief evaluation pattern definition for SIMD
*/
constexpr int CEIL_N_PATTERN_FEATURES = 64;     // ceil2^n(N_PATTERN_FEATURES)
constexpr int N_PATTERN_PARAMS_RAW = 612360;    // sum of pattern parameters for 1 phase
constexpr int N_PATTERN_PARAMS = N_PATTERN_PARAMS_RAW + 1; // +1 for byte bound
constexpr int PATTERN4_START_IDX = 52488;       // special case
constexpr int PATTERN6_START_IDX = 78732;       // special case
constexpr int SIMD_EVAL_MAX_VALUE = 4092;       // evaluate range [-4092, 4092]
constexpr int N_EVAL_VECTORS_SIMPLE = 2;
constexpr int N_EVAL_VECTORS_COMP = 2;
constexpr int N_SIMD_EVAL_FEATURE_CELLS = 16;
constexpr int N_SIMD_EVAL_FEATURE_GROUP = 4;
constexpr int MAX_N_CELLS_GROUP[4] = {9, 10, 10, 10}; // number of cells included in the group


/*
    @brief evaluation pattern definition for SIMD move ordering end
*/
constexpr int N_PATTERN_PARAMS_MO_END = 236196 + 1; // +1 for byte bound
constexpr int SIMD_EVAL_MAX_VALUE_MO_END = 16380;
constexpr int SHIFT_EVAL_MO_END = 139968; // pattern_starts[8] - 1

constexpr Feature_to_coord feature_to_coord[CEIL_N_PATTERN_FEATURES] = {
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

constexpr Coord_to_feature coord_to_feature[HW2] = {
    {15, {{12, P30}, {14, P38}, {23, P38}, {28, P32}, {29, P39}, {34, P31}, {35, P31}, {39, P39}, {42, P34}, {43, P34}, {47, P39}, {50, P31}, {51, P31}, {55, P39}, {59, P39}}}, // COORD_H8
    {11, {{ 3, P30}, { 6, P38}, {14, P37}, {23, P37}, {29, P30}, {34, P32}, {39, P38}, {47, P35}, {50, P32}, {55, P38}, {59, P38}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_G8
    { 9, {{ 6, P37}, {11, P30}, {23, P36}, {34, P33}, {39, P37}, {42, P35}, {50, P33}, {55, P37}, {60, P34}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_F8
    { 9, {{19, P30}, {26, P35}, {34, P34}, {39, P36}, {42, P36}, {50, P34}, {54, P35}, {55, P36}, {60, P33}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_E8
    { 9, {{17, P30}, {25, P31}, {34, P35}, {38, P36}, {42, P37}, {50, P35}, {54, P36}, {55, P35}, {60, P36}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_D8
    { 9, {{ 5, P31}, { 9, P30}, {22, P36}, {34, P36}, {38, P37}, {42, P38}, {50, P36}, {54, P37}, {60, P35}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_C8
    {11, {{ 1, P30}, { 5, P30}, {13, P31}, {22, P37}, {31, P30}, {34, P37}, {38, P38}, {46, P35}, {50, P37}, {54, P38}, {58, P38}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_B8
    {15, {{13, P30}, {15, P38}, {22, P38}, {30, P32}, {31, P39}, {33, P31}, {34, P38}, {38, P39}, {41, P34}, {42, P39}, {46, P39}, {49, P31}, {50, P38}, {54, P39}, {58, P39}}}, // COORD_A8
    {11, {{ 2, P30}, { 4, P30}, {12, P31}, {23, P35}, {29, P31}, {35, P32}, {39, P35}, {47, P32}, {51, P32}, {55, P34}, {59, P37}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_H7
    {14, {{ 2, P31}, { 3, P31}, { 5, P34}, {23, P34}, {24, P30}, {26, P36}, {28, P33}, {29, P38}, {34, P30}, {35, P30}, {39, P34}, {47, P38}, {55, P33}, {59, P36}, { 0, PNO}}}, // COORD_G7
    { 9, {{ 2, P32}, {11, P31}, {14, P36}, {23, P33}, {39, P33}, {42, P30}, {47, P34}, {50, P30}, {59, P35}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_F7
    { 7, {{ 2, P33}, { 6, P36}, {19, P31}, {25, P32}, {42, P31}, {59, P34}, {60, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_E7
    { 7, {{ 2, P34}, { 5, P32}, {17, P31}, {26, P34}, {42, P32}, {58, P34}, {60, P37}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_D7
    { 9, {{ 2, P35}, { 9, P31}, {13, P32}, {22, P33}, {38, P33}, {42, P33}, {46, P34}, {50, P39}, {58, P35}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_C7
    {14, {{ 1, P31}, { 2, P36}, { 6, P34}, {22, P34}, {25, P30}, {27, P36}, {30, P33}, {31, P38}, {33, P30}, {34, P39}, {38, P34}, {46, P38}, {54, P33}, {58, P36}, { 0, PNO}}}, // COORD_B7
    {11, {{ 2, P37}, { 7, P38}, {15, P37}, {22, P35}, {31, P31}, {33, P32}, {38, P35}, {46, P32}, {49, P32}, {54, P34}, {58, P37}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_A7
    { 9, {{ 4, P31}, {10, P30}, {23, P32}, {35, P33}, {39, P32}, {43, P35}, {51, P33}, {55, P32}, {63, P35}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_H6
    { 9, {{ 3, P32}, {10, P31}, {12, P32}, {23, P31}, {39, P31}, {43, P30}, {47, P31}, {51, P30}, {59, P33}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_G6
    {10, {{10, P32}, {11, P32}, {23, P30}, {25, P33}, {28, P34}, {29, P37}, {47, P37}, {59, P32}, {60, P30}, {63, P39}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_F6
    { 6, {{ 5, P33}, {10, P33}, {14, P35}, {19, P32}, {47, P33}, {60, P31}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_E6
    { 6, {{ 6, P35}, {10, P34}, {13, P33}, {17, P32}, {46, P33}, {60, P38}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_D6
    {10, {{ 9, P32}, {10, P35}, {22, P30}, {26, P33}, {30, P34}, {31, P37}, {46, P37}, {58, P32}, {60, P39}, {61, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_C6
    { 9, {{ 1, P32}, {10, P36}, {15, P36}, {22, P31}, {38, P31}, {41, P30}, {46, P31}, {49, P30}, {58, P33}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_B6
    { 9, {{ 7, P37}, {10, P37}, {22, P32}, {33, P33}, {38, P32}, {41, P35}, {49, P33}, {54, P32}, {61, P34}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_A6
    { 9, {{18, P30}, {24, P31}, {35, P34}, {39, P30}, {43, P36}, {51, P34}, {53, P30}, {55, P31}, {63, P36}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_H5
    { 7, {{ 3, P33}, { 4, P32}, {18, P31}, {25, P34}, {43, P31}, {59, P31}, {63, P37}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_G5
    { 6, {{ 5, P35}, {11, P33}, {12, P33}, {18, P32}, {47, P30}, {63, P38}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_F5
    { 7, {{13, P34}, {18, P33}, {19, P33}, {28, P35}, {29, P36}, {47, P36}, {59, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_E5
    { 7, {{14, P34}, {17, P33}, {18, P34}, {30, P35}, {31, P36}, {46, P36}, {58, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_D5
    { 6, {{ 6, P33}, { 9, P33}, {15, P35}, {18, P35}, {46, P30}, {61, P31}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_C5
    { 7, {{ 1, P33}, { 7, P36}, {18, P36}, {26, P32}, {41, P31}, {58, P31}, {61, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_B5
    { 9, {{18, P37}, {27, P35}, {33, P34}, {38, P30}, {41, P36}, {49, P34}, {52, P30}, {54, P31}, {61, P33}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_A5
    { 9, {{16, P30}, {25, P35}, {35, P35}, {37, P30}, {43, P37}, {51, P35}, {53, P31}, {55, P30}, {63, P33}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_H4
    { 7, {{ 3, P34}, { 5, P36}, {16, P31}, {24, P32}, {43, P32}, {57, P31}, {63, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_G4
    { 6, {{ 4, P33}, {11, P34}, {13, P35}, {16, P32}, {45, P30}, {63, P31}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_F4
    { 7, {{12, P34}, {16, P33}, {19, P34}, {30, P36}, {31, P35}, {45, P36}, {57, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_E4
    { 7, {{15, P34}, {16, P34}, {17, P34}, {28, P36}, {29, P35}, {44, P36}, {56, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_D4
    { 6, {{ 7, P35}, { 9, P34}, {14, P33}, {16, P35}, {44, P30}, {61, P38}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_C4
    { 7, {{ 1, P34}, { 6, P32}, {16, P36}, {27, P34}, {41, P32}, {56, P31}, {61, P37}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_B4
    { 9, {{16, P37}, {26, P31}, {33, P35}, {36, P30}, {41, P37}, {49, P35}, {52, P31}, {54, P30}, {61, P36}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_A4
    { 9, {{ 5, P37}, { 8, P30}, {21, P32}, {35, P36}, {37, P32}, {43, P38}, {51, P36}, {53, P32}, {63, P34}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_H3
    { 9, {{ 3, P35}, { 8, P31}, {13, P36}, {21, P31}, {37, P31}, {43, P33}, {45, P31}, {51, P39}, {57, P33}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_G3
    {10, {{ 8, P32}, {11, P35}, {21, P30}, {24, P33}, {30, P37}, {31, P34}, {45, P37}, {57, P32}, {62, P39}, {63, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_F3
    { 6, {{ 4, P35}, { 8, P33}, {15, P33}, {19, P35}, {45, P33}, {62, P38}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_E3
    { 6, {{ 7, P33}, { 8, P34}, {12, P35}, {17, P35}, {44, P33}, {62, P31}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_D3
    {10, {{ 8, P35}, { 9, P35}, {20, P30}, {27, P33}, {28, P37}, {29, P34}, {44, P37}, {56, P32}, {61, P39}, {62, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_C3
    { 9, {{ 1, P35}, { 8, P36}, {14, P32}, {20, P31}, {36, P31}, {41, P33}, {44, P31}, {49, P39}, {56, P33}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_B3
    { 9, {{ 6, P31}, { 8, P37}, {20, P32}, {33, P36}, {36, P32}, {41, P38}, {49, P36}, {52, P32}, {61, P35}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_A3
    {11, {{ 0, P30}, { 5, P38}, {13, P37}, {21, P35}, {30, P31}, {35, P37}, {37, P35}, {45, P32}, {51, P37}, {53, P34}, {57, P37}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_H2
    {14, {{ 0, P31}, { 3, P36}, { 4, P34}, {21, P34}, {25, P36}, {27, P30}, {30, P38}, {31, P33}, {32, P30}, {35, P39}, {37, P34}, {45, P38}, {53, P33}, {57, P36}, { 0, PNO}}}, // COORD_G2
    { 9, {{ 0, P32}, {11, P36}, {15, P32}, {21, P33}, {37, P33}, {40, P30}, {45, P34}, {48, P30}, {57, P35}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_F2
    { 7, {{ 0, P33}, { 7, P32}, {19, P36}, {24, P34}, {40, P31}, {57, P34}, {62, P37}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_E2
    { 7, {{ 0, P34}, { 4, P36}, {17, P36}, {27, P32}, {40, P32}, {56, P34}, {62, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_D2
    { 9, {{ 0, P35}, { 9, P36}, {12, P36}, {20, P33}, {36, P33}, {40, P33}, {44, P34}, {48, P39}, {56, P35}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_C2
    {14, {{ 0, P36}, { 1, P36}, { 7, P34}, {20, P34}, {24, P36}, {26, P30}, {28, P38}, {29, P33}, {32, P39}, {33, P39}, {36, P34}, {44, P38}, {52, P33}, {56, P36}, { 0, PNO}}}, // COORD_B2
    {11, {{ 0, P37}, { 6, P30}, {14, P31}, {20, P35}, {28, P31}, {33, P37}, {36, P35}, {44, P32}, {49, P37}, {52, P34}, {56, P37}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_A2
    {15, {{13, P38}, {15, P30}, {21, P38}, {30, P39}, {31, P32}, {32, P31}, {35, P38}, {37, P39}, {40, P34}, {43, P39}, {45, P39}, {48, P31}, {51, P38}, {53, P39}, {57, P39}}}, // COORD_H1
    {11, {{ 3, P37}, { 7, P30}, {15, P31}, {21, P37}, {30, P30}, {32, P32}, {37, P38}, {45, P35}, {48, P32}, {53, P38}, {57, P38}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_G1
    { 9, {{ 7, P31}, {11, P37}, {21, P36}, {32, P33}, {37, P37}, {40, P35}, {48, P33}, {53, P37}, {62, P35}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_F1
    { 9, {{19, P37}, {27, P31}, {32, P34}, {37, P36}, {40, P36}, {48, P34}, {52, P35}, {53, P36}, {62, P36}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_E1
    { 9, {{17, P37}, {24, P35}, {32, P35}, {36, P36}, {40, P37}, {48, P35}, {52, P36}, {53, P35}, {62, P33}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_D1
    { 9, {{ 4, P37}, { 9, P37}, {20, P36}, {32, P36}, {36, P37}, {40, P38}, {48, P36}, {52, P37}, {62, P34}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_C1
    {11, {{ 1, P37}, { 4, P38}, {12, P37}, {20, P37}, {28, P30}, {32, P37}, {36, P38}, {44, P35}, {48, P37}, {52, P38}, {56, P38}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_B1
    {15, {{12, P38}, {14, P30}, {20, P38}, {28, P39}, {29, P32}, {32, P38}, {33, P38}, {36, P39}, {40, P39}, {41, P39}, {44, P39}, {48, P38}, {49, P38}, {52, P39}, {56, P39}}}  // COORD_A1
};

constexpr int pattern_starts[N_PATTERNS] = {
    1, 6562, 26245, 32806,              // features[0] h2, d6+2C+X, h3, d7+2corner (from 0)
    1, 6562,                            // features[1] h4, corner9 (from PATTERN4_START_IDX)
    1, 2188,                            // features[1] d5+2X, d8+2C (from PATTERN6_START_IDX)
    139969, 199018, 258067, 317116,     // features[2] (from 0)
    376165, 435214, 494263, 553312      // features[3] (from 0)
};

/*
    @brief constants used for evaluation function with SIMD
*/
__m256i eval_lower_mask;
__m256i feature_to_coord_simd_mul[N_EVAL_VECTORS][MAX_PATTERN_CELLS - 1];
__m256i feature_to_coord_simd_cell[N_EVAL_VECTORS][MAX_PATTERN_CELLS][2];
__m256i coord_to_feature_simd[HW2][N_EVAL_VECTORS];
__m256i eval_move_unflipped_16bit[N_16BIT][N_SIMD_EVAL_FEATURE_GROUP][N_EVAL_VECTORS];
__m256i eval_simd_offsets_simple[N_EVAL_VECTORS_SIMPLE]; // 16bit * 16 * N
__m256i eval_simd_offsets_comp[N_EVAL_VECTORS_COMP * 2]; // 32bit * 8 * N


/*
    @brief evaluation parameters
*/
// normal
int16_t pattern_arr[N_PHASES][N_PATTERN_PARAMS];
int16_t eval_num_arr[N_PHASES][MAX_STONE_NUM];
// move ordering evaluation
int16_t pattern_move_ordering_end_arr[N_PATTERN_PARAMS_MO_END];

inline bool load_eval_file(const char* file, bool show_log) {
    if (show_log) {
        std::cerr << "evaluation file " << file << std::endl;
    }
    bool failed = false;
    std::vector<int16_t> unzipped_params = load_unzip_egev2(file, show_log, &failed);
    if (failed) {
        return false;
    }
    size_t param_idx = 0;
    for (int phase_idx = 0; phase_idx < N_PHASES; ++phase_idx) {
        pattern_arr[phase_idx][0] = 0; // memory bound
        std::memcpy(pattern_arr[phase_idx] + 1, &unzipped_params[param_idx], sizeof(short) * N_PATTERN_PARAMS_RAW);
        param_idx += N_PATTERN_PARAMS_RAW;
        std::memcpy(eval_num_arr[phase_idx], &unzipped_params[param_idx], sizeof(short) * MAX_STONE_NUM);
        param_idx += MAX_STONE_NUM;
    }
    // check max value
    for (int phase_idx = 0; phase_idx < N_PHASES; ++phase_idx) {
        for (int i = 1; i < N_PATTERN_PARAMS; ++i) {
            if (pattern_arr[phase_idx][i] < -SIMD_EVAL_MAX_VALUE) {
                std::cerr << "[ERROR] evaluation value too low. you can ignore this error. phase " << phase_idx << " index " << i << " found " << pattern_arr[phase_idx][i] << std::endl;
                pattern_arr[phase_idx][i] = -SIMD_EVAL_MAX_VALUE;
            }
            if (pattern_arr[phase_idx][i] > SIMD_EVAL_MAX_VALUE) {
                std::cerr << "[ERROR] evaluation value too high. you can ignore this error. phase " << phase_idx << " index " << i << " found " << pattern_arr[phase_idx][i] << std::endl;
                pattern_arr[phase_idx][i] = SIMD_EVAL_MAX_VALUE;
            }
            pattern_arr[phase_idx][i] += SIMD_EVAL_MAX_VALUE;
        }
    }
    return true;
}

inline bool load_eval_move_ordering_end_file(const char* file, bool show_log) {
    if (show_log) {
        std::cerr << "evaluation for move ordering end file " << file << std::endl;
    }
    FILE* fp;
    if (!file_open(&fp, file, "rb")) {
        std::cerr << "[ERROR] [FATAL] can't open eval " << file << std::endl;
        return false;
    }
    pattern_move_ordering_end_arr[0] = 0; // memory bound
    if (fread(pattern_move_ordering_end_arr + 1, 2, N_PATTERN_PARAMS_MO_END - 1, fp) < N_PATTERN_PARAMS_MO_END - 1) {
        std::cerr << "[ERROR] [FATAL] evaluation file for move ordering end broken" << std::endl;
        fclose(fp);
        return false;
    }
    // check max value
    for (int i = 1; i < N_PATTERN_PARAMS_MO_END; ++i) {
        if (pattern_move_ordering_end_arr[i] < -SIMD_EVAL_MAX_VALUE_MO_END) {
            std::cerr << "[ERROR] evaluation value too low. you can ignore this error. index " << i << " found " << pattern_move_ordering_end_arr[i] << std::endl;
            pattern_move_ordering_end_arr[i] = -SIMD_EVAL_MAX_VALUE_MO_END;
        }
        if (pattern_move_ordering_end_arr[i] > SIMD_EVAL_MAX_VALUE_MO_END) {
            std::cerr << "[ERROR] evaluation value too high. you can ignore this error. index " << i << " found " << pattern_move_ordering_end_arr[i] << std::endl;
            pattern_move_ordering_end_arr[i] = SIMD_EVAL_MAX_VALUE_MO_END;
        }
        pattern_move_ordering_end_arr[i] += SIMD_EVAL_MAX_VALUE_MO_END;
    }
    return true;
}

inline void pre_calculate_eval_constant() {
    { // calc_eval_features initialization
        int16_t f2c[16];
        for (int i = 0; i < N_EVAL_VECTORS; ++i) {
            for (int j = 0; j < MAX_PATTERN_CELLS - 1; ++j) {
                for (int k = 0; k < 16; ++k) {
                    f2c[k] = (j < feature_to_coord[i * 16 + k].n_cells - 1) ? 3 : 1;
                }
                feature_to_coord_simd_mul[i][j] = _mm256_set_epi16(
                    f2c[0], f2c[1], f2c[2], f2c[3], 
                    f2c[4], f2c[5], f2c[6], f2c[7], 
                    f2c[8], f2c[9], f2c[10], f2c[11], 
                    f2c[12], f2c[13], f2c[14], f2c[15]
                );
            }
        }
        int32_t f2c32[8];
        for (int i = 0; i < N_EVAL_VECTORS; ++i) {
            for (int j = 0; j < MAX_PATTERN_CELLS; ++j) {
                for (int k = 0; k < 8; ++k) {
                    f2c32[k] = feature_to_coord[i * 16 + k * 2 + 1].cells[j];
                }
                feature_to_coord_simd_cell[i][j][0] = _mm256_set_epi32(
                    f2c32[0], f2c32[1], f2c32[2], f2c32[3], 
                    f2c32[4], f2c32[5], f2c32[6], f2c32[7]
                );
                for (int k = 0; k < 8; ++k) {
                    f2c32[k] = feature_to_coord[i * 16 + k * 2].cells[j];
                }
                feature_to_coord_simd_cell[i][j][1] = _mm256_set_epi32(
                    f2c32[0], f2c32[1], f2c32[2], f2c32[3], 
                    f2c32[4], f2c32[5], f2c32[6], f2c32[7]
                );
            }
        }
        eval_simd_offsets_simple[0] = _mm256_set_epi16(
            (int16_t)pattern_starts[0], (int16_t)pattern_starts[0], (int16_t)pattern_starts[0], (int16_t)pattern_starts[0], 
            (int16_t)pattern_starts[1], (int16_t)pattern_starts[1], (int16_t)pattern_starts[1], (int16_t)pattern_starts[1], 
            (int16_t)pattern_starts[2], (int16_t)pattern_starts[2], (int16_t)pattern_starts[2], (int16_t)pattern_starts[2], 
            (int16_t)pattern_starts[3], (int16_t)pattern_starts[3], (int16_t)pattern_starts[3], (int16_t)pattern_starts[3]
        );
        eval_simd_offsets_simple[1] = _mm256_set_epi16(
            (int16_t)pattern_starts[4], (int16_t)pattern_starts[4], (int16_t)pattern_starts[4], (int16_t)pattern_starts[4], 
            (int16_t)pattern_starts[5], (int16_t)pattern_starts[5], (int16_t)pattern_starts[5], (int16_t)pattern_starts[5], 
            (int16_t)pattern_starts[6], (int16_t)pattern_starts[6], (int16_t)pattern_starts[6], (int16_t)pattern_starts[6], 
            (int16_t)pattern_starts[7], (int16_t)pattern_starts[7], (int16_t)pattern_starts[7], (int16_t)pattern_starts[7]
        );
    }
    { // eval_move initialization
        uint16_t c2f[CEIL_N_PATTERN_FEATURES];
        for (int cell = 0; cell < HW2; ++cell) { // 0 for h8, 63 for a1
            for (int i = 0; i < CEIL_N_PATTERN_FEATURES; ++i) {
                c2f[i] = 0;
            }
            for (int i = 0; i < coord_to_feature[cell].n_features; ++i) {
                c2f[coord_to_feature[cell].features[i].feature] = coord_to_feature[cell].features[i].x;
            }
            for (int i = 0; i < N_EVAL_VECTORS; ++i) {
                int idx = i * 16;
                coord_to_feature_simd[cell][i] = _mm256_set_epi16(
                    c2f[idx], c2f[idx + 1], c2f[idx + 2], c2f[idx + 3], 
                    c2f[idx + 4], c2f[idx + 5], c2f[idx + 6], c2f[idx + 7], 
                    c2f[idx + 8], c2f[idx + 9], c2f[idx + 10], c2f[idx + 11], 
                    c2f[idx + 12], c2f[idx + 13], c2f[idx + 14], c2f[idx + 15]
                );
            }
        }
        for (int bits = 0; bits < N_16BIT; ++bits) { // 1: unflipped discs, 0: others
            for (int group = 0; group < N_SIMD_EVAL_FEATURE_GROUP; ++group) { // a1-h2, a3-h4, ..., a7-h8
                for (int i = 0; i < CEIL_N_PATTERN_FEATURES; ++i) {
                    c2f[i] = 0;
                }
                for (int cell = 0; cell < N_SIMD_EVAL_FEATURE_CELLS; ++cell) {
                    if (1 & (bits >> cell)) {
                        int global_cell = group * N_SIMD_EVAL_FEATURE_CELLS + cell;
                        for (int i = 0; i < coord_to_feature[global_cell].n_features; ++i) {
                            c2f[coord_to_feature[global_cell].features[i].feature] += coord_to_feature[global_cell].features[i].x;
                        }
                    }
                }
                for (int simd_feature_idx = 0; simd_feature_idx < N_EVAL_VECTORS; ++simd_feature_idx) {
                    int idx = simd_feature_idx * 16;
                    eval_move_unflipped_16bit[bits][group][simd_feature_idx] = _mm256_set_epi16(
                        c2f[idx], c2f[idx + 1], c2f[idx + 2], c2f[idx + 3], 
                        c2f[idx + 4], c2f[idx + 5], c2f[idx + 6], c2f[idx + 7], 
                        c2f[idx + 8], c2f[idx + 9], c2f[idx + 10], c2f[idx + 11], 
                        c2f[idx + 12], c2f[idx + 13], c2f[idx + 14], c2f[idx + 15]
                    );
                }
            }
        }
        for (int i = 0; i < N_EVAL_VECTORS_COMP; ++i) {
            int i4 = i * 4;
            eval_simd_offsets_comp[i * 2] = _mm256_set_epi32(
                pattern_starts[10 + i4], pattern_starts[10 + i4], pattern_starts[10 + i4], pattern_starts[10 + i4], 
                pattern_starts[11 + i4], pattern_starts[11 + i4], pattern_starts[11 + i4], pattern_starts[11 + i4]
            );
            eval_simd_offsets_comp[i * 2 + 1] = _mm256_set_epi32(
                pattern_starts[8 + i4], pattern_starts[8 + i4], pattern_starts[8 + i4], pattern_starts[8 + i4], 
                pattern_starts[9 + i4], pattern_starts[9 + i4], pattern_starts[9 + i4], pattern_starts[9 + i4]
            );
        }
        eval_lower_mask = _mm256_set1_epi32(0x0000FFFF);
    }
}

/*
    @brief initialize the evaluation function

    @param file                 evaluation file name
    @param show_log             debug information?
    @return evaluation function conpletely initialized?
*/
inline bool evaluate_init(const char* file, const char* mo_end_nws_file, bool show_log) {
    bool eval_loaded = load_eval_file(file, show_log);
    if (!eval_loaded) {
        std::cerr << "[ERROR] [FATAL] evaluation file not loaded" << std::endl;
        return false;
    }
    bool eval_move_ordering_end_nws_loaded = load_eval_move_ordering_end_file(mo_end_nws_file, show_log);
    if (!eval_move_ordering_end_nws_loaded) {
        std::cerr << "[ERROR] [FATAL] evaluation file for move ordering end not loaded" << std::endl;
        return false;
    }
    pre_calculate_eval_constant();
    if (show_log) {
        std::cerr << "evaluation function initialized" << std::endl;
    }
    return true;
}

/*
    @brief Wrapper of evaluation initializing

    @param file                 evaluation file name
    @return evaluation function conpletely initialized?
*/
bool evaluate_init(const std::string file, std::string mo_end_nws_file, bool show_log) {
    return evaluate_init(file.c_str(), mo_end_nws_file.c_str(), show_log);
}

/*
    @brief Wrapper of evaluation initializing

    @return evaluation function conpletely initialized?
*/
bool evaluate_init(bool show_log) {
    return evaluate_init(EXE_DIRECTORY_PATH + "resources/eval.egev2", EXE_DIRECTORY_PATH + "resources/eval_move_ordering_end.egev", show_log);
}

/*
    @brief pattern evaluation

    @param phase_idx            evaluation phase
    @param search               search information
    @return pattern evaluation value
*/
inline __m256i calc_idx8_comp(const __m128i feature, const int i) {
    return _mm256_add_epi32(_mm256_cvtepu16_epi32(feature), eval_simd_offsets_comp[i]);
}

inline __m256i gather_eval(const int *start_addr, const __m256i idx8) {
    return _mm256_i32gather_epi32(start_addr, idx8, 2); // stride is 2 byte, because 16 bit array used, HACK: if (SIMD_EVAL_MAX_VALUE * 2) * (N_ADD=8) < 2 ^ 16, AND is unnecessary
    // return _mm256_and_si256(_mm256_i32gather_epi32(start_addr, idx8, 2), eval_lower_mask);
}

inline int calc_pattern(const int phase_idx, Eval_features *features) {
    const int *start_addr0 = (int*)pattern_arr[phase_idx];
    const int *start_addr4 = (int*)&pattern_arr[phase_idx][PATTERN4_START_IDX];
    const int *start_addr6 = (int*)&pattern_arr[phase_idx][PATTERN6_START_IDX];
    __m256i res256 =                  gather_eval(start_addr0, _mm256_cvtepu16_epi32(features->f128[0]));   // hv3 d7+2Corner
    res256 = _mm256_add_epi32(res256, gather_eval(start_addr0, _mm256_cvtepu16_epi32(features->f128[1])));  // hv2 d6+2C+X
    res256 = _mm256_add_epi32(res256, gather_eval(start_addr6, _mm256_cvtepu16_epi32(features->f128[2])));  // d5+2X d8+wC
    res256 = _mm256_add_epi32(res256, gather_eval(start_addr4, _mm256_cvtepu16_epi32(features->f128[3])));  // hv4 corner9
    res256 = _mm256_add_epi32(res256, gather_eval(start_addr0, calc_idx8_comp(features->f128[4], 0)));      // corner+block cross
    res256 = _mm256_add_epi32(res256, gather_eval(start_addr0, calc_idx8_comp(features->f128[5], 1)));      // edge+2X triangle
    res256 = _mm256_add_epi32(res256, gather_eval(start_addr0, calc_idx8_comp(features->f128[6], 2)));      // fish kite
    res256 = _mm256_add_epi32(res256, gather_eval(start_addr0, calc_idx8_comp(features->f128[7], 3)));      // edge+2Y narrow_triangle
    res256 = _mm256_and_si256(res256, eval_lower_mask);
    __m128i res128 = _mm_add_epi32(_mm256_castsi256_si128(res256), _mm256_extracti128_si256(res256, 1));
    res128 = _mm_hadd_epi32(res128, res128);
    return _mm_cvtsi128_si32(res128) + _mm_extract_epi32(res128, 1) - SIMD_EVAL_MAX_VALUE * N_PATTERN_FEATURES;
}

inline int calc_pattern_move_ordering_end(Eval_features *features) {
    const int *start_addr = (int*)(pattern_move_ordering_end_arr - SHIFT_EVAL_MO_END);
    __m256i res256 =                  gather_eval(start_addr, calc_idx8_comp(features->f128[4], 0));        // corner+block cross
    res256 = _mm256_add_epi32(res256, gather_eval(start_addr, calc_idx8_comp(features->f128[5], 1)));       // edge+2X triangle
    res256 = _mm256_and_si256(res256, eval_lower_mask);
    __m128i res128 = _mm_add_epi32(_mm256_castsi256_si128(res256), _mm256_extracti128_si256(res256, 1));
    res128 = _mm_hadd_epi32(res128, res128);
    return _mm_cvtsi128_si32(res128) + _mm_extract_epi32(res128, 1) - SIMD_EVAL_MAX_VALUE_MO_END * N_PATTERN_FEATURES_MO_END;
}

inline void calc_eval_features(Board *board, Eval_search *eval);

/*
    @brief midgame evaluation function

    @param b                    board
    @return evaluation value
*/
inline int mid_evaluate(Board *board) {
    Search search(board);
    calc_eval_features(&(search.board), &(search.eval));
    int phase_idx, num0;
    phase_idx = search.phase();
    num0 = pop_count_ull(search.board.player);
    int res = calc_pattern(phase_idx, &search.eval.features[search.eval.feature_idx]) + eval_num_arr[phase_idx][num0];
    res += res >= 0 ? STEP_2 : -STEP_2;
    res /= STEP;
    res = std::clamp(res, -SCORE_MAX, SCORE_MAX);
    return res;
}

/*
    @brief midgame evaluation function

    @param search               search information
    @return evaluation value
*/
inline int mid_evaluate_diff(Search *search) {
    int phase_idx, num0;
    phase_idx = search->phase();
    num0 = pop_count_ull(search->board.player);
    int res = calc_pattern(phase_idx, &search->eval.features[search->eval.feature_idx]) + eval_num_arr[phase_idx][num0];
    res += res >= 0 ? STEP_2 : -STEP_2;
    res /= STEP;
    res = std::clamp(res, -SCORE_MAX, SCORE_MAX);
    return res;
}

/*
    @brief midgame evaluation function

    @param search               search information
    @return evaluation value
*/
inline int mid_evaluate_move_ordering_end(Search *search) {
    int res = calc_pattern_move_ordering_end(&search->eval.features[search->eval.feature_idx]);
    res += res >= 0 ? STEP_2 : -STEP_2;
    res /= STEP;
    return res;
}

inline void calc_feature_vector(__m256i &f, const int *b_arr_int, const int i, const int n) {
    f = _mm256_set1_epi16(0);
    for (int j = 0; j < n; ++j) { // n: max n_cells in pattern - 1
        f = _mm256_add_epi16(f, _mm256_i32gather_epi32(b_arr_int, feature_to_coord_simd_cell[i][j][0], 4));
        f = _mm256_add_epi16(f, _mm256_slli_epi32(_mm256_i32gather_epi32(b_arr_int, feature_to_coord_simd_cell[i][j][1], 4), 16));
        f = _mm256_mullo_epi16(f, feature_to_coord_simd_mul[i][j]);
    }
    f = _mm256_add_epi16(f, _mm256_i32gather_epi32(b_arr_int, feature_to_coord_simd_cell[i][n][0], 4));
    f = _mm256_add_epi16(f, _mm256_slli_epi32(_mm256_i32gather_epi32(b_arr_int, feature_to_coord_simd_cell[i][n][1], 4), 16));
}

/*
    @brief calculate features for pattern evaluation

    @param search               search information
*/
inline void calc_eval_features(Board *board, Eval_search *eval) {
    int b_arr_int[HW2 + 1];
    board->translate_to_arr_player_rev(b_arr_int);
    b_arr_int[COORD_NO] = 0;
    calc_feature_vector(eval->features[0].f256[0], b_arr_int, 0, MAX_N_CELLS_GROUP[0] - 1);
    calc_feature_vector(eval->features[0].f256[1], b_arr_int, 1, MAX_N_CELLS_GROUP[1] - 1);
    calc_feature_vector(eval->features[0].f256[2], b_arr_int, 2, MAX_N_CELLS_GROUP[2] - 1);
    calc_feature_vector(eval->features[0].f256[3], b_arr_int, 3, MAX_N_CELLS_GROUP[3] - 1);
    eval->feature_idx = 0;
    eval->features[eval->feature_idx].f256[0] = _mm256_add_epi16(eval->features[eval->feature_idx].f256[0], eval_simd_offsets_simple[0]); // global index
    eval->features[eval->feature_idx].f256[1] = _mm256_add_epi16(eval->features[eval->feature_idx].f256[1], eval_simd_offsets_simple[1]); // global index
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
inline void eval_move(Eval_search *eval, const Flip *flip, const Board *board) {
    const uint16_t *flipped_group = (uint16_t*)&(flip->flip);
    const uint16_t *player_group = (uint16_t*)&(board->player);
    const uint16_t *opponent_group = (uint16_t*)&(board->opponent);
    __m256i f0, f1, f2, f3;
    uint16_t unflipped_p;
    uint16_t unflipped_o;
    // put cell 2 -> 1
    f0 = _mm256_sub_epi16(eval->features[eval->feature_idx].f256[0], coord_to_feature_simd[flip->pos][0]);
    f1 = _mm256_sub_epi16(eval->features[eval->feature_idx].f256[1], coord_to_feature_simd[flip->pos][1]);
    f2 = _mm256_sub_epi16(eval->features[eval->feature_idx].f256[2], coord_to_feature_simd[flip->pos][2]);
    f3 = _mm256_sub_epi16(eval->features[eval->feature_idx].f256[3], coord_to_feature_simd[flip->pos][3]);
    for (int i = 0; i < HW2 / 16; ++i) { // 64 bit / 16 bit = 4
        // player discs 0 -> 1
        unflipped_p = ~flipped_group[i] & player_group[i];
        f0 = _mm256_add_epi16(f0, eval_move_unflipped_16bit[unflipped_p][i][0]);
        f1 = _mm256_add_epi16(f1, eval_move_unflipped_16bit[unflipped_p][i][1]);
        f2 = _mm256_add_epi16(f2, eval_move_unflipped_16bit[unflipped_p][i][2]);
        f3 = _mm256_add_epi16(f3, eval_move_unflipped_16bit[unflipped_p][i][3]);
        // opponent discs 1 -> 0
        unflipped_o = ~flipped_group[i] & opponent_group[i];
        f0 = _mm256_sub_epi16(f0, eval_move_unflipped_16bit[unflipped_o][i][0]);
        f1 = _mm256_sub_epi16(f1, eval_move_unflipped_16bit[unflipped_o][i][1]);
        f2 = _mm256_sub_epi16(f2, eval_move_unflipped_16bit[unflipped_o][i][2]);
        f3 = _mm256_sub_epi16(f3, eval_move_unflipped_16bit[unflipped_o][i][3]);
    }
    ++eval->feature_idx;
    eval->features[eval->feature_idx].f256[0] = f0;
    eval->features[eval->feature_idx].f256[1] = f1;
    eval->features[eval->feature_idx].f256[2] = f2;
    eval->features[eval->feature_idx].f256[3] = f3;
}

/*
    @brief undo evaluation features

    @param eval                 evaluation features
*/
inline void eval_undo(Eval_search *eval) {
    --eval->feature_idx;
}

/*
    @brief pass evaluation features

        player discs    0 -> 1 (player -> opponent) add
        opponent discs  1 -> 0 (player -> opponent) sub

    @param eval                 evaluation features
*/
inline void eval_pass(Eval_search *eval, const Board *board) {
    const uint16_t *player_group = (uint16_t*)&(board->player);
    const uint16_t *opponent_group = (uint16_t*)&(board->opponent);
    __m256i f0, f1, f2, f3;
    f0 = eval->features[eval->feature_idx].f256[0];
    f1 = eval->features[eval->feature_idx].f256[1];
    f2 = eval->features[eval->feature_idx].f256[2];
    f3 = eval->features[eval->feature_idx].f256[3];
    for (int i = 0; i < HW2 / 16; ++i) { // 64 bit / 16 bit = 4
        f0 = _mm256_add_epi16(f0, eval_move_unflipped_16bit[player_group[i]][i][0]);
        f1 = _mm256_add_epi16(f1, eval_move_unflipped_16bit[player_group[i]][i][1]);
        f2 = _mm256_add_epi16(f2, eval_move_unflipped_16bit[player_group[i]][i][2]);
        f3 = _mm256_add_epi16(f3, eval_move_unflipped_16bit[player_group[i]][i][3]);
        f0 = _mm256_sub_epi16(f0, eval_move_unflipped_16bit[opponent_group[i]][i][0]);
        f1 = _mm256_sub_epi16(f1, eval_move_unflipped_16bit[opponent_group[i]][i][1]);
        f2 = _mm256_sub_epi16(f2, eval_move_unflipped_16bit[opponent_group[i]][i][2]);
        f3 = _mm256_sub_epi16(f3, eval_move_unflipped_16bit[opponent_group[i]][i][3]);
    }
    eval->features[eval->feature_idx].f256[0] = f0;
    eval->features[eval->feature_idx].f256[1] = f1;
    eval->features[eval->feature_idx].f256[2] = f2;
    eval->features[eval->feature_idx].f256[3] = f3;
}




// only corner+block cross edge+2X triangle
inline void eval_move_endsearch(Eval_search *eval, const Flip *flip, const Board *board) {
    const uint16_t *flipped_group = (uint16_t*)&(flip->flip);
    const uint16_t *player_group = (uint16_t*)&(board->player);
    const uint16_t *opponent_group = (uint16_t*)&(board->opponent);
    // put cell 2 -> 1
    __m256i f2 = _mm256_sub_epi16(eval->features[eval->feature_idx].f256[2], coord_to_feature_simd[flip->pos][2]);
    for (int i = 0; i < N_SIMD_EVAL_FEATURE_GROUP; ++i) {
        // player discs 0 -> 1
        f2 = _mm256_add_epi16(f2, eval_move_unflipped_16bit[~flipped_group[i] & player_group[i]][i][2]);
        // opponent discs 1 -> 0
        f2 = _mm256_sub_epi16(f2, eval_move_unflipped_16bit[~flipped_group[i] & opponent_group[i]][i][2]);
    }
    ++eval->feature_idx;
    eval->features[eval->feature_idx].f256[2] = f2;
}

inline void eval_undo_endsearch(Eval_search *eval) {
    --eval->feature_idx;
}

inline void eval_pass_endsearch(Eval_search *eval, const Board *board) {
    const uint16_t *player_group = (uint16_t*)&(board->player);
    const uint16_t *opponent_group = (uint16_t*)&(board->opponent);
    __m256i f2 = eval->features[eval->feature_idx].f256[2];
    for (int i = 0; i < N_SIMD_EVAL_FEATURE_GROUP; ++i) {
        f2 = _mm256_add_epi16(f2, eval_move_unflipped_16bit[player_group[i]][i][2]);
        f2 = _mm256_sub_epi16(f2, eval_move_unflipped_16bit[opponent_group[i]][i][2]);
    }
    eval->features[eval->feature_idx].f256[2] = f2;
}