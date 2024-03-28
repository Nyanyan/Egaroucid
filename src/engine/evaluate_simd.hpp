/*
    Egaroucid Project

    @file evaluate_simd.hpp
        Evaluation function with AVX2
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

/*
    @brief evaluation pattern definition for SIMD
*/
#define CEIL_N_SYMMETRY_PATTERNS 80         // N_SYMMETRY_PATTRENS + dummy
#define SIMD_EVAL_MAX_VALUE 3270            // evaluate range [-4092, 4092]
#define N_PATTERN_PARAMS1_LOAD1 37908
#define N_PATTERN_PARAMS1_LOAD2 26244
#define SIMD_EVAL_DUMMY_ADDR 37909
#define N_PATTERN_PARAMS1 (N_PATTERN_PARAMS1_LOAD1 + N_PATTERN_PARAMS1_LOAD2 + 2)       // +2 for byte bound & dummy for d8
#define N_PATTERN_PARAMS2_LOAD 52488
#define N_PATTERN_PARAMS2 (N_PATTERN_PARAMS2_LOAD + 1)       // +1 for byte bound
#define N_SIMD_EVAL_FEATURE_CELLS 16
#define N_SIMD_EVAL_FEATURE_GROUP 4

/*
    @brief evaluation pattern definition for SIMD move ordering end
*/
/*
#define N_PATTERN_PARAMS_MO_END (236196 + 1) // +1 for byte bound
#define SIMD_EVAL_MAX_VALUE_MO_END 16380
#define SHIFT_EVAL_MO_END 49087 // pattern_starts[8]
*/

constexpr Feature_to_coord feature_to_coord[CEIL_N_SYMMETRY_PATTERNS] = {
    // 0 hv1
    {8, {COORD_A1, COORD_B1, COORD_C1, COORD_D1, COORD_E1, COORD_F1, COORD_G1, COORD_H1}},
    {8, {COORD_H1, COORD_H2, COORD_H3, COORD_H4, COORD_H5, COORD_H6, COORD_H7, COORD_H8}},
    {8, {COORD_H8, COORD_G8, COORD_F8, COORD_E8, COORD_D8, COORD_C8, COORD_B8, COORD_A8}},
    {8, {COORD_A8, COORD_A7, COORD_A6, COORD_A5, COORD_A4, COORD_A3, COORD_A2, COORD_A1}},

    // 1 hv2
    {8, {COORD_A2, COORD_B2, COORD_C2, COORD_D2, COORD_E2, COORD_F2, COORD_G2, COORD_H2}},
    {8, {COORD_B1, COORD_B2, COORD_B3, COORD_B4, COORD_B5, COORD_B6, COORD_B7, COORD_B8}},
    {8, {COORD_A7, COORD_B7, COORD_C7, COORD_D7, COORD_E7, COORD_F7, COORD_G7, COORD_H7}},
    {8, {COORD_G1, COORD_G2, COORD_G3, COORD_G4, COORD_G5, COORD_G6, COORD_G7, COORD_G8}},

    // 2 hv3
    {8, {COORD_A3, COORD_B3, COORD_C3, COORD_D3, COORD_E3, COORD_F3, COORD_G3, COORD_H3}},
    {8, {COORD_C1, COORD_C2, COORD_C3, COORD_C4, COORD_C5, COORD_C6, COORD_C7, COORD_C8}},
    {8, {COORD_A6, COORD_B6, COORD_C6, COORD_D6, COORD_E6, COORD_F6, COORD_G6, COORD_H6}},
    {8, {COORD_F1, COORD_F2, COORD_F3, COORD_F4, COORD_F5, COORD_F6, COORD_F7, COORD_F8}},

    // 3 hv4
    {8, {COORD_A4, COORD_B4, COORD_C4, COORD_D4, COORD_E4, COORD_F4, COORD_G4, COORD_H4}},
    {8, {COORD_D1, COORD_D2, COORD_D3, COORD_D4, COORD_D5, COORD_D6, COORD_D7, COORD_D8}},
    {8, {COORD_A5, COORD_B5, COORD_C5, COORD_D5, COORD_E5, COORD_F5, COORD_G5, COORD_H5}},
    {8, {COORD_E1, COORD_E2, COORD_E3, COORD_E4, COORD_E5, COORD_E6, COORD_E7, COORD_E8}},

    // 4 d5 + corner + X
    {7, {COORD_D1, COORD_E2, COORD_F3, COORD_G4, COORD_H5, COORD_H1, COORD_G2, COORD_NO}},
    {7, {COORD_E1, COORD_D2, COORD_C3, COORD_B4, COORD_A5, COORD_A1, COORD_B2, COORD_NO}},
    {7, {COORD_A4, COORD_B5, COORD_C6, COORD_D7, COORD_E8, COORD_A8, COORD_B7, COORD_NO}},
    {7, {COORD_H4, COORD_G5, COORD_F6, COORD_E7, COORD_D8, COORD_H8, COORD_G7, COORD_NO}},

    // 5 d6
    {6, {COORD_C1, COORD_D2, COORD_E3, COORD_F4, COORD_G5, COORD_H6, COORD_NO, COORD_NO}},
    {6, {COORD_F1, COORD_E2, COORD_D3, COORD_C4, COORD_B5, COORD_A6, COORD_NO, COORD_NO}},
    {6, {COORD_A3, COORD_B4, COORD_C5, COORD_D6, COORD_E7, COORD_F8, COORD_NO, COORD_NO}},
    {6, {COORD_H3, COORD_G4, COORD_F5, COORD_E6, COORD_D7, COORD_C8, COORD_NO, COORD_NO}},

    // 6 d7
    {7, {COORD_B1, COORD_C2, COORD_D3, COORD_E4, COORD_F5, COORD_G6, COORD_H7, COORD_NO}},
    {7, {COORD_G1, COORD_F2, COORD_E3, COORD_D4, COORD_C5, COORD_B6, COORD_A7, COORD_NO}},
    {7, {COORD_A2, COORD_B3, COORD_C4, COORD_D5, COORD_E6, COORD_F7, COORD_G8, COORD_NO}},
    {7, {COORD_H2, COORD_G3, COORD_F4, COORD_E5, COORD_D6, COORD_C7, COORD_B8, COORD_NO}},

    // 7 d8
    {8, {COORD_A1, COORD_B2, COORD_C3, COORD_D4, COORD_E5, COORD_F6, COORD_G7, COORD_H8}},
    {8, {COORD_H1, COORD_G2, COORD_F3, COORD_E4, COORD_D5, COORD_C6, COORD_B7, COORD_A8}},

    // dummy
    {0, {COORD_NO, COORD_NO, COORD_NO, COORD_NO, COORD_NO, COORD_NO, COORD_NO, COORD_NO}},
    {0, {COORD_NO, COORD_NO, COORD_NO, COORD_NO, COORD_NO, COORD_NO, COORD_NO, COORD_NO}},

    // 8 corner-edge + 2x
    {8, {COORD_B2, COORD_A1, COORD_B1, COORD_C1, COORD_F1, COORD_G1, COORD_H1, COORD_G2}},
    {8, {COORD_B2, COORD_A1, COORD_A2, COORD_A3, COORD_A6, COORD_A7, COORD_A8, COORD_B7}},
    {8, {COORD_B7, COORD_A8, COORD_B8, COORD_C8, COORD_F8, COORD_G8, COORD_H8, COORD_G7}},
    {8, {COORD_G2, COORD_H1, COORD_H2, COORD_H3, COORD_H6, COORD_H7, COORD_H8, COORD_G7}},

    // 9 small triangle
    {8, {COORD_A1, COORD_B1, COORD_C1, COORD_D1, COORD_A2, COORD_B2, COORD_A3, COORD_A4}},
    {8, {COORD_H1, COORD_G1, COORD_F1, COORD_E1, COORD_H2, COORD_G2, COORD_H3, COORD_H4}},
    {8, {COORD_A8, COORD_B8, COORD_C8, COORD_D8, COORD_A7, COORD_B7, COORD_A6, COORD_A5}},
    {8, {COORD_H8, COORD_G8, COORD_F8, COORD_E8, COORD_H7, COORD_G7, COORD_H6, COORD_H5}},

    // 10 corner + small-block
    {8, {COORD_A1, COORD_C1, COORD_D1, COORD_E1, COORD_F1, COORD_H1, COORD_C2, COORD_F2}},
    {8, {COORD_A1, COORD_A3, COORD_A4, COORD_A5, COORD_A6, COORD_A8, COORD_B3, COORD_B6}},
    {8, {COORD_A8, COORD_C8, COORD_D8, COORD_E8, COORD_F8, COORD_H8, COORD_C7, COORD_F7}},
    {8, {COORD_H1, COORD_H3, COORD_H4, COORD_H5, COORD_H6, COORD_H8, COORD_G3, COORD_G6}},

    // 11 corner8
    {8, {COORD_A1, COORD_B1, COORD_C1, COORD_A2, COORD_B2, COORD_C2, COORD_A3, COORD_B3}},
    {8, {COORD_H1, COORD_G1, COORD_F1, COORD_H2, COORD_G2, COORD_F2, COORD_H3, COORD_G3}},
    {8, {COORD_A8, COORD_B8, COORD_C8, COORD_A7, COORD_B7, COORD_C7, COORD_A6, COORD_B6}},
    {8, {COORD_H8, COORD_G8, COORD_F8, COORD_H7, COORD_G7, COORD_F7, COORD_H6, COORD_G6}},

    // 12 corner-stability + 2 corner
    {8, {COORD_A1, COORD_A6, COORD_A7, COORD_A8, COORD_B7, COORD_B8, COORD_C8, COORD_H8}},
    {8, {COORD_H8, COORD_H3, COORD_H2, COORD_H1, COORD_G2, COORD_G1, COORD_F1, COORD_A1}},
    {8, {COORD_H1, COORD_H6, COORD_H7, COORD_H8, COORD_G7, COORD_G8, COORD_F8, COORD_A8}},
    {8, {COORD_A8, COORD_A3, COORD_A2, COORD_A1, COORD_B2, COORD_B1, COORD_C1, COORD_H1}},

    // 13 half center-block
    {8, {COORD_C3, COORD_D3, COORD_C4, COORD_D4, COORD_D5, COORD_C5, COORD_D6, COORD_C6}},
    {8, {COORD_F6, COORD_E6, COORD_F5, COORD_E5, COORD_E4, COORD_F4, COORD_E3, COORD_F3}},
    {8, {COORD_C3, COORD_C4, COORD_D3, COORD_D4, COORD_E4, COORD_E3, COORD_F4, COORD_F3}},
    {8, {COORD_F6, COORD_F5, COORD_E6, COORD_E5, COORD_D5, COORD_D6, COORD_C5, COORD_C6}},

    // 14 hamburger
    {8, {COORD_C2, COORD_D1, COORD_D2, COORD_D3, COORD_E3, COORD_E2, COORD_E1, COORD_F2}},
    {8, {COORD_G3, COORD_H4, COORD_G4, COORD_F4, COORD_F5, COORD_G5, COORD_H5, COORD_G6}},
    {8, {COORD_F7, COORD_E8, COORD_E7, COORD_E6, COORD_D6, COORD_D7, COORD_D8, COORD_C7}},
    {8, {COORD_B6, COORD_A5, COORD_B5, COORD_C5, COORD_C4, COORD_B4, COORD_A4, COORD_B3}},

    // 15 mid block
    {8, {COORD_C2, COORD_C3, COORD_D2, COORD_D3, COORD_E3, COORD_E2, COORD_F3, COORD_F2}},
    {8, {COORD_G3, COORD_F3, COORD_G4, COORD_F4, COORD_F5, COORD_G5, COORD_F6, COORD_G6}},
    {8, {COORD_F7, COORD_F6, COORD_E7, COORD_E6, COORD_D6, COORD_D7, COORD_C6, COORD_C7}},
    {8, {COORD_B6, COORD_C6, COORD_B5, COORD_C5, COORD_C4, COORD_B4, COORD_C3, COORD_B3}},

    // 16 I
    {8, {COORD_D1, COORD_D2, COORD_D3, COORD_D4, COORD_E4, COORD_E3, COORD_E2, COORD_E1}},
    {8, {COORD_H4, COORD_G4, COORD_F4, COORD_E4, COORD_E5, COORD_F5, COORD_G5, COORD_H5}},
    {8, {COORD_E8, COORD_E7, COORD_E6, COORD_E5, COORD_D5, COORD_D6, COORD_D7, COORD_D8}},
    {8, {COORD_A5, COORD_B5, COORD_C5, COORD_D5, COORD_D4, COORD_C4, COORD_B4, COORD_A4}},

    // 17 4x2-A
    {8, {COORD_A1, COORD_B1, COORD_A2, COORD_B2, COORD_C3, COORD_D3, COORD_C4, COORD_D4}},
    {8, {COORD_H1, COORD_H2, COORD_G1, COORD_G2, COORD_F3, COORD_F4, COORD_E3, COORD_E4}},
    {8, {COORD_H8, COORD_G8, COORD_H7, COORD_G7, COORD_F6, COORD_E6, COORD_F5, COORD_E5}},
    {8, {COORD_A8, COORD_A7, COORD_B8, COORD_B7, COORD_C6, COORD_C5, COORD_D6, COORD_D5}},

    // 18 4x2-B
    {8, {COORD_C1, COORD_D1, COORD_C2, COORD_D2, COORD_A3, COORD_B3, COORD_A4, COORD_B4}},
    {8, {COORD_H3, COORD_H4, COORD_G3, COORD_G4, COORD_F1, COORD_F2, COORD_E1, COORD_E2}},
    {8, {COORD_F8, COORD_E8, COORD_F7, COORD_E7, COORD_H6, COORD_G6, COORD_H5, COORD_G5}},
    {8, {COORD_A6, COORD_A5, COORD_B6, COORD_B5, COORD_C8, COORD_C7, COORD_D8, COORD_D7}},

    // 19 4x2-C
    {8, {COORD_A1, COORD_B1, COORD_A2, COORD_B2, COORD_B7, COORD_A7, COORD_B8, COORD_A8}},
    {8, {COORD_H1, COORD_H2, COORD_G1, COORD_G2, COORD_B2, COORD_B1, COORD_A2, COORD_A1}},
    {8, {COORD_H8, COORD_G8, COORD_H7, COORD_G7, COORD_G2, COORD_H2, COORD_G1, COORD_H1}},
    {8, {COORD_A8, COORD_A7, COORD_B8, COORD_B7, COORD_G7, COORD_G8, COORD_H7, COORD_H8}}
};

constexpr Coord_to_feature coord_to_feature[HW2] = {
    {16, {{ 1, P30}, { 2, P37}, {19, P31}, {28, P30}, {34, P31}, {35, P31}, {39, P37}, {42, P32}, {43, P32}, {47, P37}, {48, P30}, {49, P37}, {50, P34}, {70, P37}, {78, P37}, {79, P30}}}, // COORD_H8
    {10, {{ 2, P36}, { 7, P30}, {26, P30}, {34, P32}, {39, P36}, {47, P36}, {50, P32}, {70, P36}, {78, P36}, {79, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_G8
    { 9, {{ 2, P35}, {11, P30}, {22, P30}, {34, P33}, {39, P35}, {42, P33}, {47, P35}, {50, P31}, {74, P37}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_F8
    { 8, {{ 2, P34}, {15, P30}, {18, P32}, {39, P34}, {42, P34}, {58, P36}, {66, P37}, {74, P36}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_E8
    { 8, {{ 2, P33}, {13, P30}, {19, P32}, {38, P34}, {42, P35}, {58, P31}, {66, P30}, {75, P31}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_D8
    { 9, {{ 2, P32}, { 9, P30}, {23, P30}, {34, P34}, {38, P35}, {42, P36}, {46, P35}, {48, P31}, {75, P33}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_C8
    {10, {{ 2, P31}, { 5, P30}, {27, P30}, {34, P35}, {38, P36}, {46, P36}, {48, P32}, {71, P35}, {76, P31}, {79, P35}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_B8
    {16, {{ 2, P30}, { 3, P37}, {18, P31}, {29, P30}, {33, P31}, {34, P36}, {38, P37}, {41, P32}, {42, P37}, {46, P37}, {48, P34}, {50, P30}, {51, P37}, {71, P37}, {76, P30}, {79, P37}}}, // COORD_A8
    {10, {{ 1, P31}, { 6, P30}, {24, P30}, {35, P32}, {39, P33}, {47, P34}, {50, P35}, {70, P35}, {78, P35}, {79, P31}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_H7
    {12, {{ 6, P31}, { 7, P31}, {19, P30}, {28, P31}, {34, P30}, {35, P30}, {39, P32}, {47, P33}, {50, P33}, {70, P34}, {78, P34}, {79, P33}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_G7
    { 8, {{ 6, P32}, {11, P31}, {26, P31}, {42, P30}, {47, P32}, {58, P37}, {62, P37}, {74, P35}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_F7
    { 8, {{ 6, P33}, {15, P31}, {19, P33}, {22, P31}, {58, P35}, {62, P35}, {66, P36}, {74, P34}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_E7
    { 8, {{ 6, P34}, {13, P31}, {18, P33}, {23, P31}, {58, P32}, {62, P32}, {66, P31}, {75, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_D7
    { 8, {{ 6, P35}, { 9, P31}, {27, P31}, {42, P31}, {46, P32}, {58, P30}, {62, P30}, {75, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_C7
    {12, {{ 5, P31}, { 6, P36}, {18, P30}, {29, P31}, {33, P30}, {34, P37}, {38, P32}, {46, P33}, {48, P33}, {71, P34}, {76, P33}, {79, P34}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_B7
    {10, {{ 3, P36}, { 6, P37}, {25, P30}, {33, P32}, {38, P33}, {46, P34}, {48, P35}, {71, P36}, {76, P32}, {79, P36}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_A7
    { 9, {{ 1, P32}, {10, P30}, {20, P30}, {35, P33}, {39, P31}, {43, P33}, {47, P31}, {50, P36}, {74, P33}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_H6
    { 8, {{ 7, P32}, {10, P31}, {24, P31}, {43, P30}, {47, P30}, {57, P30}, {61, P30}, {74, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_G6
    { 9, {{10, P32}, {11, P32}, {19, P34}, {28, P32}, {53, P37}, {55, P37}, {61, P31}, {62, P36}, {70, P33}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_F6
    {10, {{10, P33}, {15, P32}, {23, P32}, {26, P32}, {53, P36}, {55, P35}, {58, P34}, {62, P34}, {66, P35}, {70, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_E6
    {10, {{10, P34}, {13, P32}, {22, P32}, {27, P32}, {52, P31}, {55, P32}, {58, P33}, {62, P33}, {66, P32}, {71, P31}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_D6
    { 9, {{ 9, P32}, {10, P35}, {18, P34}, {29, P32}, {52, P30}, {55, P30}, {62, P31}, {63, P36}, {71, P33}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_C6
    { 8, {{ 5, P32}, {10, P36}, {25, P31}, {41, P30}, {46, P30}, {59, P37}, {63, P37}, {75, P35}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_B6
    { 9, {{ 3, P35}, {10, P37}, {21, P30}, {33, P33}, {38, P31}, {41, P33}, {46, P31}, {48, P36}, {75, P37}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_A6
    { 8, {{ 1, P33}, {14, P30}, {16, P32}, {39, P30}, {43, P34}, {57, P31}, {65, P30}, {74, P31}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_H5
    { 8, {{ 7, P33}, {14, P31}, {19, P35}, {20, P31}, {57, P32}, {61, P32}, {65, P31}, {74, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_G5
    {10, {{11, P33}, {14, P32}, {23, P33}, {24, P32}, {53, P35}, {55, P36}, {57, P33}, {61, P33}, {65, P32}, {70, P31}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_F5
    { 9, {{14, P33}, {15, P33}, {27, P33}, {28, P33}, {53, P34}, {55, P34}, {65, P33}, {66, P34}, {70, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_E5
    { 9, {{13, P33}, {14, P34}, {26, P33}, {29, P33}, {52, P33}, {55, P33}, {66, P33}, {67, P34}, {71, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_D5
    {10, {{ 9, P33}, {14, P35}, {22, P33}, {25, P32}, {52, P32}, {55, P31}, {59, P34}, {63, P34}, {67, P35}, {71, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_C5
    { 8, {{ 5, P33}, {14, P36}, {18, P35}, {21, P31}, {59, P35}, {63, P35}, {67, P36}, {75, P34}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_B5
    { 8, {{ 3, P34}, {14, P37}, {17, P32}, {38, P30}, {41, P34}, {59, P36}, {67, P37}, {75, P36}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_A5
    { 8, {{ 1, P34}, {12, P30}, {19, P36}, {37, P30}, {43, P35}, {57, P36}, {65, P37}, {73, P36}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_H4
    { 8, {{ 7, P34}, {12, P31}, {16, P33}, {23, P34}, {57, P35}, {61, P35}, {65, P36}, {73, P34}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_G4
    {10, {{11, P34}, {12, P32}, {20, P32}, {27, P34}, {53, P32}, {54, P31}, {57, P34}, {61, P34}, {65, P35}, {69, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_F4
    { 9, {{12, P33}, {15, P34}, {24, P33}, {29, P34}, {53, P33}, {54, P33}, {64, P33}, {65, P34}, {69, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_E4
    { 9, {{12, P34}, {13, P34}, {25, P33}, {28, P34}, {52, P34}, {54, P34}, {64, P34}, {67, P33}, {68, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_D4
    {10, {{ 9, P34}, {12, P35}, {21, P32}, {26, P34}, {52, P35}, {54, P36}, {59, P33}, {63, P33}, {67, P32}, {68, P31}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_C4
    { 8, {{ 5, P34}, {12, P36}, {17, P33}, {22, P34}, {59, P32}, {63, P32}, {67, P31}, {72, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_B4
    { 8, {{ 3, P33}, {12, P37}, {18, P36}, {36, P30}, {41, P35}, {59, P31}, {67, P30}, {72, P31}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_A4
    { 9, {{ 1, P35}, { 8, P30}, {23, P35}, {35, P34}, {37, P31}, {43, P36}, {45, P31}, {49, P36}, {73, P37}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_H3
    { 8, {{ 7, P35}, { 8, P31}, {27, P35}, {43, P31}, {45, P30}, {57, P37}, {61, P37}, {73, P35}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_G3
    { 9, {{ 8, P32}, {11, P35}, {16, P34}, {29, P35}, {53, P30}, {54, P30}, {60, P31}, {61, P36}, {69, P33}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_F3
    {10, {{ 8, P33}, {15, P35}, {20, P33}, {25, P34}, {53, P31}, {54, P32}, {56, P33}, {60, P33}, {64, P32}, {69, P31}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_E3
    {10, {{ 8, P34}, {13, P35}, {21, P33}, {24, P34}, {52, P36}, {54, P35}, {56, P34}, {60, P34}, {64, P35}, {68, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_D3
    { 9, {{ 8, P35}, { 9, P35}, {17, P34}, {28, P35}, {52, P37}, {54, P37}, {60, P36}, {63, P31}, {68, P33}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_C3
    { 8, {{ 5, P35}, { 8, P36}, {26, P35}, {41, P31}, {44, P30}, {59, P30}, {63, P30}, {72, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_B3
    { 9, {{ 3, P32}, { 8, P37}, {22, P35}, {33, P34}, {36, P31}, {41, P36}, {44, P31}, {51, P36}, {72, P33}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_A3
    {10, {{ 1, P36}, { 4, P30}, {27, P36}, {35, P35}, {37, P33}, {45, P34}, {49, P35}, {69, P36}, {77, P36}, {78, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_H2
    {12, {{ 4, P31}, { 7, P36}, {16, P30}, {29, P36}, {32, P30}, {35, P37}, {37, P32}, {45, P33}, {49, P33}, {69, P34}, {77, P34}, {78, P33}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_G2
    { 8, {{ 4, P32}, {11, P36}, {25, P35}, {40, P30}, {45, P32}, {56, P30}, {60, P30}, {73, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_F2
    { 8, {{ 4, P33}, {15, P36}, {16, P35}, {21, P34}, {56, P32}, {60, P32}, {64, P31}, {73, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_E2
    { 8, {{ 4, P34}, {13, P36}, {17, P35}, {20, P34}, {56, P35}, {60, P35}, {64, P36}, {72, P34}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_D2
    { 8, {{ 4, P35}, { 9, P36}, {24, P35}, {40, P31}, {44, P32}, {56, P37}, {60, P37}, {72, P35}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_C2
    {12, {{ 4, P36}, { 5, P36}, {17, P30}, {28, P36}, {32, P37}, {33, P37}, {36, P32}, {44, P33}, {51, P33}, {68, P34}, {76, P34}, {77, P33}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_B2
    {10, {{ 3, P31}, { 4, P37}, {26, P36}, {33, P35}, {36, P33}, {44, P34}, {51, P35}, {68, P35}, {76, P35}, {77, P31}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_A2
    {16, {{ 0, P30}, { 1, P37}, {16, P31}, {29, P37}, {32, P31}, {35, P36}, {37, P37}, {40, P32}, {43, P37}, {45, P37}, {49, P34}, {50, P37}, {51, P30}, {69, P37}, {77, P37}, {78, P30}}}, // COORD_H1
    {10, {{ 0, P31}, { 7, P37}, {25, P36}, {32, P32}, {37, P36}, {45, P36}, {49, P32}, {69, P35}, {77, P35}, {78, P31}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_G1
    { 9, {{ 0, P32}, {11, P37}, {21, P35}, {32, P33}, {37, P35}, {40, P33}, {45, P35}, {49, P31}, {73, P33}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_F1
    { 8, {{ 0, P33}, {15, P37}, {17, P36}, {37, P34}, {40, P34}, {56, P31}, {64, P30}, {73, P31}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_E1
    { 8, {{ 0, P34}, {13, P37}, {16, P36}, {36, P34}, {40, P35}, {56, P36}, {64, P37}, {72, P36}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_D1
    { 9, {{ 0, P35}, { 9, P37}, {20, P35}, {32, P34}, {36, P35}, {40, P36}, {44, P35}, {51, P31}, {72, P37}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_C1
    {10, {{ 0, P36}, { 5, P37}, {24, P36}, {32, P35}, {36, P36}, {44, P36}, {51, P32}, {68, P36}, {76, P36}, {77, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_B1
    {16, {{ 0, P37}, { 3, P30}, {17, P31}, {28, P37}, {32, P36}, {33, P36}, {36, P37}, {40, P37}, {41, P37}, {44, P37}, {48, P37}, {49, P30}, {51, P34}, {68, P37}, {76, P37}, {77, P30}}}, // COORD_A1
};

/*
    @brief constants used for evaluation function with SIMD
*/
__m256i eval_lower_mask;
__m256i feature_to_coord_simd_mul[N_SIMD_EVAL_FEATURES][MAX_PATTERN_CELLS - 1];
__m256i feature_to_coord_simd_cell[N_SIMD_EVAL_FEATURES][MAX_PATTERN_CELLS][2];
__m256i coord_to_feature_simd[HW2][N_SIMD_EVAL_FEATURES];
__m256i eval_move_unflipped_16bit[N_16BIT][N_SIMD_EVAL_FEATURE_GROUP][N_SIMD_EVAL_FEATURES];
__m256i eval_simd_offsets[N_SIMD_EVAL_FEATURES]; // 16bit * 16 * N
__m256i eval_surround_mask;
__m128i eval_surround_shift1879;

/*
    @brief evaluation parameters
*/
int16_t pattern_arr1[N_PHASES][N_PATTERN_PARAMS1];
int16_t pattern_arr2[N_PHASES][N_PATTERN_PARAMS2];
int16_t eval_num_arr[N_PHASES][MAX_STONE_NUM];
int16_t eval_sur0_sur1_arr[N_PHASES][MAX_SURROUND][MAX_SURROUND];
//int16_t pattern_arr_move_ordering_end[N_PATTERN_PARAMS_MO_END];

inline bool load_eval_file(const char* file, bool show_log){
    if (show_log)
        std::cerr << "evaluation file " << file << std::endl;
    FILE* fp;
    if (!file_open(&fp, file, "rb")){
        std::cerr << "[ERROR] [FATAL] can't open eval " << file << std::endl;
        return false;
    }
    for (int phase_idx = 0; phase_idx < N_PHASES; ++phase_idx){
        pattern_arr1[phase_idx][0] = 0; // memory bound
        if (fread(pattern_arr1[phase_idx] + 1, 2, N_PATTERN_PARAMS1_LOAD1, fp) < N_PATTERN_PARAMS1_LOAD1){
            std::cerr << "[ERROR] [FATAL] evaluation file broken" << std::endl;
            fclose(fp);
            return false;
        }
        pattern_arr1[phase_idx][SIMD_EVAL_DUMMY_ADDR] = 0; // dummy for d8
        if (fread(pattern_arr1[phase_idx] + 1 + SIMD_EVAL_DUMMY_ADDR, 2, N_PATTERN_PARAMS1_LOAD2, fp) < N_PATTERN_PARAMS1_LOAD2){
            std::cerr << "[ERROR] [FATAL] evaluation file broken" << std::endl;
            fclose(fp);
            return false;
        }
        pattern_arr2[phase_idx][0] = 0; // memory bound
        if (fread(pattern_arr2[phase_idx] + 1, 2, N_PATTERN_PARAMS2_LOAD, fp) < N_PATTERN_PARAMS2_LOAD){
            std::cerr << "[ERROR] [FATAL] evaluation file broken" << std::endl;
            fclose(fp);
            return false;
        }
        if (fread(eval_num_arr[phase_idx], 2, MAX_STONE_NUM, fp) < MAX_STONE_NUM){
            std::cerr << "[ERROR] [FATAL] evaluation file broken" << std::endl;
            fclose(fp);
            return false;
        }
        if (fread(eval_sur0_sur1_arr[phase_idx], 2, MAX_SURROUND * MAX_SURROUND, fp) < MAX_SURROUND * MAX_SURROUND){
            std::cerr << "[ERROR] [FATAL] evaluation file broken" << std::endl;
            fclose(fp);
            return false;
        }
    }
    // check max value
    for (int phase_idx = 0; phase_idx < N_PHASES; ++phase_idx){
        for (int i = 1; i < N_PATTERN_PARAMS1; ++i){
            if (i == SIMD_EVAL_DUMMY_ADDR) // dummy
                continue;
            if (pattern_arr1[phase_idx][i] < -SIMD_EVAL_MAX_VALUE){
                std::cerr << "[ERROR] evaluation value too low. you can ignore this error. phase " << phase_idx << " index " << i << " found " << pattern_arr1[phase_idx][i] << std::endl;
                pattern_arr1[phase_idx][i] = -SIMD_EVAL_MAX_VALUE;
            }
            if (pattern_arr1[phase_idx][i] > SIMD_EVAL_MAX_VALUE){
                std::cerr << "[ERROR] evaluation value too high. you can ignore this error. phase " << phase_idx << " index " << i << " found " << pattern_arr1[phase_idx][i] << std::endl;
                pattern_arr1[phase_idx][i] = SIMD_EVAL_MAX_VALUE;
            }
            pattern_arr1[phase_idx][i] += SIMD_EVAL_MAX_VALUE;
        }
        for (int i = 1; i < N_PATTERN_PARAMS2; ++i){
            if (pattern_arr2[phase_idx][i] < -SIMD_EVAL_MAX_VALUE){
                std::cerr << "[ERROR] evaluation value too low. you can ignore this error. phase " << phase_idx << " index " << i << " found " << pattern_arr2[phase_idx][i] << std::endl;
                pattern_arr2[phase_idx][i] = -SIMD_EVAL_MAX_VALUE;
            }
            if (pattern_arr2[phase_idx][i] > SIMD_EVAL_MAX_VALUE){
                std::cerr << "[ERROR] evaluation value too high. you can ignore this error. phase " << phase_idx << " index " << i << " found " << pattern_arr2[phase_idx][i] << std::endl;
                pattern_arr2[phase_idx][i] = SIMD_EVAL_MAX_VALUE;
            }
            pattern_arr2[phase_idx][i] += SIMD_EVAL_MAX_VALUE;
        }
    }
    return true;
}
/*
inline bool load_eval_move_ordering_end_file(const char* file, bool show_log){
    if (show_log)
        std::cerr << "evaluation for move ordering end file " << file << std::endl;
    FILE* fp;
    if (!file_open(&fp, file, "rb")){
        std::cerr << "[ERROR] [FATAL] can't open eval " << file << std::endl;
        return false;
    }
    pattern_arr_move_ordering_end[0] = 0; // memory bound
    if (fread(pattern_arr_move_ordering_end + 1, 2, N_PATTERN_PARAMS_MO_END - 1, fp) < N_PATTERN_PARAMS_MO_END - 1){
        std::cerr << "[ERROR] [FATAL] evaluation file for move ordering end broken" << std::endl;
        fclose(fp);
        return false;
    }
    // check max value
    for (int i = 1; i < N_PATTERN_PARAMS_MO_END; ++i){
        if (pattern_arr_move_ordering_end[i] < -SIMD_EVAL_MAX_VALUE_MO_END){
            std::cerr << "[ERROR] evaluation value too low. you can ignore this error. index " << i << " found " << pattern_arr_move_ordering_end[i] << std::endl;
            pattern_arr_move_ordering_end[i] = -SIMD_EVAL_MAX_VALUE_MO_END;
        }
        if (pattern_arr_move_ordering_end[i] > SIMD_EVAL_MAX_VALUE_MO_END){
            std::cerr << "[ERROR] evaluation value too high. you can ignore this error. index " << i << " found " << pattern_arr_move_ordering_end[i] << std::endl;
            pattern_arr_move_ordering_end[i] = SIMD_EVAL_MAX_VALUE_MO_END;
        }
        pattern_arr_move_ordering_end[i] += SIMD_EVAL_MAX_VALUE_MO_END;
    }
    return true;
}
*/

inline void pre_calculate_eval_constant(){
    { // calc_eval_features initialization
        int16_t f2c[16];
        for (int i = 0; i < N_SIMD_EVAL_FEATURES; ++i){
            for (int j = 0; j < MAX_PATTERN_CELLS - 1; ++j){
                for (int k = 0; k < 16; ++k)
                    f2c[k] = (j < feature_to_coord[i * 16 + k].n_cells - 1) ? 3 : 1;
                feature_to_coord_simd_mul[i][j] = _mm256_set_epi16(
                    f2c[0], f2c[1], f2c[2], f2c[3], 
                    f2c[4], f2c[5], f2c[6], f2c[7], 
                    f2c[8], f2c[9], f2c[10], f2c[11], 
                    f2c[12], f2c[13], f2c[14], f2c[15]
                );
            }
        }
        int32_t f2c32[8];
        for (int i = 0; i < N_SIMD_EVAL_FEATURES; ++i){
            for (int j = 0; j < MAX_PATTERN_CELLS; ++j){
                for (int k = 0; k < 8; ++k)
                    f2c32[k] = feature_to_coord[i * 16 + k * 2 + 1].cells[j];
                feature_to_coord_simd_cell[i][j][0] = _mm256_set_epi32(
                    f2c32[0], f2c32[1], f2c32[2], f2c32[3], 
                    f2c32[4], f2c32[5], f2c32[6], f2c32[7]
                );
                for (int k = 0; k < 8; ++k)
                    f2c32[k] = feature_to_coord[i * 16 + k * 2].cells[j];
                feature_to_coord_simd_cell[i][j][1] = _mm256_set_epi32(
                    f2c32[0], f2c32[1], f2c32[2], f2c32[3], 
                    f2c32[4], f2c32[5], f2c32[6], f2c32[7]
                );
            }
        }
        constexpr int pattern_starts[N_PATTERNS] = {
            1, 6562, 13123, 19684,                      // features[0]
            26245, 28432, 29161, 31348, /*dummy 37909*/ // features[1]
            37910, 44471, 51032, 57593,                 // features[2]

            1, 6562, 13123, 19684,                      // features[3]
            26245, 32806, 39367, 45928                  // features[4]
        };
        eval_simd_offsets[0] = _mm256_set_epi16(
            (int16_t)pattern_starts[0], (int16_t)pattern_starts[0], (int16_t)pattern_starts[0], (int16_t)pattern_starts[0], 
            (int16_t)pattern_starts[1], (int16_t)pattern_starts[1], (int16_t)pattern_starts[1], (int16_t)pattern_starts[1], 
            (int16_t)pattern_starts[2], (int16_t)pattern_starts[2], (int16_t)pattern_starts[2], (int16_t)pattern_starts[2], 
            (int16_t)pattern_starts[3], (int16_t)pattern_starts[3], (int16_t)pattern_starts[3], (int16_t)pattern_starts[3]
        );
        eval_simd_offsets[1] = _mm256_set_epi16(
            (int16_t)pattern_starts[4], (int16_t)pattern_starts[4], (int16_t)pattern_starts[4], (int16_t)pattern_starts[4], 
            (int16_t)pattern_starts[5], (int16_t)pattern_starts[5], (int16_t)pattern_starts[5], (int16_t)pattern_starts[5], 
            (int16_t)pattern_starts[6], (int16_t)pattern_starts[6], (int16_t)pattern_starts[6], (int16_t)pattern_starts[6], 
            (int16_t)pattern_starts[7], (int16_t)pattern_starts[7],       SIMD_EVAL_DUMMY_ADDR,       SIMD_EVAL_DUMMY_ADDR
        );
        eval_simd_offsets[2] = _mm256_set_epi16(
            (int16_t)pattern_starts[8], (int16_t)pattern_starts[8], (int16_t)pattern_starts[8], (int16_t)pattern_starts[8], 
            (int16_t)pattern_starts[9], (int16_t)pattern_starts[9], (int16_t)pattern_starts[9], (int16_t)pattern_starts[9], 
            (int16_t)pattern_starts[10], (int16_t)pattern_starts[10], (int16_t)pattern_starts[10], (int16_t)pattern_starts[10], 
            (int16_t)pattern_starts[11], (int16_t)pattern_starts[11], (int16_t)pattern_starts[11], (int16_t)pattern_starts[11]
        );
        eval_simd_offsets[3] = _mm256_set_epi16(
            (int16_t)pattern_starts[12], (int16_t)pattern_starts[12], (int16_t)pattern_starts[12], (int16_t)pattern_starts[12], 
            (int16_t)pattern_starts[13], (int16_t)pattern_starts[13], (int16_t)pattern_starts[13], (int16_t)pattern_starts[13], 
            (int16_t)pattern_starts[14], (int16_t)pattern_starts[14], (int16_t)pattern_starts[14], (int16_t)pattern_starts[14], 
            (int16_t)pattern_starts[15], (int16_t)pattern_starts[15], (int16_t)pattern_starts[15], (int16_t)pattern_starts[15]
        );
        eval_simd_offsets[4] = _mm256_set_epi16(
            (int16_t)pattern_starts[16], (int16_t)pattern_starts[16], (int16_t)pattern_starts[16], (int16_t)pattern_starts[16], 
            (int16_t)pattern_starts[17], (int16_t)pattern_starts[17], (int16_t)pattern_starts[17], (int16_t)pattern_starts[17], 
            (int16_t)pattern_starts[18], (int16_t)pattern_starts[18], (int16_t)pattern_starts[18], (int16_t)pattern_starts[18], 
            (int16_t)pattern_starts[19], (int16_t)pattern_starts[19], (int16_t)pattern_starts[19], (int16_t)pattern_starts[19]
        );
    }
    { // eval_move initialization
        uint16_t c2f[CEIL_N_SYMMETRY_PATTERNS];
        for (int cell = 0; cell < HW2; ++cell){ // 0 for h8, 63 for a1
            for (int i = 0; i < CEIL_N_SYMMETRY_PATTERNS; ++i)
                c2f[i] = 0;
            for (int i = 0; i < coord_to_feature[cell].n_features; ++i)
                c2f[coord_to_feature[cell].features[i].feature] = coord_to_feature[cell].features[i].x;
            for (int i = 0; i < N_SIMD_EVAL_FEATURES; ++i){
                int idx = i * 16;
                coord_to_feature_simd[cell][i] = _mm256_set_epi16(
                    c2f[idx], c2f[idx + 1], c2f[idx + 2], c2f[idx + 3], 
                    c2f[idx + 4], c2f[idx + 5], c2f[idx + 6], c2f[idx + 7], 
                    c2f[idx + 8], c2f[idx + 9], c2f[idx + 10], c2f[idx + 11], 
                    c2f[idx + 12], c2f[idx + 13], c2f[idx + 14], c2f[idx + 15]
                );
            }
        }
        for (int bits = 0; bits < N_16BIT; ++bits){ // 1: unflipped discs, 0: others
            for (int group = 0; group < N_SIMD_EVAL_FEATURE_GROUP; ++group){ // a1-h2, a3-h4, ..., a7-h8
                for (int i = 0; i < CEIL_N_SYMMETRY_PATTERNS; ++i)
                    c2f[i] = 0;
                for (int cell = 0; cell < N_SIMD_EVAL_FEATURE_CELLS; ++cell){
                    if (1 & (bits >> cell)){
                        int global_cell = group * N_SIMD_EVAL_FEATURE_CELLS + cell;
                        for (int i = 0; i < coord_to_feature[global_cell].n_features; ++i)
                            c2f[coord_to_feature[global_cell].features[i].feature] += coord_to_feature[global_cell].features[i].x;
                    }
                }
                for (int simd_feature_idx = 0; simd_feature_idx < N_SIMD_EVAL_FEATURES; ++simd_feature_idx){
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
        eval_lower_mask = _mm256_set1_epi32(0x0000FFFF);
    }
    { // calc_surround initialization
        eval_surround_mask = _mm256_set_epi64x(0x7E7E7E7E7E7E7E7EULL, 0x00FFFFFFFFFFFF00ULL, 0x007E7E7E7E7E7E00ULL, 0x007E7E7E7E7E7E00ULL);
        eval_surround_shift1879 = _mm_set_epi32(1, HW, HW_M1, HW_P1);
    }
}

/*
    @brief initialize the evaluation function

    @param file                 evaluation file name
    @param show_log             debug information?
    @return evaluation function conpletely initialized?
*/
inline bool evaluate_init(const char* file, const char* mo_end_nws_file, bool show_log){
    bool eval_loaded = load_eval_file(file, show_log);
    if (!eval_loaded){
        std::cerr << "[ERROR] [FATAL] evaluation file not loaded" << std::endl;
        return false;
    }
    /*
    bool eval_move_ordering_end_nws_loaded = load_eval_move_ordering_end_file(mo_end_nws_file, show_log);
    if (!eval_move_ordering_end_nws_loaded){
        std::cerr << "[ERROR] [FATAL] evaluation file for move ordering end not loaded" << std::endl;
        return false;
    }
    */
    pre_calculate_eval_constant();
    if (show_log)
        std::cerr << "evaluation function initialized" << std::endl;
    return true;
}

/*
    @brief Wrapper of evaluation initializing

    @param file                 evaluation file name
    @return evaluation function conpletely initialized?
*/
bool evaluate_init(const std::string file, std::string mo_end_nws_file, bool show_log){
    return evaluate_init(file.c_str(), mo_end_nws_file.c_str(), show_log);
}

/*
    @brief Wrapper of evaluation initializing

    @return evaluation function conpletely initialized?
*/
bool evaluate_init(bool show_log){
    return evaluate_init("resources/eval.egev", "resources/eval_move_ordering_end.egev", show_log);
}

/*
    @brief calculate surround value used in evaluation function

    @param discs                a bitboard representing discs
    @param empties              a bitboard representing empties
    @return surround value
*/
inline int calc_surround(const uint64_t discs, const uint64_t empties){
    __m256i pl = _mm256_set1_epi64x(discs);
    pl = _mm256_and_si256(pl, eval_surround_mask);
    pl = _mm256_or_si256(_mm256_sll_epi64(pl, eval_surround_shift1879), _mm256_srl_epi64(pl, eval_surround_shift1879));
    __m128i res = _mm_or_si128(_mm256_castsi256_si128(pl), _mm256_extracti128_si256(pl, 1));
    res = _mm_or_si128(res, _mm_shuffle_epi32(res, 0x4e));
    return pop_count_ull(_mm_cvtsi128_si64(res));
}
#define CALC_SURROUND_FUNCTION

inline __m256i gather_eval(const int *start_addr, const __m256i idx8){
    return _mm256_i32gather_epi32(start_addr, idx8, 2); // stride is 2 byte, because 16 bit array used, HACK: if (SIMD_EVAL_MAX_VALUE * 2) * (N_ADD=10) < 2 ^ 16, AND is unnecessary
    // return _mm256_and_si256(_mm256_i32gather_epi32(start_addr, idx8, 2), eval_lower_mask);
}

inline int calc_pattern(const int phase_idx, Eval_features *features){
    const int *start_addr = (int*)pattern_arr1[phase_idx];
    const int *start_addr2 = (int*)pattern_arr2[phase_idx];
    __m256i res256 =                  gather_eval(start_addr, _mm256_cvtepu16_epi32(features->f128[0]));    // hv4 d5
    res256 = _mm256_add_epi32(res256, gather_eval(start_addr, _mm256_cvtepu16_epi32(features->f128[1])));   // hv2 hv3
    res256 = _mm256_add_epi32(res256, gather_eval(start_addr, _mm256_cvtepu16_epi32(features->f128[2])));
    res256 = _mm256_add_epi32(res256, gather_eval(start_addr, _mm256_cvtepu16_epi32(features->f128[3])));
    res256 = _mm256_add_epi32(res256, gather_eval(start_addr, _mm256_cvtepu16_epi32(features->f128[4])));
    res256 = _mm256_add_epi32(res256, gather_eval(start_addr, _mm256_cvtepu16_epi32(features->f128[5])));
    res256 = _mm256_add_epi32(res256, gather_eval(start_addr2, _mm256_cvtepu16_epi32(features->f128[6])));
    res256 = _mm256_add_epi32(res256, gather_eval(start_addr2, _mm256_cvtepu16_epi32(features->f128[7])));
    res256 = _mm256_add_epi32(res256, gather_eval(start_addr2, _mm256_cvtepu16_epi32(features->f128[8])));
    res256 = _mm256_add_epi32(res256, gather_eval(start_addr2, _mm256_cvtepu16_epi32(features->f128[9])));
    res256 = _mm256_and_si256(res256, eval_lower_mask);
    __m128i res128 = _mm_add_epi32(_mm256_castsi256_si128(res256), _mm256_extracti128_si256(res256, 1));
    res128 = _mm_hadd_epi32(res128, res128);
    return _mm_cvtsi128_si32(res128) + _mm_extract_epi32(res128, 1) - SIMD_EVAL_MAX_VALUE * N_SYMMETRY_PATTERNS;
}
/*
inline int calc_pattern_move_ordering_end(Eval_features *features){
    const int *start_addr = (int*)(pattern_arr_move_ordering_end - SHIFT_EVAL_MO_END);
    __m256i res256 =                  gather_eval(start_addr, calc_idx8_comp(features->f128[4], 0));        // corner+block cross
    res256 = _mm256_add_epi32(res256, gather_eval(start_addr, calc_idx8_comp(features->f128[5], 1)));       // edge+2X triangle
    res256 = _mm256_and_si256(res256, eval_lower_mask);
    __m128i res128 = _mm_add_epi32(_mm256_castsi256_si128(res256), _mm256_extracti128_si256(res256, 1));
    res128 = _mm_hadd_epi32(res128, res128);
    return _mm_cvtsi128_si32(res128) + _mm_extract_epi32(res128, 1) - SIMD_EVAL_MAX_VALUE_MO_END * N_SYMMETRY_PATTERNS_MO_END;
}
*/
inline void calc_eval_features(Board *board, Eval_search *eval);

/*
    @brief midgame evaluation function

    @param b                    board
    @return evaluation value
*/
inline int mid_evaluate(Board *board){
    Search search;
    search.init_board(board);
    calc_eval_features(&(search.board), &(search.eval));
    int phase_idx, sur0, sur1, num0;
    uint64_t empties;
    phase_idx = search.phase();
    empties = ~(search.board.player | search.board.opponent);
    sur0 = calc_surround(search.board.player, empties);
    sur1 = calc_surround(search.board.opponent, empties);
    num0 = pop_count_ull(search.board.player);
    int res = calc_pattern(phase_idx, &search.eval.features[search.eval.feature_idx]) + 
        eval_num_arr[phase_idx][num0] + 
        eval_sur0_sur1_arr[phase_idx][sur0][sur1];
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
inline int mid_evaluate_diff(Search *search){
    int phase_idx, sur0, sur1, num0;
    uint64_t empties;
    phase_idx = search->phase();
    empties = ~(search->board.player | search->board.opponent);
    sur0 = calc_surround(search->board.player, empties);
    sur1 = calc_surround(search->board.opponent, empties);
    num0 = pop_count_ull(search->board.player);
    int res = calc_pattern(phase_idx, &search->eval.features[search->eval.feature_idx]) + 
        eval_num_arr[phase_idx][num0] + 
        eval_sur0_sur1_arr[phase_idx][sur0][sur1];
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
/*
inline int mid_evaluate_move_ordering_end(Search *search){
    int res = calc_pattern_move_ordering_end(&search->eval.features[search->eval.feature_idx]);
    res += res >= 0 ? STEP_2 : -STEP_2;
    res /= STEP;
    return res;
}
*/

inline void calc_feature_vector(__m256i &f, const int *b_arr_int, const int i, const int n){
    f = _mm256_set1_epi16(0);
    for (int j = 0; j < n; ++j){ // n: max n_cells in pattern - 1
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
inline void calc_eval_features(Board *board, Eval_search *eval){
    int b_arr_int[HW2 + 1];
    board->translate_to_arr_player_rev(b_arr_int);
    b_arr_int[COORD_NO] = 0;
    for (int i = 0; i < N_SIMD_EVAL_FEATURES; ++i){
        calc_feature_vector(eval->features[0].f256[i], b_arr_int, i, 7);
        eval->features[eval->feature_idx].f256[i] = _mm256_add_epi16(eval->features[eval->feature_idx].f256[i], eval_simd_offsets[i]);
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
    const uint16_t *flipped_group = (uint16_t*)&(flip->flip);
    const uint16_t *player_group = (uint16_t*)&(board->player);
    const uint16_t *opponent_group = (uint16_t*)&(board->opponent);
    __m256i f[N_SIMD_EVAL_FEATURES];
    uint16_t unflipped_p;
    uint16_t unflipped_o;
    // put cell 2 -> 1
    for (int j = 0; j < N_SIMD_EVAL_FEATURES; ++j){
        f[j] = _mm256_sub_epi16(eval->features[eval->feature_idx].f256[j], coord_to_feature_simd[flip->pos][j]);
    }
    for (int i = 0; i < N_SIMD_EVAL_FEATURE_GROUP; ++i){
        // player discs 0 -> 1
        unflipped_p = ~flipped_group[i] & player_group[i];
        for (int j = 0; j < N_SIMD_EVAL_FEATURES; ++j){
            f[j] = _mm256_add_epi16(f[j], eval_move_unflipped_16bit[unflipped_p][i][j]);
        }
        // opponent discs 1 -> 0
        unflipped_o = ~flipped_group[i] & opponent_group[i];
        for (int j = 0; j < N_SIMD_EVAL_FEATURES; ++j){
            f[j] = _mm256_sub_epi16(f[j], eval_move_unflipped_16bit[unflipped_o][i][j]);
        }
    }
    ++eval->feature_idx;
    for (int j = 0; j < N_SIMD_EVAL_FEATURES; ++j){
        eval->features[eval->feature_idx].f256[j] = f[j];
    }
}

/*
    @brief undo evaluation features

    @param eval                 evaluation features
*/
inline void eval_undo(Eval_search *eval){
    --eval->feature_idx;
}

/*
    @brief pass evaluation features

        player discs    0 -> 1 (player -> opponent) add
        opponent discs  1 -> 0 (player -> opponent) sub

    @param eval                 evaluation features
*/
inline void eval_pass(Eval_search *eval, const Board *board){
    const uint16_t *player_group = (uint16_t*)&(board->player);
    const uint16_t *opponent_group = (uint16_t*)&(board->opponent);
    __m256i f[N_SIMD_EVAL_FEATURES];
    for (int j = 0; j < N_SIMD_EVAL_FEATURES; ++j){
        f[j] = eval->features[eval->feature_idx].f256[j];
    }
    for (int i = 0; i < N_SIMD_EVAL_FEATURE_GROUP; ++i){
        for (int j = 0; j < N_SIMD_EVAL_FEATURES; ++j){
            f[j] = _mm256_add_epi16(f[j], eval_move_unflipped_16bit[player_group[i]][i][j]);
            f[j] = _mm256_sub_epi16(f[j], eval_move_unflipped_16bit[opponent_group[i]][i][j]);
        }
    }
    for (int j = 0; j < N_SIMD_EVAL_FEATURES; ++j){
        eval->features[eval->feature_idx].f256[j] = f[j];
    }
}





/*
// only corner+block cross edge+2X triangle
inline void eval_move_endsearch(Eval_search *eval, const Flip *flip, const Board *board){
    const uint16_t *flipped_group = (uint16_t*)&(flip->flip);
    const uint16_t *player_group = (uint16_t*)&(board->player);
    const uint16_t *opponent_group = (uint16_t*)&(board->opponent);
    // put cell 2 -> 1
    __m256i f2 = _mm256_sub_epi16(eval->features[eval->feature_idx].f256[2], coord_to_feature_simd[flip->pos][2]);
    for (int i = 0; i < N_SIMD_EVAL_FEATURE_GROUP; ++i){
        // player discs 0 -> 1
        f2 = _mm256_add_epi16(f2, eval_move_unflipped_16bit[~flipped_group[i] & player_group[i]][i][2]);
        // opponent discs 1 -> 0
        f2 = _mm256_sub_epi16(f2, eval_move_unflipped_16bit[~flipped_group[i] & opponent_group[i]][i][2]);
    }
    ++eval->feature_idx;
    eval->features[eval->feature_idx].f256[2] = f2;
}

inline void eval_undo_endsearch(Eval_search *eval){
    --eval->feature_idx;
}

inline void eval_pass_endsearch(Eval_search *eval, const Board *board){
    const uint16_t *player_group = (uint16_t*)&(board->player);
    const uint16_t *opponent_group = (uint16_t*)&(board->opponent);
    __m256i f2 = eval->features[eval->feature_idx].f256[2];
    for (int i = 0; i < N_SIMD_EVAL_FEATURE_GROUP; ++i){
        f2 = _mm256_add_epi16(f2, eval_move_unflipped_16bit[player_group[i]][i][2]);
        f2 = _mm256_sub_epi16(f2, eval_move_unflipped_16bit[opponent_group[i]][i][2]);
    }
    eval->features[eval->feature_idx].f256[2] = f2;
}
*/
