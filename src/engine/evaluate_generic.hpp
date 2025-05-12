/*
    Egaroucid Project

    @file evaluate_generic.hpp
        Evaluation function without AVX2
    @date 2021-2025
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once
#include <iostream>
#include <fstream>
#include <cstring>
#include "setting.hpp"
#include "common.hpp"
#include "board.hpp"
#include "search.hpp"
#include "util.hpp"
#include "evaluate_common.hpp"

constexpr int EVAL_IDX_START_MOVE_ORDERING_END = 16;
constexpr int EVAL_IDX_END_MOVE_ORDERING_END = 32;
constexpr int EVAL_FEATURE_START_MOVE_ORDERING_END = 4;
constexpr int MAX_CELL_PATTERNS_MOVE_ORDERING_END = 7;


constexpr Feature_to_coord feature_to_coord[N_PATTERN_FEATURES] = {
    // 0 hv2
    {8, {COORD_A2, COORD_B2, COORD_C2, COORD_D2, COORD_E2, COORD_F2, COORD_G2, COORD_H2, COORD_NO, COORD_NO}},
    {8, {COORD_B1, COORD_B2, COORD_B3, COORD_B4, COORD_B5, COORD_B6, COORD_B7, COORD_B8, COORD_NO, COORD_NO}},
    {8, {COORD_A7, COORD_B7, COORD_C7, COORD_D7, COORD_E7, COORD_F7, COORD_G7, COORD_H7, COORD_NO, COORD_NO}},
    {8, {COORD_G1, COORD_G2, COORD_G3, COORD_G4, COORD_G5, COORD_G6, COORD_G7, COORD_G8, COORD_NO, COORD_NO}},

    // 1 hv3
    {8, {COORD_A3, COORD_B3, COORD_C3, COORD_D3, COORD_E3, COORD_F3, COORD_G3, COORD_H3, COORD_NO, COORD_NO}},
    {8, {COORD_C1, COORD_C2, COORD_C3, COORD_C4, COORD_C5, COORD_C6, COORD_C7, COORD_C8, COORD_NO, COORD_NO}},
    {8, {COORD_A6, COORD_B6, COORD_C6, COORD_D6, COORD_E6, COORD_F6, COORD_G6, COORD_H6, COORD_NO, COORD_NO}},
    {8, {COORD_F1, COORD_F2, COORD_F3, COORD_F4, COORD_F5, COORD_F6, COORD_F7, COORD_F8, COORD_NO, COORD_NO}},

    // 2 hv4
    {8, {COORD_A4, COORD_B4, COORD_C4, COORD_D4, COORD_E4, COORD_F4, COORD_G4, COORD_H4, COORD_NO, COORD_NO}},
    {8, {COORD_D1, COORD_D2, COORD_D3, COORD_D4, COORD_D5, COORD_D6, COORD_D7, COORD_D8, COORD_NO, COORD_NO}},
    {8, {COORD_A5, COORD_B5, COORD_C5, COORD_D5, COORD_E5, COORD_F5, COORD_G5, COORD_H5, COORD_NO, COORD_NO}},
    {8, {COORD_E1, COORD_E2, COORD_E3, COORD_E4, COORD_E5, COORD_E6, COORD_E7, COORD_E8, COORD_NO, COORD_NO}},

    // 3 corner9
    {9, {COORD_A1, COORD_B1, COORD_C1, COORD_A2, COORD_B2, COORD_C2, COORD_A3, COORD_B3, COORD_C3, COORD_NO}},
    {9, {COORD_H1, COORD_G1, COORD_F1, COORD_H2, COORD_G2, COORD_F2, COORD_H3, COORD_G3, COORD_F3, COORD_NO}},
    {9, {COORD_A8, COORD_B8, COORD_C8, COORD_A7, COORD_B7, COORD_C7, COORD_A6, COORD_B6, COORD_C6, COORD_NO}},
    {9, {COORD_H8, COORD_G8, COORD_F8, COORD_H7, COORD_G7, COORD_F7, COORD_H6, COORD_G6, COORD_F6, COORD_NO}},

    // 4 edge + 2x
    {10, {COORD_B2, COORD_A1, COORD_B1, COORD_C1, COORD_D1, COORD_E1, COORD_F1, COORD_G1, COORD_H1, COORD_G2}},
    {10, {COORD_B2, COORD_A1, COORD_A2, COORD_A3, COORD_A4, COORD_A5, COORD_A6, COORD_A7, COORD_A8, COORD_B7}},
    {10, {COORD_B7, COORD_A8, COORD_B8, COORD_C8, COORD_D8, COORD_E8, COORD_F8, COORD_G8, COORD_H8, COORD_G7}},
    {10, {COORD_G2, COORD_H1, COORD_H2, COORD_H3, COORD_H4, COORD_H5, COORD_H6, COORD_H7, COORD_H8, COORD_G7}},

    // 5 d7+2corner+X
    {10, {COORD_A1, COORD_B1, COORD_C2, COORD_D3, COORD_E4, COORD_F5, COORD_G6, COORD_H7, COORD_H8, COORD_B7}},
    {10, {COORD_H1, COORD_H2, COORD_G3, COORD_F4, COORD_E5, COORD_D6, COORD_C7, COORD_B8, COORD_A8, COORD_B2}},
    {10, {COORD_H8, COORD_G8, COORD_F7, COORD_E6, COORD_D5, COORD_C4, COORD_B3, COORD_A2, COORD_A1, COORD_G2}},
    {10, {COORD_A8, COORD_A7, COORD_B6, COORD_C5, COORD_D4, COORD_E3, COORD_F2, COORD_G1, COORD_H1, COORD_G7}},

    // 6 corner + block
    {10, {COORD_A1, COORD_C1, COORD_D1, COORD_E1, COORD_F1, COORD_H1, COORD_C2, COORD_D2, COORD_E2, COORD_F2}},
    {10, {COORD_A1, COORD_A3, COORD_A4, COORD_A5, COORD_A6, COORD_A8, COORD_B3, COORD_B4, COORD_B5, COORD_B6}},
    {10, {COORD_A8, COORD_C8, COORD_D8, COORD_E8, COORD_F8, COORD_H8, COORD_C7, COORD_D7, COORD_E7, COORD_F7}},
    {10, {COORD_H1, COORD_H3, COORD_H4, COORD_H5, COORD_H6, COORD_H8, COORD_G3, COORD_G4, COORD_G5, COORD_G6}},

    // 7 cross
    {10, {COORD_A1, COORD_B2, COORD_C3, COORD_D4, COORD_B1, COORD_C2, COORD_D3, COORD_A2, COORD_B3, COORD_C4}},
    {10, {COORD_H1, COORD_G2, COORD_F3, COORD_E4, COORD_G1, COORD_F2, COORD_E3, COORD_H2, COORD_G3, COORD_F4}},
    {10, {COORD_A8, COORD_B7, COORD_C6, COORD_D5, COORD_B8, COORD_C7, COORD_D6, COORD_A7, COORD_B6, COORD_C5}},
    {10, {COORD_H8, COORD_G7, COORD_F6, COORD_E5, COORD_G8, COORD_F7, COORD_E6, COORD_H7, COORD_G6, COORD_F5}},

    // 8 d4+d5+BC
    {10, {COORD_A4, COORD_B3, COORD_C2, COORD_D1, COORD_E2, COORD_F3, COORD_G4, COORD_H5, COORD_C1, COORD_B1}},
    {10, {COORD_E8, COORD_F7, COORD_G6, COORD_H5, COORD_G4, COORD_F3, COORD_E2, COORD_D1, COORD_H6, COORD_H7}},
    {10, {COORD_D1, COORD_C2, COORD_B3, COORD_A4, COORD_B5, COORD_C6, COORD_D7, COORD_E8, COORD_A3, COORD_A2}},
    {10, {COORD_H5, COORD_G6, COORD_F7, COORD_E8, COORD_D7, COORD_C6, COORD_B5, COORD_A4, COORD_F8, COORD_G8}},
    {10, {COORD_H4, COORD_G3, COORD_F2, COORD_E1, COORD_D2, COORD_C3, COORD_B4, COORD_A5, COORD_F1, COORD_G1}},
    {10, {COORD_A5, COORD_B6, COORD_C7, COORD_D8, COORD_E7, COORD_F6, COORD_G5, COORD_H4, COORD_C8, COORD_B8}},
    {10, {COORD_E1, COORD_F2, COORD_G3, COORD_H4, COORD_G5, COORD_F6, COORD_E7, COORD_D8, COORD_H3, COORD_H2}},
    {10, {COORD_D8, COORD_C7, COORD_B6, COORD_A5, COORD_B4, COORD_C3, COORD_D2, COORD_E1, COORD_A6, COORD_A7}},

    // 9 d3+d6+corner+C
    {10, {COORD_A3, COORD_B2, COORD_C1, COORD_D2, COORD_E3, COORD_F4, COORD_G5, COORD_H6, COORD_A1, COORD_B1}},
    {10, {COORD_F8, COORD_G7, COORD_H6, COORD_G5, COORD_F4, COORD_E3, COORD_D2, COORD_C1, COORD_H8, COORD_H7}},
    {10, {COORD_C1, COORD_B2, COORD_A3, COORD_B4, COORD_C5, COORD_D6, COORD_E7, COORD_F8, COORD_A1, COORD_A2}},
    {10, {COORD_H6, COORD_G7, COORD_F8, COORD_E7, COORD_D6, COORD_C5, COORD_B4, COORD_A3, COORD_H8, COORD_G8}},
    {10, {COORD_H3, COORD_G2, COORD_F1, COORD_E2, COORD_D3, COORD_C4, COORD_B5, COORD_A6, COORD_H1, COORD_G1}},
    {10, {COORD_A6, COORD_B7, COORD_C8, COORD_D7, COORD_E6, COORD_F5, COORD_G4, COORD_H3, COORD_A8, COORD_B8}},
    {10, {COORD_F1, COORD_G2, COORD_H3, COORD_G4, COORD_F5, COORD_E6, COORD_D7, COORD_C8, COORD_H1, COORD_H2}},
    {10, {COORD_C8, COORD_B7, COORD_A6, COORD_B5, COORD_C4, COORD_D3, COORD_E2, COORD_F1, COORD_A8, COORD_A7}},

    // 10 d8+BC
    {10, {COORD_A1, COORD_B2, COORD_C3, COORD_D4, COORD_E5, COORD_F6, COORD_G7, COORD_H8, COORD_B1, COORD_C1}},
    {10, {COORD_H8, COORD_G7, COORD_F6, COORD_E5, COORD_D4, COORD_C3, COORD_B2, COORD_A1, COORD_H7, COORD_H6}},
    {10, {COORD_A1, COORD_B2, COORD_C3, COORD_D4, COORD_E5, COORD_F6, COORD_G7, COORD_H8, COORD_A2, COORD_A3}},
    {10, {COORD_H8, COORD_G7, COORD_F6, COORD_E5, COORD_D4, COORD_C3, COORD_B2, COORD_A1, COORD_G8, COORD_F8}},
    {10, {COORD_H1, COORD_G2, COORD_F3, COORD_E4, COORD_D5, COORD_C6, COORD_B7, COORD_A8, COORD_G1, COORD_F1}},
    {10, {COORD_A8, COORD_B7, COORD_C6, COORD_D5, COORD_E4, COORD_F3, COORD_G2, COORD_H1, COORD_B8, COORD_C8}},
    {10, {COORD_H1, COORD_G2, COORD_F3, COORD_E4, COORD_D5, COORD_C6, COORD_B7, COORD_A8, COORD_H2, COORD_H3}},
    {10, {COORD_A8, COORD_B7, COORD_C6, COORD_D5, COORD_E4, COORD_F3, COORD_G2, COORD_H1, COORD_A7, COORD_A6}},

    // 11 corner+inner triangle
    {10, {COORD_A1, COORD_B2, COORD_C2, COORD_D2, COORD_E2, COORD_C3, COORD_D3, COORD_E3, COORD_D4, COORD_E4}},
    {10, {COORD_H8, COORD_G7, COORD_G6, COORD_G5, COORD_G4, COORD_F6, COORD_F5, COORD_F4, COORD_E5, COORD_E4}},
    {10, {COORD_A1, COORD_B2, COORD_B3, COORD_B4, COORD_B5, COORD_C3, COORD_C4, COORD_C5, COORD_D4, COORD_D5}},
    {10, {COORD_H8, COORD_G7, COORD_F7, COORD_E7, COORD_D7, COORD_F6, COORD_E6, COORD_D6, COORD_E5, COORD_D5}},
    {10, {COORD_H1, COORD_G2, COORD_F2, COORD_E2, COORD_D2, COORD_F3, COORD_E3, COORD_D3, COORD_E4, COORD_D4}},
    {10, {COORD_A8, COORD_B7, COORD_C7, COORD_D7, COORD_E7, COORD_C6, COORD_D6, COORD_E6, COORD_D5, COORD_E5}},
    {10, {COORD_H1, COORD_G2, COORD_G3, COORD_G4, COORD_G5, COORD_F3, COORD_F4, COORD_F5, COORD_E4, COORD_E5}},
    {10, {COORD_A8, COORD_B7, COORD_B6, COORD_B5, COORD_B4, COORD_C6, COORD_C5, COORD_C4, COORD_D5, COORD_D4}},

    // 12 edge shape
    {10, {COORD_A1, COORD_B1, COORD_C1, COORD_D1, COORD_E1, COORD_F1, COORD_G1, COORD_A2, COORD_B2, COORD_C2}},
    {10, {COORD_H8, COORD_H7, COORD_H6, COORD_H5, COORD_H4, COORD_H3, COORD_H2, COORD_G8, COORD_G7, COORD_G6}},
    {10, {COORD_A1, COORD_A2, COORD_A3, COORD_A4, COORD_A5, COORD_A6, COORD_A7, COORD_B1, COORD_B2, COORD_B3}},
    {10, {COORD_H8, COORD_G8, COORD_F8, COORD_E8, COORD_D8, COORD_C8, COORD_B8, COORD_H7, COORD_G7, COORD_F7}},
    {10, {COORD_H1, COORD_G1, COORD_F1, COORD_E1, COORD_D1, COORD_C1, COORD_B1, COORD_H2, COORD_G2, COORD_F2}},
    {10, {COORD_A8, COORD_B8, COORD_C8, COORD_D8, COORD_E8, COORD_F8, COORD_G8, COORD_A7, COORD_B7, COORD_C7}},
    {10, {COORD_H1, COORD_H2, COORD_H3, COORD_H4, COORD_H5, COORD_H6, COORD_H7, COORD_G1, COORD_G2, COORD_G3}},
    {10, {COORD_A8, COORD_A7, COORD_A6, COORD_A5, COORD_A4, COORD_A3, COORD_A2, COORD_B8, COORD_B7, COORD_B6}},

    // 13 corner10
    {10, {COORD_A1, COORD_B1, COORD_C1, COORD_D1, COORD_E1, COORD_A2, COORD_B2, COORD_C2, COORD_D2, COORD_E2}},
    {10, {COORD_H8, COORD_H7, COORD_H6, COORD_H5, COORD_H4, COORD_G8, COORD_G7, COORD_G6, COORD_G5, COORD_G4}},
    {10, {COORD_A1, COORD_A2, COORD_A3, COORD_A4, COORD_A5, COORD_B1, COORD_B2, COORD_B3, COORD_B4, COORD_B5}},
    {10, {COORD_H8, COORD_G8, COORD_F8, COORD_E8, COORD_D8, COORD_H7, COORD_G7, COORD_F7, COORD_E7, COORD_D7}},
    {10, {COORD_H1, COORD_G1, COORD_F1, COORD_E1, COORD_D1, COORD_H2, COORD_G2, COORD_F2, COORD_E2, COORD_D2}},
    {10, {COORD_A8, COORD_B8, COORD_C8, COORD_D8, COORD_E8, COORD_A7, COORD_B7, COORD_C7, COORD_D7, COORD_E7}},
    {10, {COORD_H1, COORD_H2, COORD_H3, COORD_H4, COORD_H5, COORD_G1, COORD_G2, COORD_G3, COORD_G4, COORD_G5}},
    {10, {COORD_A8, COORD_A7, COORD_A6, COORD_A5, COORD_A4, COORD_B8, COORD_B7, COORD_B6, COORD_B5, COORD_B4}}
};

constexpr Coord_to_feature coord_to_feature[HW2] = {
    {20, {{15, P38}, {18, P31}, {19, P31}, {20, P31}, {22, P39}, {26, P34}, {27, P34}, {31, P39}, {41, P31}, {43, P31}, {48, P32}, {49, P39}, {50, P32}, {51, P39}, {57, P39}, {59, P39}, {65, P39}, {67, P39}, {73, P39}, {75, P39}}}, // COORD_H8
    {13, {{ 3, P30}, {15, P37}, {18, P32}, {22, P38}, {31, P35}, {35, P30}, {43, P30}, {51, P31}, {65, P32}, {67, P38}, {69, P33}, {73, P34}, {75, P38}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_G8
    {12, {{ 7, P30}, {15, P36}, {18, P33}, {26, P35}, {35, P31}, {41, P39}, {42, P32}, {43, P37}, {51, P30}, {67, P37}, {69, P34}, {75, P37}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_F8
    {10, {{11, P30}, {18, P34}, {26, P36}, {33, P39}, {34, P32}, {35, P36}, {67, P36}, {69, P35}, {75, P36}, {77, P35}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_E8
    {10, {{ 9, P30}, {18, P35}, {26, P37}, {37, P36}, {38, P32}, {39, P39}, {67, P35}, {69, P36}, {75, P35}, {77, P36}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_D8
    {12, {{ 5, P30}, {14, P36}, {18, P36}, {26, P38}, {37, P31}, {45, P37}, {46, P32}, {47, P39}, {53, P30}, {67, P34}, {69, P37}, {77, P37}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_C8
    {13, {{ 1, P30}, {14, P37}, {18, P37}, {21, P32}, {30, P35}, {37, P30}, {45, P30}, {53, P31}, {67, P33}, {69, P38}, {71, P32}, {77, P38}, {79, P34}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_B8
    {20, {{14, P38}, {17, P31}, {18, P38}, {21, P31}, {23, P39}, {25, P34}, {26, P39}, {30, P39}, {45, P31}, {47, P31}, {52, P32}, {53, P39}, {54, P32}, {55, P39}, {61, P39}, {63, P39}, {69, P39}, {71, P39}, {77, P39}, {79, P39}}}, // COORD_A8
    {13, {{ 2, P30}, {15, P35}, {19, P32}, {20, P32}, {31, P32}, {33, P30}, {41, P30}, {49, P31}, {65, P38}, {67, P32}, {70, P33}, {73, P38}, {75, P34}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_H7
    {19, {{ 2, P31}, { 3, P31}, {15, P34}, {18, P30}, {19, P30}, {23, P30}, {31, P38}, {41, P38}, {43, P38}, {48, P33}, {49, P38}, {50, P33}, {51, P38}, {57, P38}, {59, P38}, {65, P31}, {67, P31}, {73, P33}, {75, P33}, { 0, PNO}}}, // COORD_G7
    {11, {{ 2, P32}, { 7, P31}, {15, P33}, {22, P37}, {26, P30}, {31, P34}, {33, P38}, {35, P37}, {59, P37}, {67, P30}, {75, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_F7
    {11, {{ 2, P33}, {11, P31}, {26, P31}, {37, P35}, {38, P33}, {42, P33}, {43, P36}, {59, P36}, {61, P35}, {75, P31}, {77, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_E7
    {11, {{ 2, P34}, { 9, P31}, {26, P32}, {34, P33}, {35, P35}, {45, P36}, {46, P33}, {59, P35}, {61, P36}, {75, P30}, {77, P31}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_D7
    {11, {{ 2, P35}, { 5, P31}, {14, P33}, {21, P33}, {26, P33}, {30, P34}, {37, P37}, {39, P38}, {61, P37}, {69, P30}, {77, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_C7
    {19, {{ 1, P31}, { 2, P36}, {14, P34}, {17, P30}, {18, P39}, {20, P30}, {30, P38}, {45, P38}, {47, P38}, {52, P33}, {53, P38}, {54, P33}, {55, P38}, {61, P38}, {63, P38}, {69, P31}, {71, P31}, {77, P33}, {79, P33}, { 0, PNO}}}, // COORD_B7
    {13, {{ 2, P37}, {14, P35}, {17, P32}, {23, P38}, {30, P32}, {39, P30}, {47, P30}, {55, P31}, {66, P33}, {69, P32}, {71, P38}, {77, P34}, {79, P38}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_A7
    {12, {{ 6, P30}, {15, P32}, {19, P33}, {27, P35}, {33, P31}, {40, P32}, {41, P37}, {43, P39}, {49, P30}, {65, P37}, {70, P34}, {73, P37}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_H6
    {11, {{ 3, P32}, { 6, P31}, {15, P31}, {20, P33}, {27, P30}, {31, P31}, {33, P37}, {35, P38}, {57, P37}, {65, P30}, {73, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_G6
    {12, {{ 6, P32}, { 7, P32}, {15, P30}, {31, P37}, {37, P34}, {38, P34}, {48, P34}, {49, P37}, {50, P34}, {51, P37}, {57, P34}, {59, P34}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_F6
    { 8, {{ 6, P33}, {11, P32}, {22, P36}, {31, P33}, {45, P35}, {46, P34}, {59, P33}, {61, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_E6
    { 8, {{ 6, P34}, { 9, P32}, {21, P34}, {30, P33}, {42, P34}, {43, P35}, {59, P32}, {61, P33}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_D6
    {12, {{ 5, P32}, { 6, P35}, {14, P30}, {30, P37}, {34, P34}, {35, P34}, {52, P34}, {53, P37}, {54, P34}, {55, P37}, {61, P34}, {63, P34}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_C6
    {11, {{ 1, P32}, { 6, P36}, {14, P31}, {23, P37}, {25, P30}, {30, P31}, {37, P38}, {39, P37}, {63, P37}, {71, P30}, {79, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_B6
    {12, {{ 6, P37}, {14, P32}, {17, P33}, {25, P35}, {39, P31}, {44, P32}, {45, P39}, {47, P37}, {55, P30}, {66, P34}, {71, P37}, {79, P37}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_A6
    {10, {{10, P30}, {19, P34}, {27, P36}, {32, P32}, {33, P36}, {35, P39}, {65, P36}, {70, P35}, {73, P36}, {78, P35}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_H5
    {11, {{ 3, P33}, {10, P31}, {27, P31}, {37, P33}, {38, P35}, {40, P33}, {41, P36}, {57, P36}, {62, P35}, {73, P31}, {78, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_G5
    { 8, {{ 7, P33}, {10, P32}, {20, P34}, {31, P30}, {45, P34}, {46, P35}, {57, P33}, {62, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_F5
    {12, {{10, P33}, {11, P33}, {21, P35}, {31, P36}, {48, P35}, {49, P36}, {50, P35}, {51, P36}, {57, P31}, {59, P31}, {61, P30}, {62, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_E5
    {12, {{ 9, P33}, {10, P34}, {22, P35}, {30, P36}, {52, P35}, {53, P36}, {54, P35}, {55, P36}, {58, P30}, {59, P30}, {61, P31}, {63, P31}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_D5
    { 8, {{ 5, P33}, {10, P35}, {23, P36}, {30, P30}, {42, P35}, {43, P34}, {58, P32}, {63, P33}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_C5
    {11, {{ 1, P33}, {10, P36}, {25, P31}, {34, P35}, {35, P33}, {44, P33}, {47, P36}, {58, P35}, {63, P36}, {74, P30}, {79, P31}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_B5
    {10, {{10, P37}, {17, P34}, {25, P36}, {36, P32}, {37, P39}, {39, P36}, {66, P35}, {71, P36}, {74, P35}, {79, P36}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_A5
    {10, {{ 8, P30}, {19, P35}, {27, P37}, {36, P39}, {37, P32}, {38, P36}, {65, P35}, {70, P36}, {73, P35}, {78, P36}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_H4
    {11, {{ 3, P34}, { 8, P31}, {27, P32}, {32, P33}, {33, P35}, {45, P33}, {46, P36}, {57, P35}, {62, P36}, {73, P30}, {78, P31}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_G4
    { 8, {{ 7, P34}, { 8, P32}, {21, P36}, {29, P30}, {40, P34}, {41, P35}, {57, P32}, {62, P33}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_F4
    {12, {{ 8, P33}, {11, P34}, {20, P35}, {29, P36}, {52, P36}, {53, P35}, {54, P36}, {55, P35}, {56, P30}, {57, P30}, {60, P31}, {62, P31}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_E4
    {12, {{ 8, P34}, { 9, P34}, {23, P35}, {28, P36}, {48, P36}, {49, P35}, {50, P36}, {51, P35}, {56, P31}, {58, P31}, {60, P30}, {63, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_D4
    { 8, {{ 5, P34}, { 8, P35}, {22, P34}, {28, P30}, {44, P34}, {47, P35}, {58, P33}, {63, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_C4
    {11, {{ 1, P34}, { 8, P36}, {25, P32}, {36, P33}, {39, P35}, {42, P36}, {43, P33}, {58, P36}, {63, P35}, {74, P31}, {79, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_B4
    {10, {{ 8, P37}, {17, P35}, {25, P37}, {32, P39}, {34, P36}, {35, P32}, {66, P36}, {71, P35}, {74, P36}, {79, P35}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_A4
    {12, {{ 4, P30}, {13, P32}, {19, P36}, {27, P38}, {38, P31}, {44, P39}, {45, P32}, {46, P37}, {54, P30}, {65, P34}, {70, P37}, {78, P37}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_H3
    {11, {{ 3, P35}, { 4, P31}, {13, P31}, {21, P37}, {27, P33}, {29, P31}, {36, P38}, {38, P37}, {62, P37}, {70, P30}, {78, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_G3
    {12, {{ 4, P32}, { 7, P35}, {13, P30}, {29, P37}, {32, P34}, {33, P34}, {52, P37}, {53, P34}, {54, P37}, {55, P34}, {60, P34}, {62, P34}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_F3
    { 8, {{ 4, P33}, {11, P35}, {23, P34}, {29, P33}, {40, P35}, {41, P34}, {56, P32}, {60, P33}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_E3
    { 8, {{ 4, P34}, { 9, P35}, {20, P36}, {28, P33}, {44, P35}, {47, P34}, {56, P33}, {60, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_D3
    {12, {{ 4, P35}, { 5, P35}, {12, P30}, {28, P37}, {36, P34}, {39, P34}, {48, P37}, {49, P34}, {50, P37}, {51, P34}, {56, P34}, {58, P34}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_C3
    {11, {{ 1, P35}, { 4, P36}, {12, P31}, {22, P33}, {25, P33}, {28, P31}, {32, P38}, {34, P37}, {58, P37}, {66, P30}, {74, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_B3
    {12, {{ 4, P37}, {12, P32}, {17, P36}, {25, P38}, {34, P31}, {40, P39}, {42, P37}, {43, P32}, {50, P30}, {66, P37}, {71, P34}, {74, P37}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_A3
    {13, {{ 0, P30}, {13, P35}, {19, P37}, {21, P38}, {29, P32}, {38, P30}, {46, P30}, {54, P31}, {65, P33}, {68, P32}, {70, P38}, {76, P34}, {78, P38}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_H2
    {19, {{ 0, P31}, { 3, P36}, {13, P34}, {16, P30}, {19, P39}, {22, P30}, {29, P38}, {44, P38}, {46, P38}, {52, P38}, {53, P33}, {54, P38}, {55, P33}, {60, P38}, {62, P38}, {68, P31}, {70, P31}, {76, P33}, {78, P33}, { 0, PNO}}}, // COORD_G2
    {11, {{ 0, P32}, { 7, P36}, {13, P33}, {23, P33}, {24, P30}, {29, P34}, {36, P37}, {38, P38}, {60, P37}, {68, P30}, {76, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_F2
    {11, {{ 0, P33}, {11, P36}, {24, P31}, {32, P35}, {33, P33}, {44, P36}, {47, P33}, {56, P35}, {60, P36}, {72, P30}, {76, P31}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_E2
    {11, {{ 0, P34}, { 9, P36}, {24, P32}, {36, P35}, {39, P33}, {40, P36}, {41, P33}, {56, P36}, {60, P35}, {72, P31}, {76, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_D2
    {11, {{ 0, P35}, { 5, P36}, {12, P33}, {20, P37}, {24, P33}, {28, P34}, {32, P37}, {34, P38}, {56, P37}, {64, P30}, {72, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_C2
    {19, {{ 0, P36}, { 1, P36}, {12, P34}, {16, P39}, {17, P39}, {21, P30}, {28, P38}, {40, P38}, {42, P38}, {48, P38}, {49, P33}, {50, P38}, {51, P33}, {56, P38}, {58, P38}, {64, P31}, {66, P31}, {72, P33}, {74, P33}, { 0, PNO}}}, // COORD_B2
    {13, {{ 0, P37}, {12, P35}, {17, P37}, {22, P32}, {28, P32}, {34, P30}, {42, P30}, {50, P31}, {64, P32}, {66, P38}, {71, P33}, {72, P34}, {74, P38}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_A2
    {20, {{13, P38}, {16, P31}, {19, P38}, {21, P39}, {23, P31}, {24, P34}, {27, P39}, {29, P39}, {44, P31}, {46, P31}, {52, P39}, {53, P32}, {54, P39}, {55, P32}, {60, P39}, {62, P39}, {68, P39}, {70, P39}, {76, P39}, {78, P39}}}, // COORD_H1
    {13, {{ 3, P37}, {13, P37}, {16, P32}, {23, P32}, {29, P35}, {36, P30}, {44, P30}, {52, P31}, {64, P33}, {68, P38}, {70, P32}, {76, P38}, {78, P34}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_G1
    {12, {{ 7, P37}, {13, P36}, {16, P33}, {24, P35}, {36, P31}, {44, P37}, {46, P39}, {47, P32}, {52, P30}, {64, P34}, {68, P37}, {76, P37}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_F1
    {10, {{11, P37}, {16, P34}, {24, P36}, {36, P36}, {38, P39}, {39, P32}, {64, P35}, {68, P36}, {72, P35}, {76, P36}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_E1
    {10, {{ 9, P37}, {16, P35}, {24, P37}, {32, P36}, {33, P32}, {34, P39}, {64, P36}, {68, P35}, {72, P36}, {76, P35}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_D1
    {12, {{ 5, P37}, {12, P36}, {16, P36}, {24, P38}, {32, P31}, {40, P37}, {41, P32}, {42, P39}, {48, P30}, {64, P37}, {68, P34}, {72, P37}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_C1
    {13, {{ 1, P37}, {12, P37}, {16, P37}, {20, P38}, {28, P35}, {32, P30}, {40, P30}, {48, P31}, {64, P38}, {66, P32}, {68, P33}, {72, P38}, {74, P34}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_B1
    {20, {{12, P38}, {16, P38}, {17, P38}, {20, P39}, {22, P31}, {24, P39}, {25, P39}, {28, P39}, {40, P31}, {42, P31}, {48, P39}, {49, P32}, {50, P39}, {51, P32}, {56, P39}, {58, P39}, {64, P39}, {66, P39}, {72, P39}, {74, P39}}}, // COORD_A1
};

constexpr Coord_to_feature coord_to_feature_move_ordering_end[HW2] = {
    { 7, {{ 2, P31}, { 3, P31}, { 4, P31}, { 6, P39}, {10, P34}, {11, P34}, {15, P39}}}, // COORD_H8
    { 3, {{ 2, P32}, { 6, P38}, {15, P35}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_G8
    { 2, {{ 2, P33}, {10, P35}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_F8
    { 2, {{ 2, P34}, {10, P36}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_E8
    { 2, {{ 2, P35}, {10, P37}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_D8
    { 2, {{ 2, P36}, {10, P38}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_C8
    { 3, {{ 2, P37}, { 5, P32}, {14, P35}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_B8
    { 7, {{ 1, P31}, { 2, P38}, { 5, P31}, { 7, P39}, { 9, P34}, {10, P39}, {14, P39}}}, // COORD_A8
    { 3, {{ 3, P32}, { 4, P32}, {15, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_H7
    { 4, {{ 2, P30}, { 3, P30}, { 7, P30}, {15, P38}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_G7
    { 3, {{ 6, P37}, {10, P30}, {15, P34}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_F7
    { 1, {{10, P31}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_E7
    { 1, {{10, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_D7
    { 3, {{ 5, P33}, {10, P33}, {14, P34}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_C7
    { 4, {{ 1, P30}, { 2, P39}, { 4, P30}, {14, P38}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_B7
    { 3, {{ 1, P32}, { 7, P38}, {14, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_A7
    { 2, {{ 3, P33}, {11, P35}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_H6
    { 3, {{ 4, P33}, {11, P30}, {15, P31}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_G6
    { 1, {{15, P37}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_F6
    { 2, {{ 6, P36}, {15, P33}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_E6
    { 2, {{ 5, P34}, {14, P33}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_D6
    { 1, {{14, P37}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_C6
    { 3, {{ 7, P37}, { 9, P30}, {14, P31}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_B6
    { 2, {{ 1, P33}, { 9, P35}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_A6
    { 2, {{ 3, P34}, {11, P36}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_H5
    { 1, {{11, P31}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_G5
    { 2, {{ 4, P34}, {15, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_F5
    { 2, {{ 5, P35}, {15, P36}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_E5
    { 2, {{ 6, P35}, {14, P36}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_D5
    { 2, {{ 7, P36}, {14, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_C5
    { 1, {{ 9, P31}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_B5
    { 2, {{ 1, P34}, { 9, P36}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_A5
    { 2, {{ 3, P35}, {11, P37}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_H4
    { 1, {{11, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_G4
    { 2, {{ 5, P36}, {13, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_F4
    { 2, {{ 4, P35}, {13, P36}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_E4
    { 2, {{ 7, P35}, {12, P36}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_D4
    { 2, {{ 6, P34}, {12, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_C4
    { 1, {{ 9, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_B4
    { 2, {{ 1, P35}, { 9, P37}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_A4
    { 2, {{ 3, P36}, {11, P38}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_H3
    { 3, {{ 5, P37}, {11, P33}, {13, P31}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_G3
    { 1, {{13, P37}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_F3
    { 2, {{ 7, P34}, {13, P33}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_E3
    { 2, {{ 4, P36}, {12, P33}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_D3
    { 1, {{12, P37}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_C3
    { 3, {{ 6, P33}, { 9, P33}, {12, P31}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_B3
    { 2, {{ 1, P36}, { 9, P38}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_A3
    { 3, {{ 3, P37}, { 5, P38}, {13, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_H2
    { 4, {{ 0, P30}, { 3, P39}, { 6, P30}, {13, P38}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_G2
    { 3, {{ 7, P33}, { 8, P30}, {13, P34}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_F2
    { 1, {{ 8, P31}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_E2
    { 1, {{ 8, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_D2
    { 3, {{ 4, P37}, { 8, P33}, {12, P34}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_C2
    { 4, {{ 0, P39}, { 1, P39}, { 5, P30}, {12, P38}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_B2
    { 3, {{ 1, P37}, { 6, P32}, {12, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_A2
    { 7, {{ 0, P31}, { 3, P38}, { 5, P39}, { 7, P31}, { 8, P34}, {11, P39}, {13, P39}}}, // COORD_H1
    { 3, {{ 0, P32}, { 7, P32}, {13, P35}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_G1
    { 2, {{ 0, P33}, { 8, P35}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_F1
    { 2, {{ 0, P34}, { 8, P36}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_E1
    { 2, {{ 0, P35}, { 8, P37}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_D1
    { 2, {{ 0, P36}, { 8, P38}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_C1
    { 3, {{ 0, P37}, { 4, P38}, {12, P35}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_B1
    { 7, {{ 0, P38}, { 1, P38}, { 4, P39}, { 6, P31}, { 8, P39}, { 9, P39}, {12, P39}}}, // COORD_A1
};

constexpr int pattern_sizes[N_PATTERNS] = {8, 8, 8, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10};

/*
    @brief feature to pattern
*/
constexpr int feature_to_pattern[N_PATTERN_FEATURES] = {
    0, 0, 0, 0, 
    1, 1, 1, 1, 
    2, 2, 2, 2, 
    3, 3, 3, 3, 
    4, 4, 4, 4, 
    5, 5, 5, 5, 
    6, 6, 6, 6, 
    7, 7, 7, 7, 
    8, 8, 8, 8, 8, 8, 8, 8, 
    9, 9, 9, 9, 9, 9, 9, 9, 
    10, 10, 10, 10, 10, 10, 10, 10, 
    11, 11, 11, 11, 11, 11, 11, 11, 
    12, 12, 12, 12, 12, 12, 12, 12, 
    13, 13, 13, 13, 13, 13, 13, 13
};

/*
    @brief evaluation parameters
*/
int16_t pattern_arr[2][N_PHASES][N_PATTERNS][MAX_EVALUATE_IDX];
int16_t eval_num_arr[N_PHASES][MAX_STONE_NUM];
int16_t pattern_arr_move_ordering_end[2][N_PATTERNS][MAX_EVALUATE_IDX];

/*
    @brief used for unzipping the evaluation function

    This function swaps the player in the index

    @param i                    index representing the discs in a pattern
    @param pattern_size         size of the pattern
    @return swapped index
*/
inline int swap_player_idx(int i, int pattern_size) {
    int j, ri = i;
    for (j = 0; j < pattern_size; ++j) {
        if ((i / pow3[j]) % 3 == 0) {
            ri += pow3[j];
        } else if ((i / pow3[j]) % 3 == 1) {
            ri -= pow3[j];
        }
    }
    return ri;
}

/*
    @brief used for unzipping the evaluation function

    @param phase_idx            evaluation phase
    @param pattern_idx          evaluation pattern's index
    @param siz                  size of the pattern
*/
void init_pattern_arr_rev(int phase_idx, int pattern_idx, int siz) {
    for (int i = 0; i < (int)pow3[siz]; ++i) {
        int ri = swap_player_idx(i, siz);
        pattern_arr[1][phase_idx][pattern_idx][ri] = pattern_arr[0][phase_idx][pattern_idx][i];
    }
}

/*
    @brief initialize the evaluation function

    @param file                 evaluation file name
    @return evaluation function conpletely initialized?
*/
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
        for (int pattern_idx = 0; pattern_idx < N_PATTERNS; ++pattern_idx) {
            std::memcpy(pattern_arr[0][phase_idx][pattern_idx], &unzipped_params[param_idx], sizeof(short) * pow3[pattern_sizes[pattern_idx]]);
            param_idx += pow3[pattern_sizes[pattern_idx]];
        }
        std::memcpy(eval_num_arr[phase_idx], &unzipped_params[param_idx], sizeof(short) * MAX_STONE_NUM);
        param_idx += MAX_STONE_NUM;
    }
    if (thread_pool.size() >= 2) {
        std::future<void> tasks[N_PHASES * N_PATTERNS];
        int i = 0;
        for (int phase_idx = 0; phase_idx < N_PHASES; ++phase_idx) {
            for (int pattern_idx = 0; pattern_idx < N_PATTERNS; ++pattern_idx) {
                bool pushed = false;
                while (!pushed) {
                    tasks[i] = thread_pool.push(&pushed, std::bind(init_pattern_arr_rev, phase_idx, pattern_idx, pattern_sizes[pattern_idx]));
                }
                ++i;
            }
        }
        for (std::future<void> &task: tasks) {
            task.get();
        }
    } else{
        for (int phase_idx = 0; phase_idx < N_PHASES; ++phase_idx) {
            for (int pattern_idx = 0; pattern_idx < N_PATTERNS; ++pattern_idx) {
                init_pattern_arr_rev(phase_idx, pattern_idx, pattern_sizes[pattern_idx]);
            }
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
    constexpr int pattern_sizes[N_PATTERNS_MO_END] = {10, 10, 10, 10};
    for (int pattern_idx = 0; pattern_idx < N_PATTERNS_MO_END; ++pattern_idx) {
        if (fread(pattern_arr_move_ordering_end[0][pattern_idx], 2, pow3[pattern_sizes[pattern_idx]], fp) < pow3[pattern_sizes[pattern_idx]]) {
            std::cerr << "[ERROR] [FATAL] evaluation file file for move ordering broken" << std::endl;
            fclose(fp);
            return false;
        }
    }
    for (int pattern_idx = 0; pattern_idx < N_PATTERNS_MO_END; ++pattern_idx) {
        for (int i = 0; i < (int)pow3[pattern_sizes[pattern_idx]]; ++i) {
            int ri = swap_player_idx(i, pattern_sizes[pattern_idx]);
            pattern_arr_move_ordering_end[1][pattern_idx][ri] = pattern_arr_move_ordering_end[0][pattern_idx][i];
        }
    }
    return true;
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
inline int calc_pattern(const int phase_idx, Eval_search *eval) {
    int res = 0;
    for (int i = 0; i < N_PATTERN_FEATURES; ++i) {
        res += pattern_arr[eval->reversed[eval->feature_idx]][phase_idx][feature_to_pattern[i]][eval->features[eval->feature_idx][i]];
    }
    return res;
}

/*
    @brief pattern evaluation

    @param phase_idx            evaluation phase
    @param search               search information
    @return pattern evaluation value
*/
inline int calc_pattern_move_ordering_end(Eval_search *eval) {
    int res = 0;
    for (int i = EVAL_IDX_START_MOVE_ORDERING_END; i < EVAL_IDX_END_MOVE_ORDERING_END; ++i) {
        res += pattern_arr_move_ordering_end[eval->reversed[eval->feature_idx]][feature_to_pattern[i] - EVAL_FEATURE_START_MOVE_ORDERING_END][eval->features[eval->feature_idx][i]];
    }
    return res;
}

inline void calc_eval_features(Board *board, Eval_search *eval);

/*
    @brief midgame evaluation function

    @param b                    board
    @return evaluation value
*/
inline int mid_evaluate(Board *board) {
    Search search(board);
    calc_eval_features(board, &search.eval);
    int phase_idx, num0;
    phase_idx = search.phase();
    num0 = pop_count_ull(search.board.player);
    int res = calc_pattern(phase_idx, &search.eval) + eval_num_arr[phase_idx][num0];
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
    int res = calc_pattern(phase_idx, &search->eval) + eval_num_arr[phase_idx][num0];
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
    int res = calc_pattern_move_ordering_end(&search->eval);
    res += res >= 0 ? STEP_2 : -STEP_2;
    res /= STEP;
    return res;
}

inline uint_fast16_t pick_pattern_idx(const uint_fast8_t b_arr[], const Feature_to_coord *f) {
    uint_fast16_t res = 0;
    for (int i = 0; i < f->n_cells; ++i) {
        res *= 3;
        res += b_arr[HW2_M1 - f->cells[i]];
    }
    return res;
}

inline void calc_eval_features(Board *board, Eval_search *eval) {
    uint_fast8_t b_arr[HW2];
    board->translate_to_arr_player(b_arr);
    for (int i = 0; i < N_PATTERN_FEATURES; ++i) {
        eval->features[0][i] = pick_pattern_idx(b_arr, &feature_to_coord[i]);
    }
    eval->reversed[0] = 0;
    eval->feature_idx = 0;
}

inline void eval_move(Eval_search *eval, const Flip *flip) {
    uint_fast8_t i, cell;
    uint64_t f;
    for (i = 0; i < N_PATTERN_FEATURES; ++i) {
        eval->features[eval->feature_idx + 1][i] = eval->features[eval->feature_idx][i];
    }
    if (eval->reversed[eval->feature_idx]) {
        for (i = 0; i < MAX_CELL_PATTERNS && coord_to_feature[flip->pos].features[i].x; ++i) {
            eval->features[eval->feature_idx + 1][coord_to_feature[flip->pos].features[i].feature] -= coord_to_feature[flip->pos].features[i].x;
        }
        f = flip->flip;
        for (cell = first_bit(&f); f; cell = next_bit(&f)) {
            for (i = 0; i < MAX_CELL_PATTERNS && coord_to_feature[cell].features[i].x; ++i) {
                eval->features[eval->feature_idx + 1][coord_to_feature[cell].features[i].feature] += coord_to_feature[cell].features[i].x;
            }
        }
    } else{
        for (i = 0; i < MAX_CELL_PATTERNS && coord_to_feature[flip->pos].features[i].x; ++i) {
            eval->features[eval->feature_idx + 1][coord_to_feature[flip->pos].features[i].feature] -= 2 * coord_to_feature[flip->pos].features[i].x;
        }
        f = flip->flip;
        for (cell = first_bit(&f); f; cell = next_bit(&f)) {
            for (i = 0; i < MAX_CELL_PATTERNS && coord_to_feature[cell].features[i].x; ++i) {
                eval->features[eval->feature_idx + 1][coord_to_feature[cell].features[i].feature] -= coord_to_feature[cell].features[i].x;
            }
        }
    }
    eval->reversed[eval->feature_idx + 1] = eval->reversed[eval->feature_idx] ^ 1;
    ++eval->feature_idx;
}

inline void eval_undo(Eval_search *eval) {
    --eval->feature_idx;
}

/*
    @brief pass evaluation features

    @param search               search information
*/
inline void eval_pass(Eval_search *eval) {
    eval->reversed[eval->feature_idx] ^= 1;
}



inline void eval_move_endsearch(Eval_search *eval, const Flip *flip) {
    uint_fast8_t i, cell;
    uint64_t f;
    for (i = EVAL_IDX_START_MOVE_ORDERING_END; i < EVAL_IDX_END_MOVE_ORDERING_END; ++i) {
        eval->features[eval->feature_idx + 1][i] = eval->features[eval->feature_idx][i];
    }
    if (eval->reversed[eval->feature_idx]) {
        for (i = 0; i < MAX_CELL_PATTERNS_MOVE_ORDERING_END && coord_to_feature_move_ordering_end[flip->pos].features[i].x; ++i) {
            eval->features[eval->feature_idx + 1][coord_to_feature_move_ordering_end[flip->pos].features[i].feature + EVAL_IDX_START_MOVE_ORDERING_END] -= coord_to_feature_move_ordering_end[flip->pos].features[i].x;
        }
        f = flip->flip;
        for (cell = first_bit(&f); f; cell = next_bit(&f)) {
            for (i = 0; i < MAX_CELL_PATTERNS_MOVE_ORDERING_END && coord_to_feature_move_ordering_end[cell].features[i].x; ++i) {
                eval->features[eval->feature_idx + 1][coord_to_feature_move_ordering_end[cell].features[i].feature + EVAL_IDX_START_MOVE_ORDERING_END] += coord_to_feature_move_ordering_end[cell].features[i].x;
            }
        }
    } else{
        for (i = 0; i < MAX_CELL_PATTERNS_MOVE_ORDERING_END && coord_to_feature_move_ordering_end[flip->pos].features[i].x; ++i) {
            eval->features[eval->feature_idx + 1][coord_to_feature_move_ordering_end[flip->pos].features[i].feature + EVAL_IDX_START_MOVE_ORDERING_END] -= 2 * coord_to_feature_move_ordering_end[flip->pos].features[i].x;
        }
        f = flip->flip;
        for (cell = first_bit(&f); f; cell = next_bit(&f)) {
            for (i = 0; i < MAX_CELL_PATTERNS_MOVE_ORDERING_END && coord_to_feature_move_ordering_end[cell].features[i].x; ++i) {
                eval->features[eval->feature_idx + 1][coord_to_feature_move_ordering_end[cell].features[i].feature + EVAL_IDX_START_MOVE_ORDERING_END] -= coord_to_feature_move_ordering_end[cell].features[i].x;
            }
        }
    }
    eval->reversed[eval->feature_idx + 1] = eval->reversed[eval->feature_idx] ^ 1;
    ++eval->feature_idx;
}

inline void eval_undo_endsearch(Eval_search *eval) {
    --eval->feature_idx;
}

/*
    @brief pass evaluation features

    @param search               search information
*/
inline void eval_pass_endsearch(Eval_search *eval) {
    eval->reversed[eval->feature_idx] ^= 1;
}