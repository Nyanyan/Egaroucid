/*
    Egaroucid Project

    @date 2021-2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once
#include <iostream>
#include <fstream>
#include "setting.hpp"
#include "common.hpp"
#include "board.hpp"
#include "search.hpp"
#include "util.hpp"
#include "evaluate_const.hpp"

using namespace std;

#ifndef N_SYMMETRY_PATTERNS
    #define N_SYMMETRY_PATTERNS 46
#endif
#define MAX_PATTERN_CELLS 9
#define MAX_CELL_PATTERNS 14

#define STEP 256
#define STEP_2 128

#define SCORE_MAX 64

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
#define P31m 2
#define P32m 8
#define P33m 26
#define P34m 80
#define P35m 242
#define P36m 728
#define P37m 2186
#define P38m 6560
#define P39m 19682
#define P310m 59048

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

struct Feature_to_coord{
    int n_cells;
    uint_fast8_t cells[MAX_PATTERN_CELLS];
};

constexpr Feature_to_coord feature_to_coord[N_SYMMETRY_PATTERNS] = {
    {8, {COORD_A1, COORD_B1, COORD_C1, COORD_D1, COORD_E1, COORD_F1, COORD_G1, COORD_H1, COORD_NO}},
    {8, {COORD_H1, COORD_H2, COORD_H3, COORD_H4, COORD_H5, COORD_H6, COORD_H7, COORD_H8, COORD_NO}},
    {8, {COORD_H8, COORD_G8, COORD_F8, COORD_E8, COORD_D8, COORD_C8, COORD_B8, COORD_A8, COORD_NO}},
    {8, {COORD_A8, COORD_A7, COORD_A6, COORD_A5, COORD_A4, COORD_A3, COORD_A2, COORD_A1, COORD_NO}},
    {8, {COORD_A2, COORD_B2, COORD_C2, COORD_D2, COORD_E2, COORD_F2, COORD_G2, COORD_H2, COORD_NO}},
    {8, {COORD_B1, COORD_B2, COORD_B3, COORD_B4, COORD_B5, COORD_B6, COORD_B7, COORD_B8, COORD_NO}},
    {8, {COORD_A7, COORD_B7, COORD_C7, COORD_D7, COORD_E7, COORD_F7, COORD_G7, COORD_H7, COORD_NO}},
    {8, {COORD_G1, COORD_G2, COORD_G3, COORD_G4, COORD_G5, COORD_G6, COORD_G7, COORD_G8, COORD_NO}},
    {8, {COORD_A3, COORD_B3, COORD_C3, COORD_D3, COORD_E3, COORD_F3, COORD_G3, COORD_H3, COORD_NO}},
    {8, {COORD_C1, COORD_C2, COORD_C3, COORD_C4, COORD_C5, COORD_C6, COORD_C7, COORD_C8, COORD_NO}},
    {8, {COORD_A6, COORD_B6, COORD_C6, COORD_D6, COORD_E6, COORD_F6, COORD_G6, COORD_H6, COORD_NO}},
    {8, {COORD_F1, COORD_F2, COORD_F3, COORD_F4, COORD_F5, COORD_F6, COORD_F7, COORD_F8, COORD_NO}},
    {8, {COORD_A4, COORD_B4, COORD_C4, COORD_D4, COORD_E4, COORD_F4, COORD_G4, COORD_H4, COORD_NO}},
    {8, {COORD_D1, COORD_D2, COORD_D3, COORD_D4, COORD_D5, COORD_D6, COORD_D7, COORD_D8, COORD_NO}},
    {8, {COORD_A5, COORD_B5, COORD_C5, COORD_D5, COORD_E5, COORD_F5, COORD_G5, COORD_H5, COORD_NO}},
    {8, {COORD_E1, COORD_E2, COORD_E3, COORD_E4, COORD_E5, COORD_E6, COORD_E7, COORD_E8, COORD_NO}},
    {8, {COORD_A3, COORD_B2, COORD_C1, COORD_D2, COORD_E3, COORD_F4, COORD_G5, COORD_H6, COORD_NO}},
    {8, {COORD_F8, COORD_G7, COORD_H6, COORD_G5, COORD_F4, COORD_E3, COORD_D2, COORD_C1, COORD_NO}},
    {8, {COORD_C1, COORD_B2, COORD_A3, COORD_B4, COORD_C5, COORD_D6, COORD_E7, COORD_F8, COORD_NO}},
    {8, {COORD_H6, COORD_G7, COORD_F8, COORD_E7, COORD_D6, COORD_C5, COORD_B4, COORD_A3, COORD_NO}},
    {8, {COORD_H3, COORD_G2, COORD_F1, COORD_E2, COORD_D3, COORD_C4, COORD_B5, COORD_A6, COORD_NO}},
    {8, {COORD_A6, COORD_B7, COORD_C8, COORD_D7, COORD_E6, COORD_F5, COORD_G4, COORD_H3, COORD_NO}},
    {8, {COORD_F1, COORD_G2, COORD_H3, COORD_G4, COORD_F5, COORD_E6, COORD_D7, COORD_C8, COORD_NO}},
    {8, {COORD_C8, COORD_B7, COORD_A6, COORD_B5, COORD_C4, COORD_D3, COORD_E2, COORD_F1, COORD_NO}},
    {8, {COORD_A4, COORD_B3, COORD_C2, COORD_D1, COORD_E2, COORD_F3, COORD_G4, COORD_H5, COORD_NO}},
    {8, {COORD_E8, COORD_F7, COORD_G6, COORD_H5, COORD_G4, COORD_F3, COORD_E2, COORD_D1, COORD_NO}},
    {8, {COORD_D1, COORD_C2, COORD_B3, COORD_A4, COORD_B5, COORD_C6, COORD_D7, COORD_E8, COORD_NO}},
    {8, {COORD_H5, COORD_G6, COORD_F7, COORD_E8, COORD_D7, COORD_C6, COORD_B5, COORD_A4, COORD_NO}},
    {8, {COORD_H4, COORD_G3, COORD_F2, COORD_E1, COORD_D2, COORD_C3, COORD_B4, COORD_A5, COORD_NO}},
    {8, {COORD_A5, COORD_B6, COORD_C7, COORD_D8, COORD_E7, COORD_F6, COORD_G5, COORD_H4, COORD_NO}},
    {8, {COORD_E1, COORD_F2, COORD_G3, COORD_H4, COORD_G5, COORD_F6, COORD_E7, COORD_D8, COORD_NO}},
    {8, {COORD_D8, COORD_C7, COORD_B6, COORD_A5, COORD_B4, COORD_C3, COORD_D2, COORD_E1, COORD_NO}},
    {9, {COORD_A1, COORD_B1, COORD_C2, COORD_D3, COORD_E4, COORD_F5, COORD_G6, COORD_H7, COORD_H8}},
    {9, {COORD_H1, COORD_H2, COORD_G3, COORD_F4, COORD_E5, COORD_D6, COORD_C7, COORD_B8, COORD_A8}},
    {9, {COORD_H8, COORD_G8, COORD_F7, COORD_E6, COORD_D5, COORD_C4, COORD_B3, COORD_A2, COORD_A1}},
    {9, {COORD_A8, COORD_A7, COORD_B6, COORD_C5, COORD_D4, COORD_E3, COORD_F2, COORD_G1, COORD_H1}},
    {9, {COORD_A1, COORD_B2, COORD_C3, COORD_D4, COORD_E5, COORD_F6, COORD_G7, COORD_H8, COORD_G2}},
    {9, {COORD_H1, COORD_G2, COORD_F3, COORD_E4, COORD_D5, COORD_C6, COORD_B7, COORD_A8, COORD_G7}},
    {9, {COORD_H8, COORD_G7, COORD_F6, COORD_E5, COORD_D4, COORD_C3, COORD_B2, COORD_A1, COORD_B7}},
    {9, {COORD_A8, COORD_B7, COORD_C6, COORD_D5, COORD_E4, COORD_F3, COORD_G2, COORD_H1, COORD_B2}},
    {9, {COORD_A1, COORD_B1, COORD_C1, COORD_A2, COORD_B2, COORD_C2, COORD_A3, COORD_B3, COORD_C3}},
    {9, {COORD_H1, COORD_G1, COORD_F1, COORD_H2, COORD_G2, COORD_F2, COORD_H3, COORD_G3, COORD_F3}},
    {9, {COORD_A8, COORD_B8, COORD_C8, COORD_A7, COORD_B7, COORD_C7, COORD_A6, COORD_B6, COORD_C6}},
    {9, {COORD_H8, COORD_G8, COORD_F8, COORD_H7, COORD_G7, COORD_F7, COORD_H6, COORD_G6, COORD_F6}},
    {9, {COORD_E1, COORD_D1, COORD_C1, COORD_B1, COORD_A1, COORD_A2, COORD_A3, COORD_A4, COORD_A5}},
    {9, {COORD_H5, COORD_H4, COORD_H3, COORD_H2, COORD_H1, COORD_G1, COORD_F1, COORD_E1, COORD_D1}},
    {9, {COORD_D8, COORD_E8, COORD_F8, COORD_G8, COORD_H8, COORD_H7, COORD_H6, COORD_H5, COORD_H4}},
    {9, {COORD_A4, COORD_A5, COORD_A6, COORD_A7, COORD_A8, COORD_B8, COORD_C8, COORD_D8, COORD_E8}},
    {9, {COORD_A1, COORD_B1, COORD_C1, COORD_D1, COORD_A2, COORD_B2, COORD_C2, COORD_D2, COORD_H1}},
    {9, {COORD_H8, COORD_H7, COORD_H6, COORD_H5, COORD_G8, COORD_G7, COORD_G6, COORD_G5, COORD_H1}},
    {9, {COORD_A1, COORD_A2, COORD_A3, COORD_A4, COORD_B1, COORD_B2, COORD_B3, COORD_B4, COORD_A8}},
    {9, {COORD_H8, COORD_G8, COORD_F8, COORD_E8, COORD_H7, COORD_G7, COORD_F7, COORD_E7, COORD_A8}},
    {9, {COORD_H1, COORD_G1, COORD_F1, COORD_E1, COORD_H2, COORD_G2, COORD_F2, COORD_E2, COORD_A1}},
    {9, {COORD_A8, COORD_B8, COORD_C8, COORD_D8, COORD_A7, COORD_B7, COORD_C7, COORD_D7, COORD_H8}},
    {9, {COORD_H1, COORD_H2, COORD_H3, COORD_H4, COORD_G1, COORD_G2, COORD_G3, COORD_G4, COORD_H8}},
    {9, {COORD_A8, COORD_A7, COORD_A6, COORD_A5, COORD_B8, COORD_B7, COORD_B6, COORD_B5, COORD_A1}},
    {9, {COORD_A1, COORD_B1, COORD_C1, COORD_D1, COORD_E1, COORD_F1, COORD_G1, COORD_B2, COORD_C2}},
    {9, {COORD_H8, COORD_H7, COORD_H6, COORD_H5, COORD_H4, COORD_H3, COORD_H2, COORD_G7, COORD_G6}},
    {9, {COORD_A1, COORD_A2, COORD_A3, COORD_A4, COORD_A5, COORD_A6, COORD_A7, COORD_B2, COORD_B3}},
    {9, {COORD_H8, COORD_G8, COORD_F8, COORD_E8, COORD_D8, COORD_C8, COORD_B8, COORD_G7, COORD_F7}},
    {9, {COORD_H1, COORD_G1, COORD_F1, COORD_E1, COORD_D1, COORD_C1, COORD_B1, COORD_G2, COORD_F2}},
    {9, {COORD_A8, COORD_B8, COORD_C8, COORD_D8, COORD_E8, COORD_F8, COORD_G8, COORD_B7, COORD_C7}},
    {9, {COORD_H1, COORD_H2, COORD_H3, COORD_H4, COORD_H5, COORD_H6, COORD_H7, COORD_G2, COORD_G3}},
    {9, {COORD_A8, COORD_A7, COORD_A6, COORD_A5, COORD_A4, COORD_A3, COORD_A2, COORD_B7, COORD_B6}}
};

struct Coord_feature{
    uint_fast8_t feature;
    int x;
};

struct Coord_to_feature{
    int n_features;
    Coord_feature features[MAX_CELL_PATTERNS];
};

constexpr Coord_to_feature coord_to_feature[HW2] = {
    {14, {{ 1, P30}, { 2, P37}, {32, P30}, {34, P38}, {36, P31}, {38, P38}, {43, P38}, {46, P34}, {49, P38}, {51, P38}, {53, P30}, {54, P30}, {57, P38}, {59, P38}}}, // COORD_H8
    { 9, {{ 2, P36}, { 7, P30}, {34, P37}, {43, P37}, {46, P35}, {49, P34}, {51, P37}, {59, P37}, {61, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_G8
    {10, {{ 2, P35}, {11, P30}, {17, P37}, {18, P30}, {19, P35}, {43, P36}, {46, P36}, {51, P36}, {59, P36}, {61, P33}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_F8
    {10, {{ 2, P34}, {15, P30}, {25, P37}, {26, P30}, {27, P34}, {46, P37}, {47, P30}, {51, P35}, {59, P35}, {61, P34}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_E8
    {10, {{ 2, P33}, {13, P30}, {29, P34}, {30, P30}, {31, P37}, {46, P38}, {47, P31}, {53, P35}, {59, P34}, {61, P35}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_D8
    {10, {{ 2, P32}, { 9, P30}, {21, P35}, {22, P30}, {23, P37}, {42, P36}, {47, P32}, {53, P36}, {59, P33}, {61, P36}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_C8
    { 9, {{ 2, P31}, { 5, P30}, {33, P31}, {42, P37}, {47, P33}, {53, P37}, {55, P34}, {59, P32}, {61, P37}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_B8
    {14, {{ 2, P30}, { 3, P37}, {33, P30}, {35, P38}, {37, P31}, {39, P38}, {42, P38}, {47, P34}, {50, P30}, {51, P30}, {53, P38}, {55, P38}, {61, P38}, {63, P38}}}, // COORD_A8
    { 9, {{ 1, P31}, { 6, P30}, {32, P31}, {43, P35}, {46, P33}, {49, P37}, {51, P34}, {57, P37}, {62, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_H7
    {12, {{ 6, P31}, { 7, P31}, {17, P36}, {19, P36}, {36, P32}, {37, P30}, {38, P37}, {43, P34}, {49, P33}, {51, P33}, {57, P31}, {59, P31}, { 0, PNO}, { 0, PNO}}}, // COORD_G7
    { 8, {{ 6, P32}, {11, P31}, {25, P36}, {27, P35}, {34, P36}, {43, P33}, {51, P32}, {59, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_F7
    { 7, {{ 6, P33}, {15, P31}, {18, P31}, {19, P34}, {29, P33}, {30, P31}, {51, P31}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_E7
    { 7, {{ 6, P34}, {13, P31}, {21, P34}, {22, P31}, {26, P31}, {27, P33}, {53, P31}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_D7
    { 8, {{ 6, P35}, { 9, P31}, {29, P35}, {31, P36}, {33, P32}, {42, P33}, {53, P32}, {61, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_C7
    {12, {{ 5, P31}, { 6, P36}, {21, P36}, {23, P36}, {37, P32}, {38, P30}, {39, P37}, {42, P34}, {53, P33}, {55, P33}, {61, P31}, {63, P31}, { 0, PNO}, { 0, PNO}}}, // COORD_B7
    { 9, {{ 3, P36}, { 6, P37}, {35, P37}, {42, P35}, {47, P35}, {53, P34}, {55, P37}, {58, P32}, {63, P37}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_A7
    {10, {{ 1, P32}, {10, P30}, {16, P30}, {17, P35}, {19, P37}, {43, P32}, {46, P32}, {49, P36}, {57, P36}, {62, P33}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_H6
    { 8, {{ 7, P32}, {10, P31}, {25, P35}, {27, P36}, {32, P32}, {43, P31}, {49, P32}, {57, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_G6
    { 7, {{10, P32}, {11, P32}, {29, P32}, {30, P32}, {36, P33}, {38, P36}, {43, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_F6
    { 5, {{10, P33}, {15, P32}, {21, P33}, {22, P32}, {34, P35}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_E6
    { 5, {{10, P34}, {13, P32}, {18, P32}, {19, P33}, {33, P33}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_D6
    { 7, {{ 9, P32}, {10, P35}, {26, P32}, {27, P32}, {37, P33}, {39, P36}, {42, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_C6
    { 8, {{ 5, P32}, {10, P36}, {29, P36}, {31, P35}, {35, P36}, {42, P31}, {55, P32}, {63, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_B6
    {10, {{ 3, P35}, {10, P37}, {20, P30}, {21, P37}, {23, P35}, {42, P32}, {47, P36}, {55, P36}, {58, P33}, {63, P36}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_A6
    {10, {{ 1, P33}, {14, P30}, {24, P30}, {25, P34}, {27, P37}, {45, P38}, {46, P31}, {49, P35}, {57, P35}, {62, P34}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_H5
    { 7, {{ 7, P33}, {14, P31}, {16, P31}, {17, P34}, {29, P31}, {30, P33}, {49, P31}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_G5
    { 5, {{11, P33}, {14, P32}, {21, P32}, {22, P33}, {32, P33}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_F5
    { 5, {{14, P33}, {15, P33}, {33, P34}, {36, P34}, {38, P35}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_E5
    { 5, {{13, P33}, {14, P34}, {34, P34}, {37, P34}, {39, P35}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_D5
    { 5, {{ 9, P33}, {14, P35}, {18, P33}, {19, P32}, {35, P35}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_C5
    { 7, {{ 5, P33}, {14, P36}, {20, P31}, {23, P34}, {26, P33}, {27, P31}, {55, P31}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_B5
    {10, {{ 3, P34}, {14, P37}, {28, P30}, {29, P37}, {31, P34}, {44, P30}, {47, P37}, {55, P35}, {58, P34}, {63, P35}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_A5
    {10, {{ 1, P34}, {12, P30}, {28, P37}, {29, P30}, {30, P34}, {45, P37}, {46, P30}, {54, P35}, {57, P34}, {62, P35}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_H4
    { 7, {{ 7, P34}, {12, P31}, {21, P31}, {22, P34}, {24, P31}, {25, P33}, {54, P31}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_G4
    { 5, {{11, P34}, {12, P32}, {16, P32}, {17, P33}, {33, P35}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_F4
    { 5, {{12, P33}, {15, P34}, {32, P34}, {37, P35}, {39, P34}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_E4
    { 5, {{12, P34}, {13, P34}, {35, P34}, {36, P35}, {38, P34}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_D4
    { 5, {{ 9, P34}, {12, P35}, {20, P32}, {23, P33}, {34, P33}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_C4
    { 7, {{ 5, P34}, {12, P36}, {18, P34}, {19, P31}, {28, P31}, {31, P33}, {50, P31}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_B4
    {10, {{ 3, P33}, {12, P37}, {24, P37}, {26, P34}, {27, P30}, {44, P31}, {47, P38}, {50, P35}, {58, P35}, {63, P34}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_A4
    {10, {{ 1, P35}, { 8, P30}, {20, P37}, {21, P30}, {22, P35}, {41, P32}, {45, P36}, {54, P36}, {57, P33}, {62, P36}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_H3
    { 8, {{ 7, P35}, { 8, P31}, {28, P36}, {30, P35}, {33, P36}, {41, P31}, {54, P32}, {62, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_G3
    { 7, {{ 8, P32}, {11, P35}, {24, P32}, {25, P32}, {37, P36}, {39, P33}, {41, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_F3
    { 5, {{ 8, P33}, {15, P35}, {16, P33}, {17, P32}, {35, P33}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_E3
    { 5, {{ 8, P34}, {13, P35}, {20, P33}, {23, P32}, {32, P35}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_D3
    { 7, {{ 8, P35}, { 9, P35}, {28, P32}, {31, P32}, {36, P36}, {38, P33}, {40, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_C3
    { 8, {{ 5, P35}, { 8, P36}, {24, P36}, {26, P35}, {34, P32}, {40, P31}, {50, P32}, {58, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_B3
    {10, {{ 3, P32}, { 8, P37}, {16, P37}, {18, P35}, {19, P30}, {40, P32}, {44, P32}, {50, P36}, {58, P36}, {63, P33}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_A3
    { 9, {{ 1, P36}, { 4, P30}, {33, P37}, {41, P35}, {45, P35}, {52, P34}, {54, P37}, {57, P32}, {62, P37}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_H2
    {12, {{ 4, P31}, { 7, P36}, {20, P36}, {22, P36}, {36, P30}, {37, P37}, {39, P32}, {41, P34}, {52, P33}, {54, P33}, {60, P31}, {62, P31}, { 0, PNO}, { 0, PNO}}}, // COORD_G2
    { 8, {{ 4, P32}, {11, P36}, {28, P35}, {30, P36}, {35, P32}, {41, P33}, {52, P32}, {60, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_F2
    { 7, {{ 4, P33}, {15, P36}, {20, P34}, {23, P31}, {24, P33}, {25, P31}, {52, P31}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_E2
    { 7, {{ 4, P34}, {13, P36}, {16, P34}, {17, P31}, {28, P33}, {31, P31}, {48, P31}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_D2
    { 8, {{ 4, P35}, { 9, P36}, {24, P35}, {26, P36}, {32, P36}, {40, P33}, {48, P32}, {56, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_C2
    {12, {{ 4, P36}, { 5, P36}, {16, P36}, {18, P36}, {36, P37}, {38, P32}, {39, P30}, {40, P34}, {48, P33}, {50, P33}, {56, P31}, {58, P31}, { 0, PNO}, { 0, PNO}}}, // COORD_B2
    { 9, {{ 3, P31}, { 4, P37}, {34, P31}, {40, P35}, {44, P33}, {48, P34}, {50, P37}, {58, P37}, {63, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_A2
    {14, {{ 0, P30}, { 1, P37}, {33, P38}, {35, P30}, {37, P38}, {39, P31}, {41, P38}, {45, P34}, {48, P30}, {49, P30}, {52, P38}, {54, P38}, {60, P38}, {62, P38}}}, // COORD_H1
    { 9, {{ 0, P31}, { 7, P37}, {35, P31}, {41, P37}, {45, P33}, {52, P37}, {54, P34}, {56, P32}, {60, P37}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_G1
    {10, {{ 0, P32}, {11, P37}, {20, P35}, {22, P37}, {23, P30}, {41, P36}, {45, P32}, {52, P36}, {56, P33}, {60, P36}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_F1
    {10, {{ 0, P33}, {15, P37}, {28, P34}, {30, P37}, {31, P30}, {44, P38}, {45, P31}, {52, P35}, {56, P34}, {60, P35}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_E1
    {10, {{ 0, P34}, {13, P37}, {24, P34}, {25, P30}, {26, P37}, {44, P37}, {45, P30}, {48, P35}, {56, P35}, {60, P34}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_D1
    {10, {{ 0, P35}, { 9, P37}, {16, P35}, {17, P30}, {18, P37}, {40, P36}, {44, P36}, {48, P36}, {56, P36}, {60, P33}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_C1
    { 9, {{ 0, P36}, { 5, P37}, {32, P37}, {40, P37}, {44, P35}, {48, P37}, {50, P34}, {56, P37}, {60, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_B1
    {14, {{ 0, P37}, { 3, P30}, {32, P38}, {34, P30}, {36, P38}, {38, P31}, {40, P38}, {44, P34}, {48, P38}, {50, P38}, {52, P30}, {55, P30}, {56, P38}, {58, P38}}}  // COORD_A1
};

int16_t pattern_arr[2][N_PHASES][N_PATTERNS][MAX_EVALUATE_IDX];

constexpr uint_fast16_t pow3[11] = {1, P31, P32, P33, P34, P35, P36, P37, P38, P39, P310};

string create_line(int b, int w){
    string res = "";
    for (int i = 0; i < HW; ++i){
        if ((b >> i) & 1)
            res += "X";
        else if ((w >> i) & 1)
            res += "O";
        else
            res += ".";
    }
    return res;
}

float leaky_relu(float x){
    return max((float)(x * 0.01), x);
}

// predict
float predict(float *h0r, int ps, float ph, float d0[][19], float dn[][ND][ND], float bn[][ND], float *d3, float b3){
    float h0[ND], h1[ND];
    // dense 0 (phase)
    for (int i = 0; i < ND; ++i) {
        h0[i] = leaky_relu(h0r[i] + ph * d0[i][ps * 2]); // add phase, activate
    }
    // dense 1
    for (int i = 0; i < ND; ++i) {
        h1[i] = bn[0][i]; // bias
        for (int j = 0; j < ND; ++j) {
            h1[i] += h0[j] * dn[0][i][j];
        }
        h1[i] = leaky_relu(h1[i]); // activation
    }
    // dense 2 + last layer
    float h2, e = b3;
    for (int i = 0; i < ND; ++i) {
        h2 = bn[1][i];
        for (int j = 0; j < ND; ++j) {
            h2 += h1[j] * dn[1][i][j];
        }
        e += leaky_relu(h2) * d3[i];
    }
    e = tanh(e); // last activation
    return e * 4091; // multiply
}

int irp(int i, int s) {
    int j, r = i;
    for (j = 0; j < s; ++j) {
        r += ((i / pow3[j] + 2) % 3 - 1) * pow3[j];
    }
    return r;
}

//int irp_arr[2][MAX_EVALUATE_IDX];

void evaluate_init() {
    float in_arr[18], d0[ND][19], b0[ND], dn[3][ND][ND], bn[2][ND], d3[ND], b3, h0r[ND]; // in_arr max 9 cells, 18 input nodes
    constexpr int n_symmetry[N_PATTERNS] = {4, 4, 4, 4, 8, 8, 4, 4, 4, 4, 8, 8};
    // for (int i = 0; i < 2; ++i) {
    //     for (int j = 0; j < pow3[8 + i]; ++j) {
    //         irp_arr[i][j] = irp(j, 8 + i);
    //     }
    // }
    // read param & predict
    int param_ix = 0;
    for (int i = 0; i < N_PATTERNS; ++i) { // for each pattern
        //std::cerr << "pattern " << i << std::endl;
        // dense 0
        for (int j = 0; j < ND; ++j) {
            for (int k = 0; k < n_discs_in_pattern[i] * 2 + 1; ++k) {
                d0[j][k] = eval_params[param_ix++];
            }
        }
        // bias 0
        for (int j = 0; j < ND; ++j) {
            b0[j] = eval_params[param_ix++];
        }
        // dense 1, 2
        for (int l = 0; l < 2; ++l) { // for each dence
            for (int j = 0; j < ND; ++j) {
                for (int k = 0; k < ND; ++k) {
                    dn[l][j][k] = eval_params[param_ix++];
                }
            }
            // bias 1, 2
            for (int j = 0; j < ND; ++j) {
                bn[l][j] = eval_params[param_ix++];
            }
        }
        // dense 3
        for (int j = 0; j < ND; ++j) {
            d3[j] = eval_params[param_ix++];
        }
        // bias 3
        b3 = eval_params[param_ix++];
        // predict
        for (int j = 0; j < pow3[n_discs_in_pattern[i]]; ++j) {
            // set input
            for (int k = 0; k < n_discs_in_pattern[i]; ++k) {
                in_arr[k] = 0;
                in_arr[k + n_discs_in_pattern[i]] = 0;
                int c = (j / pow3[n_discs_in_pattern[i] - 1 - k]) % 3;
                if (c == 0)
                    in_arr[k] = 1;
                else if (c == 1)
                    in_arr[k + n_discs_in_pattern[i]] = 1;
            }
            // pre calculation of dense 0
            for (int k = 0; k < ND; ++k) {
                h0r[k] = b0[k]; // bias of dense 0
                for (int l = 0; l < n_discs_in_pattern[i] * 2; ++l) {
                    h0r[k] += in_arr[l] * d0[k][l]; // dense 0
                }
            }
            // predict each phase
            for (int k = 0; k < N_PHASES; ++k) {
                pattern_arr[0][k][i][j] = round(predict(h0r, n_discs_in_pattern[i], (float)k / N_PHASES, d0, dn, bn, d3, b3));
                //pattern_arr[1][k][i][irp_arr[n_discs_in_pattern[i] - 8][j]] = pattern_arr[0][k][i][j];
                pattern_arr[1][k][i][irp(j, n_discs_in_pattern[i])] = pattern_arr[0][k][i][j];
            }
        }
    }
    //std::cerr << "eval initialized " << param_ix << std::endl;
    std::cerr << "eval initialized" << std::endl;
}


inline int calc_pattern_diff(const int phase_idx, Search *search) {
    constexpr int n_symmetry[N_PATTERNS] = {4, 4, 4, 4, 8, 8, 4, 4, 4, 4, 8, 8};
    int feature_idx = 0;
    int res = 0;
    for (int i = 0; i < N_PATTERNS; ++i) {
        for (int j = 0; j < n_symmetry[i]; ++j) {
            res += pattern_arr[search->eval_feature_reversed][phase_idx][i][search->eval_features[feature_idx]];
            ++feature_idx;
        }
    }
    return res;
}

inline int end_evaluate(Board *b){
    return b->score_player();
}

inline int mid_evaluate_diff(Search *search){
    int phase_idx = search->phase();
    int res = calc_pattern_diff(phase_idx, search);
    res += res > 0 ? STEP_2 : (res < 0 ? -STEP_2 : 0);
    res /= STEP;
    return max(-SCORE_MAX, min(SCORE_MAX, res));
}

inline int pick_pattern_idx(const uint_fast8_t b_arr[], const Feature_to_coord *f){
    int res = 0;
    for (int i = 0; i < f->n_cells; ++i){
        res *= 3;
        res += b_arr[HW2_M1 - f->cells[i]];
    }
    return res;
}

inline void calc_features(Search *search){
    uint_fast8_t b_arr[HW2];
    search->board.translate_to_arr_player(b_arr);
    for (int i = 0; i < N_SYMMETRY_PATTERNS; ++i){
        search->eval_features[i] = pick_pattern_idx(b_arr, &feature_to_coord[i]);
    }
    search->eval_feature_reversed = 0;
}

inline bool check_features(Search *search){
    uint_fast8_t b_arr[HW2];
    search->board.translate_to_arr_player(b_arr);
    for (int i = 0; i < N_SYMMETRY_PATTERNS; ++i){
        if (search->eval_feature_reversed) {
            if (search->eval_features[i] != irp(pick_pattern_idx(b_arr, &feature_to_coord[i]), feature_to_coord[i].n_cells)) {
                cerr << i << " " << search->eval_features[i] << " " << irp(pick_pattern_idx(b_arr, &feature_to_coord[i]), feature_to_coord[i].n_cells) << endl;
                return true;
            }
        } else {
            if (search->eval_features[i] != pick_pattern_idx(b_arr, &feature_to_coord[i])) {
                cerr << i << " " << search->eval_features[i] << " " << pick_pattern_idx(b_arr, &feature_to_coord[i]) << endl;
                return true;
            }
        }
    }
    return false;
}

inline void print_features(Search *search){
    std::cerr << "reversed " << (int)search->eval_feature_reversed << std::endl;
    for (int i = 0; i < N_SYMMETRY_PATTERNS; ++i){
        std::cerr << i << " " << search->eval_features[i] << std::endl;
    }
}

inline void eval_move(Search *search, const Flip *flip){
    uint_fast8_t i, cell;
    uint64_t f;
    if (search->eval_feature_reversed){
        for (i = 0; i < coord_to_feature[flip->pos].n_features; ++i)
            search->eval_features[coord_to_feature[flip->pos].features[i].feature] -= coord_to_feature[flip->pos].features[i].x;
        f = flip->flip;
        for (cell = first_bit(&f); f; cell = next_bit(&f)){
            for (i = 0; i < coord_to_feature[cell].n_features; ++i)
                search->eval_features[coord_to_feature[cell].features[i].feature] += coord_to_feature[cell].features[i].x;
        }
    } else{
        for (i = 0; i < coord_to_feature[flip->pos].n_features; ++i)
            search->eval_features[coord_to_feature[flip->pos].features[i].feature] -= 2 * coord_to_feature[flip->pos].features[i].x;
        f = flip->flip;
        for (cell = first_bit(&f); f; cell = next_bit(&f)){
            for (i = 0; i < coord_to_feature[cell].n_features; ++i)
                search->eval_features[coord_to_feature[cell].features[i].feature] -= coord_to_feature[cell].features[i].x;
        }
    }
    search->eval_feature_reversed ^= 1;
}

inline void eval_undo(Search *search, const Flip *flip){
    search->eval_feature_reversed ^= 1;
    uint_fast8_t i, cell;
    uint64_t f;
    if (search->eval_feature_reversed){
        for (i = 0; i < coord_to_feature[flip->pos].n_features; ++i)
            search->eval_features[coord_to_feature[flip->pos].features[i].feature] += coord_to_feature[flip->pos].features[i].x;
        f = flip->flip;
        for (cell = first_bit(&f); f; cell = next_bit(&f)){
            for (i = 0; i < coord_to_feature[cell].n_features; ++i)
                search->eval_features[coord_to_feature[cell].features[i].feature] -= coord_to_feature[cell].features[i].x;
        }
    } else{
        for (i = 0; i < coord_to_feature[flip->pos].n_features; ++i)
            search->eval_features[coord_to_feature[flip->pos].features[i].feature] += 2 * coord_to_feature[flip->pos].features[i].x;
        f = flip->flip;
        for (cell = first_bit(&f); f; cell = next_bit(&f)){
            for (i = 0; i < coord_to_feature[cell].n_features; ++i)
                search->eval_features[coord_to_feature[cell].features[i].feature] += coord_to_feature[cell].features[i].x;
        }
    }
}

inline int mid_evaluate(Board *board){
    Search search;
    search.init_board(board);
    calc_features(&search);
    return mid_evaluate_diff(&search);
}