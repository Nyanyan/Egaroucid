/*
    Egaroucid Project

    @date 2021-2023
    @author Takuto Yamana
    @license GPL-3.0 license
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
#define MAX_PATTERN_CELLS 10
#define MAX_CELL_PATTERNS 8

#define STEP 256
#define STEP_2 128
#define STEP_SHIFT 8

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

#define P40 1
#define P41 4
#define P42 16
#define P43 64
#define P44 256
#define P45 1024
#define P46 4096
#define P47 16384
#ifndef P48
    #define P48 65536
#endif

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
    // 0 hv2
    {8, {COORD_A2, COORD_B2, COORD_C2, COORD_D2, COORD_E2, COORD_F2, COORD_G2, COORD_H2, COORD_NO, COORD_NO}}, // 0
    {8, {COORD_B1, COORD_B2, COORD_B3, COORD_B4, COORD_B5, COORD_B6, COORD_B7, COORD_B8, COORD_NO, COORD_NO}}, // 1
    {8, {COORD_A7, COORD_B7, COORD_C7, COORD_D7, COORD_E7, COORD_F7, COORD_G7, COORD_H7, COORD_NO, COORD_NO}}, // 2
    {8, {COORD_G1, COORD_G2, COORD_G3, COORD_G4, COORD_G5, COORD_G6, COORD_G7, COORD_G8, COORD_NO, COORD_NO}}, // 3

    // 1 hv3
    {8, {COORD_A3, COORD_B3, COORD_C3, COORD_D3, COORD_E3, COORD_F3, COORD_G3, COORD_H3, COORD_NO, COORD_NO}}, // 4
    {8, {COORD_C1, COORD_C2, COORD_C3, COORD_C4, COORD_C5, COORD_C6, COORD_C7, COORD_C8, COORD_NO, COORD_NO}}, // 5
    {8, {COORD_A6, COORD_B6, COORD_C6, COORD_D6, COORD_E6, COORD_F6, COORD_G6, COORD_H6, COORD_NO, COORD_NO}}, // 6
    {8, {COORD_F1, COORD_F2, COORD_F3, COORD_F4, COORD_F5, COORD_F6, COORD_F7, COORD_F8, COORD_NO, COORD_NO}}, // 7

    // 2 hv4
    {8, {COORD_A4, COORD_B4, COORD_C4, COORD_D4, COORD_E4, COORD_F4, COORD_G4, COORD_H4, COORD_NO, COORD_NO}}, // 8
    {8, {COORD_D1, COORD_D2, COORD_D3, COORD_D4, COORD_D5, COORD_D6, COORD_D7, COORD_D8, COORD_NO, COORD_NO}}, // 9
    {8, {COORD_A5, COORD_B5, COORD_C5, COORD_D5, COORD_E5, COORD_F5, COORD_G5, COORD_H5, COORD_NO, COORD_NO}}, // 10
    {8, {COORD_E1, COORD_E2, COORD_E3, COORD_E4, COORD_E5, COORD_E6, COORD_E7, COORD_E8, COORD_NO, COORD_NO}}, // 11

    // 3 d5
    {5, {COORD_D1, COORD_E2, COORD_F3, COORD_G4, COORD_H5, COORD_NO, COORD_NO, COORD_NO, COORD_NO, COORD_NO}}, // 12
    {5, {COORD_E1, COORD_D2, COORD_C3, COORD_B4, COORD_A5, COORD_NO, COORD_NO, COORD_NO, COORD_NO, COORD_NO}}, // 13
    {5, {COORD_A4, COORD_B5, COORD_C6, COORD_D7, COORD_E8, COORD_NO, COORD_NO, COORD_NO, COORD_NO, COORD_NO}}, // 14
    {5, {COORD_H4, COORD_G5, COORD_F6, COORD_E7, COORD_D8, COORD_NO, COORD_NO, COORD_NO, COORD_NO, COORD_NO}}, // 15

    // 4 d6
    {6, {COORD_C1, COORD_D2, COORD_E3, COORD_F4, COORD_G5, COORD_H6, COORD_NO, COORD_NO, COORD_NO, COORD_NO}}, // 16
    {6, {COORD_F1, COORD_E2, COORD_D3, COORD_C4, COORD_B5, COORD_A6, COORD_NO, COORD_NO, COORD_NO, COORD_NO}}, // 17
    {6, {COORD_A3, COORD_B4, COORD_C5, COORD_D6, COORD_E7, COORD_F8, COORD_NO, COORD_NO, COORD_NO, COORD_NO}}, // 18
    {6, {COORD_H3, COORD_G4, COORD_F5, COORD_E6, COORD_D7, COORD_C8, COORD_NO, COORD_NO, COORD_NO, COORD_NO}}, // 19

    // 5 d7
    {7, {COORD_B1, COORD_C2, COORD_D3, COORD_E4, COORD_F5, COORD_G6, COORD_H7, COORD_NO, COORD_NO, COORD_NO}}, // 20
    {7, {COORD_G1, COORD_F2, COORD_E3, COORD_D4, COORD_C5, COORD_B6, COORD_A7, COORD_NO, COORD_NO, COORD_NO}}, // 21
    {7, {COORD_A2, COORD_B3, COORD_C4, COORD_D5, COORD_E6, COORD_F7, COORD_G8, COORD_NO, COORD_NO, COORD_NO}}, // 22
    {7, {COORD_H2, COORD_G3, COORD_F4, COORD_E5, COORD_D6, COORD_C7, COORD_B8, COORD_NO, COORD_NO, COORD_NO}}, // 23

    // 6 d8
    {8, {COORD_A1, COORD_B2, COORD_C3, COORD_D4, COORD_E5, COORD_F6, COORD_G7, COORD_H8, COORD_NO, COORD_NO}}, // 24
    {8, {COORD_H1, COORD_G2, COORD_F3, COORD_E4, COORD_D5, COORD_C6, COORD_B7, COORD_A8, COORD_NO, COORD_NO}}, // 25

    // 7 edge + 2x
    {10, {COORD_B2, COORD_A1, COORD_B1, COORD_C1, COORD_D1, COORD_E1, COORD_F1, COORD_G1, COORD_H1, COORD_G2}}, // 26
    {10, {COORD_B2, COORD_A1, COORD_A2, COORD_A3, COORD_A4, COORD_A5, COORD_A6, COORD_A7, COORD_A8, COORD_B7}}, // 27
    {10, {COORD_B7, COORD_A8, COORD_B8, COORD_C8, COORD_D8, COORD_E8, COORD_F8, COORD_G8, COORD_H8, COORD_G7}}, // 28
    {10, {COORD_G2, COORD_H1, COORD_H2, COORD_H3, COORD_H4, COORD_H5, COORD_H6, COORD_H7, COORD_H8, COORD_G7}}, // 29

    // 8 triangle
    {10, {COORD_A1, COORD_B1, COORD_C1, COORD_D1, COORD_A2, COORD_B2, COORD_C2, COORD_A3, COORD_B3, COORD_A4}}, // 30
    {10, {COORD_H1, COORD_G1, COORD_F1, COORD_E1, COORD_H2, COORD_G2, COORD_F2, COORD_H3, COORD_G3, COORD_H4}}, // 31
    {10, {COORD_A8, COORD_B8, COORD_C8, COORD_D8, COORD_A7, COORD_B7, COORD_C7, COORD_A6, COORD_B6, COORD_A5}}, // 32
    {10, {COORD_H8, COORD_G8, COORD_F8, COORD_E8, COORD_H7, COORD_G7, COORD_F7, COORD_H6, COORD_G6, COORD_H5}}, // 33

    // 9 corner + block
    {10, {COORD_A1, COORD_C1, COORD_D1, COORD_E1, COORD_F1, COORD_H1, COORD_C2, COORD_D2, COORD_E2, COORD_F2}}, // 34
    {10, {COORD_A1, COORD_A3, COORD_A4, COORD_A5, COORD_A6, COORD_A8, COORD_B3, COORD_B4, COORD_B5, COORD_B6}}, // 35
    {10, {COORD_A8, COORD_C8, COORD_D8, COORD_E8, COORD_F8, COORD_H8, COORD_C7, COORD_D7, COORD_E7, COORD_F7}}, // 36
    {10, {COORD_H1, COORD_H3, COORD_H4, COORD_H5, COORD_H6, COORD_H8, COORD_G3, COORD_G4, COORD_G5, COORD_G6}}, // 37

    // 10 cross
    {10, {COORD_A1, COORD_B2, COORD_C3, COORD_D4, COORD_B1, COORD_C2, COORD_D3, COORD_A2, COORD_B3, COORD_C4}}, // 38
    {10, {COORD_H1, COORD_G2, COORD_F3, COORD_E4, COORD_G1, COORD_F2, COORD_E3, COORD_H2, COORD_G3, COORD_F4}}, // 39
    {10, {COORD_A8, COORD_B7, COORD_C6, COORD_D5, COORD_B8, COORD_C7, COORD_D6, COORD_A7, COORD_B6, COORD_C5}}, // 40
    {10, {COORD_H8, COORD_G7, COORD_F6, COORD_E5, COORD_G8, COORD_F7, COORD_E6, COORD_H7, COORD_G6, COORD_F5}}, // 41

    // 11 corner9
    {9, {COORD_A1, COORD_B1, COORD_C1, COORD_A2, COORD_B2, COORD_C2, COORD_A3, COORD_B3, COORD_C3, COORD_NO}}, // 42
    {9, {COORD_H1, COORD_G1, COORD_F1, COORD_H2, COORD_G2, COORD_F2, COORD_H3, COORD_G3, COORD_F3, COORD_NO}}, // 43
    {9, {COORD_A8, COORD_B8, COORD_C8, COORD_A7, COORD_B7, COORD_C7, COORD_A6, COORD_B6, COORD_C6, COORD_NO}}, // 44
    {9, {COORD_H8, COORD_G8, COORD_F8, COORD_H7, COORD_G7, COORD_F7, COORD_H6, COORD_G6, COORD_F6, COORD_NO}}  // 45
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
    { 7, {{24, P30}, {28, P31}, {29, P31}, {33, P39}, {36, P34}, {37, P34}, {41, P39}, { 0, PNO}}}, // COORD_H8
    { 6, {{ 3, P30}, {22, P30}, {28, P32}, {33, P38}, {41, P35}, {45, P37}, { 0, PNO}, { 0, PNO}}}, // COORD_G8
    { 6, {{ 7, P30}, {18, P30}, {28, P33}, {33, P37}, {36, P35}, {45, P36}, { 0, PNO}, { 0, PNO}}}, // COORD_F8
    { 5, {{11, P30}, {14, P30}, {28, P34}, {33, P36}, {36, P36}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_E8
    { 5, {{ 9, P30}, {15, P30}, {28, P35}, {32, P36}, {36, P37}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_D8
    { 6, {{ 5, P30}, {19, P30}, {28, P36}, {32, P37}, {36, P38}, {44, P36}, { 0, PNO}, { 0, PNO}}}, // COORD_C8
    { 6, {{ 1, P30}, {23, P30}, {28, P37}, {32, P38}, {40, P35}, {44, P37}, { 0, PNO}, { 0, PNO}}}, // COORD_B8
    { 7, {{25, P30}, {27, P31}, {28, P38}, {32, P39}, {35, P34}, {36, P39}, {40, P39}, { 0, PNO}}}, // COORD_A8
    { 6, {{ 2, P30}, {20, P30}, {29, P32}, {33, P35}, {41, P32}, {45, P35}, { 0, PNO}, { 0, PNO}}}, // COORD_H7
    { 8, {{ 2, P31}, { 3, P31}, {24, P31}, {28, P30}, {29, P30}, {33, P34}, {41, P38}, {45, P34}}}, // COORD_G7
    { 7, {{ 2, P32}, { 7, P31}, {22, P31}, {33, P33}, {36, P30}, {41, P34}, {45, P33}, { 0, PNO}}}, // COORD_F7
    { 5, {{ 2, P33}, {11, P31}, {15, P31}, {18, P31}, {36, P31}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_E7
    { 5, {{ 2, P34}, { 9, P31}, {14, P31}, {19, P31}, {36, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_D7
    { 7, {{ 2, P35}, { 5, P31}, {23, P31}, {32, P33}, {36, P33}, {40, P34}, {44, P33}, { 0, PNO}}}, // COORD_C7
    { 8, {{ 1, P31}, { 2, P36}, {25, P31}, {27, P30}, {28, P39}, {32, P34}, {40, P38}, {44, P34}}}, // COORD_B7
    { 6, {{ 2, P37}, {21, P30}, {27, P32}, {32, P35}, {40, P32}, {44, P35}, { 0, PNO}, { 0, PNO}}}, // COORD_A7
    { 6, {{ 6, P30}, {16, P30}, {29, P33}, {33, P32}, {37, P35}, {45, P32}, { 0, PNO}, { 0, PNO}}}, // COORD_H6
    { 7, {{ 3, P32}, { 6, P31}, {20, P31}, {33, P31}, {37, P30}, {41, P31}, {45, P31}, { 0, PNO}}}, // COORD_G6
    { 6, {{ 6, P32}, { 7, P32}, {15, P32}, {24, P32}, {41, P37}, {45, P30}, { 0, PNO}, { 0, PNO}}}, // COORD_F6
    { 5, {{ 6, P33}, {11, P32}, {19, P32}, {22, P32}, {41, P33}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_E6
    { 5, {{ 6, P34}, { 9, P32}, {18, P32}, {23, P32}, {40, P33}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_D6
    { 6, {{ 5, P32}, { 6, P35}, {14, P32}, {25, P32}, {40, P37}, {44, P30}, { 0, PNO}, { 0, PNO}}}, // COORD_C6
    { 7, {{ 1, P32}, { 6, P36}, {21, P31}, {32, P31}, {35, P30}, {40, P31}, {44, P31}, { 0, PNO}}}, // COORD_B6
    { 6, {{ 6, P37}, {17, P30}, {27, P33}, {32, P32}, {35, P35}, {44, P32}, { 0, PNO}, { 0, PNO}}}, // COORD_A6
    { 5, {{10, P30}, {12, P30}, {29, P34}, {33, P30}, {37, P36}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_H5
    { 5, {{ 3, P33}, {10, P31}, {15, P33}, {16, P31}, {37, P31}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_G5
    { 5, {{ 7, P33}, {10, P32}, {19, P33}, {20, P32}, {41, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_F5
    { 5, {{10, P33}, {11, P33}, {23, P33}, {24, P33}, {41, P36}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_E5
    { 5, {{ 9, P33}, {10, P34}, {22, P33}, {25, P33}, {40, P36}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_D5
    { 5, {{ 5, P33}, {10, P35}, {18, P33}, {21, P32}, {40, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_C5
    { 5, {{ 1, P33}, {10, P36}, {14, P33}, {17, P31}, {35, P31}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_B5
    { 5, {{10, P37}, {13, P30}, {27, P34}, {32, P30}, {35, P36}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_A5
    { 5, {{ 8, P30}, {15, P34}, {29, P35}, {31, P30}, {37, P37}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_H4
    { 5, {{ 3, P34}, { 8, P31}, {12, P31}, {19, P34}, {37, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_G4
    { 5, {{ 7, P34}, { 8, P32}, {16, P32}, {23, P34}, {39, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_F4
    { 5, {{ 8, P33}, {11, P34}, {20, P33}, {25, P34}, {39, P36}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_E4
    { 5, {{ 8, P34}, { 9, P34}, {21, P33}, {24, P34}, {38, P36}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_D4
    { 5, {{ 5, P34}, { 8, P35}, {17, P32}, {22, P34}, {38, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_C4
    { 5, {{ 1, P34}, { 8, P36}, {13, P31}, {18, P34}, {35, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_B4
    { 5, {{ 8, P37}, {14, P34}, {27, P35}, {30, P30}, {35, P37}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_A4
    { 6, {{ 4, P30}, {19, P35}, {29, P36}, {31, P32}, {37, P38}, {43, P32}, { 0, PNO}, { 0, PNO}}}, // COORD_H3
    { 7, {{ 3, P35}, { 4, P31}, {23, P35}, {31, P31}, {37, P33}, {39, P31}, {43, P31}, { 0, PNO}}}, // COORD_G3
    { 6, {{ 4, P32}, { 7, P35}, {12, P32}, {25, P35}, {39, P37}, {43, P30}, { 0, PNO}, { 0, PNO}}}, // COORD_F3
    { 5, {{ 4, P33}, {11, P35}, {16, P33}, {21, P34}, {39, P33}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_E3
    { 5, {{ 4, P34}, { 9, P35}, {17, P33}, {20, P34}, {38, P33}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_D3
    { 6, {{ 4, P35}, { 5, P35}, {13, P32}, {24, P35}, {38, P37}, {42, P30}, { 0, PNO}, { 0, PNO}}}, // COORD_C3
    { 7, {{ 1, P35}, { 4, P36}, {22, P35}, {30, P31}, {35, P33}, {38, P31}, {42, P31}, { 0, PNO}}}, // COORD_B3
    { 6, {{ 4, P37}, {18, P35}, {27, P36}, {30, P32}, {35, P38}, {42, P32}, { 0, PNO}, { 0, PNO}}}, // COORD_A3
    { 6, {{ 0, P30}, {23, P36}, {29, P37}, {31, P35}, {39, P32}, {43, P35}, { 0, PNO}, { 0, PNO}}}, // COORD_H2
    { 8, {{ 0, P31}, { 3, P36}, {25, P36}, {26, P30}, {29, P39}, {31, P34}, {39, P38}, {43, P34}}}, // COORD_G2
    { 7, {{ 0, P32}, { 7, P36}, {21, P35}, {31, P33}, {34, P30}, {39, P34}, {43, P33}, { 0, PNO}}}, // COORD_F2
    { 5, {{ 0, P33}, {11, P36}, {12, P33}, {17, P34}, {34, P31}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_E2
    { 5, {{ 0, P34}, { 9, P36}, {13, P33}, {16, P34}, {34, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_D2
    { 7, {{ 0, P35}, { 5, P36}, {20, P35}, {30, P33}, {34, P33}, {38, P34}, {42, P33}, { 0, PNO}}}, // COORD_C2
    { 8, {{ 0, P36}, { 1, P36}, {24, P36}, {26, P39}, {27, P39}, {30, P34}, {38, P38}, {42, P34}}}, // COORD_B2
    { 6, {{ 0, P37}, {22, P36}, {27, P37}, {30, P35}, {38, P32}, {42, P35}, { 0, PNO}, { 0, PNO}}}, // COORD_A2
    { 7, {{25, P37}, {26, P31}, {29, P38}, {31, P39}, {34, P34}, {37, P39}, {39, P39}, { 0, PNO}}}, // COORD_H1
    { 6, {{ 3, P37}, {21, P36}, {26, P32}, {31, P38}, {39, P35}, {43, P37}, { 0, PNO}, { 0, PNO}}}, // COORD_G1
    { 6, {{ 7, P37}, {17, P35}, {26, P33}, {31, P37}, {34, P35}, {43, P36}, { 0, PNO}, { 0, PNO}}}, // COORD_F1
    { 5, {{11, P37}, {13, P34}, {26, P34}, {31, P36}, {34, P36}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_E1
    { 5, {{ 9, P37}, {12, P34}, {26, P35}, {30, P36}, {34, P37}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_D1
    { 6, {{ 5, P37}, {16, P35}, {26, P36}, {30, P37}, {34, P38}, {42, P36}, { 0, PNO}, { 0, PNO}}}, // COORD_C1
    { 6, {{ 1, P37}, {20, P36}, {26, P37}, {30, P38}, {38, P35}, {42, P37}, { 0, PNO}, { 0, PNO}}}, // COORD_B1
    { 7, {{24, P37}, {26, P38}, {27, P38}, {30, P39}, {34, P39}, {35, P39}, {38, P39}, { 0, PNO}}}  // COORD_A1
};

constexpr uint_fast16_t pow3[11] = {1, P31, P32, P33, P34, P35, P36, P37, P38, P39, P310};
//int16_t pattern_arr[2][N_PHASES][N_PATTERNS][MAX_EVALUATE_IDX];
//int16_t eval_sur0_sur1_arr[N_PHASES][MAX_SURROUND][MAX_SURROUND];
//int16_t eval_canput0_canput1_arr[N_PHASES][MAX_CANPUT][MAX_CANPUT];
//int16_t eval_num0_num1_arr[N_PHASES][MAX_STONE_NUM][MAX_STONE_NUM];
//int16_t eval_canput_pattern[N_PHASES][N_CANPUT_PATTERNS][P48];

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

inline int swap_player_idx(int i, int pattern_size){
    int j, ri = i;
    for (j = 0; j < pattern_size; ++j){
        if ((i / pow3[j]) % 3 == 0)
            ri += pow3[j];
        else if ((i / pow3[j]) % 3 == 1)
            ri -= pow3[j];
    }
    return ri;
}

void init_pattern_arr_rev(int id, int phase_idx, int pattern_idx, int siz){
    int ri;
    for (int i = 0; i < pow3[siz]; ++i){
        ri = swap_player_idx(i, siz);
        pattern_arr[1][phase_idx][pattern_idx][ri] = pattern_arr[0][phase_idx][pattern_idx][i];
    }
}

inline bool init_evaluation_calc(){
    constexpr int pattern_sizes[N_PATTERNS] = {8, 8, 8, 5, 6, 7, 8, 10, 10, 10, 10, 9};
    int phase_idx, pattern_idx;
    for (phase_idx = 0; phase_idx < N_PHASES; ++phase_idx){
        for (pattern_idx = 0; pattern_idx < N_PATTERNS; ++pattern_idx)
            init_pattern_arr_rev(0, phase_idx, pattern_idx, pattern_sizes[pattern_idx]);
    }
    cerr << "evaluation function initialized" << endl;
    return true;
}

bool evaluate_init(){
    return init_evaluation_calc();
}

inline uint64_t calc_surround_part(const uint64_t player, const int dr){
    return (player << dr) | (player >> dr);
}

inline int calc_surround(const uint64_t player, const uint64_t empties){
    return pop_count_ull(empties & (
        calc_surround_part(player & 0b0111111001111110011111100111111001111110011111100111111001111110ULL, 1) | 
        calc_surround_part(player & 0b0000000011111111111111111111111111111111111111111111111100000000ULL, HW) | 
        calc_surround_part(player & 0b0000000001111110011111100111111001111110011111100111111000000000ULL, HW_M1) | 
        calc_surround_part(player & 0b0000000001111110011111100111111001111110011111100111111000000000ULL, HW_P1)
    ));
}

inline int calc_pattern_diff(const int phase_idx, Search *search){
    return 
        pattern_arr[search->eval_feature_reversed][phase_idx][0][search->eval_features[0]] + pattern_arr[search->eval_feature_reversed][phase_idx][0][search->eval_features[1]] + pattern_arr[search->eval_feature_reversed][phase_idx][0][search->eval_features[2]] + pattern_arr[search->eval_feature_reversed][phase_idx][0][search->eval_features[3]] + 
        pattern_arr[search->eval_feature_reversed][phase_idx][1][search->eval_features[4]] + pattern_arr[search->eval_feature_reversed][phase_idx][1][search->eval_features[5]] + pattern_arr[search->eval_feature_reversed][phase_idx][1][search->eval_features[6]] + pattern_arr[search->eval_feature_reversed][phase_idx][1][search->eval_features[7]] + 
        pattern_arr[search->eval_feature_reversed][phase_idx][2][search->eval_features[8]] + pattern_arr[search->eval_feature_reversed][phase_idx][2][search->eval_features[9]] + pattern_arr[search->eval_feature_reversed][phase_idx][2][search->eval_features[10]] + pattern_arr[search->eval_feature_reversed][phase_idx][2][search->eval_features[11]] + 
        pattern_arr[search->eval_feature_reversed][phase_idx][3][search->eval_features[12]] + pattern_arr[search->eval_feature_reversed][phase_idx][3][search->eval_features[13]] + pattern_arr[search->eval_feature_reversed][phase_idx][3][search->eval_features[14]] + pattern_arr[search->eval_feature_reversed][phase_idx][3][search->eval_features[15]] + 
        pattern_arr[search->eval_feature_reversed][phase_idx][4][search->eval_features[16]] + pattern_arr[search->eval_feature_reversed][phase_idx][4][search->eval_features[17]] + pattern_arr[search->eval_feature_reversed][phase_idx][4][search->eval_features[18]] + pattern_arr[search->eval_feature_reversed][phase_idx][4][search->eval_features[19]] + 
        pattern_arr[search->eval_feature_reversed][phase_idx][5][search->eval_features[20]] + pattern_arr[search->eval_feature_reversed][phase_idx][5][search->eval_features[21]] + pattern_arr[search->eval_feature_reversed][phase_idx][5][search->eval_features[22]] + pattern_arr[search->eval_feature_reversed][phase_idx][5][search->eval_features[23]] + 
        pattern_arr[search->eval_feature_reversed][phase_idx][6][search->eval_features[24]] + pattern_arr[search->eval_feature_reversed][phase_idx][6][search->eval_features[25]] + 
        pattern_arr[search->eval_feature_reversed][phase_idx][7][search->eval_features[26]] + pattern_arr[search->eval_feature_reversed][phase_idx][7][search->eval_features[27]] + pattern_arr[search->eval_feature_reversed][phase_idx][7][search->eval_features[28]] + pattern_arr[search->eval_feature_reversed][phase_idx][7][search->eval_features[29]] + 
        pattern_arr[search->eval_feature_reversed][phase_idx][8][search->eval_features[30]] + pattern_arr[search->eval_feature_reversed][phase_idx][8][search->eval_features[31]] + pattern_arr[search->eval_feature_reversed][phase_idx][8][search->eval_features[32]] + pattern_arr[search->eval_feature_reversed][phase_idx][8][search->eval_features[33]] + 
        pattern_arr[search->eval_feature_reversed][phase_idx][9][search->eval_features[34]] + pattern_arr[search->eval_feature_reversed][phase_idx][9][search->eval_features[35]] + pattern_arr[search->eval_feature_reversed][phase_idx][9][search->eval_features[36]] + pattern_arr[search->eval_feature_reversed][phase_idx][9][search->eval_features[37]] + 
        pattern_arr[search->eval_feature_reversed][phase_idx][10][search->eval_features[38]] + pattern_arr[search->eval_feature_reversed][phase_idx][10][search->eval_features[39]] + pattern_arr[search->eval_feature_reversed][phase_idx][10][search->eval_features[40]] + pattern_arr[search->eval_feature_reversed][phase_idx][10][search->eval_features[41]] + 
        pattern_arr[search->eval_feature_reversed][phase_idx][11][search->eval_features[42]] + pattern_arr[search->eval_feature_reversed][phase_idx][11][search->eval_features[43]] + pattern_arr[search->eval_feature_reversed][phase_idx][11][search->eval_features[44]] + pattern_arr[search->eval_feature_reversed][phase_idx][11][search->eval_features[45]];
}

inline int create_canput_line_h(uint64_t b, uint64_t w, int t){
    return (join_h_line(w, t) << HW) | join_h_line(b, t);
    //return (((w >> (HW * t)) & 0b11111111) << HW) | ((b >> (HW * t)) & 0b11111111);
}

inline int create_canput_line_v(uint64_t b, uint64_t w, int t){
    return (join_v_line(w, t) << HW) | join_v_line(b, t);
}

inline int end_evaluate(Board *b){
    return b->score_player();
}

inline int mid_evaluate_diff(Search *search){
    uint64_t player_mobility, opponent_mobility;
    player_mobility = calc_legal(search->board.player, search->board.opponent);
    opponent_mobility = calc_legal(search->board.opponent, search->board.player);
    if ((player_mobility | opponent_mobility) == 0ULL)
        return end_evaluate(&search->board);
    int phase_idx, sur0, sur1, canput0, canput1, num0, num1;
    uint64_t empties;
    phase_idx = search->phase();
    canput0 = min(MAX_CANPUT - 1, pop_count_ull(player_mobility));
    canput1 = min(MAX_CANPUT - 1, pop_count_ull(opponent_mobility));
    empties = ~(search->board.player | search->board.opponent);
    sur0 = min(MAX_SURROUND - 1, calc_surround(search->board.player, empties));
    sur1 = min(MAX_SURROUND - 1, calc_surround(search->board.opponent, empties));
    num0 = pop_count_ull(search->board.player);
    num1 = pop_count_ull(search->board.opponent);
    int res = calc_pattern_diff(phase_idx, search) + 
        eval_sur0_sur1_arr[phase_idx][sur0][sur1] + 
        eval_canput0_canput1_arr[phase_idx][canput0][canput1] + 
        eval_num0_num1_arr[phase_idx][num0][num1];
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
        if (search->eval_features[i] != pick_pattern_idx(b_arr, &feature_to_coord[i])){
            cerr << i << " " << search->eval_features[i] << " " << pick_pattern_idx(b_arr, &feature_to_coord[i]) << endl;
            return true;
        }
    }
    return false;
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