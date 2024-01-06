/*
    Egaroucid Project

    @date 2021-2024
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
#define MAX_CELL_PATTERNS 6

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
    {9, {COORD_A1, COORD_B1, COORD_A2, COORD_B2, COORD_C1, COORD_A3, COORD_C2, COORD_B3, COORD_C3}},
    {9, {COORD_A8, COORD_A7, COORD_B8, COORD_B7, COORD_A6, COORD_C8, COORD_B6, COORD_C7, COORD_C6}},
    {9, {COORD_H8, COORD_G8, COORD_H7, COORD_G7, COORD_F8, COORD_H6, COORD_F7, COORD_G6, COORD_F6}},
    {9, {COORD_H1, COORD_H2, COORD_G1, COORD_G2, COORD_H3, COORD_F1, COORD_G3, COORD_F2, COORD_F3}},
    {10, {COORD_A5, COORD_A4, COORD_A3, COORD_A2, COORD_A1, COORD_B2, COORD_B1, COORD_C1, COORD_D1, COORD_E1}},
    {10, {COORD_E8, COORD_D8, COORD_C8, COORD_B8, COORD_A8, COORD_B7, COORD_A7, COORD_A6, COORD_A5, COORD_A4}},
    {10, {COORD_H4, COORD_H5, COORD_H6, COORD_H7, COORD_H8, COORD_G7, COORD_G8, COORD_F8, COORD_E8, COORD_D8}},
    {10, {COORD_D1, COORD_E1, COORD_F1, COORD_G1, COORD_H1, COORD_G2, COORD_H2, COORD_H3, COORD_H4, COORD_H5}},
    {10, {COORD_B2, COORD_A1, COORD_B1, COORD_C1, COORD_D1, COORD_E1, COORD_F1, COORD_G1, COORD_H1, COORD_G2}},
    {10, {COORD_B7, COORD_A8, COORD_A7, COORD_A6, COORD_A5, COORD_A4, COORD_A3, COORD_A2, COORD_A1, COORD_B2}},
    {10, {COORD_G7, COORD_H8, COORD_G8, COORD_F8, COORD_E8, COORD_D8, COORD_C8, COORD_B8, COORD_A8, COORD_B7}},
    {10, {COORD_G2, COORD_H1, COORD_H2, COORD_H3, COORD_H4, COORD_H5, COORD_H6, COORD_H7, COORD_H8, COORD_G7}},
    {10, {COORD_A1, COORD_C1, COORD_D1, COORD_C2, COORD_D2, COORD_E2, COORD_F2, COORD_E1, COORD_F1, COORD_H1}},
    {10, {COORD_A8, COORD_A6, COORD_A5, COORD_B6, COORD_B5, COORD_B4, COORD_B3, COORD_A4, COORD_A3, COORD_A1}},
    {10, {COORD_H8, COORD_F8, COORD_E8, COORD_F7, COORD_E7, COORD_D7, COORD_C7, COORD_D8, COORD_C8, COORD_A8}},
    {10, {COORD_H1, COORD_H3, COORD_H4, COORD_G3, COORD_G4, COORD_G5, COORD_G6, COORD_H5, COORD_H6, COORD_H8}},
    {8, {COORD_A2, COORD_B2, COORD_C2, COORD_D2, COORD_E2, COORD_F2, COORD_G2, COORD_H2}},
    {8, {COORD_B8, COORD_B7, COORD_B6, COORD_B5, COORD_B4, COORD_B3, COORD_B2, COORD_B1}},
    {8, {COORD_H7, COORD_G7, COORD_F7, COORD_E7, COORD_D7, COORD_C7, COORD_B7, COORD_A7}},
    {8, {COORD_G1, COORD_G2, COORD_G3, COORD_G4, COORD_G5, COORD_G6, COORD_G7, COORD_G8}},
    {8, {COORD_A3, COORD_B3, COORD_C3, COORD_D3, COORD_E3, COORD_F3, COORD_G3, COORD_H3}},
    {8, {COORD_C8, COORD_C7, COORD_C6, COORD_C5, COORD_C4, COORD_C3, COORD_C2, COORD_C1}},
    {8, {COORD_H6, COORD_G6, COORD_F6, COORD_E6, COORD_D6, COORD_C6, COORD_B6, COORD_A6}},
    {8, {COORD_F1, COORD_F2, COORD_F3, COORD_F4, COORD_F5, COORD_F6, COORD_F7, COORD_F8}},
    {8, {COORD_A4, COORD_B4, COORD_C4, COORD_D4, COORD_E4, COORD_F4, COORD_G4, COORD_H4}},
    {8, {COORD_D8, COORD_D7, COORD_D6, COORD_D5, COORD_D4, COORD_D3, COORD_D2, COORD_D1}},
    {8, {COORD_H5, COORD_G5, COORD_F5, COORD_E5, COORD_D5, COORD_C5, COORD_B5, COORD_A5}},
    {8, {COORD_E1, COORD_E2, COORD_E3, COORD_E4, COORD_E5, COORD_E6, COORD_E7, COORD_E8}},
    {8, {COORD_A1, COORD_B2, COORD_C3, COORD_D4, COORD_E5, COORD_F6, COORD_G7, COORD_H8}},
    {8, {COORD_A8, COORD_B7, COORD_C6, COORD_D5, COORD_E4, COORD_F3, COORD_G2, COORD_H1}},
    {7, {COORD_B1, COORD_C2, COORD_D3, COORD_E4, COORD_F5, COORD_G6, COORD_H7}},
    {7, {COORD_A7, COORD_B6, COORD_C5, COORD_D4, COORD_E3, COORD_F2, COORD_G1}},
    {7, {COORD_G8, COORD_F7, COORD_E6, COORD_D5, COORD_C4, COORD_B3, COORD_A2}},
    {7, {COORD_H2, COORD_G3, COORD_F4, COORD_E5, COORD_D6, COORD_C7, COORD_B8}},
    {6, {COORD_C1, COORD_D2, COORD_E3, COORD_F4, COORD_G5, COORD_H6}},
    {6, {COORD_A6, COORD_B5, COORD_C4, COORD_D3, COORD_E2, COORD_F1}},
    {6, {COORD_F8, COORD_E7, COORD_D6, COORD_C5, COORD_B4, COORD_A3}},
    {6, {COORD_H3, COORD_G4, COORD_F5, COORD_E6, COORD_D7, COORD_C8}},
    {5, {COORD_D1, COORD_E2, COORD_F3, COORD_G4, COORD_H5}},
    {5, {COORD_A5, COORD_B4, COORD_C3, COORD_D2, COORD_E1}},
    {5, {COORD_E8, COORD_D7, COORD_C6, COORD_B5, COORD_A4}},
    {5, {COORD_H4, COORD_G5, COORD_F6, COORD_E7, COORD_D8}},
    {4, {COORD_E1, COORD_F2, COORD_G3, COORD_H4}},
    {4, {COORD_A4, COORD_B3, COORD_C2, COORD_D1}},
    {4, {COORD_D8, COORD_C7, COORD_B6, COORD_A5}},
    {4, {COORD_H5, COORD_G6, COORD_F7, COORD_E8}}
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
    { 5, {{ 2, P38}, { 6, P35}, {10, P38}, {11, P31}, {14, P39}, { 0, PNO}}}, // COORD_H8
    { 4, {{ 2, P37}, { 6, P33}, {10, P37}, {32, P36}, { 0, PNO}, { 0, PNO}}}, // COORD_G8
    { 5, {{ 2, P34}, { 6, P32}, {10, P36}, {14, P38}, {36, P35}, { 0, PNO}}}, // COORD_F8
    { 6, {{ 5, P39}, { 6, P31}, {10, P35}, {14, P37}, {40, P34}, {45, P30}}}, // COORD_E8
    { 5, {{ 5, P38}, {10, P34}, {14, P32}, {25, P37}, {44, P33}, { 0, PNO}}}, // COORD_D8
    { 5, {{ 1, P33}, { 5, P37}, {10, P33}, {14, P31}, {21, P37}, { 0, PNO}}}, // COORD_C8
    { 4, {{ 1, P36}, { 5, P36}, {10, P32}, {17, P37}, { 0, PNO}, { 0, PNO}}}, // COORD_B8
    { 6, {{ 1, P38}, { 5, P35}, { 9, P38}, {10, P31}, {13, P39}, {29, P37}}}, // COORD_A8
    { 4, {{ 2, P36}, { 6, P36}, {11, P32}, {18, P37}, { 0, PNO}, { 0, PNO}}}, // COORD_H7
    { 6, {{ 2, P35}, { 6, P34}, {10, P39}, {18, P36}, {19, P31}, {28, P31}}}, // COORD_G7
    { 6, {{ 2, P32}, {14, P36}, {18, P35}, {23, P31}, {32, P35}, {45, P31}}}, // COORD_F7
    { 5, {{14, P35}, {18, P34}, {27, P31}, {36, P34}, {41, P31}, { 0, PNO}}}, // COORD_E7
    { 5, {{14, P34}, {18, P33}, {25, P36}, {37, P31}, {40, P33}, { 0, PNO}}}, // COORD_D7
    { 6, {{ 1, P31}, {14, P33}, {18, P32}, {21, P36}, {33, P31}, {44, P32}}}, // COORD_C7
    { 6, {{ 1, P35}, { 5, P34}, { 9, P39}, {17, P36}, {18, P31}, {29, P36}}}, // COORD_B7
    { 4, {{ 1, P37}, { 5, P33}, { 9, P37}, {31, P36}, { 0, PNO}, { 0, PNO}}}, // COORD_A7
    { 5, {{ 2, P33}, { 6, P37}, {11, P33}, {15, P31}, {22, P37}, { 0, PNO}}}, // COORD_H6
    { 6, {{ 2, P31}, {15, P33}, {19, P32}, {22, P36}, {30, P31}, {45, P32}}}, // COORD_G6
    { 4, {{22, P35}, {23, P32}, {28, P32}, {41, P32}, { 0, PNO}, { 0, PNO}}}, // COORD_F6
    { 4, {{22, P34}, {27, P32}, {32, P34}, {37, P32}, { 0, PNO}, { 0, PNO}}}, // COORD_E6
    { 4, {{22, P33}, {25, P35}, {33, P32}, {36, P33}, { 0, PNO}, { 0, PNO}}}, // COORD_D6
    { 4, {{21, P35}, {22, P32}, {29, P35}, {40, P32}, { 0, PNO}, { 0, PNO}}}, // COORD_C6
    { 6, {{ 1, P32}, {13, P36}, {17, P35}, {22, P31}, {31, P35}, {44, P31}}}, // COORD_B6
    { 5, {{ 1, P34}, { 5, P32}, { 9, P36}, {13, P38}, {35, P35}, { 0, PNO}}}, // COORD_A6
    { 5, {{ 6, P38}, {11, P34}, {15, P32}, {26, P37}, {45, P33}, { 0, PNO}}}, // COORD_H5
    { 5, {{15, P34}, {19, P33}, {26, P36}, {34, P31}, {41, P33}, { 0, PNO}}}, // COORD_G5
    { 4, {{23, P33}, {26, P35}, {30, P32}, {37, P33}, { 0, PNO}, { 0, PNO}}}, // COORD_F5
    { 4, {{26, P34}, {27, P33}, {28, P33}, {33, P33}, { 0, PNO}, { 0, PNO}}}, // COORD_E5
    { 4, {{25, P34}, {26, P33}, {29, P34}, {32, P33}, { 0, PNO}, { 0, PNO}}}, // COORD_D5
    { 4, {{21, P34}, {26, P32}, {31, P34}, {36, P32}, { 0, PNO}, { 0, PNO}}}, // COORD_C5
    { 5, {{13, P35}, {17, P34}, {26, P31}, {35, P34}, {40, P31}, { 0, PNO}}}, // COORD_B5
    { 5, {{ 4, P39}, { 5, P31}, { 9, P35}, {13, P37}, {39, P34}, { 0, PNO}}}, // COORD_A5
    { 5, {{ 6, P39}, { 7, P31}, {11, P35}, {15, P37}, {41, P34}, { 0, PNO}}}, // COORD_H4
    { 5, {{15, P35}, {19, P34}, {24, P31}, {37, P34}, {38, P31}, { 0, PNO}}}, // COORD_G4
    { 4, {{23, P34}, {24, P32}, {33, P34}, {34, P32}, { 0, PNO}, { 0, PNO}}}, // COORD_F4
    { 4, {{24, P33}, {27, P34}, {29, P33}, {30, P33}, { 0, PNO}, { 0, PNO}}}, // COORD_E4
    { 4, {{24, P34}, {25, P33}, {28, P34}, {31, P33}, { 0, PNO}, { 0, PNO}}}, // COORD_D4
    { 4, {{21, P33}, {24, P35}, {32, P32}, {35, P33}, { 0, PNO}, { 0, PNO}}}, // COORD_C4
    { 5, {{13, P34}, {17, P33}, {24, P36}, {36, P31}, {39, P33}, { 0, PNO}}}, // COORD_B4
    { 5, {{ 4, P38}, { 9, P34}, {13, P32}, {24, P37}, {43, P33}, { 0, PNO}}}, // COORD_A4
    { 5, {{ 3, P34}, { 7, P32}, {11, P36}, {15, P38}, {37, P35}, { 0, PNO}}}, // COORD_H3
    { 6, {{ 3, P32}, {15, P36}, {19, P35}, {20, P31}, {33, P35}, {42, P31}}}, // COORD_G3
    { 4, {{20, P32}, {23, P35}, {29, P32}, {38, P32}, { 0, PNO}, { 0, PNO}}}, // COORD_F3
    { 4, {{20, P33}, {27, P35}, {31, P32}, {34, P33}, { 0, PNO}, { 0, PNO}}}, // COORD_E3
    { 4, {{20, P34}, {25, P32}, {30, P34}, {35, P32}, { 0, PNO}, { 0, PNO}}}, // COORD_D3
    { 4, {{20, P35}, {21, P32}, {28, P35}, {39, P32}, { 0, PNO}, { 0, PNO}}}, // COORD_C3
    { 6, {{ 0, P31}, {13, P33}, {17, P32}, {20, P36}, {32, P31}, {43, P32}}}, // COORD_B3
    { 5, {{ 0, P33}, { 4, P37}, { 9, P33}, {13, P31}, {20, P37}, { 0, PNO}}}, // COORD_A3
    { 4, {{ 3, P37}, { 7, P33}, {11, P37}, {33, P36}, { 0, PNO}, { 0, PNO}}}, // COORD_H2
    { 6, {{ 3, P35}, { 7, P34}, {11, P39}, {16, P31}, {19, P36}, {29, P31}}}, // COORD_G2
    { 6, {{ 3, P31}, {12, P33}, {16, P32}, {23, P36}, {31, P31}, {42, P32}}}, // COORD_F2
    { 5, {{12, P34}, {16, P33}, {27, P36}, {35, P31}, {38, P33}, { 0, PNO}}}, // COORD_E2
    { 5, {{12, P35}, {16, P34}, {25, P31}, {34, P34}, {39, P31}, { 0, PNO}}}, // COORD_D2
    { 6, {{ 0, P32}, {12, P36}, {16, P35}, {21, P31}, {30, P35}, {43, P31}}}, // COORD_C2
    { 6, {{ 0, P35}, { 4, P34}, { 8, P39}, {16, P36}, {17, P31}, {28, P36}}}, // COORD_B2
    { 4, {{ 0, P36}, { 4, P36}, { 9, P32}, {16, P37}, { 0, PNO}, { 0, PNO}}}, // COORD_A2
    { 5, {{ 3, P38}, { 7, P35}, { 8, P31}, {11, P38}, {15, P39}, { 0, PNO}}}, // COORD_H1
    { 4, {{ 3, P36}, { 7, P36}, { 8, P32}, {19, P37}, { 0, PNO}, { 0, PNO}}}, // COORD_G1
    { 5, {{ 3, P33}, { 7, P37}, { 8, P33}, {12, P31}, {23, P37}, { 0, PNO}}}, // COORD_F1
    { 5, {{ 7, P38}, { 8, P34}, {12, P32}, {27, P37}, {42, P33}, { 0, PNO}}}, // COORD_E1
    { 5, {{ 4, P31}, { 7, P39}, { 8, P35}, {12, P37}, {38, P34}, { 0, PNO}}}, // COORD_D1
    { 5, {{ 0, P34}, { 4, P32}, { 8, P36}, {12, P38}, {34, P35}, { 0, PNO}}}, // COORD_C1
    { 4, {{ 0, P37}, { 4, P33}, { 8, P37}, {30, P36}, { 0, PNO}, { 0, PNO}}}, // COORD_B1
    { 6, {{ 0, P38}, { 4, P35}, { 8, P38}, { 9, P31}, {12, P39}, {28, P37}}}  // COORD_A1
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

float predict(float *h0r, float *h0, float *h1, int ps, float ph, float d0[][21], float d1[][ND], float *b1, float d2[][ND], float *b2, float *d3, float b3){
    int i, j;
    for (i = 0; i < ND; ++i)
        h0[i] = leaky_relu(h0r[i] + ph * d0[i][ps * 2]);
    for (i = 0; i < ND; ++i){
        h1[i] = b1[i];
        for (j = 0; j < ND; ++j)
            h1[i] += h0[j] * d1[i][j];
        h1[i] = leaky_relu(h1[i]);
    }
    float h2, e = b3;
    for (i = 0; i < ND; ++i){
        h2 = b2[i];
        for (j = 0; j < ND; ++j)
            h2 += h1[j] * d2[i][j];
        e += leaky_relu(h2) * d3[i];
    }
    return e;
}

int irp(int i, int s) {
    int j, r = i;
    for (j = 0; j < s; ++j)
        r += ((i / pow3[j] + 2) % 3 - 1) * pow3[j];
    return r;
}

inline bool init_evaluation_calc() {
    int i, j, k, l, m;
    float in_arr[20], d0[ND][21], b0[ND], d1[ND][ND], b1[ND], d2[ND][ND], b2[ND], d3[ND], b3, h0[ND], h1[ND], h0r[ND];
    constexpr int pattern_sizes[N_PATTERNS] = {9, 10, 10, 10, 8, 8, 8, 8, 7, 6, 5, 4};
    int param_ix = 0;
    for (i = 0; i < N_PATTERNS; ++i){
        for (j = 0; j < ND; ++j){
            for (k = 0; k < pattern_sizes[i] * 2 + 1; ++k)
                d0[j][k] = eval_params[param_ix++];
        }
        for (j = 0; j < ND; ++j)
            b0[j] = eval_params[param_ix++];
        for (j = 0; j < ND; ++j){
            for (k = 0; k < ND; ++k)
                d1[j][k] = eval_params[param_ix++];
        }
        for (j = 0; j < ND; ++j)
            b1[j] = eval_params[param_ix++];
        for (j = 0; j < ND; ++j){
            for (k = 0; k < ND; ++k)
                d2[j][k] = eval_params[param_ix++];
        }
        for (j = 0; j < ND; ++j)
            b2[j] = eval_params[param_ix++];
        for (j = 0; j < ND; ++j)
            d3[j] = eval_params[param_ix++];
        b3 = eval_params[param_ix++];
        for (j = 0; j < pow3[pattern_sizes[i]]; ++j){
            for (k = 0; k < pattern_sizes[i]; ++k){
                in_arr[k] = 0;
                in_arr[k + pattern_sizes[i]] = 0;
                int c = (j / pow3[pattern_sizes[i] - 1 - k]) % 3;
                if (c == 0)
                    in_arr[k] = 1;
                else if (c == 1)
                    in_arr[k + pattern_sizes[i]] = 1;
            }
            for (l = 0; l < ND; ++l){
                h0r[l] = b0[l];
                for (m = 0; m < pattern_sizes[i] * 2; ++m)
                    h0r[l] += in_arr[m] * d0[l][m];
            }
            for (k = 0; k < NP; ++k){
                pattern_arr[0][k][i][j] = round(predict(h0r, h0, h1, pattern_sizes[i], (float)k / NP, d0, d1, b1, d2, b2, d3, b3) * 16384);
                pattern_arr[1][k][i][irp(j, pattern_sizes[i])] = pattern_arr[0][k][i][j];
            }
        }
    }
    cerr << "evaluation function initialized" << endl;
    return true;
}

bool evaluate_init(){
    return init_evaluation_calc();
}

inline int calc_pattern_diff(const int phase_idx, Search *search){
    return 
        pattern_arr[search->eval_feature_reversed][phase_idx][0][search->eval_features[0]] + pattern_arr[search->eval_feature_reversed][phase_idx][0][search->eval_features[1]] + pattern_arr[search->eval_feature_reversed][phase_idx][0][search->eval_features[2]] + pattern_arr[search->eval_feature_reversed][phase_idx][0][search->eval_features[3]] + 
        pattern_arr[search->eval_feature_reversed][phase_idx][1][search->eval_features[4]] + pattern_arr[search->eval_feature_reversed][phase_idx][1][search->eval_features[5]] + pattern_arr[search->eval_feature_reversed][phase_idx][1][search->eval_features[6]] + pattern_arr[search->eval_feature_reversed][phase_idx][1][search->eval_features[7]] + 
        pattern_arr[search->eval_feature_reversed][phase_idx][2][search->eval_features[8]] + pattern_arr[search->eval_feature_reversed][phase_idx][2][search->eval_features[9]] + pattern_arr[search->eval_feature_reversed][phase_idx][2][search->eval_features[10]] + pattern_arr[search->eval_feature_reversed][phase_idx][2][search->eval_features[11]] + 
        pattern_arr[search->eval_feature_reversed][phase_idx][3][search->eval_features[12]] + pattern_arr[search->eval_feature_reversed][phase_idx][3][search->eval_features[13]] + pattern_arr[search->eval_feature_reversed][phase_idx][3][search->eval_features[14]] + pattern_arr[search->eval_feature_reversed][phase_idx][3][search->eval_features[15]] + 
        pattern_arr[search->eval_feature_reversed][phase_idx][4][search->eval_features[16]] + pattern_arr[search->eval_feature_reversed][phase_idx][4][search->eval_features[17]] + pattern_arr[search->eval_feature_reversed][phase_idx][4][search->eval_features[18]] + pattern_arr[search->eval_feature_reversed][phase_idx][4][search->eval_features[19]] + 
        pattern_arr[search->eval_feature_reversed][phase_idx][5][search->eval_features[20]] + pattern_arr[search->eval_feature_reversed][phase_idx][5][search->eval_features[21]] + pattern_arr[search->eval_feature_reversed][phase_idx][5][search->eval_features[22]] + pattern_arr[search->eval_feature_reversed][phase_idx][5][search->eval_features[23]] + 
        pattern_arr[search->eval_feature_reversed][phase_idx][6][search->eval_features[24]] + pattern_arr[search->eval_feature_reversed][phase_idx][6][search->eval_features[25]] + pattern_arr[search->eval_feature_reversed][phase_idx][6][search->eval_features[26]] + pattern_arr[search->eval_feature_reversed][phase_idx][6][search->eval_features[27]] + 
        pattern_arr[search->eval_feature_reversed][phase_idx][7][search->eval_features[28]] + pattern_arr[search->eval_feature_reversed][phase_idx][7][search->eval_features[29]] + 
        pattern_arr[search->eval_feature_reversed][phase_idx][8][search->eval_features[30]] + pattern_arr[search->eval_feature_reversed][phase_idx][8][search->eval_features[31]] + pattern_arr[search->eval_feature_reversed][phase_idx][8][search->eval_features[32]] + pattern_arr[search->eval_feature_reversed][phase_idx][8][search->eval_features[33]] + 
        pattern_arr[search->eval_feature_reversed][phase_idx][9][search->eval_features[34]] + pattern_arr[search->eval_feature_reversed][phase_idx][9][search->eval_features[35]] + pattern_arr[search->eval_feature_reversed][phase_idx][9][search->eval_features[36]] + pattern_arr[search->eval_feature_reversed][phase_idx][9][search->eval_features[37]] + 
        pattern_arr[search->eval_feature_reversed][phase_idx][10][search->eval_features[38]] + pattern_arr[search->eval_feature_reversed][phase_idx][10][search->eval_features[39]] + pattern_arr[search->eval_feature_reversed][phase_idx][10][search->eval_features[40]] + pattern_arr[search->eval_feature_reversed][phase_idx][10][search->eval_features[41]] + 
        pattern_arr[search->eval_feature_reversed][phase_idx][11][search->eval_features[42]] + pattern_arr[search->eval_feature_reversed][phase_idx][11][search->eval_features[43]] + pattern_arr[search->eval_feature_reversed][phase_idx][11][search->eval_features[44]] + pattern_arr[search->eval_feature_reversed][phase_idx][11][search->eval_features[45]];
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