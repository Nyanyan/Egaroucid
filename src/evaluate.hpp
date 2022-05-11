#pragma once
#include <iostream>
#include <fstream>
#include "setting.hpp"
#include "common.hpp"
#include "board.hpp"
#include "search.hpp"
#include "util.hpp"

using namespace std;

#define N_PATTERNS 16
#ifndef N_SYMMETRY_PATTERNS
    #define N_SYMMETRY_PATTERNS 62
#endif
#define MAX_PATTERN_CELLS 10
#define MAX_CELL_PATTERNS 13
#define MAX_SURROUND 100
#define MAX_CANPUT 50
#define MAX_STABILITY 65
#define MAX_STONE_NUM 65
#define N_CANPUT_PATTERNS 4
#define MAX_EVALUATE_IDX 59049

#define STEP 256
#define STEP_2 128

#if EVALUATION_STEP_WIDTH_MODE == 0
    #define SCORE_MAX 64
#elif EVALUATION_STEP_WIDTH_MODE == 1
    #define SCORE_MAX 32
#elif EVALUATION_STEP_WIDTH_MODE == 2
    #define SCORE_MAX 128
#elif EVALUATION_STEP_WIDTH_MODE == 3
    #define SCORE_MAX 256
#elif EVALUATION_STEP_WIDTH_MODE == 4
    #define SCORE_MAX 512
#elif EVALUATION_STEP_WIDTH_MODE == 5
    #define SCORE_MAX 1024
#elif EVALUATION_STEP_WIDTH_MODE == 6
    #define SCORE_MAX 2048
#endif

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
#define P48 65536

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
    // hv2
    {8, {COORD_A2, COORD_B2, COORD_C2, COORD_D2, COORD_E2, COORD_F2, COORD_G2, COORD_H2, COORD_NO, COORD_NO}}, // 0
    {8, {COORD_B1, COORD_B2, COORD_B3, COORD_B4, COORD_B5, COORD_B6, COORD_B7, COORD_B8, COORD_NO, COORD_NO}}, // 1
    {8, {COORD_A7, COORD_B7, COORD_C7, COORD_D7, COORD_E7, COORD_F7, COORD_G7, COORD_H7, COORD_NO, COORD_NO}}, // 2
    {8, {COORD_G1, COORD_G2, COORD_G3, COORD_G4, COORD_G5, COORD_G6, COORD_G7, COORD_G8, COORD_NO, COORD_NO}}, // 3

    // hv3
    {8, {COORD_A3, COORD_B3, COORD_C3, COORD_D3, COORD_E3, COORD_F3, COORD_G3, COORD_H3, COORD_NO, COORD_NO}}, // 4
    {8, {COORD_C1, COORD_C2, COORD_C3, COORD_C4, COORD_C5, COORD_C6, COORD_C7, COORD_C8, COORD_NO, COORD_NO}}, // 5
    {8, {COORD_A6, COORD_B6, COORD_C6, COORD_D6, COORD_E6, COORD_F6, COORD_G6, COORD_H6, COORD_NO, COORD_NO}}, // 6
    {8, {COORD_F1, COORD_F2, COORD_F3, COORD_F4, COORD_F5, COORD_F6, COORD_F7, COORD_F8, COORD_NO, COORD_NO}}, // 7

    // hv4
    {8, {COORD_A4, COORD_B4, COORD_C4, COORD_D4, COORD_E4, COORD_F4, COORD_G4, COORD_H4, COORD_NO, COORD_NO}}, // 8
    {8, {COORD_D1, COORD_D2, COORD_D3, COORD_D4, COORD_D5, COORD_D6, COORD_D7, COORD_D8, COORD_NO, COORD_NO}}, // 9
    {8, {COORD_A5, COORD_B5, COORD_C5, COORD_D5, COORD_E5, COORD_F5, COORD_G5, COORD_H5, COORD_NO, COORD_NO}}, // 10
    {8, {COORD_E1, COORD_E2, COORD_E3, COORD_E4, COORD_E5, COORD_E6, COORD_E7, COORD_E8, COORD_NO, COORD_NO}}, // 11

    // d5
    {5, {COORD_D1, COORD_E2, COORD_F3, COORD_G4, COORD_H5, COORD_NO, COORD_NO, COORD_NO, COORD_NO, COORD_NO}}, // 12
    {5, {COORD_E1, COORD_D2, COORD_C3, COORD_B4, COORD_A5, COORD_NO, COORD_NO, COORD_NO, COORD_NO, COORD_NO}}, // 13
    {5, {COORD_A4, COORD_B5, COORD_C6, COORD_D7, COORD_E8, COORD_NO, COORD_NO, COORD_NO, COORD_NO, COORD_NO}}, // 14
    {5, {COORD_H4, COORD_G5, COORD_F6, COORD_E7, COORD_D8, COORD_NO, COORD_NO, COORD_NO, COORD_NO, COORD_NO}}, // 15

    // d6
    {6, {COORD_C1, COORD_D2, COORD_E3, COORD_F4, COORD_G5, COORD_H6, COORD_NO, COORD_NO, COORD_NO, COORD_NO}}, // 16
    {6, {COORD_F1, COORD_E2, COORD_D3, COORD_C4, COORD_B5, COORD_A6, COORD_NO, COORD_NO, COORD_NO, COORD_NO}}, // 17
    {6, {COORD_A3, COORD_B4, COORD_C5, COORD_D6, COORD_E7, COORD_F8, COORD_NO, COORD_NO, COORD_NO, COORD_NO}}, // 18
    {6, {COORD_H3, COORD_G4, COORD_F5, COORD_E6, COORD_D7, COORD_C8, COORD_NO, COORD_NO, COORD_NO, COORD_NO}}, // 19

    // d7
    {7, {COORD_B1, COORD_C2, COORD_D3, COORD_E4, COORD_F5, COORD_G6, COORD_H7, COORD_NO, COORD_NO, COORD_NO}}, // 20
    {7, {COORD_G1, COORD_F2, COORD_E3, COORD_D4, COORD_C5, COORD_B6, COORD_A7, COORD_NO, COORD_NO, COORD_NO}}, // 21
    {7, {COORD_A2, COORD_B3, COORD_C4, COORD_D5, COORD_E6, COORD_F7, COORD_G8, COORD_NO, COORD_NO, COORD_NO}}, // 22
    {7, {COORD_H2, COORD_G3, COORD_F4, COORD_E5, COORD_D6, COORD_C7, COORD_B8, COORD_NO, COORD_NO, COORD_NO}}, // 23

    // d8
    {8, {COORD_A1, COORD_B2, COORD_C3, COORD_D4, COORD_E5, COORD_F6, COORD_G7, COORD_H8, COORD_NO, COORD_NO}}, // 24
    {8, {COORD_H1, COORD_G2, COORD_F3, COORD_E4, COORD_D5, COORD_C6, COORD_B7, COORD_A8, COORD_NO, COORD_NO}}, // 25

    // edge + 2x
    {10, {COORD_B2, COORD_A1, COORD_B1, COORD_C1, COORD_D1, COORD_E1, COORD_F1, COORD_G1, COORD_H1, COORD_G2}}, // 26
    {10, {COORD_B2, COORD_A1, COORD_A2, COORD_A3, COORD_A4, COORD_A5, COORD_A6, COORD_A7, COORD_A8, COORD_B7}}, // 27
    {10, {COORD_B7, COORD_A8, COORD_B8, COORD_C8, COORD_D8, COORD_E8, COORD_F8, COORD_G8, COORD_H8, COORD_G7}}, // 28
    {10, {COORD_G2, COORD_H1, COORD_H2, COORD_H3, COORD_H4, COORD_H5, COORD_H6, COORD_H7, COORD_H8, COORD_G7}}, // 29

    // triangle
    {10, {COORD_A1, COORD_B1, COORD_C1, COORD_D1, COORD_A2, COORD_B2, COORD_C2, COORD_A3, COORD_B3, COORD_A4}}, // 30
    {10, {COORD_H1, COORD_G1, COORD_F1, COORD_E1, COORD_H2, COORD_G2, COORD_F2, COORD_H3, COORD_G3, COORD_H4}}, // 31
    {10, {COORD_A8, COORD_B8, COORD_C8, COORD_D8, COORD_A7, COORD_B7, COORD_C7, COORD_A6, COORD_B6, COORD_A5}}, // 32
    {10, {COORD_H8, COORD_G8, COORD_F8, COORD_E8, COORD_H7, COORD_G7, COORD_F7, COORD_H6, COORD_G6, COORD_H5}}, // 33

    // corner + block
    {10, {COORD_A1, COORD_C1, COORD_D1, COORD_E1, COORD_F1, COORD_H1, COORD_C2, COORD_D2, COORD_E2, COORD_F2}}, // 34
    {10, {COORD_A1, COORD_A3, COORD_A4, COORD_A5, COORD_A6, COORD_A8, COORD_B3, COORD_B4, COORD_B5, COORD_B6}}, // 35
    {10, {COORD_A8, COORD_C8, COORD_D8, COORD_E8, COORD_F8, COORD_H8, COORD_C7, COORD_D7, COORD_E7, COORD_F7}}, // 36
    {10, {COORD_H1, COORD_H3, COORD_H4, COORD_H5, COORD_H6, COORD_H8, COORD_G3, COORD_G4, COORD_G5, COORD_G6}}, // 37

    // cross
    {10, {COORD_A1, COORD_B2, COORD_C3, COORD_D4, COORD_B1, COORD_C2, COORD_D3, COORD_A2, COORD_B3, COORD_C4}}, // 38
    {10, {COORD_H1, COORD_G2, COORD_F3, COORD_E4, COORD_G1, COORD_F2, COORD_E3, COORD_H2, COORD_G3, COORD_F4}}, // 39
    {10, {COORD_A8, COORD_B7, COORD_C6, COORD_D5, COORD_B8, COORD_C7, COORD_D6, COORD_A7, COORD_B6, COORD_C5}}, // 40
    {10, {COORD_H8, COORD_G7, COORD_F6, COORD_E5, COORD_G8, COORD_F7, COORD_E6, COORD_H7, COORD_G6, COORD_F5}}, // 41

    // corner9
    {9, {COORD_A1, COORD_B1, COORD_C1, COORD_A2, COORD_B2, COORD_C2, COORD_A3, COORD_B3, COORD_C3, COORD_NO}}, // 42
    {9, {COORD_H1, COORD_G1, COORD_F1, COORD_H2, COORD_G2, COORD_F2, COORD_H3, COORD_G3, COORD_F3, COORD_NO}}, // 43
    {9, {COORD_A8, COORD_B8, COORD_C8, COORD_A7, COORD_B7, COORD_C7, COORD_A6, COORD_B6, COORD_C6, COORD_NO}}, // 44
    {9, {COORD_H8, COORD_G8, COORD_F8, COORD_H7, COORD_G7, COORD_F7, COORD_H6, COORD_G6, COORD_F6, COORD_NO}}, // 45

    // edge + y
    {10, {COORD_C2, COORD_A1, COORD_B1, COORD_C1, COORD_D1, COORD_E1, COORD_F1, COORD_G1, COORD_H1, COORD_F2}}, // 46
    {10, {COORD_B3, COORD_A1, COORD_A2, COORD_A3, COORD_A4, COORD_A5, COORD_A6, COORD_A7, COORD_A8, COORD_B6}}, // 47
    {10, {COORD_C7, COORD_A8, COORD_B8, COORD_C8, COORD_D8, COORD_E8, COORD_F8, COORD_G8, COORD_H8, COORD_F7}}, // 48
    {10, {COORD_G3, COORD_H1, COORD_H2, COORD_H3, COORD_H4, COORD_H5, COORD_H6, COORD_H7, COORD_H8, COORD_G6}}, // 49

    // narrow triangle
    {10, {COORD_A1, COORD_B1, COORD_C1, COORD_D1, COORD_E1, COORD_A2, COORD_B2, COORD_A3, COORD_A4, COORD_A5}}, // 50
    {10, {COORD_H1, COORD_G1, COORD_F1, COORD_E1, COORD_D1, COORD_H2, COORD_G2, COORD_H3, COORD_H4, COORD_H5}}, // 51
    {10, {COORD_A8, COORD_B8, COORD_C8, COORD_D8, COORD_E8, COORD_A7, COORD_B7, COORD_A6, COORD_A5, COORD_A4}}, // 52
    {10, {COORD_H8, COORD_G8, COORD_F8, COORD_E8, COORD_D8, COORD_H7, COORD_G7, COORD_H6, COORD_H5, COORD_H4}}, // 53

    // fish
    {10, {COORD_A1, COORD_B1, COORD_A2, COORD_B2, COORD_C2, COORD_D2, COORD_B3, COORD_C3, COORD_B4, COORD_D4}}, // 54
    {10, {COORD_H1, COORD_G1, COORD_H2, COORD_G2, COORD_F2, COORD_E2, COORD_G3, COORD_F3, COORD_G4, COORD_E4}}, // 55
    {10, {COORD_A8, COORD_B8, COORD_A7, COORD_B7, COORD_C7, COORD_D7, COORD_B6, COORD_C6, COORD_B5, COORD_D5}}, // 56
    {10, {COORD_H8, COORD_G8, COORD_H7, COORD_G7, COORD_F7, COORD_E7, COORD_G6, COORD_F6, COORD_G5, COORD_E5}}, // 57

    // kite
    {10, {COORD_A1, COORD_B1, COORD_A2, COORD_B2, COORD_C2, COORD_D2, COORD_E2, COORD_B3, COORD_B4, COORD_B5}}, // 58
    {10, {COORD_H1, COORD_G1, COORD_H2, COORD_G2, COORD_F2, COORD_E2, COORD_D2, COORD_G3, COORD_G4, COORD_G5}}, // 59
    {10, {COORD_A8, COORD_B8, COORD_A7, COORD_B7, COORD_C7, COORD_D7, COORD_E7, COORD_B6, COORD_B5, COORD_B4}}, // 60
    {10, {COORD_H8, COORD_G8, COORD_H7, COORD_G7, COORD_F7, COORD_E7, COORD_D7, COORD_G6, COORD_G5, COORD_G4}}  // 61
};

struct Coord_feature{
    uint_fast8_t feature;
    uint_fast16_t x;
};

struct Coord_to_feature{
    int n_features;
    Coord_feature features[MAX_CELL_PATTERNS];
};

constexpr Coord_to_feature coord_to_feature[HW2] = {
    {13, {{24, P30}, {28, P31}, {29, P31}, {33, P39}, {36, P34}, {37, P34}, {41, P39}, {45, P38}, {48, P31}, {49, P31}, {53, P39}, {57, P39}, {61, P39}}}, // H8
    {10, {{ 3, P30}, {22, P30}, {28, P32}, {33, P38}, {41, P35}, {45, P37}, {48, P32}, {53, P38}, {57, P38}, {61, P38}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // G8
    { 8, {{ 7, P30}, {18, P30}, {28, P33}, {33, P37}, {36, P35}, {45, P36}, {48, P33}, {53, P37}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // F8
    { 8, {{11, P30}, {14, P30}, {28, P34}, {33, P36}, {36, P36}, {48, P34}, {52, P35}, {53, P36}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // E8
    { 8, {{ 9, P30}, {15, P30}, {28, P35}, {32, P36}, {36, P37}, {48, P35}, {52, P36}, {53, P35}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // D8
    { 8, {{ 5, P30}, {19, P30}, {28, P36}, {32, P37}, {36, P38}, {44, P36}, {48, P36}, {52, P37}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // C8
    {10, {{ 1, P30}, {23, P30}, {28, P37}, {32, P38}, {40, P35}, {44, P37}, {48, P37}, {52, P38}, {56, P38}, {60, P38}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // B8
    {13, {{25, P30}, {27, P31}, {28, P38}, {32, P39}, {35, P34}, {36, P39}, {40, P39}, {44, P38}, {47, P31}, {48, P38}, {52, P39}, {56, P39}, {60, P39}}}, // A8
    {10, {{ 2, P30}, {20, P30}, {29, P32}, {33, P35}, {41, P32}, {45, P35}, {49, P32}, {53, P34}, {57, P37}, {61, P37}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // H7
    {11, {{ 2, P31}, { 3, P31}, {24, P31}, {28, P30}, {29, P30}, {33, P34}, {41, P38}, {45, P34}, {53, P33}, {57, P36}, {61, P36}, { 0, PNO}, { 0, PNO}}}, // G7
    {10, {{ 2, P32}, { 7, P31}, {22, P31}, {33, P33}, {36, P30}, {41, P34}, {45, P33}, {48, P30}, {57, P35}, {61, P35}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // F7
    { 8, {{ 2, P33}, {11, P31}, {15, P31}, {18, P31}, {36, P31}, {57, P34}, {60, P33}, {61, P34}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // E7
    { 8, {{ 2, P34}, { 9, P31}, {14, P31}, {19, P31}, {36, P32}, {56, P34}, {60, P34}, {61, P33}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // D7
    {10, {{ 2, P35}, { 5, P31}, {23, P31}, {32, P33}, {36, P33}, {40, P34}, {44, P33}, {48, P39}, {56, P35}, {60, P35}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // C7
    {11, {{ 1, P31}, { 2, P36}, {25, P31}, {27, P30}, {28, P39}, {32, P34}, {40, P38}, {44, P34}, {52, P33}, {56, P36}, {60, P36}, { 0, PNO}, { 0, PNO}}}, // B7
    {10, {{ 2, P37}, {21, P30}, {27, P32}, {32, P35}, {40, P32}, {44, P35}, {47, P32}, {52, P34}, {56, P37}, {60, P37}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // A7
    { 8, {{ 6, P30}, {16, P30}, {29, P33}, {33, P32}, {37, P35}, {45, P32}, {49, P33}, {53, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // H6
    {10, {{ 3, P32}, { 6, P31}, {20, P31}, {33, P31}, {37, P30}, {41, P31}, {45, P31}, {49, P30}, {57, P33}, {61, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // G6
    { 7, {{ 6, P32}, { 7, P32}, {15, P32}, {24, P32}, {41, P37}, {45, P30}, {57, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // F6
    { 5, {{ 6, P33}, {11, P32}, {19, P32}, {22, P32}, {41, P33}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // E6
    { 5, {{ 6, P34}, { 9, P32}, {18, P32}, {23, P32}, {40, P33}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // D6
    { 7, {{ 5, P32}, { 6, P35}, {14, P32}, {25, P32}, {40, P37}, {44, P30}, {56, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // C6
    {10, {{ 1, P32}, { 6, P36}, {21, P31}, {32, P31}, {35, P30}, {40, P31}, {44, P31}, {47, P30}, {56, P33}, {60, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // B6
    { 8, {{ 6, P37}, {17, P30}, {27, P33}, {32, P32}, {35, P35}, {44, P32}, {47, P33}, {52, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // A6
    { 8, {{10, P30}, {12, P30}, {29, P34}, {33, P30}, {37, P36}, {49, P34}, {51, P30}, {53, P31}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // H5
    { 8, {{ 3, P33}, {10, P31}, {15, P33}, {16, P31}, {37, P31}, {57, P31}, {59, P30}, {61, P31}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // G5
    { 5, {{ 7, P33}, {10, P32}, {19, P33}, {20, P32}, {41, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // F5
    { 6, {{10, P33}, {11, P33}, {23, P33}, {24, P33}, {41, P36}, {57, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // E5
    { 6, {{ 9, P33}, {10, P34}, {22, P33}, {25, P33}, {40, P36}, {56, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // D5
    { 5, {{ 5, P33}, {10, P35}, {18, P33}, {21, P32}, {40, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // C5
    { 8, {{ 1, P33}, {10, P36}, {14, P33}, {17, P31}, {35, P31}, {56, P31}, {58, P30}, {60, P31}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // B5
    { 8, {{10, P37}, {13, P30}, {27, P34}, {32, P30}, {35, P36}, {47, P34}, {50, P30}, {52, P31}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // A5
    { 8, {{ 8, P30}, {15, P34}, {29, P35}, {31, P30}, {37, P37}, {49, P35}, {51, P31}, {53, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // H4
    { 8, {{ 3, P34}, { 8, P31}, {12, P31}, {19, P34}, {37, P32}, {55, P31}, {59, P31}, {61, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // G4
    { 5, {{ 7, P34}, { 8, P32}, {16, P32}, {23, P34}, {39, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // F4
    { 6, {{ 8, P33}, {11, P34}, {20, P33}, {25, P34}, {39, P36}, {55, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // E4
    { 6, {{ 8, P34}, { 9, P34}, {21, P33}, {24, P34}, {38, P36}, {54, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // D4
    { 5, {{ 5, P34}, { 8, P35}, {17, P32}, {22, P34}, {38, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // C4
    { 8, {{ 1, P34}, { 8, P36}, {13, P31}, {18, P34}, {35, P32}, {54, P31}, {58, P31}, {60, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // B4
    { 8, {{ 8, P37}, {14, P34}, {27, P35}, {30, P30}, {35, P37}, {47, P35}, {50, P31}, {52, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // A4
    { 8, {{ 4, P30}, {19, P35}, {29, P36}, {31, P32}, {37, P38}, {43, P32}, {49, P36}, {51, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // H3
    {10, {{ 3, P35}, { 4, P31}, {23, P35}, {31, P31}, {37, P33}, {39, P31}, {43, P31}, {49, P39}, {55, P33}, {59, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // G3
    { 7, {{ 4, P32}, { 7, P35}, {12, P32}, {25, P35}, {39, P37}, {43, P30}, {55, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // F3
    { 5, {{ 4, P33}, {11, P35}, {16, P33}, {21, P34}, {39, P33}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // E3
    { 5, {{ 4, P34}, { 9, P35}, {17, P33}, {20, P34}, {38, P33}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // D3
    { 7, {{ 4, P35}, { 5, P35}, {13, P32}, {24, P35}, {38, P37}, {42, P30}, {54, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // C3
    {10, {{ 1, P35}, { 4, P36}, {22, P35}, {30, P31}, {35, P33}, {38, P31}, {42, P31}, {47, P39}, {54, P33}, {58, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // B3
    { 8, {{ 4, P37}, {18, P35}, {27, P36}, {30, P32}, {35, P38}, {42, P32}, {47, P36}, {50, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // A3
    {10, {{ 0, P30}, {23, P36}, {29, P37}, {31, P35}, {39, P32}, {43, P35}, {49, P37}, {51, P34}, {55, P37}, {59, P37}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // H2
    {11, {{ 0, P31}, { 3, P36}, {25, P36}, {26, P30}, {29, P39}, {31, P34}, {39, P38}, {43, P34}, {51, P33}, {55, P36}, {59, P36}, { 0, PNO}, { 0, PNO}}}, // G2
    {10, {{ 0, P32}, { 7, P36}, {21, P35}, {31, P33}, {34, P30}, {39, P34}, {43, P33}, {46, P30}, {55, P35}, {59, P35}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // F2
    { 8, {{ 0, P33}, {11, P36}, {12, P33}, {17, P34}, {34, P31}, {55, P34}, {58, P33}, {59, P34}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // E2
    { 8, {{ 0, P34}, { 9, P36}, {13, P33}, {16, P34}, {34, P32}, {54, P34}, {58, P34}, {59, P33}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // D2
    {10, {{ 0, P35}, { 5, P36}, {20, P35}, {30, P33}, {34, P33}, {38, P34}, {42, P33}, {46, P39}, {54, P35}, {58, P35}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // C2
    {11, {{ 0, P36}, { 1, P36}, {24, P36}, {26, P39}, {27, P39}, {30, P34}, {38, P38}, {42, P34}, {50, P33}, {54, P36}, {58, P36}, { 0, PNO}, { 0, PNO}}}, // B2
    {10, {{ 0, P37}, {22, P36}, {27, P37}, {30, P35}, {38, P32}, {42, P35}, {47, P37}, {50, P34}, {54, P37}, {58, P37}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // A2
    {13, {{25, P37}, {26, P31}, {29, P38}, {31, P39}, {34, P34}, {37, P39}, {39, P39}, {43, P38}, {46, P31}, {49, P38}, {51, P39}, {55, P39}, {59, P39}}}, // H1
    {10, {{ 3, P37}, {21, P36}, {26, P32}, {31, P38}, {39, P35}, {43, P37}, {46, P32}, {51, P38}, {55, P38}, {59, P38}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // G1
    { 8, {{ 7, P37}, {17, P35}, {26, P33}, {31, P37}, {34, P35}, {43, P36}, {46, P33}, {51, P37}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // F1
    { 8, {{11, P37}, {13, P34}, {26, P34}, {31, P36}, {34, P36}, {46, P34}, {50, P35}, {51, P36}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // E1
    { 8, {{ 9, P37}, {12, P34}, {26, P35}, {30, P36}, {34, P37}, {46, P35}, {50, P36}, {51, P35}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // D1
    { 8, {{ 5, P37}, {16, P35}, {26, P36}, {30, P37}, {34, P38}, {42, P36}, {46, P36}, {50, P37}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // C1
    {10, {{ 1, P37}, {20, P36}, {26, P37}, {30, P38}, {38, P35}, {42, P37}, {46, P37}, {50, P38}, {54, P38}, {58, P38}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // B1
    {13, {{24, P37}, {26, P38}, {27, P38}, {30, P39}, {34, P39}, {35, P39}, {38, P39}, {42, P38}, {46, P38}, {47, P38}, {50, P39}, {54, P39}, {58, P39}}}  // A1
};

constexpr uint_fast16_t pow3[11] = {1, P31, P32, P33, P34, P35, P36, P37, P38, P39, P310};
uint64_t stability_edge_arr[N_8BIT][N_8BIT][2];
int16_t pattern_arr[2][N_PHASES][N_PATTERNS][MAX_EVALUATE_IDX];
int16_t eval_sur0_sur1_arr[N_PHASES][MAX_SURROUND][MAX_SURROUND];
int16_t eval_canput0_canput1_arr[N_PHASES][MAX_CANPUT][MAX_CANPUT];
int16_t eval_stab0_stab1_arr[N_PHASES][MAX_STABILITY][MAX_STABILITY];
int16_t eval_num0_num1_arr[N_PHASES][MAX_STONE_NUM][MAX_STONE_NUM];
int16_t eval_canput_pattern[N_PHASES][N_CANPUT_PATTERNS][P48];

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

inline void probably_move_line(int p, int o, int place, int *np, int *no){
    int i, j;
    *np = p | (1 << place);
    for (i = place - 1; i >= 0 && (1 & (o >> i)); --i);
    if (1 & (p >> i)){
        for (j = place - 1; j > i; --j)
            *np ^= 1 << j;
    }
    for (i = place + 1; i < HW && (1 & (o >> i)); ++i);
    if (1 & (p >> i)){
        for (j = place + 1; j < i; ++j)
            *np ^= 1 << j;
    }
    *no = o & ~(*np);
}

int calc_stability_line(int b, int w, int ob, int ow){
    int i, nb, nw, res = 0b11111111;
    res &= b & ob;
    res &= w & ow;
    for (i = 0; i < HW; ++i){
        if ((1 & (b >> i)) == 0 && (1 & (w >> i)) == 0){
            probably_move_line(b, w, i, &nb, &nw);
            res &= calc_stability_line(nb, nw, ob, ow);
            probably_move_line(w, b, i, &nw, &nb);
            res &= calc_stability_line(nb, nw, ob, ow);
        }
    }
    return res;
}

inline void init_evaluation_base() {
    int place, b, w, stab;
    for (b = 0; b < N_8BIT; ++b) {
        for (w = b; w < N_8BIT; ++w){
            stab = calc_stability_line(b, w, b, w);
            stability_edge_arr[b][w][0] = 0;
            stability_edge_arr[b][w][1] = 0;
            for (place = 0; place < HW; ++place){
                if (1 & (stab >> place)){
                    stability_edge_arr[b][w][0] |= 1ULL << place;
                    stability_edge_arr[b][w][1] |= 1ULL << (place * HW);
                }
            }
            stability_edge_arr[w][b][0] = stability_edge_arr[b][w][0];
            stability_edge_arr[w][b][1] = stability_edge_arr[b][w][1];
        }
    }
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

#if USE_MULTI_THREAD
    void init_pattern_arr_rev(int id, int phase_idx, int pattern_idx, int siz){
        int ri;
        for (int i = 0; i < pow3[siz]; ++i){
            ri = swap_player_idx(i, siz);
            pattern_arr[1][phase_idx][pattern_idx][ri] = pattern_arr[0][phase_idx][pattern_idx][i];
        }
    }
#endif

inline bool init_evaluation_calc(){
    FILE* fp;
    #ifdef _WIN64
        if (fopen_s(&fp, "resources/eval.egev", "rb") != 0){
            cerr << "can't open eval.egev" << endl;
            return false;
        }
    #else
        fp = fopen("resources/eval.egev", "rb");
        if (fp == NULL){
            cerr << "can't open eval.egev" << endl;
            return false;
        }
    #endif
    int phase_idx, pattern_idx;
    constexpr int pattern_sizes[N_PATTERNS] = {8, 8, 8, 5, 6, 7, 8, 10, 10, 10, 10, 9, 10, 10, 10, 10};
    for (phase_idx = 0; phase_idx < N_PHASES; ++phase_idx){
        for (pattern_idx = 0; pattern_idx < N_PATTERNS; ++pattern_idx){
            if (fread(pattern_arr[0][phase_idx][pattern_idx], 2, pow3[pattern_sizes[pattern_idx]], fp) < pow3[pattern_sizes[pattern_idx]]){
                cerr << "eval.egev broken" << endl;
                fclose(fp);
                return false;
            }
        }
        if (fread(eval_sur0_sur1_arr[phase_idx], 2, MAX_SURROUND * MAX_SURROUND, fp) < MAX_SURROUND * MAX_SURROUND){
            cerr << "eval.egev broken" << endl;
            fclose(fp);
            return false;
        }
        if (fread(eval_canput0_canput1_arr[phase_idx], 2, MAX_CANPUT * MAX_CANPUT, fp) < MAX_CANPUT * MAX_CANPUT){
            cerr << "eval.egev broken" << endl;
            fclose(fp);
            return false;
        }
        if (fread(eval_stab0_stab1_arr[phase_idx], 2, MAX_STABILITY * MAX_STABILITY, fp) < MAX_STABILITY * MAX_STABILITY){
            cerr << "eval.egev broken" << endl;
            fclose(fp);
            return false;
        }
        if (fread(eval_num0_num1_arr[phase_idx], 2, MAX_STONE_NUM * MAX_STONE_NUM, fp) < MAX_STONE_NUM * MAX_STONE_NUM){
            cerr << "eval.egev broken" << endl;
            fclose(fp);
            return false;
        }
        if (fread(eval_canput_pattern[phase_idx], 2, N_CANPUT_PATTERNS * P48, fp) < N_CANPUT_PATTERNS * P48){
            cerr << "eval.egev broken" << endl;
            fclose(fp);
            return false;
        }
    }
    int i, ri;
    vector<future<void>> tasks;
    for (phase_idx = 0; phase_idx < N_PHASES; ++phase_idx){
        for (pattern_idx = 0; pattern_idx < N_PATTERNS; ++pattern_idx)
            tasks.emplace_back(thread_pool.push(init_pattern_arr_rev, phase_idx, pattern_idx, pattern_sizes[pattern_idx]));
    }
    for (future<void> &task: tasks)
        task.get();
    cerr << "evaluation function initialized" << endl;
    return true;
}

bool evaluate_init(){
    init_evaluation_base();
    return init_evaluation_calc();
}

inline uint64_t calc_surround_part(const uint64_t player, const int dr){
    return (player << dr | player >> dr);
}

inline int calc_surround(const uint64_t player, const uint64_t empties){
    return pop_count_ull(empties & (
        calc_surround_part(player & 0b0111111001111110011111100111111001111110011111100111111001111110ULL, 1) | 
        calc_surround_part(player & 0b0000000011111111111111111111111111111111111111111111111100000000ULL, HW) | 
        calc_surround_part(player & 0b0000000001111110011111100111111001111110011111100111111000000000ULL, HW_M1) | 
        calc_surround_part(player & 0b0000000001111110011111100111111001111110011111100111111000000000ULL, HW_P1)
    ));
}

inline void calc_stability(Board *b, int *stab0, int *stab1){
    uint64_t full_h, full_v, full_d7, full_d9;
    uint64_t edge_stability = 0, player_stability = 0, opponent_stability = 0, n_stability;
    uint64_t h, v, d7, d9;
    const uint64_t player_mask = b->player & 0b0000000001111110011111100111111001111110011111100111111000000000ULL;
    const uint64_t opponent_mask = b->opponent & 0b0000000001111110011111100111111001111110011111100111111000000000ULL;
    uint8_t pl, op;
    pl = b->player & 0b11111111U;
    op = b->opponent & 0b11111111U;
    edge_stability |= stability_edge_arr[pl][op][0] << 56;
    pl = (b->player >> 56) & 0b11111111U;
    op = (b->opponent >> 56) & 0b11111111U;
    edge_stability |= stability_edge_arr[pl][op][0];
    pl = join_v_line(b->player, 0);
    op = join_v_line(b->opponent, 0);
    edge_stability |= stability_edge_arr[pl][op][1] << 7;
    pl = join_v_line(b->player, 7);
    op = join_v_line(b->opponent, 7);
    edge_stability |= stability_edge_arr[pl][op][1];
    b->full_stability(&full_h, &full_v, &full_d7, &full_d9);

    n_stability = (edge_stability & b->player) | (full_h & full_v & full_d7 & full_d9 & player_mask);
    while (n_stability & ~player_stability){
        player_stability |= n_stability;
        h = (player_stability >> 1) | (player_stability << 1) | full_h;
        v = (player_stability >> HW) | (player_stability << HW) | full_v;
        d7 = (player_stability >> HW_M1) | (player_stability << HW_M1) | full_d7;
        d9 = (player_stability >> HW_P1) | (player_stability << HW_P1) | full_d9;
        n_stability = h & v & d7 & d9 & player_mask;
    }

    n_stability = (edge_stability & b->opponent) | (full_h & full_v & full_d7 & full_d9 & opponent_mask);
    while (n_stability & ~opponent_stability){
        opponent_stability |= n_stability;
        h = (opponent_stability >> 1) | (opponent_stability << 1) | full_h;
        v = (opponent_stability >> HW) | (opponent_stability << HW) | full_v;
        d7 = (opponent_stability >> HW_M1) | (opponent_stability << HW_M1) | full_d7;
        d9 = (opponent_stability >> HW_P1) | (opponent_stability << HW_P1) | full_d9;
        n_stability = h & v & d7 & d9 & opponent_mask;
    }

    *stab0 = pop_count_ull(player_stability);
    *stab1 = pop_count_ull(opponent_stability);
}

inline void calc_stability_edge(Board *b, int *stab0, int *stab1){
    uint64_t edge_stability = 0;
    uint8_t pl, op;
    pl = b->player & 0b11111111U;
    op = b->opponent & 0b11111111U;
    edge_stability |= stability_edge_arr[pl][op][0] << 56;
    pl = (b->player >> 56) & 0b11111111U;
    op = (b->opponent >> 56) & 0b11111111U;
    edge_stability |= stability_edge_arr[pl][op][0];
    pl = join_v_line(b->player, 0);
    op = join_v_line(b->opponent, 0);
    edge_stability |= stability_edge_arr[pl][op][1] << 7;
    pl = join_v_line(b->player, 7);
    op = join_v_line(b->opponent, 7);
    edge_stability |= stability_edge_arr[pl][op][1];
    *stab0 = pop_count_ull(edge_stability & b->player);
    *stab1 = pop_count_ull(edge_stability & b->opponent);
}

inline int calc_stability_edge_player(uint64_t player, uint64_t opponent){
    uint64_t edge_stability = 0;
    uint8_t pl, op;
    pl = player & 0b11111111U;
    op = opponent & 0b11111111U;
    edge_stability |= stability_edge_arr[pl][op][0] << 56;
    pl = (player >> 56) & 0b11111111U;
    op = (opponent >> 56) & 0b11111111U;
    edge_stability |= stability_edge_arr[pl][op][0];
    pl = join_v_line(player, 0);
    op = join_v_line(opponent, 0);
    edge_stability |= stability_edge_arr[pl][op][1] << 7;
    pl = join_v_line(player, 7);
    op = join_v_line(opponent, 7);
    edge_stability |= stability_edge_arr[pl][op][1];
    return pop_count_ull(edge_stability & player);
}
/*
inline int pick_cell(const Board *b, const int c){
    return 2 - (1 & (b->player >> c)) * 2 - (1 & (b->opponent >> c));
}

inline int pick_pattern(const int phase_idx, const int pattern_idx, const Board *b, const int p0, const int p1, const int p2, const int p3, const int p4){
    return pattern_arr[phase_idx][pattern_idx][pick_cell(b, p0) * P34 + pick_cell(b, p1) * P33 + pick_cell(b, p2) * P32 + pick_cell(b, p3) * P31 + pick_cell(b, p4)];
}

inline int pick_pattern(const int phase_idx, const int pattern_idx, const Board *b, const int p0, const int p1, const int p2, const int p3, const int p4, const int p5){
    return pattern_arr[phase_idx][pattern_idx][pick_cell(b, p0) * P35 + pick_cell(b, p1) * P34 + pick_cell(b, p2) * P33 + pick_cell(b, p3) * P32 + pick_cell(b, p4) * P31 + pick_cell(b, p5)];
}

inline int pick_pattern(const int phase_idx, const int pattern_idx, const Board *b, const int p0, const int p1, const int p2, const int p3, const int p4, const int p5, const int p6){
    return pattern_arr[phase_idx][pattern_idx][pick_cell(b, p0) * P36 + pick_cell(b, p1) * P35 + pick_cell(b, p2) * P34 + pick_cell(b, p3) * P33 + pick_cell(b, p4) * P32 + pick_cell(b, p5) * P31 + pick_cell(b, p6)];
}

inline int pick_pattern(const int phase_idx, const int pattern_idx, const Board *b, const int p0, const int p1, const int p2, const int p3, const int p4, const int p5, const int p6, const int p7){
    return pattern_arr[phase_idx][pattern_idx][pick_cell(b, p0) * P37 + pick_cell(b, p1) * P36 + pick_cell(b, p2) * P35 + pick_cell(b, p3) * P34 + pick_cell(b, p4) * P33 + pick_cell(b, p5) * P32 + pick_cell(b, p6) * P31 + pick_cell(b, p7)];
}

inline int pick_pattern(const int phase_idx, const int pattern_idx, const Board *b, const int p0, const int p1, const int p2, const int p3, const int p4, const int p5, const int p6, const int p7, const int p8){
    return pattern_arr[phase_idx][pattern_idx][pick_cell(b, p0) * P38 + pick_cell(b, p1) * P37 + pick_cell(b, p2) * P36 + pick_cell(b, p3) * P35 + pick_cell(b, p4) * P34 + pick_cell(b, p5) * P33 + pick_cell(b, p6) * P32 + pick_cell(b, p7) * P31 + pick_cell(b, p8)];
}

inline int pick_pattern(const int phase_idx, const int pattern_idx, const Board *b, const int p0, const int p1, const int p2, const int p3, const int p4, const int p5, const int p6, const int p7, const int p8, const int p9){
    return pattern_arr[phase_idx][pattern_idx][pick_cell(b, p0) * P39 + pick_cell(b, p1) * P38 + pick_cell(b, p2) * P37 + pick_cell(b, p3) * P36 + pick_cell(b, p4) * P35 + pick_cell(b, p5) * P34 + pick_cell(b, p6) * P33 + pick_cell(b, p7) * P32 + pick_cell(b, p8) * P31 + pick_cell(b, p9)];
}

inline int calc_pattern(const int phase_idx, Board *b){
    return 
        pick_pattern(phase_idx, 0, b, 8, 9, 10, 11, 12, 13, 14, 15) + pick_pattern(phase_idx, 0, b, 1, 9, 17, 25, 33, 41, 49, 57) + pick_pattern(phase_idx, 0, b, 48, 49, 50, 51, 52, 53, 54, 55) + pick_pattern(phase_idx, 0, b, 6, 14, 22, 30, 38, 46, 54, 62) + 
        pick_pattern(phase_idx, 1, b, 16, 17, 18, 19, 20, 21, 22, 23) + pick_pattern(phase_idx, 1, b, 2, 10, 18, 26, 34, 42, 50, 58) + pick_pattern(phase_idx, 1, b, 40, 41, 42, 43, 44, 45, 46, 47) + pick_pattern(phase_idx, 1, b, 5, 13, 21, 29, 37, 45, 53, 61) + 
        pick_pattern(phase_idx, 2, b, 24, 25, 26, 27, 28, 29, 30, 31) + pick_pattern(phase_idx, 2, b, 3, 11, 19, 27, 35, 43, 51, 59) + pick_pattern(phase_idx, 2, b, 32, 33, 34, 35, 36, 37, 38, 39) + pick_pattern(phase_idx, 2, b, 4, 12, 20, 28, 36, 44, 52, 60) + 
        pick_pattern(phase_idx, 3, b, 3, 12, 21, 30, 39) + pick_pattern(phase_idx, 3, b, 4, 11, 18, 25, 32) + pick_pattern(phase_idx, 3, b, 24, 33, 42, 51, 60) + pick_pattern(phase_idx, 3, b, 59, 52, 45, 38, 31) + 
        pick_pattern(phase_idx, 4, b, 2, 11, 20, 29, 38, 47) + pick_pattern(phase_idx, 4, b, 5, 12, 19, 26, 33, 40) + pick_pattern(phase_idx, 4, b, 16, 25, 34, 43, 52, 61) + pick_pattern(phase_idx, 4, b, 58, 51, 44, 37, 30, 23) + 
        pick_pattern(phase_idx, 5, b, 1, 10, 19, 28, 37, 46, 55) + pick_pattern(phase_idx, 5, b, 6, 13, 20, 27, 34, 41, 48) + pick_pattern(phase_idx, 5, b, 8, 17, 26, 35, 44, 53, 62) + pick_pattern(phase_idx, 5, b, 57, 50, 43, 36, 29, 22, 15) + 
        pick_pattern(phase_idx, 6, b, 0, 9, 18, 27, 36, 45, 54, 63) + pick_pattern(phase_idx, 6, b, 7, 14, 21, 28, 35, 42, 49, 56) + 
        pick_pattern(phase_idx, 7, b, 9, 0, 1, 2, 3, 4, 5, 6, 7, 14) + pick_pattern(phase_idx, 7, b, 9, 0, 8, 16, 24, 32, 40, 48, 56, 49) + pick_pattern(phase_idx, 7, b, 49, 56, 57, 58, 59, 60, 61, 62, 63, 54) + pick_pattern(phase_idx, 7, b, 54, 63, 55, 47, 39, 31, 23, 15, 7, 14) + 
        pick_pattern(phase_idx, 8, b, 0, 1, 2, 3, 8, 9, 10, 16, 17, 24) + pick_pattern(phase_idx, 8, b, 7, 6, 5, 4, 15, 14, 13, 23, 22, 31) + pick_pattern(phase_idx, 8, b, 63, 62, 61, 60, 55, 54, 53, 47, 46, 39) + pick_pattern(phase_idx, 8, b, 56, 57, 58, 59, 48, 49, 50, 40, 41, 32) + 
        pick_pattern(phase_idx, 9, b, 0, 2, 3, 4, 5, 7, 10, 11, 12, 13) + pick_pattern(phase_idx, 9, b, 0, 16, 24, 32, 40, 56, 17, 25, 33, 41) + pick_pattern(phase_idx, 9, b, 56, 58, 59, 60, 61, 63, 50, 51, 52, 53) + pick_pattern(phase_idx, 9, b, 7, 23, 31, 39, 47, 63, 22, 30, 38, 46) + 
        pick_pattern(phase_idx, 10, b, 0, 9, 18, 27, 1, 10, 19, 8, 17, 26) + pick_pattern(phase_idx, 10, b, 7, 14, 21, 28, 6, 13, 20, 15, 22, 29) + pick_pattern(phase_idx, 10, b, 56, 49, 42, 35, 57, 50, 43, 48, 41, 34) + pick_pattern(phase_idx, 10, b, 63, 54, 45, 36, 62, 53, 44, 55, 46, 37) + 
        pick_pattern(phase_idx, 11, b, 0, 1, 2, 8, 9, 10, 16, 17, 18) + pick_pattern(phase_idx, 11, b, 7, 6, 5, 15, 14, 13, 23, 22, 21) + pick_pattern(phase_idx, 11, b, 56, 57, 58, 48, 49, 50, 40, 41, 42) + pick_pattern(phase_idx, 11, b, 63, 62, 61, 55, 54, 53, 47, 46, 45) + 
        pick_pattern(phase_idx, 12, b, 10, 0, 1, 2, 3, 4, 5, 6, 7, 13) + pick_pattern(phase_idx, 12, b, 17, 0, 8, 16, 24, 32, 40, 48, 56, 41) + pick_pattern(phase_idx, 12, b, 50, 56, 57, 58, 59, 60, 61, 62, 63, 53) + pick_pattern(phase_idx, 12, b, 46, 63, 55, 47, 39, 31, 23, 15, 7, 22) + 
        pick_pattern(phase_idx, 13, b, 0, 1, 2, 3, 4, 8, 9, 16, 24, 32) + pick_pattern(phase_idx, 13, b, 7, 6, 5, 4, 3, 15, 14, 23, 31, 39) + pick_pattern(phase_idx, 13, b, 63, 62, 61, 60, 59, 55, 54, 47, 39, 31) + pick_pattern(phase_idx, 13, b, 56, 57, 58, 59, 60, 48, 49, 40, 32, 24) + 
        pick_pattern(phase_idx, 14, b, 0, 1, 8, 9, 10, 11, 17, 18, 25, 27) + pick_pattern(phase_idx, 14, b, 7, 6, 15, 14, 13, 12, 22, 21, 30, 28) + pick_pattern(phase_idx, 14, b, 56, 57, 48, 49, 50, 51, 41, 42, 33, 35) + pick_pattern(phase_idx, 14, b, 63, 62, 55, 54, 53, 52, 46, 45, 38, 36) + 
        pick_pattern(phase_idx, 15, b, 0, 1, 8, 9, 10, 11, 12, 17, 25, 33) + pick_pattern(phase_idx, 15, b, 7, 6, 15, 14, 13, 12, 11, 22, 30, 38) + pick_pattern(phase_idx, 15, b, 56, 57, 48, 49, 50, 51, 52, 41, 33, 25) + pick_pattern(phase_idx, 15, b, 63, 62, 55, 54, 53, 52, 51, 46, 38, 30);
}
*/

inline int pick_pattern(const int phase_idx, const int pattern_idx, const uint_fast8_t b_arr[], const int p0, const int p1, const int p2, const int p3, const int p4){
    return pattern_arr[0][phase_idx][pattern_idx][b_arr[p0] * P34 + b_arr[p1] * P33 + b_arr[p2] * P32 + b_arr[p3] * P31 + b_arr[p4]];
}

inline int pick_pattern(const int phase_idx, const int pattern_idx, const uint_fast8_t b_arr[], const int p0, const int p1, const int p2, const int p3, const int p4, const int p5){
    return pattern_arr[0][phase_idx][pattern_idx][b_arr[p0] * P35 + b_arr[p1] * P34 + b_arr[p2] * P33 + b_arr[p3] * P32 + b_arr[p4] * P31 + b_arr[p5]];
}

inline int pick_pattern(const int phase_idx, const int pattern_idx, const uint_fast8_t b_arr[], const int p0, const int p1, const int p2, const int p3, const int p4, const int p5, const int p6){
    return pattern_arr[0][phase_idx][pattern_idx][b_arr[p0] * P36 + b_arr[p1] * P35 + b_arr[p2] * P34 + b_arr[p3] * P33 + b_arr[p4] * P32 + b_arr[p5] * P31 + b_arr[p6]];
}

inline int pick_pattern(const int phase_idx, const int pattern_idx, const uint_fast8_t b_arr[], const int p0, const int p1, const int p2, const int p3, const int p4, const int p5, const int p6, const int p7){
    return pattern_arr[0][phase_idx][pattern_idx][b_arr[p0] * P37 + b_arr[p1] * P36 + b_arr[p2] * P35 + b_arr[p3] * P34 + b_arr[p4] * P33 + b_arr[p5] * P32 + b_arr[p6] * P31 + b_arr[p7]];
}

inline int pick_pattern(const int phase_idx, const int pattern_idx, const uint_fast8_t b_arr[], const int p0, const int p1, const int p2, const int p3, const int p4, const int p5, const int p6, const int p7, const int p8){
    return pattern_arr[0][phase_idx][pattern_idx][b_arr[p0] * P38 + b_arr[p1] * P37 + b_arr[p2] * P36 + b_arr[p3] * P35 + b_arr[p4] * P34 + b_arr[p5] * P33 + b_arr[p6] * P32 + b_arr[p7] * P31 + b_arr[p8]];
}

inline int pick_pattern(const int phase_idx, const int pattern_idx, const uint_fast8_t b_arr[], const int p0, const int p1, const int p2, const int p3, const int p4, const int p5, const int p6, const int p7, const int p8, const int p9){
    return pattern_arr[0][phase_idx][pattern_idx][b_arr[p0] * P39 + b_arr[p1] * P38 + b_arr[p2] * P37 + b_arr[p3] * P36 + b_arr[p4] * P35 + b_arr[p5] * P34 + b_arr[p6] * P33 + b_arr[p7] * P32 + b_arr[p8] * P31 + b_arr[p9]];
}

inline int calc_pattern_first(const int phase_idx, Board *b){
    uint_fast8_t b_arr[HW2];
    b->translate_to_arr_player(b_arr);
    return 
        pick_pattern(phase_idx, 0, b_arr, 8, 9, 10, 11, 12, 13, 14, 15) + pick_pattern(phase_idx, 0, b_arr, 1, 9, 17, 25, 33, 41, 49, 57) + pick_pattern(phase_idx, 0, b_arr, 48, 49, 50, 51, 52, 53, 54, 55) + pick_pattern(phase_idx, 0, b_arr, 6, 14, 22, 30, 38, 46, 54, 62) + 
        pick_pattern(phase_idx, 1, b_arr, 16, 17, 18, 19, 20, 21, 22, 23) + pick_pattern(phase_idx, 1, b_arr, 2, 10, 18, 26, 34, 42, 50, 58) + pick_pattern(phase_idx, 1, b_arr, 40, 41, 42, 43, 44, 45, 46, 47) + pick_pattern(phase_idx, 1, b_arr, 5, 13, 21, 29, 37, 45, 53, 61) + 
        pick_pattern(phase_idx, 2, b_arr, 24, 25, 26, 27, 28, 29, 30, 31) + pick_pattern(phase_idx, 2, b_arr, 3, 11, 19, 27, 35, 43, 51, 59) + pick_pattern(phase_idx, 2, b_arr, 32, 33, 34, 35, 36, 37, 38, 39) + pick_pattern(phase_idx, 2, b_arr, 4, 12, 20, 28, 36, 44, 52, 60) + 
        pick_pattern(phase_idx, 3, b_arr, 3, 12, 21, 30, 39) + pick_pattern(phase_idx, 3, b_arr, 4, 11, 18, 25, 32) + pick_pattern(phase_idx, 3, b_arr, 24, 33, 42, 51, 60) + pick_pattern(phase_idx, 3, b_arr, 59, 52, 45, 38, 31) + 
        pick_pattern(phase_idx, 4, b_arr, 2, 11, 20, 29, 38, 47) + pick_pattern(phase_idx, 4, b_arr, 5, 12, 19, 26, 33, 40) + pick_pattern(phase_idx, 4, b_arr, 16, 25, 34, 43, 52, 61) + pick_pattern(phase_idx, 4, b_arr, 58, 51, 44, 37, 30, 23) + 
        pick_pattern(phase_idx, 5, b_arr, 1, 10, 19, 28, 37, 46, 55) + pick_pattern(phase_idx, 5, b_arr, 6, 13, 20, 27, 34, 41, 48) + pick_pattern(phase_idx, 5, b_arr, 8, 17, 26, 35, 44, 53, 62) + pick_pattern(phase_idx, 5, b_arr, 57, 50, 43, 36, 29, 22, 15) + 
        pick_pattern(phase_idx, 6, b_arr, 0, 9, 18, 27, 36, 45, 54, 63) + pick_pattern(phase_idx, 6, b_arr, 7, 14, 21, 28, 35, 42, 49, 56) + 
        pick_pattern(phase_idx, 7, b_arr, 9, 0, 1, 2, 3, 4, 5, 6, 7, 14) + pick_pattern(phase_idx, 7, b_arr, 9, 0, 8, 16, 24, 32, 40, 48, 56, 49) + pick_pattern(phase_idx, 7, b_arr, 49, 56, 57, 58, 59, 60, 61, 62, 63, 54) + pick_pattern(phase_idx, 7, b_arr, 54, 63, 55, 47, 39, 31, 23, 15, 7, 14) + 
        pick_pattern(phase_idx, 8, b_arr, 0, 1, 2, 3, 8, 9, 10, 16, 17, 24) + pick_pattern(phase_idx, 8, b_arr, 7, 6, 5, 4, 15, 14, 13, 23, 22, 31) + pick_pattern(phase_idx, 8, b_arr, 63, 62, 61, 60, 55, 54, 53, 47, 46, 39) + pick_pattern(phase_idx, 8, b_arr, 56, 57, 58, 59, 48, 49, 50, 40, 41, 32) + 
        pick_pattern(phase_idx, 9, b_arr, 0, 2, 3, 4, 5, 7, 10, 11, 12, 13) + pick_pattern(phase_idx, 9, b_arr, 0, 16, 24, 32, 40, 56, 17, 25, 33, 41) + pick_pattern(phase_idx, 9, b_arr, 56, 58, 59, 60, 61, 63, 50, 51, 52, 53) + pick_pattern(phase_idx, 9, b_arr, 7, 23, 31, 39, 47, 63, 22, 30, 38, 46) + 
        pick_pattern(phase_idx, 10, b_arr, 0, 9, 18, 27, 1, 10, 19, 8, 17, 26) + pick_pattern(phase_idx, 10, b_arr, 7, 14, 21, 28, 6, 13, 20, 15, 22, 29) + pick_pattern(phase_idx, 10, b_arr, 56, 49, 42, 35, 57, 50, 43, 48, 41, 34) + pick_pattern(phase_idx, 10, b_arr, 63, 54, 45, 36, 62, 53, 44, 55, 46, 37) + 
        pick_pattern(phase_idx, 11, b_arr, 0, 1, 2, 8, 9, 10, 16, 17, 18) + pick_pattern(phase_idx, 11, b_arr, 7, 6, 5, 15, 14, 13, 23, 22, 21) + pick_pattern(phase_idx, 11, b_arr, 56, 57, 58, 48, 49, 50, 40, 41, 42) + pick_pattern(phase_idx, 11, b_arr, 63, 62, 61, 55, 54, 53, 47, 46, 45) + 
        pick_pattern(phase_idx, 12, b_arr, 10, 0, 1, 2, 3, 4, 5, 6, 7, 13) + pick_pattern(phase_idx, 12, b_arr, 17, 0, 8, 16, 24, 32, 40, 48, 56, 41) + pick_pattern(phase_idx, 12, b_arr, 50, 56, 57, 58, 59, 60, 61, 62, 63, 53) + pick_pattern(phase_idx, 12, b_arr, 46, 63, 55, 47, 39, 31, 23, 15, 7, 22) + 
        pick_pattern(phase_idx, 13, b_arr, 0, 1, 2, 3, 4, 8, 9, 16, 24, 32) + pick_pattern(phase_idx, 13, b_arr, 7, 6, 5, 4, 3, 15, 14, 23, 31, 39) + pick_pattern(phase_idx, 13, b_arr, 63, 62, 61, 60, 59, 55, 54, 47, 39, 31) + pick_pattern(phase_idx, 13, b_arr, 56, 57, 58, 59, 60, 48, 49, 40, 32, 24) + 
        pick_pattern(phase_idx, 14, b_arr, 0, 1, 8, 9, 10, 11, 17, 18, 25, 27) + pick_pattern(phase_idx, 14, b_arr, 7, 6, 15, 14, 13, 12, 22, 21, 30, 28) + pick_pattern(phase_idx, 14, b_arr, 56, 57, 48, 49, 50, 51, 41, 42, 33, 35) + pick_pattern(phase_idx, 14, b_arr, 63, 62, 55, 54, 53, 52, 46, 45, 38, 36) + 
        pick_pattern(phase_idx, 15, b_arr, 0, 1, 8, 9, 10, 11, 12, 17, 25, 33) + pick_pattern(phase_idx, 15, b_arr, 7, 6, 15, 14, 13, 12, 11, 22, 30, 38) + pick_pattern(phase_idx, 15, b_arr, 56, 57, 48, 49, 50, 51, 52, 41, 33, 25) + pick_pattern(phase_idx, 15, b_arr, 63, 62, 55, 54, 53, 52, 51, 46, 38, 30);
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
        pattern_arr[search->eval_feature_reversed][phase_idx][11][search->eval_features[42]] + pattern_arr[search->eval_feature_reversed][phase_idx][11][search->eval_features[43]] + pattern_arr[search->eval_feature_reversed][phase_idx][11][search->eval_features[44]] + pattern_arr[search->eval_feature_reversed][phase_idx][11][search->eval_features[45]] + 
        pattern_arr[search->eval_feature_reversed][phase_idx][12][search->eval_features[46]] + pattern_arr[search->eval_feature_reversed][phase_idx][12][search->eval_features[47]] + pattern_arr[search->eval_feature_reversed][phase_idx][12][search->eval_features[48]] + pattern_arr[search->eval_feature_reversed][phase_idx][12][search->eval_features[49]] + 
        pattern_arr[search->eval_feature_reversed][phase_idx][13][search->eval_features[50]] + pattern_arr[search->eval_feature_reversed][phase_idx][13][search->eval_features[51]] + pattern_arr[search->eval_feature_reversed][phase_idx][13][search->eval_features[52]] + pattern_arr[search->eval_feature_reversed][phase_idx][13][search->eval_features[53]] + 
        pattern_arr[search->eval_feature_reversed][phase_idx][14][search->eval_features[54]] + pattern_arr[search->eval_feature_reversed][phase_idx][14][search->eval_features[55]] + pattern_arr[search->eval_feature_reversed][phase_idx][14][search->eval_features[56]] + pattern_arr[search->eval_feature_reversed][phase_idx][14][search->eval_features[57]] + 
        pattern_arr[search->eval_feature_reversed][phase_idx][15][search->eval_features[58]] + pattern_arr[search->eval_feature_reversed][phase_idx][15][search->eval_features[59]] + pattern_arr[search->eval_feature_reversed][phase_idx][15][search->eval_features[60]] + pattern_arr[search->eval_feature_reversed][phase_idx][15][search->eval_features[61]];
}

inline int create_canput_line_h(uint64_t b, uint64_t w, int t){
    return (((w >> (HW * t)) & 0b11111111) << HW) | ((b >> (HW * t)) & 0b11111111);
}

inline int create_canput_line_v(uint64_t b, uint64_t w, int t){
    return (join_v_line(w, t) << HW) | join_v_line(b, t);
}

inline int calc_canput_pattern(const int phase_idx, Board *b, const uint64_t player_mobility, const uint64_t opponent_mobility){
    return 
        eval_canput_pattern[phase_idx][0][create_canput_line_h(player_mobility, opponent_mobility, 0)] + 
        eval_canput_pattern[phase_idx][0][create_canput_line_h(player_mobility, opponent_mobility, 7)] + 
        eval_canput_pattern[phase_idx][0][create_canput_line_v(player_mobility, opponent_mobility, 0)] + 
        eval_canput_pattern[phase_idx][0][create_canput_line_v(player_mobility, opponent_mobility, 7)] + 
        eval_canput_pattern[phase_idx][1][create_canput_line_h(player_mobility, opponent_mobility, 1)] + 
        eval_canput_pattern[phase_idx][1][create_canput_line_h(player_mobility, opponent_mobility, 6)] + 
        eval_canput_pattern[phase_idx][1][create_canput_line_v(player_mobility, opponent_mobility, 1)] + 
        eval_canput_pattern[phase_idx][1][create_canput_line_v(player_mobility, opponent_mobility, 6)] + 
        eval_canput_pattern[phase_idx][2][create_canput_line_h(player_mobility, opponent_mobility, 2)] + 
        eval_canput_pattern[phase_idx][2][create_canput_line_h(player_mobility, opponent_mobility, 5)] + 
        eval_canput_pattern[phase_idx][2][create_canput_line_v(player_mobility, opponent_mobility, 2)] + 
        eval_canput_pattern[phase_idx][2][create_canput_line_v(player_mobility, opponent_mobility, 5)] + 
        eval_canput_pattern[phase_idx][3][create_canput_line_h(player_mobility, opponent_mobility, 3)] + 
        eval_canput_pattern[phase_idx][3][create_canput_line_h(player_mobility, opponent_mobility, 4)] + 
        eval_canput_pattern[phase_idx][3][create_canput_line_v(player_mobility, opponent_mobility, 3)] + 
        eval_canput_pattern[phase_idx][3][create_canput_line_v(player_mobility, opponent_mobility, 4)];
}

inline int end_evaluate(Board *b){
    int res = b->score_player();
    return score_to_value(res);
}

inline int mid_evaluate(Board *b){
    int phase_idx, sur0, sur1, canput0, canput1, stab0, stab1, num0, num1;
    uint64_t player_mobility, opponent_mobility, empties;
    player_mobility = calc_legal(b->player, b->opponent);
    opponent_mobility = calc_legal(b->opponent, b->player);
    canput0 = min(MAX_CANPUT - 1, pop_count_ull(player_mobility));
    canput1 = min(MAX_CANPUT - 1, pop_count_ull(opponent_mobility));
    if (canput0 == 0 && canput1 == 0)
        return end_evaluate(b);
    phase_idx = b->phase();
    empties = ~(b->player | b->opponent);
    sur0 = min(MAX_SURROUND - 1, calc_surround(b->player, empties));
    sur1 = min(MAX_SURROUND - 1, calc_surround(b->opponent, empties));
    calc_stability(b, &stab0, &stab1);
    num0 = pop_count_ull(b->player);
    num1 = pop_count_ull(b->opponent);
    //cerr << calc_pattern(phase_idx, b) << " " << eval_sur0_sur1_arr[phase_idx][sur0][sur1] << " " << eval_canput0_canput1_arr[phase_idx][canput0][canput1] << " "
    //    << eval_stab0_stab1_arr[phase_idx][stab0][stab1] << " " << eval_num0_num1_arr[phase_idx][num0][num1] << " " << calc_canput_pattern(phase_idx, b, player_mobility, opponent_mobility) << endl;
    int res = calc_pattern_first(phase_idx, b) + 
        eval_sur0_sur1_arr[phase_idx][sur0][sur1] + 
        eval_canput0_canput1_arr[phase_idx][canput0][canput1] + 
        eval_stab0_stab1_arr[phase_idx][stab0][stab1] + 
        eval_num0_num1_arr[phase_idx][num0][num1] + 
        calc_canput_pattern(phase_idx, b, player_mobility, opponent_mobility);
    //return score_modification(phase_idx, res);
    //cerr << res << endl;
    #if EVALUATION_STEP_WIDTH_MODE == 0
        if (res > 0)
            res += STEP_2;
        else if (res < 0)
            res -= STEP_2;
        res /= STEP;
    #elif EVALUATION_STEP_WIDTH_MODE == 1
        if (res > 0)
            res += STEP;
        else if (res < 0)
            res -= STEP;
        res /= STEP * 2;
    #elif EVALUATION_STEP_WIDTH_MODE == 2
        if (res > 0)
            res += STEP / 4;
        else if (res < 0)
            res -= STEP / 4;
        res /= STEP_2;
    
    #elif EVALUATION_STEP_WIDTH_MODE == 3
        if (res > 0)
            res += STEP / 8;
        else if (res < 0)
            res -= STEP / 8;
        res /= STEP / 4;
    #elif EVALUATION_STEP_WIDTH_MODE == 4
        if (res > 0)
            res += STEP / 16;
        else if (res < 0)
            res -= STEP / 16;
        res /= STEP / 8;
    #elif EVALUATION_STEP_WIDTH_MODE == 5
        if (res > 0)
            res += STEP / 32;
        else if (res < 0)
            res -= STEP / 32;
        res /= STEP / 16;
    #elif EVALUATION_STEP_WIDTH_MODE == 6
        if (res > 0)
            res += STEP / 64;
        else if (res < 0)
            res -= STEP / 64;
        res /= STEP / 32;
    #endif
    //cerr << res << " " << value_to_score_double(res) << endl;
    return max(-SCORE_MAX, min(SCORE_MAX, res));
}

inline int mid_evaluate_diff(Search *search, const bool *searching){
    if (!(*searching))
        return SCORE_UNDEFINED;
    int phase_idx, sur0, sur1, canput0, canput1, stab0, stab1, num0, num1;
    uint64_t player_mobility, opponent_mobility, empties;
    player_mobility = calc_legal(search->board.player, search->board.opponent);
    opponent_mobility = calc_legal(search->board.opponent, search->board.player);
    canput0 = min(MAX_CANPUT - 1, pop_count_ull(player_mobility));
    canput1 = min(MAX_CANPUT - 1, pop_count_ull(opponent_mobility));
    if (canput0 == 0 && canput1 == 0)
        return end_evaluate(&search->board);
    phase_idx = search->board.phase();
    empties = ~(search->board.player | search->board.opponent);
    sur0 = min(MAX_SURROUND - 1, calc_surround(search->board.player, empties));
    sur1 = min(MAX_SURROUND - 1, calc_surround(search->board.opponent, empties));
    calc_stability(&search->board, &stab0, &stab1);
    num0 = pop_count_ull(search->board.player);
    num1 = pop_count_ull(search->board.opponent);
    //cerr << calc_pattern(phase_idx, b) << " " << eval_sur0_sur1_arr[phase_idx][sur0][sur1] << " " << eval_canput0_canput1_arr[phase_idx][canput0][canput1] << " "
    //    << eval_stab0_stab1_arr[phase_idx][stab0][stab1] << " " << eval_num0_num1_arr[phase_idx][num0][num1] << " " << calc_canput_pattern(phase_idx, b, player_mobility, opponent_mobility) << endl;
    int res = calc_pattern_diff(phase_idx, search) + 
        eval_sur0_sur1_arr[phase_idx][sur0][sur1] + 
        eval_canput0_canput1_arr[phase_idx][canput0][canput1] + 
        eval_stab0_stab1_arr[phase_idx][stab0][stab1] + 
        eval_num0_num1_arr[phase_idx][num0][num1] + 
        calc_canput_pattern(phase_idx, &search->board, player_mobility, opponent_mobility);
    //return score_modification(phase_idx, res);
    #if EVALUATION_STEP_WIDTH_MODE == 0
        if (res > 0)
            res += STEP_2;
        else if (res < 0)
            res -= STEP_2;
        res /= STEP;
    #elif EVALUATION_STEP_WIDTH_MODE == 1
        if (res > 0)
            res += STEP;
        else if (res < 0)
            res -= STEP;
        res /= STEP * 2;
    #elif EVALUATION_STEP_WIDTH_MODE == 2
        if (res > 0)
            res += STEP / 4;
        else if (res < 0)
            res -= STEP / 4;
        res /= STEP_2;
    
    #elif EVALUATION_STEP_WIDTH_MODE == 3
        if (res > 0)
            res += STEP / 8;
        else if (res < 0)
            res -= STEP / 8;
        res /= STEP / 4;
    #elif EVALUATION_STEP_WIDTH_MODE == 4
        if (res > 0)
            res += STEP / 16;
        else if (res < 0)
            res -= STEP / 16;
        res /= STEP / 8;
    #elif EVALUATION_STEP_WIDTH_MODE == 5
        if (res > 0)
            res += STEP / 32;
        else if (res < 0)
            res -= STEP / 32;
        res /= STEP / 16;
    #elif EVALUATION_STEP_WIDTH_MODE == 6
        if (res > 0)
            res += STEP / 64;
        else if (res < 0)
            res -= STEP / 64;
        res /= STEP / 32;
    #endif
    //cerr << res << " " << value_to_score_double(res) << endl;
    return max(-SCORE_MAX, min(SCORE_MAX, res));
}

inline int pick_pattern_idx(const uint_fast8_t b_arr[], const Feature_to_coord *f){
    int res = 0;
    for (int i = 0; i < f->n_cells; ++i){
        //cerr << (int)f->cells[i] << " ";
        res *= 3;
        res += b_arr[HW2_M1 - f->cells[i]];
    }
    //cerr << endl;
    return res;
}

inline void calc_features(Search *search){
    uint_fast8_t b_arr[HW2];
    search->board.translate_to_arr_player(b_arr);
    for (int i = 0; i < N_SYMMETRY_PATTERNS; ++i){
        //cerr << i << " ";
        search->eval_features[i] = pick_pattern_idx(b_arr, &feature_to_coord[i]);
    }
    search->eval_feature_reversed = 0;
}

inline bool check_features(Search *search){
    uint_fast8_t b_arr[HW2];
    search->board.translate_to_arr_player(b_arr);
    for (int i = 0; i < N_SYMMETRY_PATTERNS; ++i){
        //cerr << i << " ";
        if (search->eval_features[i] != pick_pattern_idx(b_arr, &feature_to_coord[i])){
            cerr << i << " " << search->eval_features[i] << " " << pick_pattern_idx(b_arr, &feature_to_coord[i]) << endl;
            return true;
        }
    }
    return false;
}

inline void eval_move(Search *search, const Flip *flip){
    #if USE_FAST_DIFF_EVAL
        if (search->eval_feature_reversed){
            switch (coord_to_feature[flip->pos].n_features){
                case 13: search->eval_features[coord_to_feature[flip->pos].features[12].feature] -= coord_to_feature[flip->pos].features[12].x; [[fallthrough]];
                case 12: search->eval_features[coord_to_feature[flip->pos].features[11].feature] -= coord_to_feature[flip->pos].features[11].x; [[fallthrough]];
                case 11: search->eval_features[coord_to_feature[flip->pos].features[10].feature] -= coord_to_feature[flip->pos].features[10].x; [[fallthrough]];
                case 10: search->eval_features[coord_to_feature[flip->pos].features[ 9].feature] -= coord_to_feature[flip->pos].features[ 9].x; [[fallthrough]];
                case  9: search->eval_features[coord_to_feature[flip->pos].features[ 8].feature] -= coord_to_feature[flip->pos].features[ 8].x; [[fallthrough]];
                case  8: search->eval_features[coord_to_feature[flip->pos].features[ 7].feature] -= coord_to_feature[flip->pos].features[ 7].x; [[fallthrough]];
                case  7: search->eval_features[coord_to_feature[flip->pos].features[ 6].feature] -= coord_to_feature[flip->pos].features[ 6].x; [[fallthrough]];
                case  6: search->eval_features[coord_to_feature[flip->pos].features[ 5].feature] -= coord_to_feature[flip->pos].features[ 5].x; [[fallthrough]];
                case  5: search->eval_features[coord_to_feature[flip->pos].features[ 4].feature] -= coord_to_feature[flip->pos].features[ 4].x; [[fallthrough]];
                case  4: search->eval_features[coord_to_feature[flip->pos].features[ 3].feature] -= coord_to_feature[flip->pos].features[ 3].x; [[fallthrough]];
                case  3: search->eval_features[coord_to_feature[flip->pos].features[ 2].feature] -= coord_to_feature[flip->pos].features[ 2].x; [[fallthrough]];
                case  2: search->eval_features[coord_to_feature[flip->pos].features[ 1].feature] -= coord_to_feature[flip->pos].features[ 1].x; [[fallthrough]];
                case  1: search->eval_features[coord_to_feature[flip->pos].features[ 0].feature] -= coord_to_feature[flip->pos].features[ 0].x; [[fallthrough]];
                case  0: break;
            }
            uint64_t f = flip->flip;
            for (uint_fast8_t cell = first_bit(&f); f; cell = next_bit(&f)){
                switch (coord_to_feature[cell].n_features){
                    case 13: search->eval_features[coord_to_feature[cell].features[12].feature] += coord_to_feature[cell].features[12].x; [[fallthrough]];
                    case 12: search->eval_features[coord_to_feature[cell].features[11].feature] += coord_to_feature[cell].features[11].x; [[fallthrough]];
                    case 11: search->eval_features[coord_to_feature[cell].features[10].feature] += coord_to_feature[cell].features[10].x; [[fallthrough]];
                    case 10: search->eval_features[coord_to_feature[cell].features[ 9].feature] += coord_to_feature[cell].features[ 9].x; [[fallthrough]];
                    case  9: search->eval_features[coord_to_feature[cell].features[ 8].feature] += coord_to_feature[cell].features[ 8].x; [[fallthrough]];
                    case  8: search->eval_features[coord_to_feature[cell].features[ 7].feature] += coord_to_feature[cell].features[ 7].x; [[fallthrough]];
                    case  7: search->eval_features[coord_to_feature[cell].features[ 6].feature] += coord_to_feature[cell].features[ 6].x; [[fallthrough]];
                    case  6: search->eval_features[coord_to_feature[cell].features[ 5].feature] += coord_to_feature[cell].features[ 5].x; [[fallthrough]];
                    case  5: search->eval_features[coord_to_feature[cell].features[ 4].feature] += coord_to_feature[cell].features[ 4].x; [[fallthrough]];
                    case  4: search->eval_features[coord_to_feature[cell].features[ 3].feature] += coord_to_feature[cell].features[ 3].x; [[fallthrough]];
                    case  3: search->eval_features[coord_to_feature[cell].features[ 2].feature] += coord_to_feature[cell].features[ 2].x; [[fallthrough]];
                    case  2: search->eval_features[coord_to_feature[cell].features[ 1].feature] += coord_to_feature[cell].features[ 1].x; [[fallthrough]];
                    case  1: search->eval_features[coord_to_feature[cell].features[ 0].feature] += coord_to_feature[cell].features[ 0].x; [[fallthrough]];
                    case  0: break;
                }
            }
        } else{
            switch (coord_to_feature[flip->pos].n_features){
                case 13: search->eval_features[coord_to_feature[flip->pos].features[12].feature] -= 2 * coord_to_feature[flip->pos].features[12].x; [[fallthrough]];
                case 12: search->eval_features[coord_to_feature[flip->pos].features[11].feature] -= 2 * coord_to_feature[flip->pos].features[11].x; [[fallthrough]];
                case 11: search->eval_features[coord_to_feature[flip->pos].features[10].feature] -= 2 * coord_to_feature[flip->pos].features[10].x; [[fallthrough]];
                case 10: search->eval_features[coord_to_feature[flip->pos].features[ 9].feature] -= 2 * coord_to_feature[flip->pos].features[ 9].x; [[fallthrough]];
                case  9: search->eval_features[coord_to_feature[flip->pos].features[ 8].feature] -= 2 * coord_to_feature[flip->pos].features[ 8].x; [[fallthrough]];
                case  8: search->eval_features[coord_to_feature[flip->pos].features[ 7].feature] -= 2 * coord_to_feature[flip->pos].features[ 7].x; [[fallthrough]];
                case  7: search->eval_features[coord_to_feature[flip->pos].features[ 6].feature] -= 2 * coord_to_feature[flip->pos].features[ 6].x; [[fallthrough]];
                case  6: search->eval_features[coord_to_feature[flip->pos].features[ 5].feature] -= 2 * coord_to_feature[flip->pos].features[ 5].x; [[fallthrough]];
                case  5: search->eval_features[coord_to_feature[flip->pos].features[ 4].feature] -= 2 * coord_to_feature[flip->pos].features[ 4].x; [[fallthrough]];
                case  4: search->eval_features[coord_to_feature[flip->pos].features[ 3].feature] -= 2 * coord_to_feature[flip->pos].features[ 3].x; [[fallthrough]];
                case  3: search->eval_features[coord_to_feature[flip->pos].features[ 2].feature] -= 2 * coord_to_feature[flip->pos].features[ 2].x; [[fallthrough]];
                case  2: search->eval_features[coord_to_feature[flip->pos].features[ 1].feature] -= 2 * coord_to_feature[flip->pos].features[ 1].x; [[fallthrough]];
                case  1: search->eval_features[coord_to_feature[flip->pos].features[ 0].feature] -= 2 * coord_to_feature[flip->pos].features[ 0].x; [[fallthrough]];
                case  0: break;
            }
            uint64_t f = flip->flip;
            for (uint_fast8_t cell = first_bit(&f); f; cell = next_bit(&f)){
                switch (coord_to_feature[cell].n_features){
                    case 13: search->eval_features[coord_to_feature[cell].features[12].feature] -= coord_to_feature[cell].features[12].x; [[fallthrough]];
                    case 12: search->eval_features[coord_to_feature[cell].features[11].feature] -= coord_to_feature[cell].features[11].x; [[fallthrough]];
                    case 11: search->eval_features[coord_to_feature[cell].features[10].feature] -= coord_to_feature[cell].features[10].x; [[fallthrough]];
                    case 10: search->eval_features[coord_to_feature[cell].features[ 9].feature] -= coord_to_feature[cell].features[ 9].x; [[fallthrough]];
                    case  9: search->eval_features[coord_to_feature[cell].features[ 8].feature] -= coord_to_feature[cell].features[ 8].x; [[fallthrough]];
                    case  8: search->eval_features[coord_to_feature[cell].features[ 7].feature] -= coord_to_feature[cell].features[ 7].x; [[fallthrough]];
                    case  7: search->eval_features[coord_to_feature[cell].features[ 6].feature] -= coord_to_feature[cell].features[ 6].x; [[fallthrough]];
                    case  6: search->eval_features[coord_to_feature[cell].features[ 5].feature] -= coord_to_feature[cell].features[ 5].x; [[fallthrough]];
                    case  5: search->eval_features[coord_to_feature[cell].features[ 4].feature] -= coord_to_feature[cell].features[ 4].x; [[fallthrough]];
                    case  4: search->eval_features[coord_to_feature[cell].features[ 3].feature] -= coord_to_feature[cell].features[ 3].x; [[fallthrough]];
                    case  3: search->eval_features[coord_to_feature[cell].features[ 2].feature] -= coord_to_feature[cell].features[ 2].x; [[fallthrough]];
                    case  2: search->eval_features[coord_to_feature[cell].features[ 1].feature] -= coord_to_feature[cell].features[ 1].x; [[fallthrough]];
                    case  1: search->eval_features[coord_to_feature[cell].features[ 0].feature] -= coord_to_feature[cell].features[ 0].x; [[fallthrough]];
                    case  0: break;
                }
            }
        }
    #else
        int i;
        if (search->eval_feature_reversed){
            for (i = 0; i < coord_to_feature[flip->pos].n_features; ++i)
                search->eval_features[coord_to_feature[flip->pos].features[i].feature] -= coord_to_feature[flip->pos].features[i].x;
            uint64_t f = flip->flip;
            for (uint_fast8_t cell = first_bit(&f); f; cell = next_bit(&f)){
                for (i = 0; i < coord_to_feature[cell].n_features; ++i)
                    search->eval_features[coord_to_feature[cell].features[i].feature] += coord_to_feature[cell].features[i].x;
            }
        } else{
            for (i = 0; i < coord_to_feature[flip->pos].n_features; ++i)
                search->eval_features[coord_to_feature[flip->pos].features[i].feature] -= 2 * coord_to_feature[flip->pos].features[i].x;
            uint64_t f = flip->flip;
            for (uint_fast8_t cell = first_bit(&f); f; cell = next_bit(&f)){
                for (i = 0; i < coord_to_feature[cell].n_features; ++i)
                    search->eval_features[coord_to_feature[cell].features[i].feature] -= coord_to_feature[cell].features[i].x;
            }
        }
    #endif
    search->eval_feature_reversed ^= 1;
    //search->board.move(flip);
    //search->board.print();
    //if (search->eval_feature_reversed == 0 && check_features(search))
    //    cerr << "error" << endl;
    //search->board.undo(flip);
}

inline void eval_undo(Search *search, const Flip *flip){
    search->eval_feature_reversed ^= 1;
    #if USE_FAST_DIFF_EVAL
        if (search->eval_feature_reversed){
            switch (coord_to_feature[flip->pos].n_features){
                case 13: search->eval_features[coord_to_feature[flip->pos].features[12].feature] += coord_to_feature[flip->pos].features[12].x; [[fallthrough]];
                case 12: search->eval_features[coord_to_feature[flip->pos].features[11].feature] += coord_to_feature[flip->pos].features[11].x; [[fallthrough]];
                case 11: search->eval_features[coord_to_feature[flip->pos].features[10].feature] += coord_to_feature[flip->pos].features[10].x; [[fallthrough]];
                case 10: search->eval_features[coord_to_feature[flip->pos].features[ 9].feature] += coord_to_feature[flip->pos].features[ 9].x; [[fallthrough]];
                case  9: search->eval_features[coord_to_feature[flip->pos].features[ 8].feature] += coord_to_feature[flip->pos].features[ 8].x; [[fallthrough]];
                case  8: search->eval_features[coord_to_feature[flip->pos].features[ 7].feature] += coord_to_feature[flip->pos].features[ 7].x; [[fallthrough]];
                case  7: search->eval_features[coord_to_feature[flip->pos].features[ 6].feature] += coord_to_feature[flip->pos].features[ 6].x; [[fallthrough]];
                case  6: search->eval_features[coord_to_feature[flip->pos].features[ 5].feature] += coord_to_feature[flip->pos].features[ 5].x; [[fallthrough]];
                case  5: search->eval_features[coord_to_feature[flip->pos].features[ 4].feature] += coord_to_feature[flip->pos].features[ 4].x; [[fallthrough]];
                case  4: search->eval_features[coord_to_feature[flip->pos].features[ 3].feature] += coord_to_feature[flip->pos].features[ 3].x; [[fallthrough]];
                case  3: search->eval_features[coord_to_feature[flip->pos].features[ 2].feature] += coord_to_feature[flip->pos].features[ 2].x; [[fallthrough]];
                case  2: search->eval_features[coord_to_feature[flip->pos].features[ 1].feature] += coord_to_feature[flip->pos].features[ 1].x; [[fallthrough]];
                case  1: search->eval_features[coord_to_feature[flip->pos].features[ 0].feature] += coord_to_feature[flip->pos].features[ 0].x; [[fallthrough]];
                case  0: break;
            }
            uint64_t f = flip->flip;
            for (uint_fast8_t cell = first_bit(&f); f; cell = next_bit(&f)){
                switch (coord_to_feature[cell].n_features){
                    case 13: search->eval_features[coord_to_feature[cell].features[12].feature] -= coord_to_feature[cell].features[12].x; [[fallthrough]];
                    case 12: search->eval_features[coord_to_feature[cell].features[11].feature] -= coord_to_feature[cell].features[11].x; [[fallthrough]];
                    case 11: search->eval_features[coord_to_feature[cell].features[10].feature] -= coord_to_feature[cell].features[10].x; [[fallthrough]];
                    case 10: search->eval_features[coord_to_feature[cell].features[ 9].feature] -= coord_to_feature[cell].features[ 9].x; [[fallthrough]];
                    case  9: search->eval_features[coord_to_feature[cell].features[ 8].feature] -= coord_to_feature[cell].features[ 8].x; [[fallthrough]];
                    case  8: search->eval_features[coord_to_feature[cell].features[ 7].feature] -= coord_to_feature[cell].features[ 7].x; [[fallthrough]];
                    case  7: search->eval_features[coord_to_feature[cell].features[ 6].feature] -= coord_to_feature[cell].features[ 6].x; [[fallthrough]];
                    case  6: search->eval_features[coord_to_feature[cell].features[ 5].feature] -= coord_to_feature[cell].features[ 5].x; [[fallthrough]];
                    case  5: search->eval_features[coord_to_feature[cell].features[ 4].feature] -= coord_to_feature[cell].features[ 4].x; [[fallthrough]];
                    case  4: search->eval_features[coord_to_feature[cell].features[ 3].feature] -= coord_to_feature[cell].features[ 3].x; [[fallthrough]];
                    case  3: search->eval_features[coord_to_feature[cell].features[ 2].feature] -= coord_to_feature[cell].features[ 2].x; [[fallthrough]];
                    case  2: search->eval_features[coord_to_feature[cell].features[ 1].feature] -= coord_to_feature[cell].features[ 1].x; [[fallthrough]];
                    case  1: search->eval_features[coord_to_feature[cell].features[ 0].feature] -= coord_to_feature[cell].features[ 0].x; [[fallthrough]];
                    case  0: break;
                }
            }
        } else{
            switch (coord_to_feature[flip->pos].n_features){
                case 13: search->eval_features[coord_to_feature[flip->pos].features[12].feature] += 2 * coord_to_feature[flip->pos].features[12].x; [[fallthrough]];
                case 12: search->eval_features[coord_to_feature[flip->pos].features[11].feature] += 2 * coord_to_feature[flip->pos].features[11].x; [[fallthrough]];
                case 11: search->eval_features[coord_to_feature[flip->pos].features[10].feature] += 2 * coord_to_feature[flip->pos].features[10].x; [[fallthrough]];
                case 10: search->eval_features[coord_to_feature[flip->pos].features[ 9].feature] += 2 * coord_to_feature[flip->pos].features[ 9].x; [[fallthrough]];
                case  9: search->eval_features[coord_to_feature[flip->pos].features[ 8].feature] += 2 * coord_to_feature[flip->pos].features[ 8].x; [[fallthrough]];
                case  8: search->eval_features[coord_to_feature[flip->pos].features[ 7].feature] += 2 * coord_to_feature[flip->pos].features[ 7].x; [[fallthrough]];
                case  7: search->eval_features[coord_to_feature[flip->pos].features[ 6].feature] += 2 * coord_to_feature[flip->pos].features[ 6].x; [[fallthrough]];
                case  6: search->eval_features[coord_to_feature[flip->pos].features[ 5].feature] += 2 * coord_to_feature[flip->pos].features[ 5].x; [[fallthrough]];
                case  5: search->eval_features[coord_to_feature[flip->pos].features[ 4].feature] += 2 * coord_to_feature[flip->pos].features[ 4].x; [[fallthrough]];
                case  4: search->eval_features[coord_to_feature[flip->pos].features[ 3].feature] += 2 * coord_to_feature[flip->pos].features[ 3].x; [[fallthrough]];
                case  3: search->eval_features[coord_to_feature[flip->pos].features[ 2].feature] += 2 * coord_to_feature[flip->pos].features[ 2].x; [[fallthrough]];
                case  2: search->eval_features[coord_to_feature[flip->pos].features[ 1].feature] += 2 * coord_to_feature[flip->pos].features[ 1].x; [[fallthrough]];
                case  1: search->eval_features[coord_to_feature[flip->pos].features[ 0].feature] += 2 * coord_to_feature[flip->pos].features[ 0].x; [[fallthrough]];
                case  0: break;
            }
            uint64_t f = flip->flip;
            for (uint_fast8_t cell = first_bit(&f); f; cell = next_bit(&f)){
                switch (coord_to_feature[cell].n_features){
                    case 13: search->eval_features[coord_to_feature[cell].features[12].feature] += coord_to_feature[cell].features[12].x; [[fallthrough]];
                    case 12: search->eval_features[coord_to_feature[cell].features[11].feature] += coord_to_feature[cell].features[11].x; [[fallthrough]];
                    case 11: search->eval_features[coord_to_feature[cell].features[10].feature] += coord_to_feature[cell].features[10].x; [[fallthrough]];
                    case 10: search->eval_features[coord_to_feature[cell].features[ 9].feature] += coord_to_feature[cell].features[ 9].x; [[fallthrough]];
                    case  9: search->eval_features[coord_to_feature[cell].features[ 8].feature] += coord_to_feature[cell].features[ 8].x; [[fallthrough]];
                    case  8: search->eval_features[coord_to_feature[cell].features[ 7].feature] += coord_to_feature[cell].features[ 7].x; [[fallthrough]];
                    case  7: search->eval_features[coord_to_feature[cell].features[ 6].feature] += coord_to_feature[cell].features[ 6].x; [[fallthrough]];
                    case  6: search->eval_features[coord_to_feature[cell].features[ 5].feature] += coord_to_feature[cell].features[ 5].x; [[fallthrough]];
                    case  5: search->eval_features[coord_to_feature[cell].features[ 4].feature] += coord_to_feature[cell].features[ 4].x; [[fallthrough]];
                    case  4: search->eval_features[coord_to_feature[cell].features[ 3].feature] += coord_to_feature[cell].features[ 3].x; [[fallthrough]];
                    case  3: search->eval_features[coord_to_feature[cell].features[ 2].feature] += coord_to_feature[cell].features[ 2].x; [[fallthrough]];
                    case  2: search->eval_features[coord_to_feature[cell].features[ 1].feature] += coord_to_feature[cell].features[ 1].x; [[fallthrough]];
                    case  1: search->eval_features[coord_to_feature[cell].features[ 0].feature] += coord_to_feature[cell].features[ 0].x; [[fallthrough]];
                    case  0: break;
                }
            }
        }
    #else
        int i;
        if (search->eval_feature_reversed){
            for (i = 0; i < coord_to_feature[flip->pos].n_features; ++i)
                search->eval_features[coord_to_feature[flip->pos].features[i].feature] += coord_to_feature[flip->pos].features[i].x;
            uint64_t f = flip->flip;
            for (uint_fast8_t cell = first_bit(&f); f; cell = next_bit(&f)){
                for (i = 0; i < coord_to_feature[cell].n_features; ++i)
                    search->eval_features[coord_to_feature[cell].features[i].feature] -= coord_to_feature[cell].features[i].x;
            }
        } else{
            for (i = 0; i < coord_to_feature[flip->pos].n_features; ++i)
                search->eval_features[coord_to_feature[flip->pos].features[i].feature] += 2 * coord_to_feature[flip->pos].features[i].x;
            uint64_t f = flip->flip;
            for (uint_fast8_t cell = first_bit(&f); f; cell = next_bit(&f)){
                for (i = 0; i < coord_to_feature[cell].n_features; ++i)
                    search->eval_features[coord_to_feature[cell].features[i].feature] += coord_to_feature[cell].features[i].x;
            }
        }
    #endif
    //if (search->eval_feature_reversed == 0 && check_features(search))
    //    cerr << "error" << endl;
}