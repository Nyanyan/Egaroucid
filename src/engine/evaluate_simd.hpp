/*
    Egaroucid Project

    @file evaluate_generic.hpp
        Evaluation function with AVX2
    @date 2021-2023
    @author Takuto Yamana (a.k.a. Nyanyan)
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


/*
    @brief evaluation pattern definition
*/
#define N_PATTERNS 16
#ifndef N_SYMMETRY_PATTERNS
    #define N_SYMMETRY_PATTERNS 62
#endif
#define CEIL_N_SYMMETRY_PATTERNS 64
#define SIMD_EVAL_OFFSET 16384
#define MAX_PATTERN_CELLS 10
#define MAX_CELL_PATTERNS 13
#define MAX_SURROUND 100
#define MAX_CANPUT 50
#define MAX_STONE_NUM 65
#define N_CANPUT_PATTERNS 4
#define MAX_EVALUATE_IDX 59049
#define N_PATTERN_PARAMS 521478

/*
    @brief value definition

    Raw score is STEP times larger than the real score.
*/
#define STEP 256
#define STEP_2 128
#define STEP_SHIFT 8

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

constexpr Feature_to_coord feature_to_coord[CEIL_N_SYMMETRY_PATTERNS] = {
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
    {9, {COORD_H8, COORD_G8, COORD_F8, COORD_H7, COORD_G7, COORD_F7, COORD_H6, COORD_G6, COORD_F6, COORD_NO}}, // 45

    // 12 edge + y
    {10, {COORD_C2, COORD_A1, COORD_B1, COORD_C1, COORD_D1, COORD_E1, COORD_F1, COORD_G1, COORD_H1, COORD_F2}}, // 46
    {10, {COORD_B3, COORD_A1, COORD_A2, COORD_A3, COORD_A4, COORD_A5, COORD_A6, COORD_A7, COORD_A8, COORD_B6}}, // 47
    {10, {COORD_C7, COORD_A8, COORD_B8, COORD_C8, COORD_D8, COORD_E8, COORD_F8, COORD_G8, COORD_H8, COORD_F7}}, // 48
    {10, {COORD_G3, COORD_H1, COORD_H2, COORD_H3, COORD_H4, COORD_H5, COORD_H6, COORD_H7, COORD_H8, COORD_G6}}, // 49

    // 13 narrow triangle
    {10, {COORD_A1, COORD_B1, COORD_C1, COORD_D1, COORD_E1, COORD_A2, COORD_B2, COORD_A3, COORD_A4, COORD_A5}}, // 50
    {10, {COORD_H1, COORD_G1, COORD_F1, COORD_E1, COORD_D1, COORD_H2, COORD_G2, COORD_H3, COORD_H4, COORD_H5}}, // 51
    {10, {COORD_A8, COORD_B8, COORD_C8, COORD_D8, COORD_E8, COORD_A7, COORD_B7, COORD_A6, COORD_A5, COORD_A4}}, // 52
    {10, {COORD_H8, COORD_G8, COORD_F8, COORD_E8, COORD_D8, COORD_H7, COORD_G7, COORD_H6, COORD_H5, COORD_H4}}, // 53

    // 14 fish
    {10, {COORD_A1, COORD_B1, COORD_A2, COORD_B2, COORD_C2, COORD_D2, COORD_B3, COORD_C3, COORD_B4, COORD_D4}}, // 54
    {10, {COORD_H1, COORD_G1, COORD_H2, COORD_G2, COORD_F2, COORD_E2, COORD_G3, COORD_F3, COORD_G4, COORD_E4}}, // 55
    {10, {COORD_A8, COORD_B8, COORD_A7, COORD_B7, COORD_C7, COORD_D7, COORD_B6, COORD_C6, COORD_B5, COORD_D5}}, // 56
    {10, {COORD_H8, COORD_G8, COORD_H7, COORD_G7, COORD_F7, COORD_E7, COORD_G6, COORD_F6, COORD_G5, COORD_E5}}, // 57

    // 15 kite
    {10, {COORD_A1, COORD_B1, COORD_A2, COORD_B2, COORD_C2, COORD_D2, COORD_E2, COORD_B3, COORD_B4, COORD_B5}}, // 58
    {10, {COORD_H1, COORD_G1, COORD_H2, COORD_G2, COORD_F2, COORD_E2, COORD_D2, COORD_G3, COORD_G4, COORD_G5}}, // 59
    {10, {COORD_A8, COORD_B8, COORD_A7, COORD_B7, COORD_C7, COORD_D7, COORD_E7, COORD_B6, COORD_B5, COORD_B4}}, // 60
    {10, {COORD_H8, COORD_G8, COORD_H7, COORD_G7, COORD_F7, COORD_E7, COORD_D7, COORD_G6, COORD_G5, COORD_G4}}, // 61

    // dummy
    {0, {COORD_NO, COORD_NO, COORD_NO, COORD_NO, COORD_NO, COORD_NO, COORD_NO, COORD_NO, COORD_NO, COORD_NO}}, // 62
    {0, {COORD_NO, COORD_NO, COORD_NO, COORD_NO, COORD_NO, COORD_NO, COORD_NO, COORD_NO, COORD_NO, COORD_NO}}  // 63
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

/*
    @brief constants used for evaluation function with SIMD
*/
__m256i eval_lower_mask;
__m256i feature_to_coord_simd_mul[N_SIMD_EVAL_FEATURES][MAX_PATTERN_CELLS - 1];
__m256i feature_to_coord_simd_cell[N_SIMD_EVAL_FEATURES][MAX_PATTERN_CELLS][2];
__m256i coord_to_feature_simd[HW2][N_SIMD_EVAL_FEATURES];
__m256i coord_to_feature_simd2[HW2][N_SIMD_EVAL_FEATURES];
__m256i eval_simd_offsets[N_SIMD_EVAL_FEATURES];

/*
    @brief constants of 3 ^ N
*/
constexpr uint_fast16_t pow3[11] = {1, P31, P32, P33, P34, P35, P36, P37, P38, P39, P310};

/*
    @brief evaluation parameters
*/
int16_t pattern_arr[2][N_PHASES][N_PATTERN_PARAMS + 2];
int16_t eval_sur0_sur1_arr[N_PHASES][MAX_SURROUND][MAX_SURROUND];
int16_t eval_canput0_canput1_arr[N_PHASES][MAX_CANPUT][MAX_CANPUT];
int16_t eval_num0_num1_arr[N_PHASES][MAX_STONE_NUM][MAX_STONE_NUM];
int16_t eval_mobility_pattern[N_PHASES][N_CANPUT_PATTERNS][P44][P44];

/*
    @brief used for unzipping the evaluation function

    This function swaps the player in the index

    @param i                    index representing the discs in a pattern
    @param pattern_size         size of the pattern
    @return swapped index
*/
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

/*
    @brief used for unzipping the evaluation function

    @param phase_idx            evaluation phase
    @param pattern_idx          evaluation pattern's index
    @param siz                  size of the pattern
    @param strt                 start position of the pattern
*/
void init_pattern_arr_rev(int phase_idx, int pattern_idx, int siz, int strt){
    int ri;
    for (int i = 0; i < (int)pow3[siz]; ++i){
        ri = swap_player_idx(i, siz);
        pattern_arr[1][phase_idx][strt + ri] = pattern_arr[0][phase_idx][strt + i];
    }
}

/*
    @brief initialize the evaluation function

    @param file                 evaluation file name
    @return evaluation function conpletely initialized?
*/
inline bool init_evaluation_calc(const char* file, bool show_log){
    if (show_log)
        std::cerr << "evaluation file " << file << std::endl;
    FILE* fp;
    if (!file_open(&fp, file, "rb")){
        std::cerr << "[ERROR] [FATAL] can't open eval " << file << std::endl;
        return false;
    }
    int phase_idx, pattern_idx;
    constexpr int pattern_sizes[N_PATTERNS] = {8, 8, 8, 5, 6, 7, 8, 10, 10, 10, 10, 9, 10, 10, 10, 10};
    constexpr int pattern_starts[N_PATTERNS] = {1, 6562, 13123, 19684, 19927, 20656, 22843, 29404, 88453, 147502, 206551, 265600, 285283, 344332, 403381, 462430};
    for (phase_idx = 0; phase_idx < N_PHASES; ++phase_idx){
        if (fread(pattern_arr[0][phase_idx] + 1, 2, N_PATTERN_PARAMS, fp) < N_PATTERN_PARAMS){
            std::cerr << "[ERROR] [FATAL] evaluation file broken" << std::endl;
            fclose(fp);
            return false;
        }
        if (fread(eval_sur0_sur1_arr[phase_idx], 2, MAX_SURROUND * MAX_SURROUND, fp) < MAX_SURROUND * MAX_SURROUND){
            std::cerr << "[ERROR] [FATAL] evaluation file broken" << std::endl;
            fclose(fp);
            return false;
        }
        if (fread(eval_canput0_canput1_arr[phase_idx], 2, MAX_CANPUT * MAX_CANPUT, fp) < MAX_CANPUT * MAX_CANPUT){
            std::cerr << "[ERROR] [FATAL] evaluation file broken" << std::endl;
            fclose(fp);
            return false;
        }
        if (fread(eval_num0_num1_arr[phase_idx], 2, MAX_STONE_NUM * MAX_STONE_NUM, fp) < MAX_STONE_NUM * MAX_STONE_NUM){
            std::cerr << "[ERROR] [FATAL] evaluation file broken" << std::endl;
            fclose(fp);
            return false;
        }
        if (fread(eval_mobility_pattern[phase_idx], 2, N_CANPUT_PATTERNS * P44 * P44, fp) < N_CANPUT_PATTERNS * P48){
            std::cerr << "[ERROR] [FATAL] evaluation file broken" << std::endl;
            fclose(fp);
            return false;
        }
    }
    int i, j, k, idx, cell;
    eval_lower_mask = _mm256_set1_epi32(0x0000FFFF);
    for (i = 0; i < 2; ++i){
        for (phase_idx = 0; phase_idx < N_PHASES; ++phase_idx){
            pattern_arr[i][phase_idx][0] = 0;
            pattern_arr[i][phase_idx][N_PATTERN_PARAMS + 1] = 0;
        }
    }
    for (phase_idx = 0; phase_idx < N_PHASES; ++phase_idx){
        for (j = 0; j < N_PATTERNS; ++j){
            for (k = 0; k < pow3[pattern_sizes[j]]; ++k){
                if (pattern_arr[0][phase_idx][pattern_starts[j] + k] < -SIMD_EVAL_OFFSET){
                    pattern_arr[0][phase_idx][pattern_starts[j] + k] = -SIMD_EVAL_OFFSET;
                    std::cerr << "[ERROR] evaluation value too low. you can ignore this error." << std::endl;
                }
                if (pattern_arr[0][phase_idx][pattern_starts[j] + k] >= 0x7FFF - SIMD_EVAL_OFFSET){
                    pattern_arr[0][phase_idx][pattern_starts[j] + k] = 0x7FFF - SIMD_EVAL_OFFSET - 1;
                    std::cerr << "[ERROR] evaluation value too high. you can ignore this error." << std::endl;
                }
                pattern_arr[0][phase_idx][pattern_starts[j] + k] += SIMD_EVAL_OFFSET;
            }
        }
    }
    if (thread_pool.size() >= 2){
        std::future<void> tasks[N_PHASES * N_PATTERNS];
        int i = 0;
        for (phase_idx = 0; phase_idx < N_PHASES; ++phase_idx){
            for (pattern_idx = 0; pattern_idx < N_PATTERNS; ++pattern_idx)
                tasks[i++] = thread_pool.push(std::bind(init_pattern_arr_rev, phase_idx, pattern_idx, pattern_sizes[pattern_idx], pattern_starts[pattern_idx]));
        }
        for (std::future<void> &task: tasks)
            task.get();
    } else{
        for (phase_idx = 0; phase_idx < N_PHASES; ++phase_idx){
            for (pattern_idx = 0; pattern_idx < N_PATTERNS; ++pattern_idx)
                init_pattern_arr_rev(phase_idx, pattern_idx, pattern_sizes[pattern_idx], pattern_starts[pattern_idx]);
        }
    }
    int16_t f2c[16];
    for (i = 0; i < N_SIMD_EVAL_FEATURES; ++i){
        for (j = 0; j < MAX_PATTERN_CELLS - 1; ++j){
            for (k = 0; k < 16; ++k)
                f2c[k] = j < feature_to_coord[i * 16 + k].n_cells - 1 ? 3 : 1;
            feature_to_coord_simd_mul[i][j] = _mm256_set_epi16(f2c[0], f2c[1], f2c[2], f2c[3], f2c[4], f2c[5], f2c[6], f2c[7], f2c[8], f2c[9], f2c[10], f2c[11], f2c[12], f2c[13], f2c[14], f2c[15]);
        }
    }
    int f2c32[8];
    for (i = 0; i < N_SIMD_EVAL_FEATURES; ++i){
        for (j = 0; j < MAX_PATTERN_CELLS; ++j){
            for (k = 0; k < 8; ++k)
                f2c32[k] = feature_to_coord[i * 16 + k * 2 + 1].cells[j];
            feature_to_coord_simd_cell[i][j][0] = _mm256_set_epi32(f2c32[0], f2c32[1], f2c32[2], f2c32[3], f2c32[4], f2c32[5], f2c32[6], f2c32[7]);
            for (k = 0; k < 8; ++k)
                f2c32[k] = feature_to_coord[i * 16 + k * 2].cells[j];
            feature_to_coord_simd_cell[i][j][1] = _mm256_set_epi32(f2c32[0], f2c32[1], f2c32[2], f2c32[3], f2c32[4], f2c32[5], f2c32[6], f2c32[7]);
        }
    }
    int16_t c2f[CEIL_N_SYMMETRY_PATTERNS];
    for (cell = 0; cell < HW2; ++cell){
        for (i = 0; i < CEIL_N_SYMMETRY_PATTERNS; ++i)
            c2f[i] = 0;
        for (i = 0; i < coord_to_feature[cell].n_features; ++i)
            c2f[coord_to_feature[cell].features[i].feature] = coord_to_feature[cell].features[i].x;
        for (i = 0; i < N_SIMD_EVAL_FEATURES; ++i){
            idx = i * 16;
            coord_to_feature_simd[cell][i] = _mm256_set_epi16(c2f[idx], c2f[idx + 1], c2f[idx + 2], c2f[idx + 3], c2f[idx + 4], c2f[idx + 5], c2f[idx + 6], c2f[idx + 7], 
                c2f[idx + 8], c2f[idx + 9], c2f[idx + 10], c2f[idx + 11], c2f[idx + 12], c2f[idx + 13], c2f[idx + 14], c2f[idx + 15]);
            coord_to_feature_simd2[cell][i] = _mm256_slli_epi16(coord_to_feature_simd[cell][i], 1);
        }
    }
    eval_simd_offsets[0] = _mm256_set_epi32(pattern_starts[0], pattern_starts[0], pattern_starts[1], pattern_starts[1], pattern_starts[2], pattern_starts[2], pattern_starts[3], pattern_starts[3]);
    eval_simd_offsets[1] = _mm256_set_epi32(pattern_starts[4], pattern_starts[4], pattern_starts[5], pattern_starts[5], pattern_starts[6], pattern_starts[7], pattern_starts[7], pattern_starts[8]);
    eval_simd_offsets[2] = _mm256_set_epi32(pattern_starts[8], pattern_starts[9], pattern_starts[9], pattern_starts[10], pattern_starts[10], pattern_starts[11], pattern_starts[11], pattern_starts[12]);
    eval_simd_offsets[3] = _mm256_set_epi32(pattern_starts[12], pattern_starts[13], pattern_starts[13], pattern_starts[14], pattern_starts[14], pattern_starts[15], pattern_starts[15], N_PATTERN_PARAMS + 1);
    if (show_log)
        std::cerr << "evaluation function initialized" << std::endl;
    return true;
}

/*
    @brief Wrapper of evaluation initializing

    @param file                 evaluation file name
    @return evaluation function conpletely initialized?
*/
bool evaluate_init(const char* file, bool show_log){
    return init_evaluation_calc(file, show_log);
}

/*
    @brief Wrapper of evaluation initializing

    @param file                 evaluation file name
    @return evaluation function conpletely initialized?
*/
bool evaluate_init(const std::string file, bool show_log){
    return init_evaluation_calc(file.c_str(), show_log);
}

/*
    @brief Wrapper of evaluation initializing

    @return evaluation function conpletely initialized?
*/
bool evaluate_init(bool show_log){
    return init_evaluation_calc("resources/eval.egev", show_log);
}

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
    int score = b->count_player() * 2 + e - HW2;
    if (score > 0)
        score += e;
    else if (score < 0)
        score -= e;
    return score;
}

/*
    @brief calculate surround value used in evaluation function

    @param player               a bitboard representing player
    @param empties              a bitboard representing empties
    @return surround value
*/
inline int calc_surround(const uint64_t player, const uint64_t empties){
    const u64_4 shift(1, HW, HW_M1, HW_P1);
    const u64_4 mask(0x7E7E7E7E7E7E7E7EULL, 0x00FFFFFFFFFFFF00ULL, 0x007E7E7E7E7E7E00ULL, 0x007E7E7E7E7E7E00ULL);
    u64_4 pl(player);
    pl = pl & mask;
    return pop_count_ull(empties & all_or((pl << shift) | (pl >> shift)));
}

/*
    @brief pattern evaluation

    @param phase_idx            evaluation phase
    @param search               search information
    @return pattern evaluation value
*/
inline __m256i calc_idx8_a(const __m256i eval_features[], const int i){
    return _mm256_add_epi32(_mm256_and_si256(eval_features[i], eval_lower_mask), eval_simd_offsets[i]);
}

inline __m256i calc_idx8_b(const __m256i eval_features[], const int i){
    return _mm256_add_epi32(_mm256_srli_epi32(eval_features[i], 16), eval_simd_offsets[i]);
}

inline __m256i gather_eval(const int *pat_com, const __m256i idx8){
    return _mm256_and_si256(_mm256_i32gather_epi32(pat_com, idx8, 2), eval_lower_mask);
}

inline int calc_pattern_diff(const int phase_idx, Search *search){
    const int *pat_com = (int*)pattern_arr[search->eval_feature_reversed][phase_idx];
    __m256i res256 =                  gather_eval(pat_com, calc_idx8_a(search->eval_features, 0));
    res256 = _mm256_add_epi32(res256, gather_eval(pat_com, calc_idx8_b(search->eval_features, 0)));
    res256 = _mm256_add_epi32(res256, gather_eval(pat_com, calc_idx8_a(search->eval_features, 1)));
    res256 = _mm256_add_epi32(res256, gather_eval(pat_com, calc_idx8_b(search->eval_features, 1)));
    res256 = _mm256_add_epi32(res256, gather_eval(pat_com, calc_idx8_a(search->eval_features, 2)));
    res256 = _mm256_add_epi32(res256, gather_eval(pat_com, calc_idx8_b(search->eval_features, 2)));
    res256 = _mm256_add_epi32(res256, gather_eval(pat_com, calc_idx8_a(search->eval_features, 3)));
    res256 = _mm256_add_epi32(res256, gather_eval(pat_com, calc_idx8_b(search->eval_features, 3)));
    return _mm256_extract_epi32(res256, 7) + 
        _mm256_extract_epi32(res256, 6) + 
        _mm256_extract_epi32(res256, 5) + 
        _mm256_extract_epi32(res256, 4) + 
        _mm256_extract_epi32(res256, 3) + 
        _mm256_extract_epi32(res256, 2) + 
        _mm256_extract_epi32(res256, 1) + 
        _mm256_extract_epi32(res256, 0) - 
        SIMD_EVAL_OFFSET * N_SYMMETRY_PATTERNS;
}

/*
    @brief mobility pattern evaluation

    @param phase_idx            evaluation phase
    @param player_mobility      player's legal moves in bitboard
    @param opponent_mobility    opponent's legal moves in bitboard
    @return mobility pattern evaluation value
*/
inline int calc_mobility_pattern(const int phase_idx, const uint64_t player_mobility, const uint64_t opponent_mobility){
    uint8_t *ph = (uint8_t*)&player_mobility;
    uint8_t *oh = (uint8_t*)&opponent_mobility;
    uint64_t p90 = rotate_90(player_mobility);
    uint64_t o90 = rotate_90(opponent_mobility);
    uint8_t *pv = (uint8_t*)&p90;
    uint8_t *ov = (uint8_t*)&o90;
    return 
        eval_mobility_pattern[phase_idx][0][oh[0]][ph[0]] + 
        eval_mobility_pattern[phase_idx][0][oh[7]][ph[7]] + 
        eval_mobility_pattern[phase_idx][0][ov[0]][pv[0]] + 
        eval_mobility_pattern[phase_idx][0][ov[7]][pv[7]] + 
        eval_mobility_pattern[phase_idx][1][oh[1]][ph[1]] + 
        eval_mobility_pattern[phase_idx][1][oh[6]][ph[6]] + 
        eval_mobility_pattern[phase_idx][1][ov[1]][pv[1]] + 
        eval_mobility_pattern[phase_idx][1][ov[6]][pv[6]] + 
        eval_mobility_pattern[phase_idx][2][oh[2]][ph[2]] + 
        eval_mobility_pattern[phase_idx][2][oh[5]][ph[5]] + 
        eval_mobility_pattern[phase_idx][2][ov[2]][pv[2]] + 
        eval_mobility_pattern[phase_idx][2][ov[5]][pv[5]] + 
        eval_mobility_pattern[phase_idx][3][oh[3]][ph[3]] + 
        eval_mobility_pattern[phase_idx][3][oh[4]][ph[4]] + 
        eval_mobility_pattern[phase_idx][3][ov[3]][pv[3]] + 
        eval_mobility_pattern[phase_idx][3][ov[4]][pv[4]];
}

/*
    @brief midgame evaluation function

    @param b                    board
    @return evaluation value
*/
inline void calc_features(Search *search);

inline int mid_evaluate(Board *board){
    Search search;
    search.init_board(board);
    calc_features(&search);
    uint64_t player_mobility, opponent_mobility;
    player_mobility = calc_legal(search.board.player, search.board.opponent);
    opponent_mobility = calc_legal(search.board.opponent, search.board.player);
    if ((player_mobility | opponent_mobility) == 0ULL)
        return end_evaluate(&search.board);
    int phase_idx, sur0, sur1, canput0, canput1, num0, num1;
    uint64_t empties;
    phase_idx = search.phase();
    canput0 = pop_count_ull(player_mobility);
    canput1 = pop_count_ull(opponent_mobility);
    empties = ~(search.board.player | search.board.opponent);
    sur0 = std::min(MAX_SURROUND - 1, calc_surround(search.board.player, empties));
    sur1 = std::min(MAX_SURROUND - 1, calc_surround(search.board.opponent, empties));
    num0 = pop_count_ull(search.board.player);
    num1 = search.n_discs - num0;
    int res = calc_pattern_diff(phase_idx, &search) + 
        eval_sur0_sur1_arr[phase_idx][sur0][sur1] + 
        eval_canput0_canput1_arr[phase_idx][canput0][canput1] + 
        eval_num0_num1_arr[phase_idx][num0][num1] + 
        calc_mobility_pattern(phase_idx, player_mobility, opponent_mobility);
    res += res >= 0 ? STEP_2 : -STEP_2;
    res /= STEP;
    //return std::max(-SCORE_MAX, std::min(SCORE_MAX, res));
    if (res > SCORE_MAX)
        return SCORE_MAX;
    if (res < -SCORE_MAX)
        return -SCORE_MAX;
    return res;
}

/*
    @brief midgame evaluation function

    @param search               search information
    @return evaluation value
*/
inline int mid_evaluate_diff(Search *search){
    uint64_t player_mobility, opponent_mobility;
    player_mobility = calc_legal(search->board.player, search->board.opponent);
    opponent_mobility = calc_legal(search->board.opponent, search->board.player);
    if ((player_mobility | opponent_mobility) == 0ULL)
        return end_evaluate(&search->board);
    int phase_idx, sur0, sur1, canput0, canput1, num0, num1;
    uint64_t empties;
    phase_idx = search->phase();
    canput0 = pop_count_ull(player_mobility);
    canput1 = pop_count_ull(opponent_mobility);
    empties = ~(search->board.player | search->board.opponent);
    sur0 = calc_surround(search->board.player, empties);
    sur1 = calc_surround(search->board.opponent, empties);
    num0 = pop_count_ull(search->board.player);
    num1 = search->n_discs - num0;
    int res = calc_pattern_diff(phase_idx, search) + 
        eval_sur0_sur1_arr[phase_idx][sur0][sur1] + 
        eval_canput0_canput1_arr[phase_idx][canput0][canput1] + 
        eval_num0_num1_arr[phase_idx][num0][num1] + 
        calc_mobility_pattern(phase_idx, player_mobility, opponent_mobility);
    res += res >= 0 ? STEP_2 : -STEP_2;
    res /= STEP;
    //return std::max(-SCORE_MAX, std::min(SCORE_MAX, res));
    if (res > SCORE_MAX)
        return SCORE_MAX;
    if (res < -SCORE_MAX)
        return -SCORE_MAX;
    return res;
}

/*
    @brief calculate features for pattern evaluation

    @param search               search information
*/
inline void calc_features(Search *search){
    int b_arr_int[HW2 + 1];
    search->board.translate_to_arr_player_rev(b_arr_int);
    b_arr_int[COORD_NO] = 0;
    int i, j;
    for (i = 0; i < N_SIMD_EVAL_FEATURES; ++i){
        search->eval_features[i] = _mm256_set1_epi16(0);
        for (j = 0; j < MAX_PATTERN_CELLS - 1; ++j){
            search->eval_features[i] = _mm256_adds_epu16(search->eval_features[i], _mm256_i32gather_epi32(b_arr_int, feature_to_coord_simd_cell[i][j][0], 4));
            search->eval_features[i] = _mm256_adds_epu16(search->eval_features[i], _mm256_slli_epi32(_mm256_i32gather_epi32(b_arr_int, feature_to_coord_simd_cell[i][j][1], 4), 16));
            search->eval_features[i] = _mm256_mullo_epi16(search->eval_features[i], feature_to_coord_simd_mul[i][j]);
        }
        search->eval_features[i] = _mm256_adds_epu16(search->eval_features[i], _mm256_i32gather_epi32(b_arr_int, feature_to_coord_simd_cell[i][MAX_PATTERN_CELLS - 1][0], 4));
        search->eval_features[i] = _mm256_adds_epu16(search->eval_features[i], _mm256_slli_epi32(_mm256_i32gather_epi32(b_arr_int, feature_to_coord_simd_cell[i][MAX_PATTERN_CELLS - 1][1], 4), 16));
    }
    search->eval_feature_reversed = 0;
}

/*
    @brief move evaluation features

    @param search               search information
    @param flip                 flip information
*/
inline void eval_move(Search *search, const Flip *flip){
    uint_fast8_t cell;
    uint64_t f;
    if (search->eval_feature_reversed){
        search->eval_features[0] = _mm256_subs_epu16(search->eval_features[0], coord_to_feature_simd[flip->pos][0]);
        search->eval_features[1] = _mm256_subs_epu16(search->eval_features[1], coord_to_feature_simd[flip->pos][1]);
        search->eval_features[2] = _mm256_subs_epu16(search->eval_features[2], coord_to_feature_simd[flip->pos][2]);
        search->eval_features[3] = _mm256_subs_epu16(search->eval_features[3], coord_to_feature_simd[flip->pos][3]);
        f = flip->flip;
        for (cell = first_bit(&f); f; cell = next_bit(&f)){
            search->eval_features[0] = _mm256_adds_epu16(search->eval_features[0], coord_to_feature_simd[cell][0]);
            search->eval_features[1] = _mm256_adds_epu16(search->eval_features[1], coord_to_feature_simd[cell][1]);
            search->eval_features[2] = _mm256_adds_epu16(search->eval_features[2], coord_to_feature_simd[cell][2]);
            search->eval_features[3] = _mm256_adds_epu16(search->eval_features[3], coord_to_feature_simd[cell][3]);
        }
    } else{
        search->eval_features[0] = _mm256_subs_epu16(search->eval_features[0], coord_to_feature_simd2[flip->pos][0]);
        search->eval_features[1] = _mm256_subs_epu16(search->eval_features[1], coord_to_feature_simd2[flip->pos][1]);
        search->eval_features[2] = _mm256_subs_epu16(search->eval_features[2], coord_to_feature_simd2[flip->pos][2]);
        search->eval_features[3] = _mm256_subs_epu16(search->eval_features[3], coord_to_feature_simd2[flip->pos][3]);
        f = flip->flip;
        for (cell = first_bit(&f); f; cell = next_bit(&f)){
            search->eval_features[0] = _mm256_subs_epu16(search->eval_features[0], coord_to_feature_simd[cell][0]);
            search->eval_features[1] = _mm256_subs_epu16(search->eval_features[1], coord_to_feature_simd[cell][1]);
            search->eval_features[2] = _mm256_subs_epu16(search->eval_features[2], coord_to_feature_simd[cell][2]);
            search->eval_features[3] = _mm256_subs_epu16(search->eval_features[3], coord_to_feature_simd[cell][3]);
        }
    }
    search->eval_feature_reversed ^= 1;
}

/*
    @brief undo evaluation features

    @param search               search information
    @param flip                 flip information
*/
inline void eval_undo(Search *search, const Flip *flip){
    search->eval_feature_reversed ^= 1;
    uint_fast8_t cell;
    uint64_t f;
    if (search->eval_feature_reversed){
        search->eval_features[0] = _mm256_adds_epu16(search->eval_features[0], coord_to_feature_simd[flip->pos][0]);
        search->eval_features[1] = _mm256_adds_epu16(search->eval_features[1], coord_to_feature_simd[flip->pos][1]);
        search->eval_features[2] = _mm256_adds_epu16(search->eval_features[2], coord_to_feature_simd[flip->pos][2]);
        search->eval_features[3] = _mm256_adds_epu16(search->eval_features[3], coord_to_feature_simd[flip->pos][3]);
        f = flip->flip;
        for (cell = first_bit(&f); f; cell = next_bit(&f)){
            search->eval_features[0] = _mm256_subs_epu16(search->eval_features[0], coord_to_feature_simd[cell][0]);
            search->eval_features[1] = _mm256_subs_epu16(search->eval_features[1], coord_to_feature_simd[cell][1]);
            search->eval_features[2] = _mm256_subs_epu16(search->eval_features[2], coord_to_feature_simd[cell][2]);
            search->eval_features[3] = _mm256_subs_epu16(search->eval_features[3], coord_to_feature_simd[cell][3]);
        }
    } else{
        search->eval_features[0] = _mm256_adds_epu16(search->eval_features[0], coord_to_feature_simd2[flip->pos][0]);
        search->eval_features[1] = _mm256_adds_epu16(search->eval_features[1], coord_to_feature_simd2[flip->pos][1]);
        search->eval_features[2] = _mm256_adds_epu16(search->eval_features[2], coord_to_feature_simd2[flip->pos][2]);
        search->eval_features[3] = _mm256_adds_epu16(search->eval_features[3], coord_to_feature_simd2[flip->pos][3]);
        f = flip->flip;
        for (cell = first_bit(&f); f; cell = next_bit(&f)){
            search->eval_features[0] = _mm256_adds_epu16(search->eval_features[0], coord_to_feature_simd[cell][0]);
            search->eval_features[1] = _mm256_adds_epu16(search->eval_features[1], coord_to_feature_simd[cell][1]);
            search->eval_features[2] = _mm256_adds_epu16(search->eval_features[2], coord_to_feature_simd[cell][2]);
            search->eval_features[3] = _mm256_adds_epu16(search->eval_features[3], coord_to_feature_simd[cell][3]);
        }
    }
}
