/*
    Egaroucid Project

    @file evaluate_generic.hpp
        Evaluation function without AVX2
    @date 2021-2023
    @author Takuto Yamana (a.k.a. Nyanyan)
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


/*
    @brief evaluation pattern definition
*/
// disc patterns
#define N_PATTERNS 16
#define MAX_PATTERN_CELLS 10
#define MAX_CELL_PATTERNS 13
#define MAX_EVALUATE_IDX 59049

// additional features
#define MAX_SURROUND 64
#define MAX_CANPUT 35
#define MAX_STONE_NUM 65

// mobility patterns
#define N_CANPUT_PATTERNS 4
#define MOBILITY_PATTERN_SIZE 256

/*
    @brief value definition

    Raw score is STEP times larger than the real score.
*/
#define STEP 128
#define STEP_2 64

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

    // 7 corner9
    {9, {COORD_A1, COORD_B1, COORD_C1, COORD_A2, COORD_B2, COORD_C2, COORD_A3, COORD_B3, COORD_C3, COORD_NO}}, // 26
    {9, {COORD_H1, COORD_G1, COORD_F1, COORD_H2, COORD_G2, COORD_F2, COORD_H3, COORD_G3, COORD_F3, COORD_NO}}, // 27
    {9, {COORD_A8, COORD_B8, COORD_C8, COORD_A7, COORD_B7, COORD_C7, COORD_A6, COORD_B6, COORD_C6, COORD_NO}}, // 28
    {9, {COORD_H8, COORD_G8, COORD_F8, COORD_H7, COORD_G7, COORD_F7, COORD_H6, COORD_G6, COORD_F6, COORD_NO}}, // 29

    // 8 edge + 2x
    {10, {COORD_B2, COORD_A1, COORD_B1, COORD_C1, COORD_D1, COORD_E1, COORD_F1, COORD_G1, COORD_H1, COORD_G2}}, // 30
    {10, {COORD_B2, COORD_A1, COORD_A2, COORD_A3, COORD_A4, COORD_A5, COORD_A6, COORD_A7, COORD_A8, COORD_B7}}, // 31
    {10, {COORD_B7, COORD_A8, COORD_B8, COORD_C8, COORD_D8, COORD_E8, COORD_F8, COORD_G8, COORD_H8, COORD_G7}}, // 32
    {10, {COORD_G2, COORD_H1, COORD_H2, COORD_H3, COORD_H4, COORD_H5, COORD_H6, COORD_H7, COORD_H8, COORD_G7}}, // 33

    // 9 triangle
    {10, {COORD_A1, COORD_B1, COORD_C1, COORD_D1, COORD_A2, COORD_B2, COORD_C2, COORD_A3, COORD_B3, COORD_A4}}, // 34
    {10, {COORD_H1, COORD_G1, COORD_F1, COORD_E1, COORD_H2, COORD_G2, COORD_F2, COORD_H3, COORD_G3, COORD_H4}}, // 35
    {10, {COORD_A8, COORD_B8, COORD_C8, COORD_D8, COORD_A7, COORD_B7, COORD_C7, COORD_A6, COORD_B6, COORD_A5}}, // 36
    {10, {COORD_H8, COORD_G8, COORD_F8, COORD_E8, COORD_H7, COORD_G7, COORD_F7, COORD_H6, COORD_G6, COORD_H5}}, // 37

    // 10 corner + block
    {10, {COORD_A1, COORD_C1, COORD_D1, COORD_E1, COORD_F1, COORD_H1, COORD_C2, COORD_D2, COORD_E2, COORD_F2}}, // 38
    {10, {COORD_A1, COORD_A3, COORD_A4, COORD_A5, COORD_A6, COORD_A8, COORD_B3, COORD_B4, COORD_B5, COORD_B6}}, // 39
    {10, {COORD_A8, COORD_C8, COORD_D8, COORD_E8, COORD_F8, COORD_H8, COORD_C7, COORD_D7, COORD_E7, COORD_F7}}, // 40
    {10, {COORD_H1, COORD_H3, COORD_H4, COORD_H5, COORD_H6, COORD_H8, COORD_G3, COORD_G4, COORD_G5, COORD_G6}}, // 41

    // 11 cross
    {10, {COORD_A1, COORD_B2, COORD_C3, COORD_D4, COORD_B1, COORD_C2, COORD_D3, COORD_A2, COORD_B3, COORD_C4}}, // 42
    {10, {COORD_H1, COORD_G2, COORD_F3, COORD_E4, COORD_G1, COORD_F2, COORD_E3, COORD_H2, COORD_G3, COORD_F4}}, // 43
    {10, {COORD_A8, COORD_B7, COORD_C6, COORD_D5, COORD_B8, COORD_C7, COORD_D6, COORD_A7, COORD_B6, COORD_C5}}, // 44
    {10, {COORD_H8, COORD_G7, COORD_F6, COORD_E5, COORD_G8, COORD_F7, COORD_E6, COORD_H7, COORD_G6, COORD_F5}}, // 45

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
    {10, {COORD_H8, COORD_G8, COORD_H7, COORD_G7, COORD_F7, COORD_E7, COORD_D7, COORD_G6, COORD_G5, COORD_G4}}  // 61
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
    {13, {{24, P30}, {29, P38}, {32, P31}, {33, P31}, {37, P39}, {40, P34}, {41, P34}, {45, P39}, {48, P31}, {49, P31}, {53, P39}, {57, P39}, {61, P39}}}, // COORD_H8
    {10, {{ 3, P30}, {22, P30}, {29, P37}, {32, P32}, {37, P38}, {45, P35}, {48, P32}, {53, P38}, {57, P38}, {61, P38}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_G8
    { 8, {{ 7, P30}, {18, P30}, {29, P36}, {32, P33}, {37, P37}, {40, P35}, {48, P33}, {53, P37}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_F8
    { 8, {{11, P30}, {14, P30}, {32, P34}, {37, P36}, {40, P36}, {48, P34}, {52, P35}, {53, P36}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_E8
    { 8, {{ 9, P30}, {15, P30}, {32, P35}, {36, P36}, {40, P37}, {48, P35}, {52, P36}, {53, P35}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_D8
    { 8, {{ 5, P30}, {19, P30}, {28, P36}, {32, P36}, {36, P37}, {40, P38}, {48, P36}, {52, P37}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_C8
    {10, {{ 1, P30}, {23, P30}, {28, P37}, {32, P37}, {36, P38}, {44, P35}, {48, P37}, {52, P38}, {56, P38}, {60, P38}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_B8
    {13, {{25, P30}, {28, P38}, {31, P31}, {32, P38}, {36, P39}, {39, P34}, {40, P39}, {44, P39}, {47, P31}, {48, P38}, {52, P39}, {56, P39}, {60, P39}}}, // COORD_A8
    {10, {{ 2, P30}, {20, P30}, {29, P35}, {33, P32}, {37, P35}, {45, P32}, {49, P32}, {53, P34}, {57, P37}, {61, P37}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_H7
    {11, {{ 2, P31}, { 3, P31}, {24, P31}, {29, P34}, {32, P30}, {33, P30}, {37, P34}, {45, P38}, {53, P33}, {57, P36}, {61, P36}, { 0, PNO}, { 0, PNO}}}, // COORD_G7
    {10, {{ 2, P32}, { 7, P31}, {22, P31}, {29, P33}, {37, P33}, {40, P30}, {45, P34}, {48, P30}, {57, P35}, {61, P35}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_F7
    { 8, {{ 2, P33}, {11, P31}, {15, P31}, {18, P31}, {40, P31}, {57, P34}, {60, P33}, {61, P34}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_E7
    { 8, {{ 2, P34}, { 9, P31}, {14, P31}, {19, P31}, {40, P32}, {56, P34}, {60, P34}, {61, P33}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_D7
    {10, {{ 2, P35}, { 5, P31}, {23, P31}, {28, P33}, {36, P33}, {40, P33}, {44, P34}, {48, P39}, {56, P35}, {60, P35}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_C7
    {11, {{ 1, P31}, { 2, P36}, {25, P31}, {28, P34}, {31, P30}, {32, P39}, {36, P34}, {44, P38}, {52, P33}, {56, P36}, {60, P36}, { 0, PNO}, { 0, PNO}}}, // COORD_B7
    {10, {{ 2, P37}, {21, P30}, {28, P35}, {31, P32}, {36, P35}, {44, P32}, {47, P32}, {52, P34}, {56, P37}, {60, P37}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_A7
    { 8, {{ 6, P30}, {16, P30}, {29, P32}, {33, P33}, {37, P32}, {41, P35}, {49, P33}, {53, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_H6
    {10, {{ 3, P32}, { 6, P31}, {20, P31}, {29, P31}, {37, P31}, {41, P30}, {45, P31}, {49, P30}, {57, P33}, {61, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_G6
    { 7, {{ 6, P32}, { 7, P32}, {15, P32}, {24, P32}, {29, P30}, {45, P37}, {57, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_F6
    { 5, {{ 6, P33}, {11, P32}, {19, P32}, {22, P32}, {45, P33}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_E6
    { 5, {{ 6, P34}, { 9, P32}, {18, P32}, {23, P32}, {44, P33}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_D6
    { 7, {{ 5, P32}, { 6, P35}, {14, P32}, {25, P32}, {28, P30}, {44, P37}, {56, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_C6
    {10, {{ 1, P32}, { 6, P36}, {21, P31}, {28, P31}, {36, P31}, {39, P30}, {44, P31}, {47, P30}, {56, P33}, {60, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_B6
    { 8, {{ 6, P37}, {17, P30}, {28, P32}, {31, P33}, {36, P32}, {39, P35}, {47, P33}, {52, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_A6
    { 8, {{10, P30}, {12, P30}, {33, P34}, {37, P30}, {41, P36}, {49, P34}, {51, P30}, {53, P31}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_H5
    { 8, {{ 3, P33}, {10, P31}, {15, P33}, {16, P31}, {41, P31}, {57, P31}, {59, P30}, {61, P31}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_G5
    { 5, {{ 7, P33}, {10, P32}, {19, P33}, {20, P32}, {45, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_F5
    { 6, {{10, P33}, {11, P33}, {23, P33}, {24, P33}, {45, P36}, {57, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_E5
    { 6, {{ 9, P33}, {10, P34}, {22, P33}, {25, P33}, {44, P36}, {56, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_D5
    { 5, {{ 5, P33}, {10, P35}, {18, P33}, {21, P32}, {44, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_C5
    { 8, {{ 1, P33}, {10, P36}, {14, P33}, {17, P31}, {39, P31}, {56, P31}, {58, P30}, {60, P31}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_B5
    { 8, {{10, P37}, {13, P30}, {31, P34}, {36, P30}, {39, P36}, {47, P34}, {50, P30}, {52, P31}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_A5
    { 8, {{ 8, P30}, {15, P34}, {33, P35}, {35, P30}, {41, P37}, {49, P35}, {51, P31}, {53, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_H4
    { 8, {{ 3, P34}, { 8, P31}, {12, P31}, {19, P34}, {41, P32}, {55, P31}, {59, P31}, {61, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_G4
    { 5, {{ 7, P34}, { 8, P32}, {16, P32}, {23, P34}, {43, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_F4
    { 6, {{ 8, P33}, {11, P34}, {20, P33}, {25, P34}, {43, P36}, {55, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_E4
    { 6, {{ 8, P34}, { 9, P34}, {21, P33}, {24, P34}, {42, P36}, {54, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_D4
    { 5, {{ 5, P34}, { 8, P35}, {17, P32}, {22, P34}, {42, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_C4
    { 8, {{ 1, P34}, { 8, P36}, {13, P31}, {18, P34}, {39, P32}, {54, P31}, {58, P31}, {60, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_B4
    { 8, {{ 8, P37}, {14, P34}, {31, P35}, {34, P30}, {39, P37}, {47, P35}, {50, P31}, {52, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_A4
    { 8, {{ 4, P30}, {19, P35}, {27, P32}, {33, P36}, {35, P32}, {41, P38}, {49, P36}, {51, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_H3
    {10, {{ 3, P35}, { 4, P31}, {23, P35}, {27, P31}, {35, P31}, {41, P33}, {43, P31}, {49, P39}, {55, P33}, {59, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_G3
    { 7, {{ 4, P32}, { 7, P35}, {12, P32}, {25, P35}, {27, P30}, {43, P37}, {55, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_F3
    { 5, {{ 4, P33}, {11, P35}, {16, P33}, {21, P34}, {43, P33}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_E3
    { 5, {{ 4, P34}, { 9, P35}, {17, P33}, {20, P34}, {42, P33}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_D3
    { 7, {{ 4, P35}, { 5, P35}, {13, P32}, {24, P35}, {26, P30}, {42, P37}, {54, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_C3
    {10, {{ 1, P35}, { 4, P36}, {22, P35}, {26, P31}, {34, P31}, {39, P33}, {42, P31}, {47, P39}, {54, P33}, {58, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_B3
    { 8, {{ 4, P37}, {18, P35}, {26, P32}, {31, P36}, {34, P32}, {39, P38}, {47, P36}, {50, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_A3
    {10, {{ 0, P30}, {23, P36}, {27, P35}, {33, P37}, {35, P35}, {43, P32}, {49, P37}, {51, P34}, {55, P37}, {59, P37}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_H2
    {11, {{ 0, P31}, { 3, P36}, {25, P36}, {27, P34}, {30, P30}, {33, P39}, {35, P34}, {43, P38}, {51, P33}, {55, P36}, {59, P36}, { 0, PNO}, { 0, PNO}}}, // COORD_G2
    {10, {{ 0, P32}, { 7, P36}, {21, P35}, {27, P33}, {35, P33}, {38, P30}, {43, P34}, {46, P30}, {55, P35}, {59, P35}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_F2
    { 8, {{ 0, P33}, {11, P36}, {12, P33}, {17, P34}, {38, P31}, {55, P34}, {58, P33}, {59, P34}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_E2
    { 8, {{ 0, P34}, { 9, P36}, {13, P33}, {16, P34}, {38, P32}, {54, P34}, {58, P34}, {59, P33}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_D2
    {10, {{ 0, P35}, { 5, P36}, {20, P35}, {26, P33}, {34, P33}, {38, P33}, {42, P34}, {46, P39}, {54, P35}, {58, P35}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_C2
    {11, {{ 0, P36}, { 1, P36}, {24, P36}, {26, P34}, {30, P39}, {31, P39}, {34, P34}, {42, P38}, {50, P33}, {54, P36}, {58, P36}, { 0, PNO}, { 0, PNO}}}, // COORD_B2
    {10, {{ 0, P37}, {22, P36}, {26, P35}, {31, P37}, {34, P35}, {42, P32}, {47, P37}, {50, P34}, {54, P37}, {58, P37}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_A2
    {13, {{25, P37}, {27, P38}, {30, P31}, {33, P38}, {35, P39}, {38, P34}, {41, P39}, {43, P39}, {46, P31}, {49, P38}, {51, P39}, {55, P39}, {59, P39}}}, // COORD_H1
    {10, {{ 3, P37}, {21, P36}, {27, P37}, {30, P32}, {35, P38}, {43, P35}, {46, P32}, {51, P38}, {55, P38}, {59, P38}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_G1
    { 8, {{ 7, P37}, {17, P35}, {27, P36}, {30, P33}, {35, P37}, {38, P35}, {46, P33}, {51, P37}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_F1
    { 8, {{11, P37}, {13, P34}, {30, P34}, {35, P36}, {38, P36}, {46, P34}, {50, P35}, {51, P36}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_E1
    { 8, {{ 9, P37}, {12, P34}, {30, P35}, {34, P36}, {38, P37}, {46, P35}, {50, P36}, {51, P35}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_D1
    { 8, {{ 5, P37}, {16, P35}, {26, P36}, {30, P36}, {34, P37}, {38, P38}, {46, P36}, {50, P37}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_C1
    {10, {{ 1, P37}, {20, P36}, {26, P37}, {30, P37}, {34, P38}, {42, P35}, {46, P37}, {50, P38}, {54, P38}, {58, P38}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_B1
    {13, {{24, P37}, {26, P38}, {30, P38}, {31, P38}, {34, P39}, {38, P39}, {39, P39}, {42, P39}, {46, P38}, {47, P38}, {50, P39}, {54, P39}, {58, P39}}}  // COORD_A1
};

/*
    @brief constants of 3 ^ N
*/
constexpr uint_fast16_t pow3[11] = {1, P31, P32, P33, P34, P35, P36, P37, P38, P39, P310};

/*
    @brief evaluation parameters
*/
int16_t pattern_arr[2][N_PHASES][N_PATTERNS][MAX_EVALUATE_IDX];
int16_t eval_sur0_sur1_arr[N_PHASES][MAX_SURROUND][MAX_SURROUND];
int16_t eval_canput0_canput1_arr[N_PHASES][MAX_CANPUT][MAX_CANPUT];
int16_t eval_num0_num1_arr[N_PHASES][MAX_STONE_NUM][MAX_STONE_NUM];
int16_t eval_mobility_pattern[N_PHASES][N_CANPUT_PATTERNS][MOBILITY_PATTERN_SIZE][MOBILITY_PATTERN_SIZE];

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
*/
void init_pattern_arr_rev(int phase_idx, int pattern_idx, int siz){
    int ri;
    for (int i = 0; i < (int)pow3[siz]; ++i){
        ri = swap_player_idx(i, siz);
        pattern_arr[1][phase_idx][pattern_idx][ri] = pattern_arr[0][phase_idx][pattern_idx][i];
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
    constexpr int pattern_sizes[N_PATTERNS] = {8, 8, 8, 5, 6, 7, 8, 9, 10, 10, 10, 10, 10, 10, 10, 10};
    for (phase_idx = 0; phase_idx < N_PHASES; ++phase_idx){
        for (pattern_idx = 0; pattern_idx < N_PATTERNS; ++pattern_idx){
            if (fread(pattern_arr[0][phase_idx][pattern_idx], 2, pow3[pattern_sizes[pattern_idx]], fp) < pow3[pattern_sizes[pattern_idx]]){
                std::cerr << "[ERROR] [FATAL] evaluation file broken" << std::endl;
                fclose(fp);
                return false;
            }
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
        if (fread(eval_mobility_pattern[phase_idx], 2, N_CANPUT_PATTERNS * MOBILITY_PATTERN_SIZE * MOBILITY_PATTERN_SIZE, fp) < N_CANPUT_PATTERNS * MOBILITY_PATTERN_SIZE * MOBILITY_PATTERN_SIZE){
            std::cerr << "[ERROR] [FATAL] evaluation file broken" << std::endl;
            fclose(fp);
            return false;
        }
    }
    if (thread_pool.size() >= 2){
        std::future<void> tasks[N_PHASES * N_PATTERNS];
        int i = 0;
        for (phase_idx = 0; phase_idx < N_PHASES; ++phase_idx){
            for (pattern_idx = 0; pattern_idx < N_PATTERNS; ++pattern_idx)
                tasks[i++] = thread_pool.push(std::bind(init_pattern_arr_rev, phase_idx, pattern_idx, pattern_sizes[pattern_idx]));
        }
        for (std::future<void> &task: tasks)
            task.get();
    } else{
        for (phase_idx = 0; phase_idx < N_PHASES; ++phase_idx){
            for (pattern_idx = 0; pattern_idx < N_PATTERNS; ++pattern_idx)
                init_pattern_arr_rev(phase_idx, pattern_idx, pattern_sizes[pattern_idx]);
        }
    }
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

/*
    @brief pattern evaluation

    @param phase_idx            evaluation phase
    @param search               search information
    @return pattern evaluation value
*/
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
    uint64_t p90 = black_line_mirror(player_mobility);
    uint64_t o90 = black_line_mirror(opponent_mobility);
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
    sur0 = calc_surround(search.board.player, empties);
    sur1 = calc_surround(search.board.opponent, empties);
    num0 = pop_count_ull(search.board.player);
    num1 = search.n_discs - num0;
    int res = calc_pattern_diff(phase_idx, &search) + 
        eval_sur0_sur1_arr[phase_idx][sur0][sur1] + 
        eval_canput0_canput1_arr[phase_idx][canput0][canput1] + 
        eval_num0_num1_arr[phase_idx][num0][num1] + 
        calc_mobility_pattern(phase_idx, player_mobility, opponent_mobility);
    res += res >= 0 ? STEP_2 : -STEP_2;
    res /= STEP;
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
    if (res > SCORE_MAX)
        return SCORE_MAX;
    if (res < -SCORE_MAX)
        return -SCORE_MAX;
    return res;
}

inline uint_fast16_t pick_pattern_idx(const uint_fast8_t b_arr[], const Feature_to_coord *f){
    uint_fast16_t res = 0;
    for (int i = 0; i < f->n_cells; ++i){
        res *= 3;
        res += b_arr[HW2_M1 - f->cells[i]];
    }
    return res;
}

inline void calc_features(Search *search){
    uint_fast8_t b_arr[HW2];
    search->board.translate_to_arr_player(b_arr);
    for (int i = 0; i < N_SYMMETRY_PATTERNS; ++i)
        search->eval_features[i] = pick_pattern_idx(b_arr, &feature_to_coord[i]);
    search->eval_feature_reversed = 0;
}

inline bool check_features(Search *search){
    uint_fast8_t b_arr[HW2];
    search->board.translate_to_arr_player(b_arr);
    for (int i = 0; i < N_SYMMETRY_PATTERNS; ++i){
        if (search->eval_features[i] != pick_pattern_idx(b_arr, &feature_to_coord[i])){
            std::cerr << i << " " << search->eval_features[i] << " " << pick_pattern_idx(b_arr, &feature_to_coord[i]) << std::endl;
            return true;
        }
    }
    return false;
}

inline void eval_move(Search *search, const Flip *flip){
    uint_fast8_t i, cell;
    uint64_t f;
    if (search->eval_feature_reversed){
        for (i = 0; i < MAX_CELL_PATTERNS && coord_to_feature[flip->pos].features[i].x; ++i)
            search->eval_features[coord_to_feature[flip->pos].features[i].feature] -= coord_to_feature[flip->pos].features[i].x;
        f = flip->flip;
        for (cell = first_bit(&f); f; cell = next_bit(&f)){
            for (i = 0; i < MAX_CELL_PATTERNS && coord_to_feature[cell].features[i].x; ++i)
                search->eval_features[coord_to_feature[cell].features[i].feature] += coord_to_feature[cell].features[i].x;
        }
    } else{
        for (i = 0; i < MAX_CELL_PATTERNS && coord_to_feature[flip->pos].features[i].x; ++i)
            search->eval_features[coord_to_feature[flip->pos].features[i].feature] -= 2 * coord_to_feature[flip->pos].features[i].x;
        f = flip->flip;
        for (cell = first_bit(&f); f; cell = next_bit(&f)){
            for (i = 0; i < MAX_CELL_PATTERNS && coord_to_feature[cell].features[i].x; ++i)
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
        for (i = 0; i < MAX_CELL_PATTERNS && coord_to_feature[flip->pos].features[i].x; ++i)
            search->eval_features[coord_to_feature[flip->pos].features[i].feature] += coord_to_feature[flip->pos].features[i].x;
        f = flip->flip;
        for (cell = first_bit(&f); f; cell = next_bit(&f)){
            for (i = 0; i < MAX_CELL_PATTERNS && coord_to_feature[cell].features[i].x; ++i)
                search->eval_features[coord_to_feature[cell].features[i].feature] -= coord_to_feature[cell].features[i].x;
        }
    } else{
        for (i = 0; i < MAX_CELL_PATTERNS && coord_to_feature[flip->pos].features[i].x; ++i)
            search->eval_features[coord_to_feature[flip->pos].features[i].feature] += 2 * coord_to_feature[flip->pos].features[i].x;
        f = flip->flip;
        for (cell = first_bit(&f); f; cell = next_bit(&f)){
            for (i = 0; i < MAX_CELL_PATTERNS && coord_to_feature[cell].features[i].x; ++i)
                search->eval_features[coord_to_feature[cell].features[i].feature] += coord_to_feature[cell].features[i].x;
        }
    }
}
