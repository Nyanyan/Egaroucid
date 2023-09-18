/*
    Egaroucid Project

    @file evaluate_generic.hpp
        Evaluation function without AVX2
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


/*
    @brief evaluation pattern definition
*/
// disc patterns
#define N_PATTERNS 20
#define MAX_PATTERN_CELLS 10
#define MAX_CELL_PATTERNS 17
#define MAX_EVALUATE_IDX 59049

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
    {10, {COORD_H8, COORD_G8, COORD_H7, COORD_G7, COORD_F7, COORD_E7, COORD_D7, COORD_G6, COORD_G5, COORD_G4}}, // 61

    // 16 boot
    {10, {COORD_A2, COORD_B2, COORD_A3, COORD_B3, COORD_C3, COORD_D3, COORD_A4, COORD_B4, COORD_C4, COORD_D4}}, // 62
    {10, {COORD_G8, COORD_G7, COORD_F8, COORD_F7, COORD_F6, COORD_F5, COORD_E8, COORD_E7, COORD_E6, COORD_E5}}, // 63
    {10, {COORD_B1, COORD_B2, COORD_C1, COORD_C2, COORD_C3, COORD_C4, COORD_D1, COORD_D2, COORD_D3, COORD_D4}}, // 64
    {10, {COORD_H7, COORD_G7, COORD_H6, COORD_G6, COORD_F6, COORD_E6, COORD_H5, COORD_G5, COORD_F5, COORD_E5}}, // 65
    {10, {COORD_H2, COORD_G2, COORD_H3, COORD_G3, COORD_F3, COORD_E3, COORD_H4, COORD_G4, COORD_F4, COORD_E4}}, // 66
    {10, {COORD_A7, COORD_B7, COORD_A6, COORD_B6, COORD_C6, COORD_D6, COORD_A5, COORD_B5, COORD_C5, COORD_D5}}, // 67
    {10, {COORD_G1, COORD_G2, COORD_F1, COORD_F2, COORD_F3, COORD_F4, COORD_E1, COORD_E2, COORD_E3, COORD_E4}}, // 68
    {10, {COORD_B8, COORD_B7, COORD_C8, COORD_C7, COORD_C6, COORD_C5, COORD_D8, COORD_D7, COORD_D6, COORD_D5}}, // 69

    // 17 thunder
    {10, {COORD_B2, COORD_C2, COORD_C3, COORD_D3, COORD_E3, COORD_F3, COORD_C4, COORD_D4, COORD_E4, COORD_F4}}, // 70
    {10, {COORD_G7, COORD_G6, COORD_F6, COORD_F5, COORD_F4, COORD_F3, COORD_E6, COORD_E5, COORD_E4, COORD_E3}}, // 71
    {10, {COORD_B2, COORD_B3, COORD_C3, COORD_C4, COORD_C5, COORD_C6, COORD_D3, COORD_D4, COORD_D5, COORD_D6}}, // 72
    {10, {COORD_G7, COORD_F7, COORD_F6, COORD_E6, COORD_D6, COORD_C6, COORD_F5, COORD_E5, COORD_D5, COORD_C5}}, // 73
    {10, {COORD_G2, COORD_F2, COORD_F3, COORD_E3, COORD_D3, COORD_C3, COORD_F4, COORD_E4, COORD_D4, COORD_C4}}, // 74
    {10, {COORD_B7, COORD_C7, COORD_C6, COORD_D6, COORD_E6, COORD_F6, COORD_C5, COORD_D5, COORD_E5, COORD_F5}}, // 75
    {10, {COORD_G2, COORD_G3, COORD_F3, COORD_F4, COORD_F5, COORD_F6, COORD_E3, COORD_E4, COORD_E5, COORD_E6}}, // 76
    {10, {COORD_B7, COORD_B6, COORD_C6, COORD_C5, COORD_C4, COORD_C3, COORD_D6, COORD_D5, COORD_D4, COORD_D3}}, // 77

    // 18 inner block
    {10, {COORD_A1, COORD_B1, COORD_C2, COORD_D2, COORD_E2, COORD_F2, COORD_C3, COORD_D3, COORD_E3, COORD_F3}}, // 78
    {10, {COORD_H8, COORD_H7, COORD_G6, COORD_G5, COORD_G4, COORD_G3, COORD_F6, COORD_F5, COORD_F4, COORD_F3}}, // 79
    {10, {COORD_A1, COORD_A2, COORD_B3, COORD_B4, COORD_B5, COORD_B6, COORD_C3, COORD_C4, COORD_C5, COORD_C6}}, // 80
    {10, {COORD_H8, COORD_G8, COORD_F7, COORD_E7, COORD_D7, COORD_C7, COORD_F6, COORD_E6, COORD_D6, COORD_C6}}, // 81
    {10, {COORD_H1, COORD_G1, COORD_F2, COORD_E2, COORD_D2, COORD_C2, COORD_F3, COORD_E3, COORD_D3, COORD_C3}}, // 82
    {10, {COORD_A8, COORD_B8, COORD_C7, COORD_D7, COORD_E7, COORD_F7, COORD_C6, COORD_D6, COORD_E6, COORD_F6}}, // 83
    {10, {COORD_H1, COORD_H2, COORD_G3, COORD_G4, COORD_G5, COORD_G6, COORD_F3, COORD_F4, COORD_F5, COORD_F6}}, // 84
    {10, {COORD_A8, COORD_A7, COORD_B6, COORD_B5, COORD_B4, COORD_B3, COORD_C6, COORD_C5, COORD_C4, COORD_C3}}, // 85

    // 19 2 edge
    {10, {COORD_C1, COORD_D1, COORD_E1, COORD_F1, COORD_B2, COORD_C2, COORD_A3, COORD_B3, COORD_A4, COORD_B4}}, // 86
    {10, {COORD_H6, COORD_H5, COORD_H4, COORD_H3, COORD_G7, COORD_G6, COORD_F8, COORD_F7, COORD_E8, COORD_E7}}, // 87
    {10, {COORD_A3, COORD_A4, COORD_A5, COORD_A6, COORD_B2, COORD_B3, COORD_C1, COORD_C2, COORD_D1, COORD_D2}}, // 88
    {10, {COORD_F8, COORD_E8, COORD_D8, COORD_C8, COORD_G7, COORD_F7, COORD_H6, COORD_G6, COORD_H5, COORD_G5}}, // 89
    {10, {COORD_F1, COORD_E1, COORD_D1, COORD_C1, COORD_G2, COORD_F2, COORD_H3, COORD_G3, COORD_H4, COORD_G4}}, // 90
    {10, {COORD_C8, COORD_D8, COORD_E8, COORD_F8, COORD_B7, COORD_C7, COORD_A6, COORD_B6, COORD_A5, COORD_B5}}, // 91
    {10, {COORD_H3, COORD_H4, COORD_H5, COORD_H6, COORD_G2, COORD_G3, COORD_F1, COORD_F2, COORD_E1, COORD_E2}}, // 92
    {10, {COORD_A6, COORD_A5, COORD_A4, COORD_A3, COORD_B7, COORD_B6, COORD_C8, COORD_C7, COORD_D8, COORD_D7}}  // 93
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
    {15, {{24, P30}, {29, P38}, {32, P31}, {33, P31}, {37, P39}, {40, P34}, {41, P34}, {45, P39}, {48, P31}, {49, P31}, {53, P39}, {57, P39}, {61, P39}, {79, P39}, {81, P39}, { 0, PNO}, { 0, PNO}}}, // COORD_H8
    {12, {{ 3, P30}, {22, P30}, {29, P37}, {32, P32}, {37, P38}, {45, P35}, {48, P32}, {53, P38}, {57, P38}, {61, P38}, {63, P39}, {81, P38}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_G8
    {12, {{ 7, P30}, {18, P30}, {29, P36}, {32, P33}, {37, P37}, {40, P35}, {48, P33}, {53, P37}, {63, P37}, {87, P33}, {89, P39}, {91, P36}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_F8
    {12, {{11, P30}, {14, P30}, {32, P34}, {37, P36}, {40, P36}, {48, P34}, {52, P35}, {53, P36}, {63, P33}, {87, P31}, {89, P38}, {91, P37}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_E8
    {12, {{ 9, P30}, {15, P30}, {32, P35}, {36, P36}, {40, P37}, {48, P35}, {52, P36}, {53, P35}, {69, P33}, {89, P37}, {91, P38}, {93, P31}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_D8
    {12, {{ 5, P30}, {19, P30}, {28, P36}, {32, P36}, {36, P37}, {40, P38}, {48, P36}, {52, P37}, {69, P37}, {89, P36}, {91, P39}, {93, P33}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_C8
    {12, {{ 1, P30}, {23, P30}, {28, P37}, {32, P37}, {36, P38}, {44, P35}, {48, P37}, {52, P38}, {56, P38}, {60, P38}, {69, P39}, {83, P38}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_B8
    {15, {{25, P30}, {28, P38}, {31, P31}, {32, P38}, {36, P39}, {39, P34}, {40, P39}, {44, P39}, {47, P31}, {48, P38}, {52, P39}, {56, P39}, {60, P39}, {83, P39}, {85, P39}, { 0, PNO}, { 0, PNO}}}, // COORD_A8
    {12, {{ 2, P30}, {20, P30}, {29, P35}, {33, P32}, {37, P35}, {45, P32}, {49, P32}, {53, P34}, {57, P37}, {61, P37}, {65, P39}, {79, P38}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_H7
    {17, {{ 2, P31}, { 3, P31}, {24, P31}, {29, P34}, {32, P30}, {33, P30}, {37, P34}, {45, P38}, {53, P33}, {57, P36}, {61, P36}, {63, P38}, {65, P38}, {71, P39}, {73, P39}, {87, P35}, {89, P35}}}, // COORD_G7
    {16, {{ 2, P32}, { 7, P31}, {22, P31}, {29, P33}, {37, P33}, {40, P30}, {45, P34}, {48, P30}, {57, P35}, {61, P35}, {63, P36}, {73, P38}, {81, P37}, {83, P34}, {87, P32}, {89, P34}, { 0, PNO}}}, // COORD_F7
    {12, {{ 2, P33}, {11, P31}, {15, P31}, {18, P31}, {40, P31}, {57, P34}, {60, P33}, {61, P34}, {63, P32}, {81, P36}, {83, P35}, {87, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_E7
    {12, {{ 2, P34}, { 9, P31}, {14, P31}, {19, P31}, {40, P32}, {56, P34}, {60, P34}, {61, P33}, {69, P32}, {81, P35}, {83, P36}, {93, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_D7
    {16, {{ 2, P35}, { 5, P31}, {23, P31}, {28, P33}, {36, P33}, {40, P33}, {44, P34}, {48, P39}, {56, P35}, {60, P35}, {69, P36}, {75, P38}, {81, P34}, {83, P37}, {91, P34}, {93, P32}, { 0, PNO}}}, // COORD_C7
    {17, {{ 1, P31}, { 2, P36}, {25, P31}, {28, P34}, {31, P30}, {32, P39}, {36, P34}, {44, P38}, {52, P33}, {56, P36}, {60, P36}, {67, P38}, {69, P38}, {75, P39}, {77, P39}, {91, P35}, {93, P35}}}, // COORD_B7
    {12, {{ 2, P37}, {21, P30}, {28, P35}, {31, P32}, {36, P35}, {44, P32}, {47, P32}, {52, P34}, {56, P37}, {60, P37}, {67, P39}, {85, P38}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_A7
    {12, {{ 6, P30}, {16, P30}, {29, P32}, {33, P33}, {37, P32}, {41, P35}, {49, P33}, {53, P32}, {65, P37}, {87, P39}, {89, P33}, {92, P36}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_H6
    {16, {{ 3, P32}, { 6, P31}, {20, P31}, {29, P31}, {37, P31}, {41, P30}, {45, P31}, {49, P30}, {57, P33}, {61, P32}, {65, P36}, {71, P38}, {79, P37}, {84, P34}, {87, P34}, {89, P32}, { 0, PNO}}}, // COORD_G6
    {17, {{ 6, P32}, { 7, P32}, {15, P32}, {24, P32}, {29, P30}, {45, P37}, {57, P32}, {63, P35}, {65, P35}, {71, P37}, {73, P37}, {75, P34}, {76, P34}, {79, P33}, {81, P33}, {83, P30}, {84, P30}}}, // COORD_F6
    {13, {{ 6, P33}, {11, P32}, {19, P32}, {22, P32}, {45, P33}, {63, P31}, {65, P34}, {71, P33}, {73, P36}, {75, P35}, {76, P30}, {81, P32}, {83, P31}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_E6
    {13, {{ 6, P34}, { 9, P32}, {18, P32}, {23, P32}, {44, P33}, {67, P34}, {69, P31}, {72, P30}, {73, P35}, {75, P36}, {77, P33}, {81, P31}, {83, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_D6
    {17, {{ 5, P32}, { 6, P35}, {14, P32}, {25, P32}, {28, P30}, {44, P37}, {56, P32}, {67, P35}, {69, P35}, {72, P34}, {73, P34}, {75, P37}, {77, P37}, {80, P30}, {81, P30}, {83, P33}, {85, P33}}}, // COORD_C6
    {16, {{ 1, P32}, { 6, P36}, {21, P31}, {28, P31}, {36, P31}, {39, P30}, {44, P31}, {47, P30}, {56, P33}, {60, P32}, {67, P36}, {77, P38}, {80, P34}, {85, P37}, {91, P32}, {93, P34}, { 0, PNO}}}, // COORD_B6
    {12, {{ 6, P37}, {17, P30}, {28, P32}, {31, P33}, {36, P32}, {39, P35}, {47, P33}, {52, P32}, {67, P37}, {88, P36}, {91, P33}, {93, P39}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_A6
    {12, {{10, P30}, {12, P30}, {33, P34}, {37, P30}, {41, P36}, {49, P34}, {51, P30}, {53, P31}, {65, P33}, {87, P38}, {89, P31}, {92, P37}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_H5
    {12, {{ 3, P33}, {10, P31}, {15, P33}, {16, P31}, {41, P31}, {57, P31}, {59, P30}, {61, P31}, {65, P32}, {79, P36}, {84, P35}, {89, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_G5
    {13, {{ 7, P33}, {10, P32}, {19, P33}, {20, P32}, {45, P30}, {63, P34}, {65, P31}, {71, P36}, {73, P33}, {75, P30}, {76, P35}, {79, P32}, {84, P31}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_F5
    {12, {{10, P33}, {11, P33}, {23, P33}, {24, P33}, {45, P36}, {57, P30}, {63, P30}, {65, P30}, {71, P32}, {73, P32}, {75, P31}, {76, P31}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_E5
    {12, {{ 9, P33}, {10, P34}, {22, P33}, {25, P33}, {44, P36}, {56, P30}, {67, P30}, {69, P30}, {72, P31}, {73, P31}, {75, P32}, {77, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_D5
    {13, {{ 5, P33}, {10, P35}, {18, P33}, {21, P32}, {44, P30}, {67, P31}, {69, P34}, {72, P35}, {73, P30}, {75, P33}, {77, P36}, {80, P31}, {85, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_C5
    {12, {{ 1, P33}, {10, P36}, {14, P33}, {17, P31}, {39, P31}, {56, P31}, {58, P30}, {60, P31}, {67, P32}, {80, P35}, {85, P36}, {91, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_B5
    {12, {{10, P37}, {13, P30}, {31, P34}, {36, P30}, {39, P36}, {47, P34}, {50, P30}, {52, P31}, {67, P33}, {88, P37}, {91, P31}, {93, P38}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_A5
    {12, {{ 8, P30}, {15, P34}, {33, P35}, {35, P30}, {41, P37}, {49, P35}, {51, P31}, {53, P30}, {66, P33}, {87, P37}, {90, P31}, {92, P38}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_H4
    {12, {{ 3, P34}, { 8, P31}, {12, P31}, {19, P34}, {41, P32}, {55, P31}, {59, P31}, {61, P30}, {66, P32}, {79, P35}, {84, P36}, {90, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_G4
    {13, {{ 7, P34}, { 8, P32}, {16, P32}, {23, P34}, {43, P30}, {66, P31}, {68, P34}, {70, P30}, {71, P35}, {74, P33}, {76, P36}, {79, P31}, {84, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_F4
    {12, {{ 8, P33}, {11, P34}, {20, P33}, {25, P34}, {43, P36}, {55, P30}, {66, P30}, {68, P30}, {70, P31}, {71, P31}, {74, P32}, {76, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_E4
    {12, {{ 8, P34}, { 9, P34}, {21, P33}, {24, P34}, {42, P36}, {54, P30}, {62, P30}, {64, P30}, {70, P32}, {72, P32}, {74, P31}, {77, P31}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_D4
    {13, {{ 5, P34}, { 8, P35}, {17, P32}, {22, P34}, {42, P30}, {62, P31}, {64, P34}, {70, P33}, {72, P36}, {74, P30}, {77, P35}, {80, P32}, {85, P31}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_C4
    {12, {{ 1, P34}, { 8, P36}, {13, P31}, {18, P34}, {39, P32}, {54, P31}, {58, P31}, {60, P30}, {62, P32}, {80, P36}, {85, P35}, {86, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_B4
    {12, {{ 8, P37}, {14, P34}, {31, P35}, {34, P30}, {39, P37}, {47, P35}, {50, P31}, {52, P30}, {62, P33}, {86, P31}, {88, P38}, {93, P37}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_A4
    {12, {{ 4, P30}, {19, P35}, {27, P32}, {33, P36}, {35, P32}, {41, P38}, {49, P36}, {51, P32}, {66, P37}, {87, P36}, {90, P33}, {92, P39}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_H3
    {16, {{ 3, P35}, { 4, P31}, {23, P35}, {27, P31}, {35, P31}, {41, P33}, {43, P31}, {49, P39}, {55, P33}, {59, P32}, {66, P36}, {76, P38}, {79, P34}, {84, P37}, {90, P32}, {92, P34}, { 0, PNO}}}, // COORD_G3
    {17, {{ 4, P32}, { 7, P35}, {12, P32}, {25, P35}, {27, P30}, {43, P37}, {55, P32}, {66, P35}, {68, P35}, {70, P34}, {71, P34}, {74, P37}, {76, P37}, {78, P30}, {79, P30}, {82, P33}, {84, P33}}}, // COORD_F3
    {13, {{ 4, P33}, {11, P35}, {16, P33}, {21, P34}, {43, P33}, {66, P34}, {68, P31}, {70, P35}, {71, P30}, {74, P36}, {76, P33}, {78, P31}, {82, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_E3
    {13, {{ 4, P34}, { 9, P35}, {17, P33}, {20, P34}, {42, P33}, {62, P34}, {64, P31}, {70, P36}, {72, P33}, {74, P35}, {77, P30}, {78, P32}, {82, P31}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_D3
    {17, {{ 4, P35}, { 5, P35}, {13, P32}, {24, P35}, {26, P30}, {42, P37}, {54, P32}, {62, P35}, {64, P35}, {70, P37}, {72, P37}, {74, P34}, {77, P34}, {78, P33}, {80, P33}, {82, P30}, {85, P30}}}, // COORD_C3
    {16, {{ 1, P35}, { 4, P36}, {22, P35}, {26, P31}, {34, P31}, {39, P33}, {42, P31}, {47, P39}, {54, P33}, {58, P32}, {62, P36}, {72, P38}, {80, P37}, {85, P34}, {86, P32}, {88, P34}, { 0, PNO}}}, // COORD_B3
    {12, {{ 4, P37}, {18, P35}, {26, P32}, {31, P36}, {34, P32}, {39, P38}, {47, P36}, {50, P32}, {62, P37}, {86, P33}, {88, P39}, {93, P36}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_A3
    {12, {{ 0, P30}, {23, P36}, {27, P35}, {33, P37}, {35, P35}, {43, P32}, {49, P37}, {51, P34}, {55, P37}, {59, P37}, {66, P39}, {84, P38}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_H2
    {17, {{ 0, P31}, { 3, P36}, {25, P36}, {27, P34}, {30, P30}, {33, P39}, {35, P34}, {43, P38}, {51, P33}, {55, P36}, {59, P36}, {66, P38}, {68, P38}, {74, P39}, {76, P39}, {90, P35}, {92, P35}}}, // COORD_G2
    {16, {{ 0, P32}, { 7, P36}, {21, P35}, {27, P33}, {35, P33}, {38, P30}, {43, P34}, {46, P30}, {55, P35}, {59, P35}, {68, P36}, {74, P38}, {78, P34}, {82, P37}, {90, P34}, {92, P32}, { 0, PNO}}}, // COORD_F2
    {12, {{ 0, P33}, {11, P36}, {12, P33}, {17, P34}, {38, P31}, {55, P34}, {58, P33}, {59, P34}, {68, P32}, {78, P35}, {82, P36}, {92, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_E2
    {12, {{ 0, P34}, { 9, P36}, {13, P33}, {16, P34}, {38, P32}, {54, P34}, {58, P34}, {59, P33}, {64, P32}, {78, P36}, {82, P35}, {88, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_D2
    {16, {{ 0, P35}, { 5, P36}, {20, P35}, {26, P33}, {34, P33}, {38, P33}, {42, P34}, {46, P39}, {54, P35}, {58, P35}, {64, P36}, {70, P38}, {78, P37}, {82, P34}, {86, P34}, {88, P32}, { 0, PNO}}}, // COORD_C2
    {17, {{ 0, P36}, { 1, P36}, {24, P36}, {26, P34}, {30, P39}, {31, P39}, {34, P34}, {42, P38}, {50, P33}, {54, P36}, {58, P36}, {62, P38}, {64, P38}, {70, P39}, {72, P39}, {86, P35}, {88, P35}}}, // COORD_B2
    {12, {{ 0, P37}, {22, P36}, {26, P35}, {31, P37}, {34, P35}, {42, P32}, {47, P37}, {50, P34}, {54, P37}, {58, P37}, {62, P39}, {80, P38}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_A2
    {15, {{25, P37}, {27, P38}, {30, P31}, {33, P38}, {35, P39}, {38, P34}, {41, P39}, {43, P39}, {46, P31}, {49, P38}, {51, P39}, {55, P39}, {59, P39}, {82, P39}, {84, P39}, { 0, PNO}, { 0, PNO}}}, // COORD_H1
    {12, {{ 3, P37}, {21, P36}, {27, P37}, {30, P32}, {35, P38}, {43, P35}, {46, P32}, {51, P38}, {55, P38}, {59, P38}, {68, P39}, {82, P38}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_G1
    {12, {{ 7, P37}, {17, P35}, {27, P36}, {30, P33}, {35, P37}, {38, P35}, {46, P33}, {51, P37}, {68, P37}, {86, P36}, {90, P39}, {92, P33}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_F1
    {12, {{11, P37}, {13, P34}, {30, P34}, {35, P36}, {38, P36}, {46, P34}, {50, P35}, {51, P36}, {68, P33}, {86, P37}, {90, P38}, {92, P31}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_E1
    {12, {{ 9, P37}, {12, P34}, {30, P35}, {34, P36}, {38, P37}, {46, P35}, {50, P36}, {51, P35}, {64, P33}, {86, P38}, {88, P31}, {90, P37}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_D1
    {12, {{ 5, P37}, {16, P35}, {26, P36}, {30, P36}, {34, P37}, {38, P38}, {46, P36}, {50, P37}, {64, P37}, {86, P39}, {88, P33}, {90, P36}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_C1
    {12, {{ 1, P37}, {20, P36}, {26, P37}, {30, P37}, {34, P38}, {42, P35}, {46, P37}, {50, P38}, {54, P38}, {58, P38}, {64, P39}, {78, P38}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_B1
    {15, {{24, P37}, {26, P38}, {30, P38}, {31, P38}, {34, P39}, {38, P39}, {39, P39}, {42, P39}, {46, P38}, {47, P38}, {50, P39}, {54, P39}, {58, P39}, {78, P39}, {80, P39}, { 0, PNO}, { 0, PNO}}}  // COORD_A1
};

/*
    @brief constants of 3 ^ N
*/
constexpr uint_fast16_t pow3[11] = {1, P31, P32, P33, P34, P35, P36, P37, P38, P39, P310};

/*
    @brief evaluation parameters
*/
int16_t pattern_arr[2][N_PHASES][N_PATTERNS][MAX_EVALUATE_IDX];

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
    constexpr int pattern_sizes[N_PATTERNS] = {
        8, 8, 8, 5, 6, 7, 8, 9, 
        10, 10, 10, 10, 10, 10, 10, 10, 
        10, 10, 10, 10
    };
    for (phase_idx = 0; phase_idx < N_PHASES; ++phase_idx){
        for (pattern_idx = 0; pattern_idx < N_PATTERNS; ++pattern_idx){
            if (fread(pattern_arr[0][phase_idx][pattern_idx], 2, pow3[pattern_sizes[pattern_idx]], fp) < pow3[pattern_sizes[pattern_idx]]){
                std::cerr << "[ERROR] [FATAL] evaluation file broken" << std::endl;
                fclose(fp);
                return false;
            }
        }
    }
    if (thread_pool.size() >= 2){
        std::future<void> tasks[N_PHASES * N_PATTERNS];
        int i = 0;
        for (phase_idx = 0; phase_idx < N_PHASES; ++phase_idx){
            for (pattern_idx = 0; pattern_idx < N_PATTERNS; ++pattern_idx){
                bool pushed = false;
                while (!pushed)
                    tasks[i] = thread_pool.push(&pushed, std::bind(init_pattern_arr_rev, phase_idx, pattern_idx, pattern_sizes[pattern_idx]));
                ++i;
            }
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
    return init_evaluation_calc("resources/eval.egev2", show_log);
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
    int score = b->count_player() * 2 + e;
    score += (((score >> 6) & 1) + (((score + HW2_M1) >> 7) & 1) - 1) * e;
    return score - HW2;
}

/*
    @brief evaluation function for game over (odd empties)

    @param b                    board
    @param e                    number of empty squares
    @return final score
*/
inline int end_evaluate_odd(Board *b, int e){
    int score = b->count_player() * 2 + e;
    score += (((score >> 5) & 2) - 1) * e;
    return score - HW2;
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
        pattern_arr[search->eval_feature_reversed][phase_idx][15][search->eval_features[58]] + pattern_arr[search->eval_feature_reversed][phase_idx][15][search->eval_features[59]] + pattern_arr[search->eval_feature_reversed][phase_idx][15][search->eval_features[60]] + pattern_arr[search->eval_feature_reversed][phase_idx][15][search->eval_features[61]] + 
        pattern_arr[search->eval_feature_reversed][phase_idx][16][search->eval_features[62]] + pattern_arr[search->eval_feature_reversed][phase_idx][16][search->eval_features[63]] + pattern_arr[search->eval_feature_reversed][phase_idx][16][search->eval_features[64]] + pattern_arr[search->eval_feature_reversed][phase_idx][16][search->eval_features[65]] + 
        pattern_arr[search->eval_feature_reversed][phase_idx][16][search->eval_features[66]] + pattern_arr[search->eval_feature_reversed][phase_idx][16][search->eval_features[67]] + pattern_arr[search->eval_feature_reversed][phase_idx][16][search->eval_features[68]] + pattern_arr[search->eval_feature_reversed][phase_idx][16][search->eval_features[69]] + 
        pattern_arr[search->eval_feature_reversed][phase_idx][17][search->eval_features[70]] + pattern_arr[search->eval_feature_reversed][phase_idx][17][search->eval_features[71]] + pattern_arr[search->eval_feature_reversed][phase_idx][17][search->eval_features[72]] + pattern_arr[search->eval_feature_reversed][phase_idx][17][search->eval_features[73]] + 
        pattern_arr[search->eval_feature_reversed][phase_idx][17][search->eval_features[74]] + pattern_arr[search->eval_feature_reversed][phase_idx][17][search->eval_features[75]] + pattern_arr[search->eval_feature_reversed][phase_idx][17][search->eval_features[76]] + pattern_arr[search->eval_feature_reversed][phase_idx][17][search->eval_features[77]] + 
        pattern_arr[search->eval_feature_reversed][phase_idx][18][search->eval_features[78]] + pattern_arr[search->eval_feature_reversed][phase_idx][18][search->eval_features[79]] + pattern_arr[search->eval_feature_reversed][phase_idx][18][search->eval_features[80]] + pattern_arr[search->eval_feature_reversed][phase_idx][18][search->eval_features[81]] + 
        pattern_arr[search->eval_feature_reversed][phase_idx][18][search->eval_features[82]] + pattern_arr[search->eval_feature_reversed][phase_idx][18][search->eval_features[83]] + pattern_arr[search->eval_feature_reversed][phase_idx][18][search->eval_features[84]] + pattern_arr[search->eval_feature_reversed][phase_idx][18][search->eval_features[85]] + 
        pattern_arr[search->eval_feature_reversed][phase_idx][19][search->eval_features[86]] + pattern_arr[search->eval_feature_reversed][phase_idx][19][search->eval_features[87]] + pattern_arr[search->eval_feature_reversed][phase_idx][19][search->eval_features[88]] + pattern_arr[search->eval_feature_reversed][phase_idx][19][search->eval_features[89]] + 
        pattern_arr[search->eval_feature_reversed][phase_idx][19][search->eval_features[90]] + pattern_arr[search->eval_feature_reversed][phase_idx][19][search->eval_features[91]] + pattern_arr[search->eval_feature_reversed][phase_idx][19][search->eval_features[92]] + pattern_arr[search->eval_feature_reversed][phase_idx][19][search->eval_features[93]];
}

inline void calc_features(Search *search);

/*
    @brief midgame evaluation function

    @param b                    board
    @return evaluation value
*/
inline int mid_evaluate(Board *board){
    Search search;
    search.init_board(board);
    calc_features(&search);
    int phase_idx = search.phase();
    int res = calc_pattern_diff(phase_idx, &search);
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
    int phase_idx = search->phase();
    int res = calc_pattern_diff(phase_idx, search);
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
