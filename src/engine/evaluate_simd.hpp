/*
    Egaroucid Project

    @file evaluate_generic.hpp
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
#define CEIL_N_SYMMETRY_PATTERNS 64         // N_SYMMETRY_PATTRENS + dummy
#define N_PATTERN_PARAMS (521478 + 2)       // +2 for byte bound & dummy
#define SIMD_EVAL_OFFSET 4092               // evaluate range
#define N_SIMD_EVAL_FEATURES_SIMPLE 2
#define N_SIMD_EVAL_FEATURES_COMP 2
#define N_PATTERN_PARAMS_BEFORE_DUMMY 29403
#define SIMD_EVAL_DUMMY_ADDR 29404
#define N_PATTERN_PARAMS_AFTER_DUMMY 492075
#define SIMD_EVAL_F1_OFFSET 19927           // pattern_starts[4]

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

    // dummy
    {0, {COORD_NO, COORD_NO, COORD_NO, COORD_NO, COORD_NO, COORD_NO, COORD_NO, COORD_NO, COORD_NO, COORD_NO}}, // 26
    {0, {COORD_NO, COORD_NO, COORD_NO, COORD_NO, COORD_NO, COORD_NO, COORD_NO, COORD_NO, COORD_NO, COORD_NO}}, // 27

    // 7 corner9
    {9, {COORD_A1, COORD_B1, COORD_C1, COORD_A2, COORD_B2, COORD_C2, COORD_A3, COORD_B3, COORD_C3, COORD_NO}}, // 28
    {9, {COORD_H1, COORD_G1, COORD_F1, COORD_H2, COORD_G2, COORD_F2, COORD_H3, COORD_G3, COORD_F3, COORD_NO}}, // 29
    {9, {COORD_A8, COORD_B8, COORD_C8, COORD_A7, COORD_B7, COORD_C7, COORD_A6, COORD_B6, COORD_C6, COORD_NO}}, // 30
    {9, {COORD_H8, COORD_G8, COORD_F8, COORD_H7, COORD_G7, COORD_F7, COORD_H6, COORD_G6, COORD_F6, COORD_NO}}, // 31

    // 8 edge + 2x
    {10, {COORD_B2, COORD_A1, COORD_B1, COORD_C1, COORD_D1, COORD_E1, COORD_F1, COORD_G1, COORD_H1, COORD_G2}}, // 32
    {10, {COORD_B2, COORD_A1, COORD_A2, COORD_A3, COORD_A4, COORD_A5, COORD_A6, COORD_A7, COORD_A8, COORD_B7}}, // 33
    {10, {COORD_B7, COORD_A8, COORD_B8, COORD_C8, COORD_D8, COORD_E8, COORD_F8, COORD_G8, COORD_H8, COORD_G7}}, // 34
    {10, {COORD_G2, COORD_H1, COORD_H2, COORD_H3, COORD_H4, COORD_H5, COORD_H6, COORD_H7, COORD_H8, COORD_G7}}, // 35

    // 9 triangle
    {10, {COORD_A1, COORD_B1, COORD_C1, COORD_D1, COORD_A2, COORD_B2, COORD_C2, COORD_A3, COORD_B3, COORD_A4}}, // 36
    {10, {COORD_H1, COORD_G1, COORD_F1, COORD_E1, COORD_H2, COORD_G2, COORD_F2, COORD_H3, COORD_G3, COORD_H4}}, // 37
    {10, {COORD_A8, COORD_B8, COORD_C8, COORD_D8, COORD_A7, COORD_B7, COORD_C7, COORD_A6, COORD_B6, COORD_A5}}, // 38
    {10, {COORD_H8, COORD_G8, COORD_F8, COORD_E8, COORD_H7, COORD_G7, COORD_F7, COORD_H6, COORD_G6, COORD_H5}}, // 39

    // 10 corner + block
    {10, {COORD_A1, COORD_C1, COORD_D1, COORD_E1, COORD_F1, COORD_H1, COORD_C2, COORD_D2, COORD_E2, COORD_F2}}, // 40
    {10, {COORD_A1, COORD_A3, COORD_A4, COORD_A5, COORD_A6, COORD_A8, COORD_B3, COORD_B4, COORD_B5, COORD_B6}}, // 41
    {10, {COORD_A8, COORD_C8, COORD_D8, COORD_E8, COORD_F8, COORD_H8, COORD_C7, COORD_D7, COORD_E7, COORD_F7}}, // 42
    {10, {COORD_H1, COORD_H3, COORD_H4, COORD_H5, COORD_H6, COORD_H8, COORD_G3, COORD_G4, COORD_G5, COORD_G6}}, // 43

    // 11 cross
    {10, {COORD_A1, COORD_B2, COORD_C3, COORD_D4, COORD_B1, COORD_C2, COORD_D3, COORD_A2, COORD_B3, COORD_C4}}, // 44
    {10, {COORD_H1, COORD_G2, COORD_F3, COORD_E4, COORD_G1, COORD_F2, COORD_E3, COORD_H2, COORD_G3, COORD_F4}}, // 45
    {10, {COORD_A8, COORD_B7, COORD_C6, COORD_D5, COORD_B8, COORD_C7, COORD_D6, COORD_A7, COORD_B6, COORD_C5}}, // 46
    {10, {COORD_H8, COORD_G7, COORD_F6, COORD_E5, COORD_G8, COORD_F7, COORD_E6, COORD_H7, COORD_G6, COORD_F5}}, // 47

    // 12 edge + y
    {10, {COORD_C2, COORD_A1, COORD_B1, COORD_C1, COORD_D1, COORD_E1, COORD_F1, COORD_G1, COORD_H1, COORD_F2}}, // 48
    {10, {COORD_B3, COORD_A1, COORD_A2, COORD_A3, COORD_A4, COORD_A5, COORD_A6, COORD_A7, COORD_A8, COORD_B6}}, // 49
    {10, {COORD_C7, COORD_A8, COORD_B8, COORD_C8, COORD_D8, COORD_E8, COORD_F8, COORD_G8, COORD_H8, COORD_F7}}, // 50
    {10, {COORD_G3, COORD_H1, COORD_H2, COORD_H3, COORD_H4, COORD_H5, COORD_H6, COORD_H7, COORD_H8, COORD_G6}}, // 51

    // 13 narrow triangle
    {10, {COORD_A1, COORD_B1, COORD_C1, COORD_D1, COORD_E1, COORD_A2, COORD_B2, COORD_A3, COORD_A4, COORD_A5}}, // 52
    {10, {COORD_H1, COORD_G1, COORD_F1, COORD_E1, COORD_D1, COORD_H2, COORD_G2, COORD_H3, COORD_H4, COORD_H5}}, // 53
    {10, {COORD_A8, COORD_B8, COORD_C8, COORD_D8, COORD_E8, COORD_A7, COORD_B7, COORD_A6, COORD_A5, COORD_A4}}, // 54
    {10, {COORD_H8, COORD_G8, COORD_F8, COORD_E8, COORD_D8, COORD_H7, COORD_G7, COORD_H6, COORD_H5, COORD_H4}}, // 55

    // 14 fish
    {10, {COORD_A1, COORD_B1, COORD_A2, COORD_B2, COORD_C2, COORD_D2, COORD_B3, COORD_C3, COORD_B4, COORD_D4}}, // 56
    {10, {COORD_H1, COORD_G1, COORD_H2, COORD_G2, COORD_F2, COORD_E2, COORD_G3, COORD_F3, COORD_G4, COORD_E4}}, // 57
    {10, {COORD_A8, COORD_B8, COORD_A7, COORD_B7, COORD_C7, COORD_D7, COORD_B6, COORD_C6, COORD_B5, COORD_D5}}, // 58
    {10, {COORD_H8, COORD_G8, COORD_H7, COORD_G7, COORD_F7, COORD_E7, COORD_G6, COORD_F6, COORD_G5, COORD_E5}}, // 59

    // 15 kite
    {10, {COORD_A1, COORD_B1, COORD_A2, COORD_B2, COORD_C2, COORD_D2, COORD_E2, COORD_B3, COORD_B4, COORD_B5}}, // 60
    {10, {COORD_H1, COORD_G1, COORD_H2, COORD_G2, COORD_F2, COORD_E2, COORD_D2, COORD_G3, COORD_G4, COORD_G5}}, // 61
    {10, {COORD_A8, COORD_B8, COORD_A7, COORD_B7, COORD_C7, COORD_D7, COORD_E7, COORD_B6, COORD_B5, COORD_B4}}, // 62
    {10, {COORD_H8, COORD_G8, COORD_H7, COORD_G7, COORD_F7, COORD_E7, COORD_D7, COORD_G6, COORD_G5, COORD_G4}}  // 63
};

constexpr Coord_to_feature coord_to_feature[HW2] = {
    {13, {{24, P30}, {31, P38}, {34, P31}, {35, P31}, {39, P39}, {42, P34}, {43, P34}, {47, P39}, {50, P31}, {51, P31}, {55, P39}, {59, P39}, {63, P39}}}, // COORD_H8
    {10, {{ 3, P30}, {22, P30}, {31, P37}, {34, P32}, {39, P38}, {47, P35}, {50, P32}, {55, P38}, {59, P38}, {63, P38}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_G8
    { 8, {{ 7, P30}, {18, P30}, {31, P36}, {34, P33}, {39, P37}, {42, P35}, {50, P33}, {55, P37}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_F8
    { 8, {{11, P30}, {14, P30}, {34, P34}, {39, P36}, {42, P36}, {50, P34}, {54, P35}, {55, P36}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_E8
    { 8, {{ 9, P30}, {15, P30}, {34, P35}, {38, P36}, {42, P37}, {50, P35}, {54, P36}, {55, P35}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_D8
    { 8, {{ 5, P30}, {19, P30}, {30, P36}, {34, P36}, {38, P37}, {42, P38}, {50, P36}, {54, P37}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_C8
    {10, {{ 1, P30}, {23, P30}, {30, P37}, {34, P37}, {38, P38}, {46, P35}, {50, P37}, {54, P38}, {58, P38}, {62, P38}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_B8
    {13, {{25, P30}, {30, P38}, {33, P31}, {34, P38}, {38, P39}, {41, P34}, {42, P39}, {46, P39}, {49, P31}, {50, P38}, {54, P39}, {58, P39}, {62, P39}}}, // COORD_A8
    {10, {{ 2, P30}, {20, P30}, {31, P35}, {35, P32}, {39, P35}, {47, P32}, {51, P32}, {55, P34}, {59, P37}, {63, P37}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_H7
    {11, {{ 2, P31}, { 3, P31}, {24, P31}, {31, P34}, {34, P30}, {35, P30}, {39, P34}, {47, P38}, {55, P33}, {59, P36}, {63, P36}, { 0, PNO}, { 0, PNO}}}, // COORD_G7
    {10, {{ 2, P32}, { 7, P31}, {22, P31}, {31, P33}, {39, P33}, {42, P30}, {47, P34}, {50, P30}, {59, P35}, {63, P35}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_F7
    { 8, {{ 2, P33}, {11, P31}, {15, P31}, {18, P31}, {42, P31}, {59, P34}, {62, P33}, {63, P34}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_E7
    { 8, {{ 2, P34}, { 9, P31}, {14, P31}, {19, P31}, {42, P32}, {58, P34}, {62, P34}, {63, P33}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_D7
    {10, {{ 2, P35}, { 5, P31}, {23, P31}, {30, P33}, {38, P33}, {42, P33}, {46, P34}, {50, P39}, {58, P35}, {62, P35}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_C7
    {11, {{ 1, P31}, { 2, P36}, {25, P31}, {30, P34}, {33, P30}, {34, P39}, {38, P34}, {46, P38}, {54, P33}, {58, P36}, {62, P36}, { 0, PNO}, { 0, PNO}}}, // COORD_B7
    {10, {{ 2, P37}, {21, P30}, {30, P35}, {33, P32}, {38, P35}, {46, P32}, {49, P32}, {54, P34}, {58, P37}, {62, P37}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_A7
    { 8, {{ 6, P30}, {16, P30}, {31, P32}, {35, P33}, {39, P32}, {43, P35}, {51, P33}, {55, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_H6
    {10, {{ 3, P32}, { 6, P31}, {20, P31}, {31, P31}, {39, P31}, {43, P30}, {47, P31}, {51, P30}, {59, P33}, {63, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_G6
    { 7, {{ 6, P32}, { 7, P32}, {15, P32}, {24, P32}, {31, P30}, {47, P37}, {59, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_F6
    { 5, {{ 6, P33}, {11, P32}, {19, P32}, {22, P32}, {47, P33}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_E6
    { 5, {{ 6, P34}, { 9, P32}, {18, P32}, {23, P32}, {46, P33}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_D6
    { 7, {{ 5, P32}, { 6, P35}, {14, P32}, {25, P32}, {30, P30}, {46, P37}, {58, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_C6
    {10, {{ 1, P32}, { 6, P36}, {21, P31}, {30, P31}, {38, P31}, {41, P30}, {46, P31}, {49, P30}, {58, P33}, {62, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_B6
    { 8, {{ 6, P37}, {17, P30}, {30, P32}, {33, P33}, {38, P32}, {41, P35}, {49, P33}, {54, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_A6
    { 8, {{10, P30}, {12, P30}, {35, P34}, {39, P30}, {43, P36}, {51, P34}, {53, P30}, {55, P31}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_H5
    { 8, {{ 3, P33}, {10, P31}, {15, P33}, {16, P31}, {43, P31}, {59, P31}, {61, P30}, {63, P31}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_G5
    { 5, {{ 7, P33}, {10, P32}, {19, P33}, {20, P32}, {47, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_F5
    { 6, {{10, P33}, {11, P33}, {23, P33}, {24, P33}, {47, P36}, {59, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_E5
    { 6, {{ 9, P33}, {10, P34}, {22, P33}, {25, P33}, {46, P36}, {58, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_D5
    { 5, {{ 5, P33}, {10, P35}, {18, P33}, {21, P32}, {46, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_C5
    { 8, {{ 1, P33}, {10, P36}, {14, P33}, {17, P31}, {41, P31}, {58, P31}, {60, P30}, {62, P31}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_B5
    { 8, {{10, P37}, {13, P30}, {33, P34}, {38, P30}, {41, P36}, {49, P34}, {52, P30}, {54, P31}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_A5
    { 8, {{ 8, P30}, {15, P34}, {35, P35}, {37, P30}, {43, P37}, {51, P35}, {53, P31}, {55, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_H4
    { 8, {{ 3, P34}, { 8, P31}, {12, P31}, {19, P34}, {43, P32}, {57, P31}, {61, P31}, {63, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_G4
    { 5, {{ 7, P34}, { 8, P32}, {16, P32}, {23, P34}, {45, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_F4
    { 6, {{ 8, P33}, {11, P34}, {20, P33}, {25, P34}, {45, P36}, {57, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_E4
    { 6, {{ 8, P34}, { 9, P34}, {21, P33}, {24, P34}, {44, P36}, {56, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_D4
    { 5, {{ 5, P34}, { 8, P35}, {17, P32}, {22, P34}, {44, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_C4
    { 8, {{ 1, P34}, { 8, P36}, {13, P31}, {18, P34}, {41, P32}, {56, P31}, {60, P31}, {62, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_B4
    { 8, {{ 8, P37}, {14, P34}, {33, P35}, {36, P30}, {41, P37}, {49, P35}, {52, P31}, {54, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_A4
    { 8, {{ 4, P30}, {19, P35}, {29, P32}, {35, P36}, {37, P32}, {43, P38}, {51, P36}, {53, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_H3
    {10, {{ 3, P35}, { 4, P31}, {23, P35}, {29, P31}, {37, P31}, {43, P33}, {45, P31}, {51, P39}, {57, P33}, {61, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_G3
    { 7, {{ 4, P32}, { 7, P35}, {12, P32}, {25, P35}, {29, P30}, {45, P37}, {57, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_F3
    { 5, {{ 4, P33}, {11, P35}, {16, P33}, {21, P34}, {45, P33}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_E3
    { 5, {{ 4, P34}, { 9, P35}, {17, P33}, {20, P34}, {44, P33}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_D3
    { 7, {{ 4, P35}, { 5, P35}, {13, P32}, {24, P35}, {28, P30}, {44, P37}, {56, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_C3
    {10, {{ 1, P35}, { 4, P36}, {22, P35}, {28, P31}, {36, P31}, {41, P33}, {44, P31}, {49, P39}, {56, P33}, {60, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_B3
    { 8, {{ 4, P37}, {18, P35}, {28, P32}, {33, P36}, {36, P32}, {41, P38}, {49, P36}, {52, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_A3
    {10, {{ 0, P30}, {23, P36}, {29, P35}, {35, P37}, {37, P35}, {45, P32}, {51, P37}, {53, P34}, {57, P37}, {61, P37}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_H2
    {11, {{ 0, P31}, { 3, P36}, {25, P36}, {29, P34}, {32, P30}, {35, P39}, {37, P34}, {45, P38}, {53, P33}, {57, P36}, {61, P36}, { 0, PNO}, { 0, PNO}}}, // COORD_G2
    {10, {{ 0, P32}, { 7, P36}, {21, P35}, {29, P33}, {37, P33}, {40, P30}, {45, P34}, {48, P30}, {57, P35}, {61, P35}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_F2
    { 8, {{ 0, P33}, {11, P36}, {12, P33}, {17, P34}, {40, P31}, {57, P34}, {60, P33}, {61, P34}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_E2
    { 8, {{ 0, P34}, { 9, P36}, {13, P33}, {16, P34}, {40, P32}, {56, P34}, {60, P34}, {61, P33}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_D2
    {10, {{ 0, P35}, { 5, P36}, {20, P35}, {28, P33}, {36, P33}, {40, P33}, {44, P34}, {48, P39}, {56, P35}, {60, P35}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_C2
    {11, {{ 0, P36}, { 1, P36}, {24, P36}, {28, P34}, {32, P39}, {33, P39}, {36, P34}, {44, P38}, {52, P33}, {56, P36}, {60, P36}, { 0, PNO}, { 0, PNO}}}, // COORD_B2
    {10, {{ 0, P37}, {22, P36}, {28, P35}, {33, P37}, {36, P35}, {44, P32}, {49, P37}, {52, P34}, {56, P37}, {60, P37}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_A2
    {13, {{25, P37}, {29, P38}, {32, P31}, {35, P38}, {37, P39}, {40, P34}, {43, P39}, {45, P39}, {48, P31}, {51, P38}, {53, P39}, {57, P39}, {61, P39}}}, // COORD_H1
    {10, {{ 3, P37}, {21, P36}, {29, P37}, {32, P32}, {37, P38}, {45, P35}, {48, P32}, {53, P38}, {57, P38}, {61, P38}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_G1
    { 8, {{ 7, P37}, {17, P35}, {29, P36}, {32, P33}, {37, P37}, {40, P35}, {48, P33}, {53, P37}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_F1
    { 8, {{11, P37}, {13, P34}, {32, P34}, {37, P36}, {40, P36}, {48, P34}, {52, P35}, {53, P36}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_E1
    { 8, {{ 9, P37}, {12, P34}, {32, P35}, {36, P36}, {40, P37}, {48, P35}, {52, P36}, {53, P35}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_D1
    { 8, {{ 5, P37}, {16, P35}, {28, P36}, {32, P36}, {36, P37}, {40, P38}, {48, P36}, {52, P37}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_C1
    {10, {{ 1, P37}, {20, P36}, {28, P37}, {32, P37}, {36, P38}, {44, P35}, {48, P37}, {52, P38}, {56, P38}, {60, P38}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_B1
    {13, {{24, P37}, {28, P38}, {32, P38}, {33, P38}, {36, P39}, {40, P39}, {41, P39}, {44, P39}, {48, P38}, {49, P38}, {52, P39}, {56, P39}, {60, P39}}}  // COORD_A1
};

/*
    @brief constants used for evaluation function with SIMD
*/
__m256i eval_lower_mask;
__m256i feature_to_coord_simd_mul[N_SIMD_EVAL_FEATURES][MAX_PATTERN_CELLS - 1];
__m256i feature_to_coord_simd_cell[N_SIMD_EVAL_FEATURES][MAX_PATTERN_CELLS][2];
__m256i coord_to_feature_simd[HW2][N_SIMD_EVAL_FEATURES];
__m256i coord_to_feature_simd2[HW2][N_SIMD_EVAL_FEATURES];
__m256i eval_simd_offsets_bef[N_SIMD_EVAL_FEATURES_SIMPLE]; // 16bit * 16
__m256i eval_simd_offsets_aft[N_SIMD_EVAL_FEATURES_COMP * 2]; // 32bit * 8
__m256i eval_surround_mask;
__m128i eval_surround_shift1879;

/*
    @brief evaluation parameters
*/
int16_t pattern_arr[2][N_PHASES][N_PATTERN_PARAMS];
int16_t eval_num_arr[N_PHASES][MAX_STONE_NUM];
int16_t eval_sur0_sur1_arr[N_PHASES][MAX_SURROUND][MAX_SURROUND];

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
void init_pattern_arr_rev(int phase_idx, int siz, int strt){
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
    constexpr int pattern_sizes[N_PATTERNS] = {8, 8, 8, 5, 6, 7, 8, 9, 10, 10, 10, 10, 10, 10, 10, 10};
    constexpr int pattern_starts[N_PATTERNS] = {
        1, 6562, 13123, 19684, // features[0]
        19927, 20656, 22843, /*dummy 29404*/ 29405, // features[1]
        49088, 108137, 167186, 226235, // features[2]
        285284, 344333, 403382, 462431  // features[3]
    };
    for (phase_idx = 0; phase_idx < N_PHASES; ++phase_idx){
        pattern_arr[0][phase_idx][0] = 0; // memory bound
        pattern_arr[1][phase_idx][0] = 0; // memory bound
        if (fread(pattern_arr[0][phase_idx] + pattern_starts[0], 2, N_PATTERN_PARAMS_BEFORE_DUMMY, fp) < N_PATTERN_PARAMS_BEFORE_DUMMY){
            std::cerr << "[ERROR] [FATAL] evaluation file broken" << std::endl;
            fclose(fp);
            return false;
        }
        pattern_arr[0][phase_idx][SIMD_EVAL_DUMMY_ADDR] = 0; // dummy
        pattern_arr[1][phase_idx][SIMD_EVAL_DUMMY_ADDR] = 0; // dummy
        if (fread(pattern_arr[0][phase_idx] + pattern_starts[7], 2, N_PATTERN_PARAMS_AFTER_DUMMY, fp) < N_PATTERN_PARAMS_AFTER_DUMMY){
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
    int i, j, k, idx, cell;
    eval_lower_mask = _mm256_set1_epi32(0x0000FFFF);
    for (phase_idx = 0; phase_idx < N_PHASES; ++phase_idx){
        for (i = 1; i < N_PATTERN_PARAMS; ++i){
            if (i == SIMD_EVAL_DUMMY_ADDR) // dummy
                continue;
            if (pattern_arr[0][phase_idx][i] < -SIMD_EVAL_OFFSET){
                std::cerr << "[ERROR] evaluation value too low. you can ignore this error. phase " << phase_idx << " index " << i << " found " << pattern_arr[0][phase_idx][i] << std::endl;
                pattern_arr[0][phase_idx][i] = -SIMD_EVAL_OFFSET;
            }
            if (pattern_arr[0][phase_idx][i] > SIMD_EVAL_OFFSET){
                std::cerr << "[ERROR] evaluation value too high. you can ignore this error. phase " << phase_idx << " index " << i << " found " << pattern_arr[0][phase_idx][i] << std::endl;
                pattern_arr[0][phase_idx][i] = SIMD_EVAL_OFFSET;
            }
            pattern_arr[0][phase_idx][i] += SIMD_EVAL_OFFSET;
        }
    }
    if (thread_pool.size() >= 2){
        std::future<void> tasks[N_PHASES * N_PATTERNS];
        int i = 0;
        for (phase_idx = 0; phase_idx < N_PHASES; ++phase_idx){
            for (pattern_idx = 0; pattern_idx < N_PATTERNS; ++pattern_idx){
                bool pushed = false;
                while (!pushed)
                    tasks[i] = thread_pool.push(&pushed, std::bind(init_pattern_arr_rev, phase_idx, pattern_sizes[pattern_idx], pattern_starts[pattern_idx]));
                ++i;
            }
        }
        for (std::future<void> &task: tasks)
            task.get();
    } else{
        for (phase_idx = 0; phase_idx < N_PHASES; ++phase_idx){
            for (pattern_idx = 0; pattern_idx < N_PATTERNS; ++pattern_idx)
                init_pattern_arr_rev(phase_idx, pattern_sizes[pattern_idx], pattern_starts[pattern_idx]);
        }
    }
    int16_t f2c[16];
    for (i = 0; i < N_SIMD_EVAL_FEATURES; ++i){
        for (j = 0; j < MAX_PATTERN_CELLS - 1; ++j){
            for (k = 0; k < 16; ++k)
                f2c[k] = (j < feature_to_coord[i * 16 + k].n_cells - 1) ? 3 : 1;
            feature_to_coord_simd_mul[i][j] = _mm256_set_epi16(f2c[0], f2c[1], f2c[2], f2c[3], f2c[4], f2c[5], f2c[6], f2c[7], f2c[8], f2c[9], f2c[10], f2c[11], f2c[12], f2c[13], f2c[14], f2c[15]);
        }
    }
    int32_t f2c32[8];
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
            coord_to_feature_simd[cell][i] = _mm256_set_epi16(
                c2f[idx], c2f[idx + 1], c2f[idx + 2], c2f[idx + 3], c2f[idx + 4], c2f[idx + 5], c2f[idx + 6], c2f[idx + 7], 
                c2f[idx + 8], c2f[idx + 9], c2f[idx + 10], c2f[idx + 11], c2f[idx + 12], c2f[idx + 13], c2f[idx + 14], c2f[idx + 15]
            );
            coord_to_feature_simd2[cell][i] = _mm256_slli_epi16(coord_to_feature_simd[cell][i], 1);
        }
    }
    eval_simd_offsets_bef[0] = _mm256_set_epi16(
        (int16_t)pattern_starts[0], (int16_t)pattern_starts[0], (int16_t)pattern_starts[0], (int16_t)pattern_starts[0], (int16_t)pattern_starts[1], (int16_t)pattern_starts[1], (int16_t)pattern_starts[1], (int16_t)pattern_starts[1], 
        (int16_t)pattern_starts[2], (int16_t)pattern_starts[2], (int16_t)pattern_starts[2], (int16_t)pattern_starts[2], (int16_t)pattern_starts[3], (int16_t)pattern_starts[3], (int16_t)pattern_starts[3], (int16_t)pattern_starts[3]
    );
    eval_simd_offsets_bef[1] = _mm256_set_epi16(
        (int16_t)pattern_starts[4], (int16_t)pattern_starts[4], (int16_t)pattern_starts[4], (int16_t)pattern_starts[4], (int16_t)pattern_starts[5], (int16_t)pattern_starts[5], (int16_t)pattern_starts[5], (int16_t)pattern_starts[5], 
        (int16_t)pattern_starts[6], (int16_t)pattern_starts[6],       SIMD_EVAL_DUMMY_ADDR,       SIMD_EVAL_DUMMY_ADDR, (int16_t)pattern_starts[7], (int16_t)pattern_starts[7], (int16_t)pattern_starts[7], (int16_t)pattern_starts[7]
    );
    eval_simd_offsets_bef[1] = _mm256_sub_epi16(eval_simd_offsets_bef[1], _mm256_set1_epi16(SIMD_EVAL_F1_OFFSET));
    int i4;
    for (i = 0; i < N_SIMD_EVAL_FEATURES_COMP; ++i){
        i4 = i * 4;
        eval_simd_offsets_aft[i * 2] = _mm256_set_epi32(
            pattern_starts[10 + i4], pattern_starts[10 + i4], pattern_starts[10 + i4], pattern_starts[10 + i4], 
            pattern_starts[11 + i4], pattern_starts[11 + i4], pattern_starts[11 + i4], pattern_starts[11 + i4]
        );
        eval_simd_offsets_aft[i * 2 + 1] = _mm256_set_epi32(
            pattern_starts[8 + i4], pattern_starts[8 + i4], pattern_starts[8 + i4], pattern_starts[8 + i4], 
            pattern_starts[9 + i4], pattern_starts[9 + i4], pattern_starts[9 + i4], pattern_starts[9 + i4]
        );
    }
    eval_surround_mask = _mm256_set_epi64x(0x7E7E7E7E7E7E7E7EULL, 0x00FFFFFFFFFFFF00ULL, 0x007E7E7E7E7E7E00ULL, 0x007E7E7E7E7E7E00ULL);
    eval_surround_shift1879 = _mm_set_epi32(1, HW, HW_M1, HW_P1);
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

/*
    @brief pattern evaluation

    @param phase_idx            evaluation phase
    @param search               search information
    @return pattern evaluation value
*/
inline __m256i calc_idx8_aft(const __m128i feature, const int i){
    return _mm256_add_epi32(_mm256_cvtepu16_epi32(feature), eval_simd_offsets_aft[i]);
}

inline __m256i gather_eval(const int *pat_com, const __m256i idx8){
    #if SIMD_EVAL_OFFSET * 16 < 65535  // HACK: if (SIMD_EVAL_OFFSET * 2) * 8 < 2 ^ 16, AND is unnecessary
        return _mm256_i32gather_epi32(pat_com, idx8, 2);
    #else
        return _mm256_and_si256(_mm256_i32gather_epi32(pat_com, idx8, 2), eval_lower_mask);
    #endif
}

inline int calc_pattern_diff(const int phase_idx, bool reversed, Eval_features *features){
    const int *pat_com = (int*)pattern_arr[reversed][phase_idx];
    const int *pat_com1 = (int*)(&pattern_arr[reversed][phase_idx][SIMD_EVAL_F1_OFFSET]);
    __m256i res256 =                  gather_eval(pat_com, _mm256_cvtepu16_epi32(features->f128[0]));
    res256 = _mm256_add_epi32(res256, gather_eval(pat_com, _mm256_cvtepu16_epi32(features->f128[1])));
    res256 = _mm256_add_epi32(res256, gather_eval(pat_com1, _mm256_cvtepu16_epi32(features->f128[2])));
    res256 = _mm256_add_epi32(res256, gather_eval(pat_com1, _mm256_cvtepu16_epi32(features->f128[3])));
    res256 = _mm256_add_epi32(res256, gather_eval(pat_com, calc_idx8_aft(features->f128[4], 0)));
    res256 = _mm256_add_epi32(res256, gather_eval(pat_com, calc_idx8_aft(features->f128[5], 1)));
    res256 = _mm256_add_epi32(res256, gather_eval(pat_com, calc_idx8_aft(features->f128[6], 2)));
    res256 = _mm256_add_epi32(res256, gather_eval(pat_com, calc_idx8_aft(features->f128[7], 3)));
    #if SIMD_EVAL_OFFSET * 16 < 65535
        res256 = _mm256_and_si256(res256, eval_lower_mask);
    #endif
    __m128i res128 = _mm_add_epi32(_mm256_castsi256_si128(res256), _mm256_extracti128_si256(res256, 1));
    res128 = _mm_hadd_epi32(res128, res128);
    return _mm_cvtsi128_si32(res128) + _mm_extract_epi32(res128, 1) - SIMD_EVAL_OFFSET * N_SYMMETRY_PATTERNS;
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
    int phase_idx, sur0, sur1, num0;
    uint64_t empties;
    phase_idx = search.phase();
    empties = ~(search.board.player | search.board.opponent);
    sur0 = calc_surround(search.board.player, empties);
    sur1 = calc_surround(search.board.opponent, empties);
    num0 = pop_count_ull(search.board.player);
    int res = calc_pattern_diff(phase_idx, search.eval_feature_reversed, &search.eval_features[search.eval_feature_idx]) + 
        eval_num_arr[phase_idx][num0] + 
        eval_sur0_sur1_arr[phase_idx][sur0][sur1];
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
    int phase_idx, sur0, sur1, num0;
    uint64_t empties;
    phase_idx = search->phase();
    empties = ~(search->board.player | search->board.opponent);
    sur0 = calc_surround(search->board.player, empties);
    sur1 = calc_surround(search->board.opponent, empties);
    num0 = pop_count_ull(search->board.player);
    int res = calc_pattern_diff(phase_idx, search->eval_feature_reversed, &search->eval_features[search->eval_feature_idx]) + 
        eval_num_arr[phase_idx][num0] + 
        eval_sur0_sur1_arr[phase_idx][sur0][sur1];
    res += res >= 0 ? STEP_2 : -STEP_2;
    res /= STEP;
    if (res > SCORE_MAX)
        return SCORE_MAX;
    if (res < -SCORE_MAX)
        return -SCORE_MAX;
    return res;
}

inline void calc_feature_vector(__m256i &f, const int *b_arr_int, const int i, const int n){
    f = _mm256_set1_epi16(0);
    for (int j = 0; j < n; ++j){
        f = _mm256_adds_epu16(f, _mm256_i32gather_epi32(b_arr_int, feature_to_coord_simd_cell[i][j][0], 4));
        f = _mm256_adds_epu16(f, _mm256_slli_epi32(_mm256_i32gather_epi32(b_arr_int, feature_to_coord_simd_cell[i][j][1], 4), 16));
        f = _mm256_mullo_epi16(f, feature_to_coord_simd_mul[i][j]);
    }
    f = _mm256_adds_epu16(f, _mm256_i32gather_epi32(b_arr_int, feature_to_coord_simd_cell[i][n][0], 4));
    f = _mm256_adds_epu16(f, _mm256_slli_epi32(_mm256_i32gather_epi32(b_arr_int, feature_to_coord_simd_cell[i][n][1], 4), 16));
}

/*
    @brief calculate features for pattern evaluation

    @param search               search information
*/
inline void calc_features(Search *search){
    int b_arr_int[HW2 + 1];
    search->board.translate_to_arr_player_rev(b_arr_int);
    b_arr_int[COORD_NO] = 0;
    calc_feature_vector(search->eval_features[0].f256[0], b_arr_int, 0, 7);
    calc_feature_vector(search->eval_features[0].f256[1], b_arr_int, 1, 8);
    calc_feature_vector(search->eval_features[0].f256[2], b_arr_int, 2, 9);
    calc_feature_vector(search->eval_features[0].f256[3], b_arr_int, 3, 9);
    search->eval_features[0].f256[0] = _mm256_adds_epu16(search->eval_features[0].f256[0], eval_simd_offsets_bef[0]);
    search->eval_features[0].f256[1] = _mm256_adds_epu16(search->eval_features[0].f256[1], eval_simd_offsets_bef[1]);
    search->eval_feature_reversed = false;
    search->eval_feature_idx = 0;
}

/*
    @brief move evaluation features

    @param search               search information
    @param flip                 flip information
*/
inline void eval_move(Search *search, const Flip *flip){
    uint_fast8_t cell;
    uint64_t f;
    __m256i f0, f1, f2, f3;
    f0 = search->eval_features[search->eval_feature_idx].f256[0];
    f1 = search->eval_features[search->eval_feature_idx].f256[1];
    f2 = search->eval_features[search->eval_feature_idx].f256[2];
    f3 = search->eval_features[search->eval_feature_idx].f256[3];
    if (search->eval_feature_reversed){
        f0 = _mm256_subs_epu16(f0, coord_to_feature_simd[flip->pos][0]);
        f1 = _mm256_subs_epu16(f1, coord_to_feature_simd[flip->pos][1]);
        f2 = _mm256_subs_epu16(f2, coord_to_feature_simd[flip->pos][2]);
        f3 = _mm256_subs_epu16(f3, coord_to_feature_simd[flip->pos][3]);
        f = flip->flip;
        for (cell = first_bit(&f); f; cell = next_bit(&f)){
            f0 = _mm256_adds_epu16(f0, coord_to_feature_simd[cell][0]);
            f1 = _mm256_adds_epu16(f1, coord_to_feature_simd[cell][1]);
            f2 = _mm256_adds_epu16(f2, coord_to_feature_simd[cell][2]);
            f3 = _mm256_adds_epu16(f3, coord_to_feature_simd[cell][3]);
        }
    } else{
        f0 = _mm256_subs_epu16(f0, coord_to_feature_simd2[flip->pos][0]);
        f1 = _mm256_subs_epu16(f1, coord_to_feature_simd2[flip->pos][1]);
        f2 = _mm256_subs_epu16(f2, coord_to_feature_simd2[flip->pos][2]);
        f3 = _mm256_subs_epu16(f3, coord_to_feature_simd2[flip->pos][3]);
        f = flip->flip;
        for (cell = first_bit(&f); f; cell = next_bit(&f)){
            f0 = _mm256_subs_epu16(f0, coord_to_feature_simd[cell][0]);
            f1 = _mm256_subs_epu16(f1, coord_to_feature_simd[cell][1]);
            f2 = _mm256_subs_epu16(f2, coord_to_feature_simd[cell][2]);
            f3 = _mm256_subs_epu16(f3, coord_to_feature_simd[cell][3]);
        }
    }
    ++search->eval_feature_idx;
    search->eval_features[search->eval_feature_idx].f256[0] = f0;
    search->eval_features[search->eval_feature_idx].f256[1] = f1;
    search->eval_features[search->eval_feature_idx].f256[2] = f2;
    search->eval_features[search->eval_feature_idx].f256[3] = f3;
    search->eval_feature_reversed ^= 1;
}

/*
    @brief undo evaluation features

    @param search               search information
    @param flip                 flip information
*/
inline void eval_undo(Search *search, const Flip *flip){
    search->eval_feature_reversed ^= 1;
    --search->eval_feature_idx;
    //search->eval_features.f256[0] = search->eval_features_memo[search->eval_feature_idx].f256[0];
    //search->eval_features.f256[1] = search->eval_features_memo[search->eval_feature_idx].f256[1];
    //search->eval_features.f256[2] = search->eval_features_memo[search->eval_feature_idx].f256[2];
    //search->eval_features.f256[3] = search->eval_features_memo[search->eval_feature_idx].f256[3];
    /*
    uint_fast8_t cell;
    uint64_t f;
    if (search->eval_feature_reversed){
        search->eval_features.f256[0] = _mm256_adds_epu16(search->eval_features.f256[0], coord_to_feature_simd[flip->pos][0]);
        search->eval_features.f256[1] = _mm256_adds_epu16(search->eval_features.f256[1], coord_to_feature_simd[flip->pos][1]);
        search->eval_features.f256[2] = _mm256_adds_epu16(search->eval_features.f256[2], coord_to_feature_simd[flip->pos][2]);
        search->eval_features.f256[3] = _mm256_adds_epu16(search->eval_features.f256[3], coord_to_feature_simd[flip->pos][3]);
        f = flip->flip;
        for (cell = first_bit(&f); f; cell = next_bit(&f)){
            search->eval_features.f256[0] = _mm256_subs_epu16(search->eval_features.f256[0], coord_to_feature_simd[cell][0]);
            search->eval_features.f256[1] = _mm256_subs_epu16(search->eval_features.f256[1], coord_to_feature_simd[cell][1]);
            search->eval_features.f256[2] = _mm256_subs_epu16(search->eval_features.f256[2], coord_to_feature_simd[cell][2]);
            search->eval_features.f256[3] = _mm256_subs_epu16(search->eval_features.f256[3], coord_to_feature_simd[cell][3]);
        }
    } else{
        search->eval_features.f256[0] = _mm256_adds_epu16(search->eval_features.f256[0], coord_to_feature_simd2[flip->pos][0]);
        search->eval_features.f256[1] = _mm256_adds_epu16(search->eval_features.f256[1], coord_to_feature_simd2[flip->pos][1]);
        search->eval_features.f256[2] = _mm256_adds_epu16(search->eval_features.f256[2], coord_to_feature_simd2[flip->pos][2]);
        search->eval_features.f256[3] = _mm256_adds_epu16(search->eval_features.f256[3], coord_to_feature_simd2[flip->pos][3]);
        f = flip->flip;
        for (cell = first_bit(&f); f; cell = next_bit(&f)){
            search->eval_features.f256[0] = _mm256_adds_epu16(search->eval_features.f256[0], coord_to_feature_simd[cell][0]);
            search->eval_features.f256[1] = _mm256_adds_epu16(search->eval_features.f256[1], coord_to_feature_simd[cell][1]);
            search->eval_features.f256[2] = _mm256_adds_epu16(search->eval_features.f256[2], coord_to_feature_simd[cell][2]);
            search->eval_features.f256[3] = _mm256_adds_epu16(search->eval_features.f256[3], coord_to_feature_simd[cell][3]);
        }
    }
    */
}
