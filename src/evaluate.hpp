#pragma once
#include <iostream>
#include <fstream>
#include "setting.hpp"
#include "common.hpp"
#include "board.hpp"
#include "search.hpp"
#include "util.hpp"

using namespace std;

#define N_PATTERNS 18
#define N_JOINED_PATTERNS 20
#ifndef N_SYMMETRY_PATTERNS
    #define N_SYMMETRY_PATTERNS 70
#endif
#define MAX_PATTERN_CELLS 10
#define MAX_CELL_PATTERNS 18
#define MAX_SURROUND 100
#define MAX_CANPUT 50
#define MAX_STONE_NUM 65
#define MAX_EVALUATE_IDX 59049

#define STEP 256
#define STEP_2 128
#define STEP_SHIFT 8

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
    // 0 hv2
    {8,  {COORD_A2, COORD_B2, COORD_C2, COORD_D2, COORD_E2, COORD_F2, COORD_G2, COORD_H2, COORD_NO, COORD_NO}}, // 0
    {8,  {COORD_B1, COORD_B2, COORD_B3, COORD_B4, COORD_B5, COORD_B6, COORD_B7, COORD_B8, COORD_NO, COORD_NO}}, // 1
    {8,  {COORD_A7, COORD_B7, COORD_C7, COORD_D7, COORD_E7, COORD_F7, COORD_G7, COORD_H7, COORD_NO, COORD_NO}}, // 2
    {8,  {COORD_G1, COORD_G2, COORD_G3, COORD_G4, COORD_G5, COORD_G6, COORD_G7, COORD_G8, COORD_NO, COORD_NO}}, // 3

    // 1 hv3
    {8,  {COORD_A3, COORD_B3, COORD_C3, COORD_D3, COORD_E3, COORD_F3, COORD_G3, COORD_H3, COORD_NO, COORD_NO}}, // 4
    {8,  {COORD_C1, COORD_C2, COORD_C3, COORD_C4, COORD_C5, COORD_C6, COORD_C7, COORD_C8, COORD_NO, COORD_NO}}, // 5
    {8,  {COORD_A6, COORD_B6, COORD_C6, COORD_D6, COORD_E6, COORD_F6, COORD_G6, COORD_H6, COORD_NO, COORD_NO}}, // 6
    {8,  {COORD_F1, COORD_F2, COORD_F3, COORD_F4, COORD_F5, COORD_F6, COORD_F7, COORD_F8, COORD_NO, COORD_NO}}, // 7

    // 2 hv4
    {8,  {COORD_A4, COORD_B4, COORD_C4, COORD_D4, COORD_E4, COORD_F4, COORD_G4, COORD_H4, COORD_NO, COORD_NO}}, // 8
    {8,  {COORD_D1, COORD_D2, COORD_D3, COORD_D4, COORD_D5, COORD_D6, COORD_D7, COORD_D8, COORD_NO, COORD_NO}}, // 9
    {8,  {COORD_A5, COORD_B5, COORD_C5, COORD_D5, COORD_E5, COORD_F5, COORD_G5, COORD_H5, COORD_NO, COORD_NO}}, // 10
    {8,  {COORD_E1, COORD_E2, COORD_E3, COORD_E4, COORD_E5, COORD_E6, COORD_E7, COORD_E8, COORD_NO, COORD_NO}}, // 11

    // 3 d5
    {5,  {COORD_D1, COORD_E2, COORD_F3, COORD_G4, COORD_H5, COORD_NO, COORD_NO, COORD_NO, COORD_NO, COORD_NO}}, // 12
    {5,  {COORD_E1, COORD_D2, COORD_C3, COORD_B4, COORD_A5, COORD_NO, COORD_NO, COORD_NO, COORD_NO, COORD_NO}}, // 13
    {5,  {COORD_A4, COORD_B5, COORD_C6, COORD_D7, COORD_E8, COORD_NO, COORD_NO, COORD_NO, COORD_NO, COORD_NO}}, // 14
    {5,  {COORD_H4, COORD_G5, COORD_F6, COORD_E7, COORD_D8, COORD_NO, COORD_NO, COORD_NO, COORD_NO, COORD_NO}}, // 15

    // 4 d6
    {6,  {COORD_C1, COORD_D2, COORD_E3, COORD_F4, COORD_G5, COORD_H6, COORD_NO, COORD_NO, COORD_NO, COORD_NO}}, // 16
    {6,  {COORD_F1, COORD_E2, COORD_D3, COORD_C4, COORD_B5, COORD_A6, COORD_NO, COORD_NO, COORD_NO, COORD_NO}}, // 17
    {6,  {COORD_A3, COORD_B4, COORD_C5, COORD_D6, COORD_E7, COORD_F8, COORD_NO, COORD_NO, COORD_NO, COORD_NO}}, // 18
    {6,  {COORD_H3, COORD_G4, COORD_F5, COORD_E6, COORD_D7, COORD_C8, COORD_NO, COORD_NO, COORD_NO, COORD_NO}}, // 19

    // 5 d7
    {7,  {COORD_B1, COORD_C2, COORD_D3, COORD_E4, COORD_F5, COORD_G6, COORD_H7, COORD_NO, COORD_NO, COORD_NO}}, // 20
    {7,  {COORD_G1, COORD_F2, COORD_E3, COORD_D4, COORD_C5, COORD_B6, COORD_A7, COORD_NO, COORD_NO, COORD_NO}}, // 21
    {7,  {COORD_A2, COORD_B3, COORD_C4, COORD_D5, COORD_E6, COORD_F7, COORD_G8, COORD_NO, COORD_NO, COORD_NO}}, // 22
    {7,  {COORD_H2, COORD_G3, COORD_F4, COORD_E5, COORD_D6, COORD_C7, COORD_B8, COORD_NO, COORD_NO, COORD_NO}}, // 23

    // 6 d8
    {8,  {COORD_A1, COORD_B2, COORD_C3, COORD_D4, COORD_E5, COORD_F6, COORD_G7, COORD_H8, COORD_NO, COORD_NO}}, // 24
    {8,  {COORD_H1, COORD_G2, COORD_F3, COORD_E4, COORD_D5, COORD_C6, COORD_B7, COORD_A8, COORD_NO, COORD_NO}}, // 25

    // 7 edge + 2x
    {10, {COORD_B2, COORD_A1, COORD_B1, COORD_C1, COORD_D1, COORD_E1, COORD_F1, COORD_G1, COORD_H1, COORD_G2}}, // 26
    {10, {COORD_B2, COORD_A1, COORD_A2, COORD_A3, COORD_A4, COORD_A5, COORD_A6, COORD_A7, COORD_A8, COORD_B7}}, // 27
    {10, {COORD_B7, COORD_A8, COORD_B8, COORD_C8, COORD_D8, COORD_E8, COORD_F8, COORD_G8, COORD_H8, COORD_G7}}, // 28
    {10, {COORD_G2, COORD_H1, COORD_H2, COORD_H3, COORD_H4, COORD_H5, COORD_H6, COORD_H7, COORD_H8, COORD_G7}}, // 29

    // 8 corner + block
    {10, {COORD_A1, COORD_C1, COORD_D1, COORD_E1, COORD_F1, COORD_H1, COORD_C2, COORD_D2, COORD_E2, COORD_F2}}, // 30
    {10, {COORD_A1, COORD_A3, COORD_A4, COORD_A5, COORD_A6, COORD_A8, COORD_B3, COORD_B4, COORD_B5, COORD_B6}}, // 31
    {10, {COORD_A8, COORD_C8, COORD_D8, COORD_E8, COORD_F8, COORD_H8, COORD_C7, COORD_D7, COORD_E7, COORD_F7}}, // 32
    {10, {COORD_H1, COORD_H3, COORD_H4, COORD_H5, COORD_H6, COORD_H8, COORD_G3, COORD_G4, COORD_G5, COORD_G6}}, // 33

    // 9 corner10
    {10, {COORD_A1, COORD_B1, COORD_C1, COORD_A2, COORD_B2, COORD_C2, COORD_A3, COORD_B3, COORD_C3, COORD_D4}}, // 34
    {10, {COORD_H1, COORD_G1, COORD_F1, COORD_H2, COORD_G2, COORD_F2, COORD_H3, COORD_G3, COORD_F3, COORD_E4}}, // 35
    {10, {COORD_A8, COORD_B8, COORD_C8, COORD_A7, COORD_B7, COORD_C7, COORD_A6, COORD_B6, COORD_C6, COORD_D5}}, // 36
    {10, {COORD_H8, COORD_G8, COORD_F8, COORD_H7, COORD_G7, COORD_F7, COORD_H6, COORD_G6, COORD_F6, COORD_E5}}, // 37

    // 10 cross
    {10, {COORD_A1, COORD_B2, COORD_C3, COORD_D4, COORD_B1, COORD_C2, COORD_D3, COORD_A2, COORD_B3, COORD_C4}}, // 38
    {10, {COORD_H1, COORD_G2, COORD_F3, COORD_E4, COORD_G1, COORD_F2, COORD_E3, COORD_H2, COORD_G3, COORD_F4}}, // 39
    {10, {COORD_A8, COORD_B7, COORD_C6, COORD_D5, COORD_B8, COORD_C7, COORD_D6, COORD_A7, COORD_B6, COORD_C5}}, // 40
    {10, {COORD_H8, COORD_G7, COORD_F6, COORD_E5, COORD_G8, COORD_F7, COORD_E6, COORD_H7, COORD_G6, COORD_F5}}, // 41

    // 11 kite
    {10, {COORD_A1, COORD_B1, COORD_A2, COORD_B2, COORD_C2, COORD_D2, COORD_E2, COORD_B3, COORD_B4, COORD_B5}}, // 42
    {10, {COORD_H1, COORD_G1, COORD_H2, COORD_G2, COORD_F2, COORD_E2, COORD_D2, COORD_G3, COORD_G4, COORD_G5}}, // 43
    {10, {COORD_A8, COORD_B8, COORD_A7, COORD_B7, COORD_C7, COORD_D7, COORD_E7, COORD_B6, COORD_B5, COORD_B4}}, // 44
    {10, {COORD_H8, COORD_G8, COORD_H7, COORD_G7, COORD_F7, COORD_E7, COORD_D7, COORD_G6, COORD_G5, COORD_G4}}, // 45

    // 12 triangle
    {10, {COORD_A1, COORD_B1, COORD_C1, COORD_D1, COORD_A2, COORD_B2, COORD_C2, COORD_A3, COORD_B3, COORD_A4}}, // 46
    {10, {COORD_H1, COORD_G1, COORD_F1, COORD_E1, COORD_H2, COORD_G2, COORD_F2, COORD_H3, COORD_G3, COORD_H4}}, // 47
    {10, {COORD_A8, COORD_B8, COORD_C8, COORD_D8, COORD_A7, COORD_B7, COORD_C7, COORD_A6, COORD_B6, COORD_A5}}, // 48
    {10, {COORD_H8, COORD_G8, COORD_F8, COORD_E8, COORD_H7, COORD_G7, COORD_F7, COORD_H6, COORD_G6, COORD_H5}}, // 49

    // 13 edge + 2Xa
    {8,  {COORD_A1, COORD_B1, COORD_G1, COORD_H1, COORD_B2, COORD_C2, COORD_F2, COORD_G2, COORD_NO, COORD_NO}}, // 50
    {8,  {COORD_A8, COORD_B8, COORD_G8, COORD_H8, COORD_B7, COORD_C7, COORD_F7, COORD_G7, COORD_NO, COORD_NO}}, // 51
    {8,  {COORD_A1, COORD_A2, COORD_A7, COORD_A8, COORD_B2, COORD_B3, COORD_B6, COORD_B7, COORD_NO, COORD_NO}}, // 52
    {8,  {COORD_H1, COORD_H2, COORD_H7, COORD_H8, COORD_G2, COORD_G3, COORD_G6, COORD_G7, COORD_NO, COORD_NO}}, // 53

    // 14 2edge + X
    {6,  {COORD_A1, COORD_A7, COORD_A8, COORD_B7, COORD_B8, COORD_H8, COORD_NO, COORD_NO, COORD_NO, COORD_NO}}, // 54
    {6,  {COORD_A8, COORD_G8, COORD_H8, COORD_G7, COORD_H7, COORD_H1, COORD_NO, COORD_NO, COORD_NO, COORD_NO}}, // 55
    {6,  {COORD_H8, COORD_H2, COORD_H1, COORD_G2, COORD_G1, COORD_A1, COORD_NO, COORD_NO, COORD_NO, COORD_NO}}, // 56
    {6,  {COORD_H1, COORD_B1, COORD_A1, COORD_B2, COORD_A2, COORD_A8, COORD_NO, COORD_NO, COORD_NO, COORD_NO}}, // 57

    // 15 edge + midedge
    {6,  {COORD_A1, COORD_B1, COORD_B2, COORD_G2, COORD_G1, COORD_H1, COORD_NO, COORD_NO, COORD_NO, COORD_NO}}, // 58
    {6,  {COORD_A8, COORD_B8, COORD_B7, COORD_G7, COORD_G8, COORD_H8, COORD_NO, COORD_NO, COORD_NO, COORD_NO}}, // 59
    {6,  {COORD_A1, COORD_A2, COORD_B2, COORD_B7, COORD_A7, COORD_A8, COORD_NO, COORD_NO, COORD_NO, COORD_NO}}, // 60
    {6,  {COORD_H1, COORD_H2, COORD_G2, COORD_G7, COORD_H7, COORD_H8, COORD_NO, COORD_NO, COORD_NO, COORD_NO}}, // 61

    // 16 2edge + corner
    {6,  {COORD_A1, COORD_B1, COORD_A2, COORD_B2, COORD_C2, COORD_B3, COORD_NO, COORD_NO, COORD_NO, COORD_NO}}, // 62
    {6,  {COORD_A8, COORD_B8, COORD_A7, COORD_B7, COORD_C7, COORD_B6, COORD_NO, COORD_NO, COORD_NO, COORD_NO}}, // 63
    {6,  {COORD_H1, COORD_G1, COORD_H2, COORD_G2, COORD_F2, COORD_G3, COORD_NO, COORD_NO, COORD_NO, COORD_NO}}, // 64
    {6,  {COORD_H8, COORD_G8, COORD_H7, COORD_G7, COORD_F7, COORD_G6, COORD_NO, COORD_NO, COORD_NO, COORD_NO}}, // 65

    // 17 corner + 3line
    {4,  {COORD_A1, COORD_B1, COORD_A2, COORD_B2, COORD_NO, COORD_NO, COORD_NO, COORD_NO, COORD_NO, COORD_NO}}, // 66
    {4,  {COORD_A8, COORD_B8, COORD_A7, COORD_B7, COORD_NO, COORD_NO, COORD_NO, COORD_NO, COORD_NO, COORD_NO}}, // 67
    {4,  {COORD_H1, COORD_G1, COORD_H2, COORD_G2, COORD_NO, COORD_NO, COORD_NO, COORD_NO, COORD_NO, COORD_NO}}, // 68
    {4,  {COORD_H8, COORD_G8, COORD_H7, COORD_G7, COORD_NO, COORD_NO, COORD_NO, COORD_NO, COORD_NO, COORD_NO}}  // 69
};

struct Joined_pattern{
    int n_joined;
    int offset;
    uint64_t mask[3];
};

constexpr Joined_pattern joined_pattern[N_JOINED_PATTERNS] = {
    {1, 8,   {0x3C00000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL}}, // 50
    {1, 8,   {0x000000000000003CULL, 0x0000000000000000ULL, 0x0000000000000000ULL}}, // 51
    {1, 8,   {0x0000808080800000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL}}, // 52
    {1, 8,   {0x0000010101010000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL}}, // 53

    {2, 64,  {0x0080808080800000ULL, 0x000000000000003EULL, 0x0000000000000000ULL}}, // 54
    {2, 64,  {0x000000000000007CULL, 0x0001010101010000ULL, 0x0000000000000000ULL}}, // 55
    {2, 64,  {0x0000010101010100ULL, 0x7C00000000000000ULL, 0x0000000000000000ULL}}, // 56
    {2, 64,  {0x3E00000000000000ULL, 0x0000808080808000ULL, 0x0000000000000000ULL}}, // 57

    {2, 64,  {0x003C000000000000ULL, 0x3C00000000000000ULL, 0x0000000000000000ULL}}, // 58
    {2, 64,  {0x0000000000003C00ULL, 0x000000000000003CULL, 0x0000000000000000ULL}}, // 59
    {2, 64,  {0x0000404040400000ULL, 0x0000808080800000ULL, 0x0000000000000000ULL}}, // 60
    {2, 64,  {0x0000020202020000ULL, 0x0000010101010000ULL, 0x0000000000000000ULL}}, // 61

    {2, 64,  {0x0000808080808080ULL, 0x3F00000000000000ULL, 0x0000000000000000ULL}}, // 62
    {2, 64,  {0x8080808080800000ULL, 0x000000000000003FULL, 0x0000000000000000ULL}}, // 63
    {2, 64,  {0x0000010101010101ULL, 0xFC00000000000000ULL, 0x0000000000000000ULL}}, // 64
    {2, 64,  {0x0101010101010000ULL, 0x00000000000000FCULL, 0x0000000000000000ULL}}, // 65

    {3, 512, {0x0000808080808080ULL, 0x0000201008040201ULL, 0x3F00000000000000ULL}}, // 66
    {3, 512, {0x8080808080800000ULL, 0x0102040810200000ULL, 0x000000000000003FULL}}, // 67
    {3, 512, {0x0000010101010101ULL, 0x0000040810204080ULL, 0xFC00000000000000ULL}}, // 68
    {3, 512, {0x0101010101010000ULL, 0x8040201008040000ULL, 0x00000000000000FCULL}}  // 69
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
    {18, {{24, P30}, {28, P31}, {29, P31}, {32, P34}, {33, P34}, {37, P39}, {41, P39}, {45, P39}, {49, P39}, {51, P34}, {53, P34}, {54, P30}, {55, P33}, {56, P35}, {59, P30}, {61, P30}, {65, P35}, {69, P33}}}, // COORD_H8
    {12, {{ 3, P30}, {22, P30}, {28, P32}, {37, P38}, {41, P35}, {45, P38}, {49, P38}, {51, P35}, {55, P34}, {59, P31}, {65, P34}, {69, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_G8       
    { 6, {{ 7, P30}, {18, P30}, {28, P33}, {32, P35}, {37, P37}, {49, P37}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_F8       
    { 5, {{11, P30}, {14, P30}, {28, P34}, {32, P36}, {49, P36}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_E8       
    { 5, {{ 9, P30}, {15, P30}, {28, P35}, {32, P37}, {48, P36}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_D8       
    { 6, {{ 5, P30}, {19, P30}, {28, P36}, {32, P38}, {36, P37}, {48, P37}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_C8       
    {12, {{ 1, P30}, {23, P30}, {28, P37}, {36, P38}, {40, P35}, {44, P38}, {48, P38}, {51, P36}, {54, P31}, {59, P34}, {63, P34}, {67, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_B8       
    {18, {{25, P30}, {27, P31}, {28, P38}, {31, P34}, {32, P39}, {36, P39}, {40, P39}, {44, P39}, {48, P39}, {51, P37}, {52, P34}, {54, P33}, {55, P35}, {57, P30}, {59, P35}, {60, P30}, {63, P35}, {67, P33}}}, // COORD_A8       
    {12, {{ 2, P30}, {20, P30}, {29, P32}, {37, P36}, {41, P32}, {45, P37}, {49, P35}, {53, P35}, {55, P31}, {61, P31}, {65, P33}, {69, P31}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_H7       
    {16, {{ 2, P31}, { 3, P31}, {24, P31}, {28, P30}, {29, P30}, {37, P35}, {41, P38}, {45, P36}, {49, P34}, {51, P30}, {53, P30}, {55, P32}, {59, P32}, {61, P32}, {65, P32}, {69, P30}, { 0, PNO}, { 0, PNO}}}, // COORD_G7       
    {10, {{ 2, P32}, { 7, P31}, {22, P31}, {32, P30}, {37, P34}, {41, P34}, {45, P35}, {49, P33}, {51, P31}, {65, P31}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_F7       
    { 7, {{ 2, P33}, {11, P31}, {15, P31}, {18, P31}, {32, P31}, {44, P33}, {45, P34}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_E7       
    { 7, {{ 2, P34}, { 9, P31}, {14, P31}, {19, P31}, {32, P32}, {44, P34}, {45, P33}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_D7       
    {10, {{ 2, P35}, { 5, P31}, {23, P31}, {32, P33}, {36, P34}, {40, P34}, {44, P35}, {48, P33}, {51, P32}, {63, P31}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_C7       
    {16, {{ 1, P31}, { 2, P36}, {25, P31}, {27, P30}, {28, P39}, {36, P35}, {40, P38}, {44, P36}, {48, P34}, {51, P33}, {52, P30}, {54, P32}, {59, P33}, {60, P32}, {63, P32}, {67, P30}, { 0, PNO}, { 0, PNO}}}, // COORD_B7       
    {12, {{ 2, P37}, {21, P30}, {27, P32}, {36, P36}, {40, P32}, {44, P37}, {48, P35}, {52, P35}, {54, P34}, {60, P31}, {63, P33}, {67, P31}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_A7       
    { 6, {{ 6, P30}, {16, P30}, {29, P33}, {33, P35}, {37, P33}, {49, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_H6       
    {10, {{ 3, P32}, { 6, P31}, {20, P31}, {33, P30}, {37, P32}, {41, P31}, {45, P32}, {49, P31}, {53, P31}, {65, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_G6       
    { 6, {{ 6, P32}, { 7, P32}, {15, P32}, {24, P32}, {37, P31}, {41, P37}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_F6       
    { 5, {{ 6, P33}, {11, P32}, {19, P32}, {22, P32}, {41, P33}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_E6       
    { 5, {{ 6, P34}, { 9, P32}, {18, P32}, {23, P32}, {40, P33}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_D6       
    { 6, {{ 5, P32}, { 6, P35}, {14, P32}, {25, P32}, {36, P31}, {40, P37}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_C6       
    {10, {{ 1, P32}, { 6, P36}, {21, P31}, {31, P30}, {36, P32}, {40, P31}, {44, P32}, {48, P31}, {52, P31}, {63, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_B6       
    { 6, {{ 6, P37}, {17, P30}, {27, P33}, {31, P35}, {36, P33}, {48, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_A6       
    { 5, {{10, P30}, {12, P30}, {29, P34}, {33, P36}, {49, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_H5       
    { 7, {{ 3, P33}, {10, P31}, {15, P33}, {16, P31}, {33, P31}, {43, P30}, {45, P31}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_G5       
    { 5, {{ 7, P33}, {10, P32}, {19, P33}, {20, P32}, {41, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_F5       
    { 6, {{10, P33}, {11, P33}, {23, P33}, {24, P33}, {37, P30}, {41, P36}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_E5       
    { 6, {{ 9, P33}, {10, P34}, {22, P33}, {25, P33}, {36, P30}, {40, P36}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_D5       
    { 5, {{ 5, P33}, {10, P35}, {18, P33}, {21, P32}, {40, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_C5       
    { 7, {{ 1, P33}, {10, P36}, {14, P33}, {17, P31}, {31, P31}, {42, P30}, {44, P31}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_B5       
    { 5, {{10, P37}, {13, P30}, {27, P34}, {31, P36}, {48, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_A5       
    { 5, {{ 8, P30}, {15, P34}, {29, P35}, {33, P37}, {47, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_H4       
    { 7, {{ 3, P34}, { 8, P31}, {12, P31}, {19, P34}, {33, P32}, {43, P31}, {45, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_G4       
    { 5, {{ 7, P34}, { 8, P32}, {16, P32}, {23, P34}, {39, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_F4       
    { 6, {{ 8, P33}, {11, P34}, {20, P33}, {25, P34}, {35, P30}, {39, P36}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_E4       
    { 6, {{ 8, P34}, { 9, P34}, {21, P33}, {24, P34}, {34, P30}, {38, P36}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_D4       
    { 5, {{ 5, P34}, { 8, P35}, {17, P32}, {22, P34}, {38, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_C4       
    { 7, {{ 1, P34}, { 8, P36}, {13, P31}, {18, P34}, {31, P32}, {42, P31}, {44, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_B4       
    { 5, {{ 8, P37}, {14, P34}, {27, P35}, {31, P37}, {46, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_A4       
    { 6, {{ 4, P30}, {19, P35}, {29, P36}, {33, P38}, {35, P33}, {47, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_H3       
    {10, {{ 3, P35}, { 4, P31}, {23, P35}, {33, P33}, {35, P32}, {39, P31}, {43, P32}, {47, P31}, {53, P32}, {64, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_G3       
    { 6, {{ 4, P32}, { 7, P35}, {12, P32}, {25, P35}, {35, P31}, {39, P37}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_F3       
    { 5, {{ 4, P33}, {11, P35}, {16, P33}, {21, P34}, {39, P33}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_E3       
    { 5, {{ 4, P34}, { 9, P35}, {17, P33}, {20, P34}, {38, P33}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_D3       
    { 6, {{ 4, P35}, { 5, P35}, {13, P32}, {24, P35}, {34, P31}, {38, P37}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_C3       
    {10, {{ 1, P35}, { 4, P36}, {22, P35}, {31, P33}, {34, P32}, {38, P31}, {42, P32}, {46, P31}, {52, P32}, {62, P30}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_B3       
    { 6, {{ 4, P37}, {18, P35}, {27, P36}, {31, P38}, {34, P33}, {46, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_A3       
    {12, {{ 0, P30}, {23, P36}, {29, P37}, {35, P36}, {39, P32}, {43, P37}, {47, P35}, {53, P36}, {56, P34}, {61, P34}, {64, P33}, {68, P31}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_H2       
    {16, {{ 0, P31}, { 3, P36}, {25, P36}, {26, P30}, {29, P39}, {35, P35}, {39, P38}, {43, P36}, {47, P34}, {50, P30}, {53, P33}, {56, P32}, {58, P32}, {61, P33}, {64, P32}, {68, P30}, { 0, PNO}, { 0, PNO}}}, // COORD_G2       
    {10, {{ 0, P32}, { 7, P36}, {21, P35}, {30, P30}, {35, P34}, {39, P34}, {43, P35}, {47, P33}, {50, P31}, {64, P31}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_F2       
    { 7, {{ 0, P33}, {11, P36}, {12, P33}, {17, P34}, {30, P31}, {42, P33}, {43, P34}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_E2       
    { 7, {{ 0, P34}, { 9, P36}, {13, P33}, {16, P34}, {30, P32}, {42, P34}, {43, P33}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_D2       
    {10, {{ 0, P35}, { 5, P36}, {20, P35}, {30, P33}, {34, P34}, {38, P34}, {42, P35}, {46, P33}, {50, P32}, {62, P31}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_C2       
    {16, {{ 0, P36}, { 1, P36}, {24, P36}, {26, P39}, {27, P39}, {34, P35}, {38, P38}, {42, P36}, {46, P34}, {50, P33}, {52, P33}, {57, P32}, {58, P33}, {60, P33}, {62, P32}, {66, P30}, { 0, PNO}, { 0, PNO}}}, // COORD_B2       
    {12, {{ 0, P37}, {22, P36}, {27, P37}, {34, P36}, {38, P32}, {42, P37}, {46, P35}, {52, P36}, {57, P31}, {60, P34}, {62, P33}, {66, P31}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_A2       
    {18, {{25, P37}, {26, P31}, {29, P38}, {30, P34}, {33, P39}, {35, P39}, {39, P39}, {43, P39}, {47, P39}, {50, P34}, {53, P37}, {55, P30}, {56, P33}, {57, P35}, {58, P30}, {61, P35}, {64, P35}, {68, P33}}}, // COORD_H1       
    {12, {{ 3, P37}, {21, P36}, {26, P32}, {35, P38}, {39, P35}, {43, P38}, {47, P38}, {50, P35}, {56, P31}, {58, P31}, {64, P34}, {68, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_G1       
    { 6, {{ 7, P37}, {17, P35}, {26, P33}, {30, P35}, {35, P37}, {47, P37}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_F1       
    { 5, {{11, P37}, {13, P34}, {26, P34}, {30, P36}, {47, P36}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_E1       
    { 5, {{ 9, P37}, {12, P34}, {26, P35}, {30, P37}, {46, P36}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_D1       
    { 6, {{ 5, P37}, {16, P35}, {26, P36}, {30, P38}, {34, P37}, {46, P37}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_C1       
    {12, {{ 1, P37}, {20, P36}, {26, P37}, {34, P38}, {38, P35}, {42, P38}, {46, P38}, {50, P36}, {57, P34}, {58, P34}, {62, P34}, {66, P32}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}, { 0, PNO}}}, // COORD_B1       
    {18, {{24, P37}, {26, P38}, {27, P38}, {30, P39}, {31, P39}, {34, P39}, {38, P39}, {42, P39}, {46, P39}, {50, P37}, {52, P37}, {54, P35}, {56, P30}, {57, P33}, {58, P35}, {60, P35}, {62, P35}, {66, P33}}}  // COORD_A1
};

constexpr uint_fast16_t pow3[11] = {1, P31, P32, P33, P34, P35, P36, P37, P38, P39, P310};
int16_t eval_pattern_arr[2][N_PHASES][N_PATTERNS][MAX_EVALUATE_IDX];
int16_t eval_sur0_sur1_arr[N_PHASES][MAX_SURROUND][MAX_SURROUND];
int16_t eval_canput0_canput1_arr[N_PHASES][MAX_CANPUT][MAX_CANPUT];
int16_t eval_num0_num1_arr[N_PHASES][MAX_STONE_NUM][MAX_STONE_NUM];

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

void init_pattern_arr_rev(int id, int16_t from[], int16_t to[], int siz, int offset){
    int ri, ri_pattern, i_pattern, i_additional;
    for (int i = 0; i < pow3[siz] * offset; ++i){
        i_pattern = i / offset;
        i_additional = i % offset;
        ri_pattern = swap_player_idx(i_pattern, siz);
        ri = ri_pattern * offset + i_additional;
        to[ri] = from[i];
    }
}

inline bool init_evaluation_calc(const char* file){
    FILE* fp;
    #ifdef _WIN64
        if (fopen_s(&fp, file, "rb") != 0){
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
    constexpr int pattern_sizes[N_PATTERNS] = {8, 8, 8, 5, 6, 7, 8, 10, 10, 10, 10, 10, 10, 8, 6, 6, 6, 4};
    constexpr int pattern_joined_offsets[N_PATTERNS] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 8, 64, 64, 64, 512};
    constexpr int pattern_actual_sizes[N_PATTERNS] = {
        P38, P38, P38, P35, P36, P37, P38, P310, P310, P310, P310, P310, P310, 
        52488, 46656, 46656, 46656, 41472
    };
    for (phase_idx = 0; phase_idx < N_PHASES; ++phase_idx){
        for (pattern_idx = 0; pattern_idx < N_PATTERNS; ++pattern_idx){
            if (fread(eval_pattern_arr[0][phase_idx][pattern_idx], 2, pattern_actual_sizes[pattern_idx], fp) < pattern_actual_sizes[pattern_idx]){
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
        if (fread(eval_num0_num1_arr[phase_idx], 2, MAX_STONE_NUM * MAX_STONE_NUM, fp) < MAX_STONE_NUM * MAX_STONE_NUM){
            cerr << "eval.egev broken" << endl;
            fclose(fp);
            return false;
        }
    }
    #if USE_MULTI_THREAD
        int i = 0;
        vector<future<void>> tasks;
        for (phase_idx = 0; phase_idx < N_PHASES; ++phase_idx){
            for (pattern_idx = 0; pattern_idx < N_PATTERNS; ++pattern_idx)
                tasks.emplace_back(thread_pool.push(init_pattern_arr_rev, eval_pattern_arr[0][phase_idx][pattern_idx], eval_pattern_arr[1][phase_idx][pattern_idx], pattern_sizes[pattern_idx], pattern_joined_offsets[pattern_idx]));
        }
        for (future<void> &task: tasks)
            task.get();
    #else
        for (phase_idx = 0; phase_idx < N_PHASES; ++phase_idx){
            for (pattern_idx = 0; pattern_idx < N_PATTERNS; ++pattern_idx)
                init_pattern_arr_rev(0, eval_pattern_arr[0][phase_idx][pattern_idx], eval_pattern_arr[1][phase_idx][pattern_idx], pattern_sizes[pattern_idx], 1);
        }
    #endif
    cerr << "evaluation function initialized" << endl;
    return true;
}

bool evaluate_init(const char* file){
    return init_evaluation_calc(file);
}

bool evaluate_init(){
    return init_evaluation_calc("resources/eval.egev");
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

inline int pick_joined_pattern(Board *b, uint64_t mask){
    int res = (int)((~(b->player | b->opponent) & mask) > 0) << 2;
    res |= (int)((b->player & mask) > 0) << 1;
    res |= (int)((b->opponent & mask) > 0);
    return res;
}

inline int calc_pattern_diff(const int phase_idx, Search *search){
    int res = 
        eval_pattern_arr[search->eval_feature_reversed][phase_idx][0][search->eval_features[0]] + eval_pattern_arr[search->eval_feature_reversed][phase_idx][0][search->eval_features[1]] + eval_pattern_arr[search->eval_feature_reversed][phase_idx][0][search->eval_features[2]] + eval_pattern_arr[search->eval_feature_reversed][phase_idx][0][search->eval_features[3]] + 
        eval_pattern_arr[search->eval_feature_reversed][phase_idx][1][search->eval_features[4]] + eval_pattern_arr[search->eval_feature_reversed][phase_idx][1][search->eval_features[5]] + eval_pattern_arr[search->eval_feature_reversed][phase_idx][1][search->eval_features[6]] + eval_pattern_arr[search->eval_feature_reversed][phase_idx][1][search->eval_features[7]] + 
        eval_pattern_arr[search->eval_feature_reversed][phase_idx][2][search->eval_features[8]] + eval_pattern_arr[search->eval_feature_reversed][phase_idx][2][search->eval_features[9]] + eval_pattern_arr[search->eval_feature_reversed][phase_idx][2][search->eval_features[10]] + eval_pattern_arr[search->eval_feature_reversed][phase_idx][2][search->eval_features[11]] + 
        eval_pattern_arr[search->eval_feature_reversed][phase_idx][3][search->eval_features[12]] + eval_pattern_arr[search->eval_feature_reversed][phase_idx][3][search->eval_features[13]] + eval_pattern_arr[search->eval_feature_reversed][phase_idx][3][search->eval_features[14]] + eval_pattern_arr[search->eval_feature_reversed][phase_idx][3][search->eval_features[15]] + 
        eval_pattern_arr[search->eval_feature_reversed][phase_idx][4][search->eval_features[16]] + eval_pattern_arr[search->eval_feature_reversed][phase_idx][4][search->eval_features[17]] + eval_pattern_arr[search->eval_feature_reversed][phase_idx][4][search->eval_features[18]] + eval_pattern_arr[search->eval_feature_reversed][phase_idx][4][search->eval_features[19]] + 
        eval_pattern_arr[search->eval_feature_reversed][phase_idx][5][search->eval_features[20]] + eval_pattern_arr[search->eval_feature_reversed][phase_idx][5][search->eval_features[21]] + eval_pattern_arr[search->eval_feature_reversed][phase_idx][5][search->eval_features[22]] + eval_pattern_arr[search->eval_feature_reversed][phase_idx][5][search->eval_features[23]] + 
        eval_pattern_arr[search->eval_feature_reversed][phase_idx][6][search->eval_features[24]] + eval_pattern_arr[search->eval_feature_reversed][phase_idx][6][search->eval_features[25]] + 
        eval_pattern_arr[search->eval_feature_reversed][phase_idx][7][search->eval_features[26]] + eval_pattern_arr[search->eval_feature_reversed][phase_idx][7][search->eval_features[27]] + eval_pattern_arr[search->eval_feature_reversed][phase_idx][7][search->eval_features[28]] + eval_pattern_arr[search->eval_feature_reversed][phase_idx][7][search->eval_features[29]] + 
        eval_pattern_arr[search->eval_feature_reversed][phase_idx][8][search->eval_features[30]] + eval_pattern_arr[search->eval_feature_reversed][phase_idx][8][search->eval_features[31]] + eval_pattern_arr[search->eval_feature_reversed][phase_idx][8][search->eval_features[32]] + eval_pattern_arr[search->eval_feature_reversed][phase_idx][8][search->eval_features[33]] + 
        eval_pattern_arr[search->eval_feature_reversed][phase_idx][9][search->eval_features[34]] + eval_pattern_arr[search->eval_feature_reversed][phase_idx][9][search->eval_features[35]] + eval_pattern_arr[search->eval_feature_reversed][phase_idx][9][search->eval_features[36]] + eval_pattern_arr[search->eval_feature_reversed][phase_idx][9][search->eval_features[37]] + 
        eval_pattern_arr[search->eval_feature_reversed][phase_idx][10][search->eval_features[38]] + eval_pattern_arr[search->eval_feature_reversed][phase_idx][10][search->eval_features[39]] + eval_pattern_arr[search->eval_feature_reversed][phase_idx][10][search->eval_features[40]] + eval_pattern_arr[search->eval_feature_reversed][phase_idx][10][search->eval_features[41]] + 
        eval_pattern_arr[search->eval_feature_reversed][phase_idx][11][search->eval_features[42]] + eval_pattern_arr[search->eval_feature_reversed][phase_idx][11][search->eval_features[43]] + eval_pattern_arr[search->eval_feature_reversed][phase_idx][11][search->eval_features[44]] + eval_pattern_arr[search->eval_feature_reversed][phase_idx][11][search->eval_features[45]] + 
        eval_pattern_arr[search->eval_feature_reversed][phase_idx][12][search->eval_features[46]] + eval_pattern_arr[search->eval_feature_reversed][phase_idx][12][search->eval_features[47]] + eval_pattern_arr[search->eval_feature_reversed][phase_idx][12][search->eval_features[48]] + eval_pattern_arr[search->eval_feature_reversed][phase_idx][12][search->eval_features[49]];
    int i;
    for (i = 50; i < 54; ++i)
        res += eval_pattern_arr[search->eval_feature_reversed][phase_idx][13][search->eval_features[i] * joined_pattern[i - 50].offset + pick_joined_pattern(&search->board, joined_pattern[i - 50].mask[0])];
    for (i = 54; i < 66; ++i)
        res += eval_pattern_arr[search->eval_feature_reversed][phase_idx][13 + (i - 50) / 4][search->eval_features[i] * joined_pattern[i - 50].offset + pick_joined_pattern(&search->board, joined_pattern[i - 50].mask[0]) * 8 + pick_joined_pattern(&search->board, joined_pattern[i - 50].mask[1])];
    for (i = 66; i < 70; ++i)
        res += eval_pattern_arr[search->eval_feature_reversed][phase_idx][17][search->eval_features[i] * joined_pattern[i - 50].offset + pick_joined_pattern(&search->board, joined_pattern[i - 50].mask[0]) * 64 + pick_joined_pattern(&search->board, joined_pattern[i - 50].mask[1]) * 8 + pick_joined_pattern(&search->board, joined_pattern[i - 50].mask[2])];
    return res;
}

inline int end_evaluate(Board *b){
    int res = b->score_player();
    return score_to_value(res);
}

inline int mid_evaluate_diff(Search *search){
    int phase_idx, sur0, sur1, canput0, canput1, num0, num1;
    uint64_t empties;
    canput0 = min(MAX_CANPUT - 1, pop_count_ull(calc_legal(search->board.player, search->board.opponent)));
    canput1 = min(MAX_CANPUT - 1, pop_count_ull(calc_legal(search->board.opponent, search->board.player)));
    if (canput0 == 0 && canput1 == 0)
        return end_evaluate(&search->board);
    phase_idx = search->board.phase();
    empties = ~(search->board.player | search->board.opponent);
    sur0 = min(MAX_SURROUND - 1, calc_surround(search->board.player, empties));
    sur1 = min(MAX_SURROUND - 1, calc_surround(search->board.opponent, empties));
    num0 = pop_count_ull(search->board.player);
    num1 = pop_count_ull(search->board.opponent);
    int res = calc_pattern_diff(phase_idx, search) + 
        eval_sur0_sur1_arr[phase_idx][sur0][sur1] + 
        eval_canput0_canput1_arr[phase_idx][canput0][canput1] + 
        eval_num0_num1_arr[phase_idx][num0][num1];
    #if EVALUATION_STEP_WIDTH_MODE == 0
        res += res > 0 ? STEP_2 : (res < 0 ? -STEP_2 : 0);
        //res += STEP_2 * min(1, max(-1, res));
        res /= STEP;
        //res >>= STEP_SHIFT;
        /*
        if (res > 0)
            res += STEP_2;
        else if (res < 0)
            res -= STEP_2;
        res /= STEP;
        */
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
    uint_fast8_t i, cell;
    uint64_t f;
    if (search->eval_feature_reversed){
        for (i = 0; i < coord_to_feature[flip->pos].n_features; ++i){
            //if (search->eval_features[coord_to_feature[flip->pos].features[i].feature] / coord_to_feature[flip->pos].features[i].x % 3 != 2)
            //    cerr << "error " << search->eval_features[coord_to_feature[flip->pos].features[i].feature] << " " << coord_to_feature[flip->pos].features[i].x << " " << flip->pos << endl;
            search->eval_features[coord_to_feature[flip->pos].features[i].feature] -= coord_to_feature[flip->pos].features[i].x;
        }
        f = flip->flip;
        for (cell = first_bit(&f); f; cell = next_bit(&f)){
            for (i = 0; i < coord_to_feature[cell].n_features; ++i){
                //if (search->eval_features[coord_to_feature[cell].features[i].feature] / coord_to_feature[cell].features[i].x % 3 != 0)
                //    cerr << "error " << search->eval_features[coord_to_feature[cell].features[i].feature] << " " << coord_to_feature[cell].features[i].x << " " << cell << endl;
                search->eval_features[coord_to_feature[cell].features[i].feature] += coord_to_feature[cell].features[i].x;
            }
        }
    } else{
        for (i = 0; i < coord_to_feature[flip->pos].n_features; ++i){
            //if (search->eval_features[coord_to_feature[flip->pos].features[i].feature] / coord_to_feature[flip->pos].features[i].x % 3 != 2)
            //    cerr << "error " << search->eval_features[coord_to_feature[flip->pos].features[i].feature] << " " << coord_to_feature[flip->pos].features[i].x << " " << flip->pos << endl;
            search->eval_features[coord_to_feature[flip->pos].features[i].feature] -= 2 * coord_to_feature[flip->pos].features[i].x;
        }
        f = flip->flip;
        for (cell = first_bit(&f); f; cell = next_bit(&f)){
            for (i = 0; i < coord_to_feature[cell].n_features; ++i){
                //if (search->eval_features[coord_to_feature[cell].features[i].feature] / coord_to_feature[cell].features[i].x % 3 != 1)
                //    cerr << "error " << search->eval_features[coord_to_feature[cell].features[i].feature] << " " << coord_to_feature[cell].features[i].x << " " << cell << endl;
                search->eval_features[coord_to_feature[cell].features[i].feature] -= coord_to_feature[cell].features[i].x;
            }
        }
    }
    search->eval_feature_reversed ^= 1;
    //search->board.move(flip);
    //if (search->eval_feature_reversed == 0 && check_features(search)){
    //    search->board.print();
    //    cerr << "error" << endl;
    //}
    //search->board.undo(flip);
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
    //if (search->eval_feature_reversed == 0 && check_features(search))
    //    cerr << "error" << endl;
}

int mid_evaluate(Board *board){
    Search search;
    search.board = board->copy();
    calc_features(&search);
    return mid_evaluate_diff(&search);
}