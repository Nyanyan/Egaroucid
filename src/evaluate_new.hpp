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
#define N_JOINED_PATTERNS 5
#ifndef N_SYMMETRY_PATTERNS
    #define N_SYMMETRY_PATTERNS 70
#endif
#define MAX_PATTERN_CELLS 10
#define MAX_CELL_PATTERNS 12
#define MAX_SURROUND 100
#define MAX_CANPUT 50
#define MAX_STONE_NUM 65
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

};

constexpr uint_fast16_t pow3[11] = {1, P31, P32, P33, P34, P35, P36, P37, P38, P39, P310};
uint64_t stability_edge_arr[N_8BIT][N_8BIT][2];
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

inline void probably_move_line(int p, int o, int place, int *np, int *no){
    int i, j;
    *np = p | (1 << place);
    for (i = place - 1; i > 0 && (1 & (o >> i)); --i);
    if (1 & (p >> i)){
        for (j = place - 1; j > i; --j)
            *np ^= 1 << j;
    }
    for (i = place + 1; i < HW_M1 && (1 & (o >> i)); ++i);
    if (1 & (p >> i)){
        for (j = place + 1; j < i; ++j)
            *np ^= 1 << j;
    }
    *no = o & ~(*np);
}

int calc_stability_line(int b, int w){
    int i, nb, nw, res = b | w;
    int empties = ~(b | w);
    for (i = 0; i < HW; ++i){
        if (1 & (empties >> i)){
            probably_move_line(b, w, i, &nb, &nw);
            res &= b | nw;
            res &= calc_stability_line(nb, nw);
            probably_move_line(w, b, i, &nw, &nb);
            res &= w | nb;
            res &= calc_stability_line(nb, nw);
        }
    }
    return res;
}

inline void init_evaluation_base() {
    int place, b, w, stab;
    for (b = 0; b < N_8BIT; ++b) {
        for (w = b; w < N_8BIT; ++w){
            if (b & w){
                stability_edge_arr[b][w][0] = 0;
                stability_edge_arr[b][w][1] = 0;
                stability_edge_arr[w][b][0] = 0;
                stability_edge_arr[w][b][1] = 0;
            } else{
                stab = calc_stability_line(b, w);
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
    for (phase_idx = 0; phase_idx < N_PHASES; ++phase_idx){
        for (pattern_idx = 0; pattern_idx < N_PATTERNS; ++pattern_idx){
            if (fread(eval_pattern_arr[0][phase_idx][pattern_idx], 2, pow3[pattern_sizes[pattern_idx]], fp) < pow3[pattern_sizes[pattern_idx]]){
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
                init_pattern_arr_rev(0, eval_pattern_arr[0][phase_idx][pattern_idx], pattern_arr[1][phase_idx][pattern_idx], pattern_sizes[pattern_idx], 1);
        }
    #endif
    cerr << "evaluation function initialized" << endl;
    return true;
}

bool evaluate_init(const char* file){
    init_evaluation_base();
    return init_evaluation_calc(file);
}

bool evaluate_init(){
    init_evaluation_base();
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

#if USE_SIMD && false
    inline int calc_stability_player(uint64_t player, uint64_t opponent){
        uint64_t full_h, full_v, full_d7, full_d9;
        uint64_t edge_stability = 0, player_stability = 0, n_stability;
        const uint64_t player_mask = player & 0b0000000001111110011111100111111001111110011111100111111000000000ULL;
        uint8_t pl, op;
        pl = player & 0b11111111U;
        op = opponent & 0b11111111U;
        edge_stability |= stability_edge_arr[pl][op][0];
        pl = (player >> 56) & 0b11111111U;
        op = (opponent >> 56) & 0b11111111U;
        edge_stability |= stability_edge_arr[pl][op][0] << 56;
        pl = join_v_line(player, 0);
        op = join_v_line(opponent, 0);
        edge_stability |= stability_edge_arr[pl][op][1];
        pl = join_v_line(player, 7);
        op = join_v_line(opponent, 7);
        edge_stability |= stability_edge_arr[pl][op][1] << 7;
        full_stability(player, opponent, &full_h, &full_v, &full_d7, &full_d9);
        u64_4 hvd7d9;
        const u64_4 shift(1, HW, HW_M1, HW_P1);
        u64_4 full(full_h, full_v, full_d7, full_d9);
        u64_4 stab;
        n_stability = (edge_stability & player) | (full_h & full_v & full_d7 & full_d9 & player_mask);
        while (n_stability & ~player_stability){
            player_stability |= n_stability;
            stab = player_stability;
            hvd7d9 = (stab >> shift) | (stab << shift) | full;
            n_stability = all_and(hvd7d9) & player_mask;
        }
        return pop_count_ull(player_stability);
    }

    inline void calc_stability(Board *b, int *stab0, int *stab1){
        uint64_t full_h, full_v, full_d7, full_d9;
        uint64_t edge_stability = 0, player_stability = 0, opponent_stability = 0, n_stability;
        const uint64_t player_mask = b->player & 0b0000000001111110011111100111111001111110011111100111111000000000ULL;
        const uint64_t opponent_mask = b->opponent & 0b0000000001111110011111100111111001111110011111100111111000000000ULL;
        uint8_t pl, op;
        pl = b->player & 0b11111111U;
        op = b->opponent & 0b11111111U;
        edge_stability |= stability_edge_arr[pl][op][0];
        pl = (b->player >> 56) & 0b11111111U;
        op = (b->opponent >> 56) & 0b11111111U;
        edge_stability |= stability_edge_arr[pl][op][0] << 56;
        pl = join_v_line(b->player, 0);
        op = join_v_line(b->opponent, 0);
        edge_stability |= stability_edge_arr[pl][op][1];
        pl = join_v_line(b->player, 7);
        op = join_v_line(b->opponent, 7);
        edge_stability |= stability_edge_arr[pl][op][1] << 7;
        b->full_stability(&full_h, &full_v, &full_d7, &full_d9);

        u64_4 hvd7d9;
        const u64_4 shift(1, HW, HW_M1, HW_P1);
        u64_4 full(full_h, full_v, full_d7, full_d9);
        u64_4 stab;

        n_stability = (edge_stability & b->player) | (full_h & full_v & full_d7 & full_d9 & player_mask);
        while (n_stability & ~player_stability){
            player_stability |= n_stability;
            stab = player_stability;
            hvd7d9 = (stab >> shift) | (stab << shift) | full;
            n_stability = all_and(hvd7d9) & player_mask;
        }

        n_stability = (edge_stability & b->opponent) | (full_h & full_v & full_d7 & full_d9 & opponent_mask);
        while (n_stability & ~opponent_stability){
            opponent_stability |= n_stability;
            stab = opponent_stability;
            hvd7d9 = (stab >> shift) | (stab << shift) | full;
            n_stability = all_and(hvd7d9) & opponent_mask;
        }

        *stab0 = pop_count_ull(player_stability);
        *stab1 = pop_count_ull(opponent_stability);
    }
#else
    inline int calc_stability_player(uint64_t player, uint64_t opponent){
        uint64_t full_h, full_v, full_d7, full_d9;
        uint64_t edge_stability = 0, player_stability = 0, n_stability;
        uint64_t h, v, d7, d9;
        const uint64_t player_mask = player & 0b0000000001111110011111100111111001111110011111100111111000000000ULL;
        uint8_t pl, op;
        pl = player & 0b11111111U;
        op = opponent & 0b11111111U;
        edge_stability |= stability_edge_arr[pl][op][0];
        pl = join_h_line(player, 7);
        op = join_h_line(opponent, 7);
        edge_stability |= stability_edge_arr[pl][op][0] << 56;
        pl = join_v_line(player, 0);
        op = join_v_line(opponent, 0);
        edge_stability |= stability_edge_arr[pl][op][1];
        pl = join_v_line(player, 7);
        op = join_v_line(opponent, 7);
        edge_stability |= stability_edge_arr[pl][op][1] << 7;
        full_stability(player, opponent, &full_h, &full_v, &full_d7, &full_d9);
        n_stability = (edge_stability & player) | (full_h & full_v & full_d7 & full_d9 & player_mask);
        while (n_stability & ~player_stability){
            player_stability |= n_stability;
            h = (player_stability >> 1) | (player_stability << 1) | full_h;
            v = (player_stability >> HW) | (player_stability << HW) | full_v;
            d7 = (player_stability >> HW_M1) | (player_stability << HW_M1) | full_d7;
            d9 = (player_stability >> HW_P1) | (player_stability << HW_P1) | full_d9;
            n_stability = h & v & d7 & d9 & player_mask;
        }
        return pop_count_ull(player_stability);
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
        edge_stability |= stability_edge_arr[pl][op][0];
        pl = join_h_line(b->player, 7);
        op = join_h_line(b->opponent, 7);
        edge_stability |= stability_edge_arr[pl][op][0] << 56;
        pl = join_v_line(b->player, 0);
        op = join_v_line(b->opponent, 0);
        edge_stability |= stability_edge_arr[pl][op][1];
        pl = join_v_line(b->player, 7);
        op = join_v_line(b->opponent, 7);
        edge_stability |= stability_edge_arr[pl][op][1] << 7;
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

    inline void calc_stability(Board *b, uint64_t edge_stability, int *stab0, int *stab1){
        uint64_t full_h, full_v, full_d7, full_d9;
        uint64_t player_stability = 0, opponent_stability = 0, n_stability;
        uint64_t h, v, d7, d9;
        const uint64_t player_mask = b->player & 0b0000000001111110011111100111111001111110011111100111111000000000ULL;
        const uint64_t opponent_mask = b->opponent & 0b0000000001111110011111100111111001111110011111100111111000000000ULL;
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
#endif

inline int calc_stability_edge(Board *b){
    uint64_t edge_stability = 0;
    uint8_t pl, op;
    pl = b->player & 0b11111111U;
    op = b->opponent & 0b11111111U;
    edge_stability |= stability_edge_arr[pl][op][0] << 56;
    pl = join_h_line(b->player, 7);
    op = join_h_line(b->opponent, 7);
    edge_stability |= stability_edge_arr[pl][op][0];
    pl = join_v_line(b->player, 0);
    op = join_v_line(b->opponent, 0);
    edge_stability |= stability_edge_arr[pl][op][1] << 7;
    pl = join_v_line(b->player, 7);
    op = join_v_line(b->opponent, 7);
    edge_stability |= stability_edge_arr[pl][op][1];
    return pop_count_ull(edge_stability & b->player) - pop_count_ull(edge_stability & b->opponent);
}

inline void calc_stability_edge(Board *b, int *stab0, int *stab1){
    uint64_t edge_stability = 0;
    uint8_t pl, op;
    pl = b->player & 0b11111111U;
    op = b->opponent & 0b11111111U;
    edge_stability |= stability_edge_arr[pl][op][0] << 56;
    pl = join_h_line(b->player, 7);
    op = join_h_line(b->opponent, 7);
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

inline void calc_stability_edge(Board *b, int *stab0, int *stab1, uint64_t *edge_stability){
    *edge_stability = 0;
    uint8_t pl, op;
    pl = b->player & 0b11111111U;
    op = b->opponent & 0b11111111U;
    *edge_stability |= stability_edge_arr[pl][op][0] << 56;
    pl = join_h_line(b->player, 7);
    op = join_h_line(b->opponent, 7);
    *edge_stability |= stability_edge_arr[pl][op][0];
    pl = join_v_line(b->player, 0);
    op = join_v_line(b->opponent, 0);
    *edge_stability |= stability_edge_arr[pl][op][1] << 7;
    pl = join_v_line(b->player, 7);
    op = join_v_line(b->opponent, 7);
    *edge_stability |= stability_edge_arr[pl][op][1];
    *stab0 = pop_count_ull(*edge_stability & b->player);
    *stab1 = pop_count_ull(*edge_stability & b->opponent);
}

inline int calc_stability_edge_player(uint64_t player, uint64_t opponent){
    uint64_t edge_stability = 0;
    uint8_t pl, op;
    pl = player & 0b11111111U;
    op = opponent & 0b11111111U;
    edge_stability |= stability_edge_arr[pl][op][0] << 56;
    pl = join_h_line(player, 7);
    op = join_h_line(opponent, 7);
    edge_stability |= stability_edge_arr[pl][op][0];
    pl = join_v_line(player, 0);
    op = join_v_line(opponent, 0);
    edge_stability |= stability_edge_arr[pl][op][1] << 7;
    pl = join_v_line(player, 7);
    op = join_v_line(opponent, 7);
    edge_stability |= stability_edge_arr[pl][op][1];
    return pop_count_ull(edge_stability & player);
}

inline int pick_joined_pattern(Board *b, uint64_t mask){
    int res = (int)((~(b->player | b->opponent) & mask) > 0) << 2;
    res |= (int)((b->player & mask) > 0) << 1;
    res |= (int)((b->opponent & mask) > 0);
    return res;
}

inline int calc_pattern_diff(const int phase_idx, Search *search){
    int res = 
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
        pattern_arr[search->eval_feature_reversed][phase_idx][12][search->eval_features[46]] + pattern_arr[search->eval_feature_reversed][phase_idx][12][search->eval_features[47]] + pattern_arr[search->eval_feature_reversed][phase_idx][12][search->eval_features[48]] + pattern_arr[search->eval_feature_reversed][phase_idx][12][search->eval_features[49]];
    int i;
    for (i = 50; i < 54; ++i)
        res += pattern_arr[search->eval_feature_reversed][phase_idx][13][search->eval_features[i] * joined_pattern[i - 50].offset + pick_joined_pattern(&search->board, joined_pattern[i - 50].mask[0])];
    for (i = 54; i < 66; ++i)
        res += pattern_arr[search->eval_feature_reversed][phase_idx][13 + (i - 50) / 4][search->eval_features[i] * joined_pattern[i - 50].offset + pick_joined_pattern(&search->board, joined_pattern[i - 50].mask[0]) * 8 + pick_joined_pattern(&search->board, joined_pattern[i - 50].mask[1])];
    for (i = 66; i < 70; ++i)
        res += pattern_arr[search->eval_feature_reversed][phase_idx][17][search->eval_features[i] * joined_pattern[i - 50].offset + pick_joined_pattern(&search->board, joined_pattern[i - 50].mask[0]) * 64 + pick_joined_pattern(&search->board, joined_pattern[i - 50].mask[1]) * 8 + pick_joined_pattern(&search->board, joined_pattern[i - 50].mask[2])];
    return res;
}

inline int end_evaluate(Board *b){
    int res = b->score_player();
    return score_to_value(res);
}

inline int mid_evaluate_diff(Search *search){
    int phase_idx, sur0, sur1, canput0, canput1, num0, num1;
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
    num0 = pop_count_ull(search->board.player);
    num1 = pop_count_ull(search->board.opponent);
    int res = calc_pattern_diff(phase_idx, search) + 
        eval_sur0_sur1_arr[phase_idx][sur0][sur1] + 
        eval_canput0_canput1_arr[phase_idx][canput0][canput1] + 
        eval_num0_num1_arr[phase_idx][num0][num1];
    #if EVALUATION_STEP_WIDTH_MODE == 0
        res += res > 0 ? STEP_2 : (res < 0 ? -STEP_2 : 0);
        //res += STEP_2 * min(1, max(-1, res));
        res >>= STEP_SHIFT;
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
    #endif
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
    #endif
    //if (search->eval_feature_reversed == 0 && check_features(search))
    //    cerr << "error" << endl;
}