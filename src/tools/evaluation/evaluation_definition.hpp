#pragma once
#ifndef OPTIMIZER_INCLUDE
    #include "./../../engine/board.hpp"
#endif

#ifndef HW
    #define HW 8
#endif

#ifndef HW2
    #define HW2 64
#endif

#ifndef SCORE_MAX
    #define SCORE_MAX 64
#endif

/*
    @brief evaluation pattern definition
*/
// disc pattern
#define ADJ_N_PATTERNS 12
#define ADJ_N_SYMMETRY_PATTERNS 96
#define ADJ_MAX_PATTERN_CELLS 10

// overall
#define ADJ_MAX_EVALUATE_IDX 59049
#define ADJ_N_EVAL 12
#define ADJ_N_FEATURES 96

// phase
#define ADJ_N_PHASES 60
#define ADJ_N_PHASE_DISCS 1 // 60 / ADJ_N_PHASES

//#define ADJ_SCORE_MAX HW2

/*
    @brief value definition

    Raw score is STEP times larger than the real score.
*/
#define ADJ_STEP 128
#define ADJ_STEP_2 64
//#define ADJ_STEP_SHIFT 7

#ifndef PNO
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
#endif

#ifndef COORD_NO
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
#endif

constexpr int adj_pow3[11] = {P30, P31, P32, P33, P34, P35, P36, P37, P38, P39, P310};

/*
    @brief definition of patterns in evaluation function

    pattern -> coordinate

    @param n_cells              number of cells included in the pattern
    @param cells                coordinates of each cell
*/
struct Adj_Feature_to_coord{
    uint_fast8_t n_cells;
    uint_fast8_t cells[ADJ_MAX_PATTERN_CELLS];
};

constexpr Adj_Feature_to_coord adj_feature_to_coord[ADJ_N_SYMMETRY_PATTERNS] = {
    // 0
    {10, {COORD_C1, COORD_C2, COORD_F2, COORD_B4, COORD_A5, COORD_A6, COORD_A8, COORD_C8, COORD_E8, COORD_H8}}, // 0
    {10, {COORD_H6, COORD_G6, COORD_G3, COORD_E7, COORD_D8, COORD_C8, COORD_A8, COORD_A6, COORD_A4, COORD_A1}}, // 1
    {10, {COORD_A3, COORD_B3, COORD_B6, COORD_D2, COORD_E1, COORD_F1, COORD_H1, COORD_H3, COORD_H5, COORD_H8}}, // 2
    {10, {COORD_F8, COORD_F7, COORD_C7, COORD_G5, COORD_H4, COORD_H3, COORD_H1, COORD_F1, COORD_D1, COORD_A1}}, // 3
    {10, {COORD_F1, COORD_F2, COORD_C2, COORD_G4, COORD_H5, COORD_H6, COORD_H8, COORD_F8, COORD_D8, COORD_A8}}, // 4
    {10, {COORD_C8, COORD_C7, COORD_F7, COORD_B5, COORD_A4, COORD_A3, COORD_A1, COORD_C1, COORD_E1, COORD_H1}}, // 5
    {10, {COORD_H3, COORD_G3, COORD_G6, COORD_E2, COORD_D1, COORD_C1, COORD_A1, COORD_A3, COORD_A5, COORD_A8}}, // 6
    {10, {COORD_A6, COORD_B6, COORD_B3, COORD_D7, COORD_E8, COORD_F8, COORD_H8, COORD_H6, COORD_H4, COORD_H1}}, // 7

    // 1
    {10, {COORD_C1, COORD_C3, COORD_D4, COORD_A5, COORD_C6, COORD_D6, COORD_C7, COORD_F7, COORD_H7, COORD_H8}}, // 8
    {10, {COORD_H6, COORD_F6, COORD_E5, COORD_D8, COORD_C6, COORD_C5, COORD_B6, COORD_B3, COORD_B1, COORD_A1}}, // 9
    {10, {COORD_A3, COORD_C3, COORD_D4, COORD_E1, COORD_F3, COORD_F4, COORD_G3, COORD_G6, COORD_G8, COORD_H8}}, // 10
    {10, {COORD_F8, COORD_F6, COORD_E5, COORD_H4, COORD_F3, COORD_E3, COORD_F2, COORD_C2, COORD_A2, COORD_A1}}, // 11
    {10, {COORD_F1, COORD_F3, COORD_E4, COORD_H5, COORD_F6, COORD_E6, COORD_F7, COORD_C7, COORD_A7, COORD_A8}}, // 12
    {10, {COORD_C8, COORD_C6, COORD_D5, COORD_A4, COORD_C3, COORD_D3, COORD_C2, COORD_F2, COORD_H2, COORD_H1}}, // 13
    {10, {COORD_H3, COORD_F3, COORD_E4, COORD_D1, COORD_C3, COORD_C4, COORD_B3, COORD_B6, COORD_B8, COORD_A8}}, // 14
    {10, {COORD_A6, COORD_C6, COORD_D5, COORD_E8, COORD_F6, COORD_F5, COORD_G6, COORD_G3, COORD_G1, COORD_H1}}, // 15

    // 2
    {10, {COORD_C2, COORD_A4, COORD_H4, COORD_A6, COORD_A7, COORD_B8, COORD_C8, COORD_E8, COORD_G8, COORD_H8}}, // 16
    {10, {COORD_G6, COORD_E8, COORD_E1, COORD_C8, COORD_B8, COORD_A7, COORD_A6, COORD_A4, COORD_A2, COORD_A1}}, // 17
    {10, {COORD_B3, COORD_D1, COORD_D8, COORD_F1, COORD_G1, COORD_H2, COORD_H3, COORD_H5, COORD_H7, COORD_H8}}, // 18
    {10, {COORD_F7, COORD_H5, COORD_A5, COORD_H3, COORD_H2, COORD_G1, COORD_F1, COORD_D1, COORD_B1, COORD_A1}}, // 19
    {10, {COORD_F2, COORD_H4, COORD_A4, COORD_H6, COORD_H7, COORD_G8, COORD_F8, COORD_D8, COORD_B8, COORD_A8}}, // 20
    {10, {COORD_C7, COORD_A5, COORD_H5, COORD_A3, COORD_A2, COORD_B1, COORD_C1, COORD_E1, COORD_G1, COORD_H1}}, // 21
    {10, {COORD_G3, COORD_E1, COORD_E8, COORD_C1, COORD_B1, COORD_A2, COORD_A3, COORD_A5, COORD_A7, COORD_A8}}, // 22
    {10, {COORD_B6, COORD_D8, COORD_D1, COORD_F8, COORD_G8, COORD_H7, COORD_H6, COORD_H4, COORD_H2, COORD_H1}}, // 23

    // 3
    {10, {COORD_D4, COORD_B5, COORD_A6, COORD_B6, COORD_D6, COORD_E6, COORD_F6, COORD_H6, COORD_G8, COORD_H8}}, // 24
    {10, {COORD_E5, COORD_D7, COORD_C8, COORD_C7, COORD_C5, COORD_C4, COORD_C3, COORD_C1, COORD_A2, COORD_A1}}, // 25
    {10, {COORD_D4, COORD_E2, COORD_F1, COORD_F2, COORD_F4, COORD_F5, COORD_F6, COORD_F8, COORD_H7, COORD_H8}}, // 26
    {10, {COORD_E5, COORD_G4, COORD_H3, COORD_G3, COORD_E3, COORD_D3, COORD_C3, COORD_A3, COORD_B1, COORD_A1}}, // 27
    {10, {COORD_E4, COORD_G5, COORD_H6, COORD_G6, COORD_E6, COORD_D6, COORD_C6, COORD_A6, COORD_B8, COORD_A8}}, // 28
    {10, {COORD_D5, COORD_B4, COORD_A3, COORD_B3, COORD_D3, COORD_E3, COORD_F3, COORD_H3, COORD_G1, COORD_H1}}, // 29
    {10, {COORD_E4, COORD_D2, COORD_C1, COORD_C2, COORD_C4, COORD_C5, COORD_C6, COORD_C8, COORD_A7, COORD_A8}}, // 30
    {10, {COORD_D5, COORD_E7, COORD_F8, COORD_F7, COORD_F5, COORD_F4, COORD_F3, COORD_F1, COORD_H2, COORD_H1}}, // 31

    // 4
    {10, {COORD_F1, COORD_B2, COORD_C2, COORD_D2, COORD_G3, COORD_G5, COORD_H5, COORD_H7, COORD_E8, COORD_H8}}, // 32
    {10, {COORD_H3, COORD_G7, COORD_G6, COORD_G5, COORD_F2, COORD_D2, COORD_D1, COORD_B1, COORD_A4, COORD_A1}}, // 33
    {10, {COORD_A6, COORD_B2, COORD_B3, COORD_B4, COORD_C7, COORD_E7, COORD_E8, COORD_G8, COORD_H5, COORD_H8}}, // 34
    {10, {COORD_C8, COORD_G7, COORD_F7, COORD_E7, COORD_B6, COORD_B4, COORD_A4, COORD_A2, COORD_D1, COORD_A1}}, // 35
    {10, {COORD_C1, COORD_G2, COORD_F2, COORD_E2, COORD_B3, COORD_B5, COORD_A5, COORD_A7, COORD_D8, COORD_A8}}, // 36
    {10, {COORD_F8, COORD_B7, COORD_C7, COORD_D7, COORD_G6, COORD_G4, COORD_H4, COORD_H2, COORD_E1, COORD_H1}}, // 37
    {10, {COORD_H6, COORD_G2, COORD_G3, COORD_G4, COORD_F7, COORD_D7, COORD_D8, COORD_B8, COORD_A5, COORD_A8}}, // 38
    {10, {COORD_A3, COORD_B7, COORD_B6, COORD_B5, COORD_C2, COORD_E2, COORD_E1, COORD_G1, COORD_H4, COORD_H1}}, // 39

    // 5
    {10, {COORD_D1, COORD_E2, COORD_E3, COORD_D4, COORD_H4, COORD_G5, COORD_D6, COORD_C7, COORD_G7, COORD_H8}}, // 40
    {10, {COORD_H5, COORD_G4, COORD_F4, COORD_E5, COORD_E1, COORD_D2, COORD_C5, COORD_B6, COORD_B2, COORD_A1}}, // 41
    {10, {COORD_A4, COORD_B5, COORD_C5, COORD_D4, COORD_D8, COORD_E7, COORD_F4, COORD_G3, COORD_G7, COORD_H8}}, // 42
    {10, {COORD_E8, COORD_D7, COORD_D6, COORD_E5, COORD_A5, COORD_B4, COORD_E3, COORD_F2, COORD_B2, COORD_A1}}, // 43
    {10, {COORD_E1, COORD_D2, COORD_D3, COORD_E4, COORD_A4, COORD_B5, COORD_E6, COORD_F7, COORD_B7, COORD_A8}}, // 44
    {10, {COORD_D8, COORD_E7, COORD_E6, COORD_D5, COORD_H5, COORD_G4, COORD_D3, COORD_C2, COORD_G2, COORD_H1}}, // 45
    {10, {COORD_H4, COORD_G5, COORD_F5, COORD_E4, COORD_E8, COORD_D7, COORD_C4, COORD_B3, COORD_B7, COORD_A8}}, // 46
    {10, {COORD_A5, COORD_B4, COORD_C4, COORD_D5, COORD_D1, COORD_E2, COORD_F5, COORD_G6, COORD_G2, COORD_H1}}, // 47

    // 6
    {10, {COORD_F1, COORD_G1, COORD_B2, COORD_D2, COORD_G3, COORD_H4, COORD_H5, COORD_H6, COORD_H7, COORD_H8}}, // 48
    {10, {COORD_H3, COORD_H2, COORD_G7, COORD_G5, COORD_F2, COORD_E1, COORD_D1, COORD_C1, COORD_B1, COORD_A1}}, // 49
    {10, {COORD_A6, COORD_A7, COORD_B2, COORD_B4, COORD_C7, COORD_D8, COORD_E8, COORD_F8, COORD_G8, COORD_H8}}, // 50
    {10, {COORD_C8, COORD_B8, COORD_G7, COORD_E7, COORD_B6, COORD_A5, COORD_A4, COORD_A3, COORD_A2, COORD_A1}}, // 51
    {10, {COORD_C1, COORD_B1, COORD_G2, COORD_E2, COORD_B3, COORD_A4, COORD_A5, COORD_A6, COORD_A7, COORD_A8}}, // 52
    {10, {COORD_F8, COORD_G8, COORD_B7, COORD_D7, COORD_G6, COORD_H5, COORD_H4, COORD_H3, COORD_H2, COORD_H1}}, // 53
    {10, {COORD_H6, COORD_H7, COORD_G2, COORD_G4, COORD_F7, COORD_E8, COORD_D8, COORD_C8, COORD_B8, COORD_A8}}, // 54
    {10, {COORD_A3, COORD_A2, COORD_B7, COORD_B5, COORD_C2, COORD_D1, COORD_E1, COORD_F1, COORD_G1, COORD_H1}}, // 55

    // 7
    {10, {COORD_C2, COORD_B3, COORD_A4, COORD_B4, COORD_E4, COORD_D5, COORD_F5, COORD_E6, COORD_F6, COORD_B8}}, // 56
    {10, {COORD_G6, COORD_F7, COORD_E8, COORD_E7, COORD_E4, COORD_D5, COORD_D3, COORD_C4, COORD_C3, COORD_A7}}, // 57
    {10, {COORD_B3, COORD_C2, COORD_D1, COORD_D2, COORD_D5, COORD_E4, COORD_E6, COORD_F5, COORD_F6, COORD_H2}}, // 58
    {10, {COORD_F7, COORD_G6, COORD_H5, COORD_G5, COORD_D5, COORD_E4, COORD_C4, COORD_D3, COORD_C3, COORD_G1}}, // 59
    {10, {COORD_F2, COORD_G3, COORD_H4, COORD_G4, COORD_D4, COORD_E5, COORD_C5, COORD_D6, COORD_C6, COORD_G8}}, // 60
    {10, {COORD_C7, COORD_B6, COORD_A5, COORD_B5, COORD_E5, COORD_D4, COORD_F4, COORD_E3, COORD_F3, COORD_B1}}, // 61
    {10, {COORD_G3, COORD_F2, COORD_E1, COORD_E2, COORD_E5, COORD_D4, COORD_D6, COORD_C5, COORD_C6, COORD_A2}}, // 62
    {10, {COORD_B6, COORD_C7, COORD_D8, COORD_D7, COORD_D4, COORD_E5, COORD_E3, COORD_F4, COORD_F3, COORD_H7}}, // 63

    // 8
    {10, {COORD_B1, COORD_C5, COORD_D6, COORD_E6, COORD_C7, COORD_E7, COORD_G7, COORD_H7, COORD_C8, COORD_G8}}, // 64
    {10, {COORD_H7, COORD_D6, COORD_C5, COORD_C4, COORD_B6, COORD_B4, COORD_B2, COORD_B1, COORD_A6, COORD_A2}}, // 65
    {10, {COORD_A2, COORD_E3, COORD_F4, COORD_F5, COORD_G3, COORD_G5, COORD_G7, COORD_G8, COORD_H3, COORD_H7}}, // 66
    {10, {COORD_G8, COORD_F4, COORD_E3, COORD_D3, COORD_F2, COORD_D2, COORD_B2, COORD_A2, COORD_F1, COORD_B1}}, // 67
    {10, {COORD_G1, COORD_F5, COORD_E6, COORD_D6, COORD_F7, COORD_D7, COORD_B7, COORD_A7, COORD_F8, COORD_B8}}, // 68
    {10, {COORD_B8, COORD_C4, COORD_D3, COORD_E3, COORD_C2, COORD_E2, COORD_G2, COORD_H2, COORD_C1, COORD_G1}}, // 69
    {10, {COORD_H2, COORD_D3, COORD_C4, COORD_C5, COORD_B3, COORD_B5, COORD_B7, COORD_B8, COORD_A3, COORD_A7}}, // 70
    {10, {COORD_A7, COORD_E6, COORD_F5, COORD_F4, COORD_G6, COORD_G4, COORD_G2, COORD_G1, COORD_H6, COORD_H2}}, // 71

    // 9
    {10, {COORD_D1, COORD_C3, COORD_B4, COORD_A6, COORD_B6, COORD_A7, COORD_B7, COORD_F7, COORD_G7, COORD_H8}}, // 72
    {10, {COORD_H5, COORD_F6, COORD_E7, COORD_C8, COORD_C7, COORD_B8, COORD_B7, COORD_B3, COORD_B2, COORD_A1}}, // 73
    {10, {COORD_A4, COORD_C3, COORD_D2, COORD_F1, COORD_F2, COORD_G1, COORD_G2, COORD_G6, COORD_G7, COORD_H8}}, // 74
    {10, {COORD_E8, COORD_F6, COORD_G5, COORD_H3, COORD_G3, COORD_H2, COORD_G2, COORD_C2, COORD_B2, COORD_A1}}, // 75
    {10, {COORD_E1, COORD_F3, COORD_G4, COORD_H6, COORD_G6, COORD_H7, COORD_G7, COORD_C7, COORD_B7, COORD_A8}}, // 76
    {10, {COORD_D8, COORD_C6, COORD_B5, COORD_A3, COORD_B3, COORD_A2, COORD_B2, COORD_F2, COORD_G2, COORD_H1}}, // 77
    {10, {COORD_H4, COORD_F3, COORD_E2, COORD_C1, COORD_C2, COORD_B1, COORD_B2, COORD_B6, COORD_B7, COORD_A8}}, // 78
    {10, {COORD_A5, COORD_C6, COORD_D7, COORD_F8, COORD_F7, COORD_G8, COORD_G7, COORD_G3, COORD_G2, COORD_H1}}, // 79

    // 10
    {10, {COORD_B2, COORD_F3, COORD_C4, COORD_D4, COORD_B5, COORD_D5, COORD_A6, COORD_D6, COORD_E6, COORD_B8}}, // 80
    {10, {COORD_G7, COORD_F3, COORD_E6, COORD_E5, COORD_D7, COORD_D5, COORD_C8, COORD_C5, COORD_C4, COORD_A7}}, // 81
    {10, {COORD_B2, COORD_C6, COORD_D3, COORD_D4, COORD_E2, COORD_E4, COORD_F1, COORD_F4, COORD_F5, COORD_H2}}, // 82
    {10, {COORD_G7, COORD_C6, COORD_F5, COORD_E5, COORD_G4, COORD_E4, COORD_H3, COORD_E3, COORD_D3, COORD_G1}}, // 83
    {10, {COORD_G2, COORD_C3, COORD_F4, COORD_E4, COORD_G5, COORD_E5, COORD_H6, COORD_E6, COORD_D6, COORD_G8}}, // 84
    {10, {COORD_B7, COORD_F6, COORD_C5, COORD_D5, COORD_B4, COORD_D4, COORD_A3, COORD_D3, COORD_E3, COORD_B1}}, // 85
    {10, {COORD_G2, COORD_F6, COORD_E3, COORD_E4, COORD_D2, COORD_D4, COORD_C1, COORD_C4, COORD_C5, COORD_A2}}, // 86
    {10, {COORD_B7, COORD_C3, COORD_D6, COORD_D5, COORD_E7, COORD_E5, COORD_F8, COORD_F5, COORD_F4, COORD_H7}}, // 87

    // 11
    {10, {COORD_H2, COORD_C3, COORD_B5, COORD_F5, COORD_G5, COORD_G6, COORD_E7, COORD_G7, COORD_H7, COORD_C8}}, // 88
    {10, {COORD_G1, COORD_F6, COORD_D7, COORD_D3, COORD_D2, COORD_C2, COORD_B4, COORD_B2, COORD_B1, COORD_A6}}, // 89
    {10, {COORD_B8, COORD_C3, COORD_E2, COORD_E6, COORD_E7, COORD_F7, COORD_G5, COORD_G7, COORD_G8, COORD_H3}}, // 90
    {10, {COORD_A7, COORD_F6, COORD_G4, COORD_C4, COORD_B4, COORD_B3, COORD_D2, COORD_B2, COORD_A2, COORD_F1}}, // 91
    {10, {COORD_A2, COORD_F3, COORD_G5, COORD_C5, COORD_B5, COORD_B6, COORD_D7, COORD_B7, COORD_A7, COORD_F8}}, // 92
    {10, {COORD_H7, COORD_C6, COORD_B4, COORD_F4, COORD_G4, COORD_G3, COORD_E2, COORD_G2, COORD_H2, COORD_C1}}, // 93
    {10, {COORD_G8, COORD_F3, COORD_D2, COORD_D6, COORD_D7, COORD_C7, COORD_B5, COORD_B7, COORD_B8, COORD_A3}}, // 94
    {10, {COORD_B1, COORD_C6, COORD_E7, COORD_E3, COORD_E2, COORD_F2, COORD_G4, COORD_G2, COORD_G1, COORD_H6}}  // 95
};

constexpr int adj_pattern_n_cells[ADJ_N_PATTERNS] = {
    10, 10, 10, 10, 
    10, 10, 10, 10, 
    10, 10, 10, 10
};

constexpr int adj_rev_patterns[ADJ_N_PATTERNS][ADJ_MAX_PATTERN_CELLS] = {
    {-1}, 
    {-1}, 
    {-1}, 
    {-1}, 
    {-1}, 
    {-1}, 
    {-1}, 
    {-1}, 
    {-1}, 
    {-1}, 
    {-1}, 
    {-1}
};

constexpr int adj_eval_sizes[ADJ_N_EVAL] = {
    P310, P310, P310, P310, 
    P310, P310, P310, P310,
    P310, P310, P310, P310
};

constexpr int adj_feature_to_eval_idx[ADJ_N_FEATURES] = {
    0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 1,
    2, 2, 2, 2, 2, 2, 2, 2,
    3, 3, 3, 3, 3, 3, 3, 3,
    4, 4, 4, 4, 4, 4, 4, 4,
    5, 5, 5, 5, 5, 5, 5, 5,
    6, 6, 6, 6, 6, 6, 6, 6,
    7, 7, 7, 7, 7, 7, 7, 7,
    8, 8, 8, 8, 8, 8, 8, 8,
    9, 9, 9, 9, 9, 9, 9, 9,
    10, 10, 10, 10, 10, 10, 10, 10,
    11, 11, 11, 11, 11, 11, 11, 11
};

int adj_pick_digit3(int num, int d, int n_digit){
    num /= adj_pow3[n_digit - 1 - d];
    return num % 3;
}

int adj_pick_digit2(int num, int d){
    return 1 & (num >> d);
}

uint16_t adj_calc_rev_idx(int feature, int idx){
    uint16_t res = 0;
    if (adj_rev_patterns[feature][0] == -1)
        res = idx;
    else{
        for (int i = 0; i < adj_pattern_n_cells[feature]; ++i){
            res += adj_pick_digit3(idx, adj_rev_patterns[feature][i], adj_pattern_n_cells[feature]) * adj_pow3[adj_pattern_n_cells[feature] - 1 - i];
        }
    }
    return res;
}

#ifndef OPTIMIZER_INCLUDE

inline int adj_pick_pattern(const uint_fast8_t b_arr[], int pattern_idx){
    int res = 0;
    for (int i = 0; i < adj_feature_to_coord[pattern_idx].n_cells; ++i){
        res *= 3;
        res += b_arr[adj_feature_to_coord[pattern_idx].cells[i]];
    }
    return res;
}

void adj_calc_features(Board *board, uint16_t res[]){
    uint_fast8_t b_arr[HW2];
    board->translate_to_arr_player(b_arr);
    int idx = 0;
    for (int i = 0; i < ADJ_N_SYMMETRY_PATTERNS; ++i)
        res[idx++] = adj_pick_pattern(b_arr, i);
}

int calc_phase(Board *board, int16_t player){
    return (pop_count_ull(board->player | board->opponent) - 4) / ADJ_N_PHASE_DISCS;
}

void evaluation_definition_init(){
    mobility_init();
}

#endif