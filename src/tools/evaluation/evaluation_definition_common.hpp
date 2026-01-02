/*
    Egaroucid Project

    @file evaluation_definition_common.hpp
        Common things for Evaluation Function Definition
    @date 2021-2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/


#ifndef HW
    #define HW 8
#endif

#ifndef HW2
    #define HW2 64
#endif

#ifndef SCORE_MAX
    #define SCORE_MAX 64
#endif

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