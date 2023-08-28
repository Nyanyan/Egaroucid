/*
    Egaroucid Project

    @file level.hpp
        definition of Egaroucid's level
    @date 2021-2023
    @author Takuto Yamana
    @license GPL-3.0 license
*/

#pragma once
#include <iostream>
#include <cmath>

/*
    @brief definition of level categories

    0                   - LIGHT_LEVEL           : light
    LIGHT_LEVEL         - STANDARD_MAX_LEVEL    : standard
    STANDARD_MAX_LEVEL  - PRAGMATIC_MAX_LEVEL   : pragmatic
    PRAGMATIC_MAX_LEVEL - ACCURATE_MAX_LEVEL    : accurate
    ACCURATE_MAX_LEVEL  - 60                    : danger
*/
#define ACCURATE_MAX_LEVEL 30
#define PRAGMATIC_MAX_LEVEL 25
#define STANDARD_MAX_LEVEL 21
#define LIGHT_LEVEL 15

#define DEFAULT_LEVEL 21

/*
    @brief constants for level definition
*/
#define N_LEVEL 61
#define N_MOVES 60
#define MPC_70_LEVEL 0
#define MPC_80_LEVEL 1
#define MPC_93_LEVEL 2
#define MPC_98_LEVEL 3
#define MPC_99_LEVEL 4
#define MPC_100_LEVEL 5
#define NODEPTH 100

#define N_SELECTIVITY_LEVEL 6
constexpr int SELECTIVITY_PERCENTAGE[N_SELECTIVITY_LEVEL] = {70, 80, 93, 98, 99, 100};
constexpr double SELECTIVITY_MPCT[N_SELECTIVITY_LEVEL] = {0.52, 0.84, 1.48, 2.05, 2.33, 9.99};

#define DOUBLE_NEAR_THRESHOLD 0.001

/*
    @brief structure of level definition

    @param mid_lookahead        lookahead depth in midgame
    @param mid_mpc_level        MPC (Multi-ProbCut) constant for midgame search
    @param complete0            lookahead depth in endgame search 1/5
    @param complete0_mpc_level  MPC (Multi-ProbCut) constant for endgame search 1/5
    @param complete1            lookahead depth in endgame search 2/5
    @param complete1_mpc_level  MPC (Multi-ProbCut) constant for endgame search 2/5
    @param complete2            lookahead depth in endgame search 3/5
    @param complete2_mpc_level  MPC (Multi-ProbCut) constant for endgame search 3/5
    @param complete3            lookahead depth in endgame search 4/5
    @param complete3_mpc_level  MPC (Multi-ProbCut) constant for endgame search 4/5
    @param complete4            lookahead depth in endgame search 5/5
    @param complete5_mpc_level  MPC (Multi-ProbCut) constant for endgame search 5/5
*/
struct Level{
    uint_fast8_t mid_lookahead;
    uint_fast8_t mid_mpc_level;
    uint_fast8_t complete0;
    uint_fast8_t complete0_mpc_level;
    uint_fast8_t complete1;
    uint_fast8_t complete1_mpc_level;
    uint_fast8_t complete2;
    uint_fast8_t complete2_mpc_level;
    uint_fast8_t complete3;
    uint_fast8_t complete3_mpc_level;
    uint_fast8_t complete4;
    uint_fast8_t complete4_mpc_level;
};

/*
    @brief level definition
*/
constexpr Level level_definition[N_LEVEL] = {
    {0, MPC_100_LEVEL, 0, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL},
    {1, MPC_100_LEVEL, 2, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL},
    {2, MPC_100_LEVEL, 4, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL},
    {3, MPC_100_LEVEL, 6, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL},
    {4, MPC_100_LEVEL, 8, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL},

    {5, MPC_100_LEVEL, 10, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL},
    {6, MPC_100_LEVEL, 12, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL},
    {7, MPC_100_LEVEL, 14, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL},
    {8, MPC_100_LEVEL, 16, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL},
    {9, MPC_100_LEVEL, 18, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL},

    {10, MPC_100_LEVEL, 20, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL},
    {11, MPC_80_LEVEL, 24, MPC_98_LEVEL, 21, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL},
    {12, MPC_80_LEVEL, 24, MPC_98_LEVEL, 21, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL},
    {13, MPC_80_LEVEL, 27, MPC_93_LEVEL, 23, MPC_98_LEVEL, 21, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL},
    {14, MPC_80_LEVEL, 27, MPC_93_LEVEL, 23, MPC_98_LEVEL, 21, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL},

    {15, MPC_80_LEVEL, 27, MPC_98_LEVEL, 24, MPC_99_LEVEL, 22, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL},
    {16, MPC_80_LEVEL, 27, MPC_98_LEVEL, 24, MPC_99_LEVEL, 22, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL},
    {17, MPC_80_LEVEL, 27, MPC_98_LEVEL, 24, MPC_99_LEVEL, 22, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL},
    {18, MPC_80_LEVEL, 28, MPC_93_LEVEL, 26, MPC_98_LEVEL, 24, MPC_99_LEVEL, 22, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL},
    {19, MPC_80_LEVEL, 28, MPC_93_LEVEL, 26, MPC_98_LEVEL, 24, MPC_99_LEVEL, 22, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL},

    {20, MPC_80_LEVEL, 30, MPC_93_LEVEL, 28, MPC_98_LEVEL, 26, MPC_99_LEVEL, 24, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL},
    {21, MPC_80_LEVEL, 30, MPC_93_LEVEL, 28, MPC_98_LEVEL, 26, MPC_99_LEVEL, 24, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL},
    {22, MPC_80_LEVEL, 30, MPC_93_LEVEL, 38, MPC_98_LEVEL, 26, MPC_99_LEVEL, 24, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL},
    {23, MPC_80_LEVEL, 32, MPC_93_LEVEL, 30, MPC_98_LEVEL, 28, MPC_99_LEVEL, 26, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL},
    {24, MPC_80_LEVEL, 32, MPC_93_LEVEL, 30, MPC_98_LEVEL, 28, MPC_99_LEVEL, 26, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL},
    
    {25, MPC_80_LEVEL, 32, MPC_93_LEVEL, 30, MPC_98_LEVEL, 28, MPC_99_LEVEL, 26, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL},
    {26, MPC_80_LEVEL, 32, MPC_93_LEVEL, 30, MPC_98_LEVEL, 28, MPC_99_LEVEL, 26, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL},
    {27, MPC_80_LEVEL, 34, MPC_93_LEVEL, 32, MPC_98_LEVEL, 30, MPC_99_LEVEL, 28, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL},
    {28, MPC_80_LEVEL, 34, MPC_93_LEVEL, 32, MPC_98_LEVEL, 30, MPC_99_LEVEL, 28, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL},
    {29, MPC_80_LEVEL, 36, MPC_93_LEVEL, 34, MPC_98_LEVEL, 32, MPC_99_LEVEL, 30, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL},

    {30, MPC_80_LEVEL, 36, MPC_93_LEVEL, 34, MPC_98_LEVEL, 32, MPC_99_LEVEL, 30, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL},
    {31, MPC_80_LEVEL, 38, MPC_80_LEVEL, 36, MPC_93_LEVEL, 34, MPC_98_LEVEL, 32, MPC_99_LEVEL, 30, MPC_100_LEVEL},
    {32, MPC_80_LEVEL, 38, MPC_80_LEVEL, 36, MPC_93_LEVEL, 34, MPC_98_LEVEL, 32, MPC_99_LEVEL, 30, MPC_100_LEVEL},
    {33, MPC_80_LEVEL, 38, MPC_93_LEVEL, 36, MPC_98_LEVEL, 33, MPC_99_LEVEL, 30, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL},
    {34, MPC_80_LEVEL, 38, MPC_93_LEVEL, 36, MPC_98_LEVEL, 33, MPC_99_LEVEL, 30, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL},

    {35, MPC_80_LEVEL, 42, MPC_80_LEVEL, 39, MPC_93_LEVEL, 36, MPC_98_LEVEL, 33, MPC_99_LEVEL, 30, MPC_100_LEVEL},
    {36, MPC_80_LEVEL, 43, MPC_80_LEVEL, 40, MPC_93_LEVEL, 37, MPC_98_LEVEL, 34, MPC_99_LEVEL, 31, MPC_100_LEVEL},
    {37, MPC_80_LEVEL, 44, MPC_80_LEVEL, 41, MPC_93_LEVEL, 38, MPC_98_LEVEL, 35, MPC_99_LEVEL, 32, MPC_100_LEVEL},
    {38, MPC_80_LEVEL, 45, MPC_80_LEVEL, 42, MPC_93_LEVEL, 39, MPC_98_LEVEL, 36, MPC_99_LEVEL, 33, MPC_100_LEVEL},
    {39, MPC_80_LEVEL, 46, MPC_80_LEVEL, 43, MPC_93_LEVEL, 40, MPC_98_LEVEL, 37, MPC_99_LEVEL, 34, MPC_100_LEVEL},

    {40, MPC_80_LEVEL, 47, MPC_80_LEVEL, 44, MPC_93_LEVEL, 41, MPC_98_LEVEL, 38, MPC_99_LEVEL, 35, MPC_100_LEVEL},
    {41, MPC_80_LEVEL, 48, MPC_80_LEVEL, 45, MPC_93_LEVEL, 42, MPC_98_LEVEL, 39, MPC_99_LEVEL, 36, MPC_100_LEVEL},
    {42, MPC_80_LEVEL, 49, MPC_80_LEVEL, 46, MPC_93_LEVEL, 43, MPC_98_LEVEL, 40, MPC_99_LEVEL, 37, MPC_100_LEVEL},
    {43, MPC_80_LEVEL, 50, MPC_80_LEVEL, 47, MPC_93_LEVEL, 44, MPC_98_LEVEL, 41, MPC_99_LEVEL, 38, MPC_100_LEVEL},
    {44, MPC_80_LEVEL, 51, MPC_80_LEVEL, 48, MPC_93_LEVEL, 45, MPC_98_LEVEL, 42, MPC_99_LEVEL, 39, MPC_100_LEVEL},

    {45, MPC_80_LEVEL, 52, MPC_80_LEVEL, 49, MPC_93_LEVEL, 46, MPC_98_LEVEL, 43, MPC_99_LEVEL, 40, MPC_100_LEVEL},
    {46, MPC_80_LEVEL, 53, MPC_80_LEVEL, 50, MPC_93_LEVEL, 47, MPC_98_LEVEL, 44, MPC_99_LEVEL, 41, MPC_100_LEVEL},
    {47, MPC_80_LEVEL, 54, MPC_80_LEVEL, 51, MPC_93_LEVEL, 48, MPC_98_LEVEL, 45, MPC_99_LEVEL, 42, MPC_100_LEVEL},
    {48, MPC_80_LEVEL, 55, MPC_80_LEVEL, 52, MPC_93_LEVEL, 49, MPC_98_LEVEL, 46, MPC_99_LEVEL, 43, MPC_100_LEVEL},
    {49, MPC_80_LEVEL, 56, MPC_80_LEVEL, 53, MPC_93_LEVEL, 50, MPC_98_LEVEL, 47, MPC_99_LEVEL, 44, MPC_100_LEVEL},

    {50, MPC_80_LEVEL, 57, MPC_80_LEVEL, 54, MPC_93_LEVEL, 51, MPC_98_LEVEL, 48, MPC_99_LEVEL, 45, MPC_100_LEVEL},
    {51, MPC_80_LEVEL, 58, MPC_80_LEVEL, 55, MPC_93_LEVEL, 52, MPC_98_LEVEL, 49, MPC_99_LEVEL, 46, MPC_100_LEVEL},
    {52, MPC_80_LEVEL, 59, MPC_80_LEVEL, 56, MPC_93_LEVEL, 53, MPC_98_LEVEL, 50, MPC_99_LEVEL, 47, MPC_100_LEVEL},
    {60, MPC_80_LEVEL, 60, MPC_80_LEVEL, 57, MPC_93_LEVEL, 54, MPC_98_LEVEL, 51, MPC_99_LEVEL, 48, MPC_100_LEVEL},
    {60, MPC_80_LEVEL, 60, MPC_80_LEVEL, 58, MPC_93_LEVEL, 55, MPC_98_LEVEL, 52, MPC_99_LEVEL, 49, MPC_100_LEVEL},
    
    {60, MPC_80_LEVEL, 60, MPC_80_LEVEL, 58, MPC_93_LEVEL, 56, MPC_98_LEVEL, 53, MPC_99_LEVEL, 50, MPC_100_LEVEL},
    {60, MPC_80_LEVEL, 60, MPC_80_LEVEL, 58, MPC_93_LEVEL, 56, MPC_98_LEVEL, 53, MPC_99_LEVEL, 51, MPC_100_LEVEL},
    {60, MPC_93_LEVEL, 60, MPC_93_LEVEL, 58, MPC_98_LEVEL, 54, MPC_99_LEVEL, 52, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL},
    {60, MPC_93_LEVEL, 60, MPC_93_LEVEL, 58, MPC_98_LEVEL, 55, MPC_99_LEVEL, 53, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL},
    {60, MPC_93_LEVEL, 60, MPC_93_LEVEL, 58, MPC_98_LEVEL, 56, MPC_99_LEVEL, 54, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL},

    {60, MPC_100_LEVEL, 60, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL}
};

/*
    @brief Get level information

    get all information

    @param level                level to search
    @param n_moves              ply
    @param is_mid_search        flag to store this search is midgame search or not
    @param depth                integer to store the depth
    @param mpc_level            MPC level
*/
void get_level(int level, int n_moves, bool *is_mid_search, int *depth, uint_fast8_t *mpc_level){
    if (level < 1)
        level = 1;
    if (level > 60)
        level = 60;
    Level level_status = level_definition[level];
    int n_empties = 60 - n_moves;
    if (n_empties > level_status.complete0){
        *is_mid_search = true;
        *depth = level_status.mid_lookahead;
        *mpc_level = level_status.mid_mpc_level;
    } else {
        *is_mid_search = false;
        *depth = n_empties;
        if (n_empties > level_status.complete1)
            *mpc_level = level_status.complete0_mpc_level;
        else if (n_empties > level_status.complete2)
            *mpc_level = level_status.complete1_mpc_level;
        else if (n_empties > level_status.complete3)
            *mpc_level = level_status.complete2_mpc_level;
        else if (n_empties > level_status.complete4)
            *mpc_level = level_status.complete3_mpc_level;
        else
            *mpc_level = level_status.complete4_mpc_level;
    }
}

/*
    @brief Get level information

    get MPC usage information

    @param level                level to search
    @param n_moves              ply
    @return use MPC (Multi-ProbCut)?
*/
bool get_level_use_mpc(int level, int n_moves){
    if (level < 1)
        level = 1;
    if (level > 60)
        level = 60;
    Level level_status = level_definition[level];
    int n_empties = 60 - n_moves;
    if (n_empties < level_status.complete0){
        return level_status.mid_mpc_level != MPC_100_LEVEL;
    } else {
        if (n_empties > level_status.complete1){
            return level_status.complete0_mpc_level != MPC_100_LEVEL;
        } else if (n_empties > level_status.complete2){
            return level_status.complete1_mpc_level != MPC_100_LEVEL;
        } else if (n_empties > level_status.complete3){
            return level_status.complete2_mpc_level != MPC_100_LEVEL;
        } else if (n_empties > level_status.complete4){
            return level_status.complete3_mpc_level != MPC_100_LEVEL;
        } else{
            return level_status.complete4_mpc_level != MPC_100_LEVEL;
        }
    }
}

/*
    @brief Get level information

    get depth information

    @param level                level to search
    @param mid_depth            integer to store midgame lookahead depth
    @param end_depth            integer to store endgame lookahead depth
*/
void get_level_depth(int level, int *mid_depth, int *end_depth){
    if (level < 1)
        level = 1;
    if (level > 60)
        level = 60;
    *mid_depth = level_definition[level].mid_lookahead;
    *end_depth = level_definition[level].complete0;
}

/*
    @brief Get level information

    get if it is midgame search

    @param level                level to search
    @param n_moves              ply
    @return midgame search?
*/
bool get_level_midsearch(int level, int n_moves){
    if (level < 1)
        level = 1;
    if (level > 60)
        level = 60;
    Level level_status = level_definition[level];
    int n_empties = 60 - n_moves;
    if (n_empties > level_status.complete0){
        return true;
    } else {
        return false;
    }
}

/*
    @brief Get level information

    get endgame search depth

    @param level                level to search
    @return endgame search depth
*/
int get_level_endsearch_depth(int level){
    return level_definition[level].complete0;
}

/*
    @brief Get level information

    get endgame complete (100%) search depth

    @param level                level to search
    @return endgame complete search depth
*/
int get_level_complete_depth(int level){
    if (level_definition[level].complete0_mpc_level == MPC_100_LEVEL)
        return level_definition[level].complete0;
    if (level_definition[level].complete1_mpc_level == MPC_100_LEVEL)
        return level_definition[level].complete1;
    if (level_definition[level].complete2_mpc_level == MPC_100_LEVEL)
        return level_definition[level].complete2;
    if (level_definition[level].complete3_mpc_level == MPC_100_LEVEL)
        return level_definition[level].complete3;
    return level_definition[level].complete4;
}