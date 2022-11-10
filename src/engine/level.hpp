/*
    Egaroucid Project

    @file level.hpp
        definition of Egaroucid's level
    @date 2021-2022
    @author Takuto Yamana (a.k.a. Nyanyan)
    @license GPL-3.0 license
*/

#pragma once
#include <iostream>

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
#define MPC_81 0.88
#define MPC_95 1.64
#define MPC_98 2.05
#define MPC_99 2.33
#define NOMPC 9.99
#define NODEPTH 100

#define DOUBLE_NEAR_THRESHOLD 0.001

/*
    @brief structure of level definition

    @param mid_lookahead        lookahead depth in midgame
    @param mid_mpct             MPC (Multi-ProbCut) constant for midgame search
    @param complete0            lookahead depth in endgame search 1/5
    @param complete0_mpct       MPC (Multi-ProbCut) constant for endgame search 1/5
    @param complete1            lookahead depth in endgame search 2/5
    @param complete1_mpct       MPC (Multi-ProbCut) constant for endgame search 2/5
    @param complete2            lookahead depth in endgame search 3/5
    @param complete2_mpct       MPC (Multi-ProbCut) constant for endgame search 3/5
    @param complete3            lookahead depth in endgame search 4/5
    @param complete3_mpct       MPC (Multi-ProbCut) constant for endgame search 4/5
    @param complete4            lookahead depth in endgame search 5/5
    @param complete5_mpct       MPC (Multi-ProbCut) constant for endgame search 5/5
*/
struct Level{
    int mid_lookahead;
    double mid_mpct;
    int complete0;
    double complete0_mpct;
    int complete1;
    double complete1_mpct;
    int complete2;
    double complete2_mpct;
    int complete3;
    double complete3_mpct;
    int complete4;
    double complete4_mpct;
};

/*
    @brief level definition
*/
constexpr Level level_definition[N_LEVEL] = {
    {0, NOMPC, 0, NOMPC, NODEPTH, NOMPC, NODEPTH, NOMPC, NODEPTH, NOMPC, NODEPTH, NOMPC},
    {1, NOMPC, 2, NOMPC, NODEPTH, NOMPC, NODEPTH, NOMPC, NODEPTH, NOMPC, NODEPTH, NOMPC},
    {2, NOMPC, 4, NOMPC, NODEPTH, NOMPC, NODEPTH, NOMPC, NODEPTH, NOMPC, NODEPTH, NOMPC},
    {3, NOMPC, 6, NOMPC, NODEPTH, NOMPC, NODEPTH, NOMPC, NODEPTH, NOMPC, NODEPTH, NOMPC},
    {4, NOMPC, 8, NOMPC, NODEPTH, NOMPC, NODEPTH, NOMPC, NODEPTH, NOMPC, NODEPTH, NOMPC},

    {5, NOMPC, 10, NOMPC, NODEPTH, NOMPC, NODEPTH, NOMPC, NODEPTH, NOMPC, NODEPTH, NOMPC},
    {6, NOMPC, 12, NOMPC, NODEPTH, NOMPC, NODEPTH, NOMPC, NODEPTH, NOMPC, NODEPTH, NOMPC},
    {7, NOMPC, 14, NOMPC, NODEPTH, NOMPC, NODEPTH, NOMPC, NODEPTH, NOMPC, NODEPTH, NOMPC},
    {8, NOMPC, 16, NOMPC, NODEPTH, NOMPC, NODEPTH, NOMPC, NODEPTH, NOMPC, NODEPTH, NOMPC},
    {9, NOMPC, 18, NOMPC, NODEPTH, NOMPC, NODEPTH, NOMPC, NODEPTH, NOMPC, NODEPTH, NOMPC},

    {10, NOMPC, 20, NOMPC, NODEPTH, NOMPC, NODEPTH, NOMPC, NODEPTH, NOMPC, NODEPTH, NOMPC},
    {11, MPC_81, 24, MPC_98, 21, NOMPC, NODEPTH, NOMPC, NODEPTH, NOMPC, NODEPTH, NOMPC},
    {12, MPC_81, 24, MPC_98, 21, NOMPC, NODEPTH, NOMPC, NODEPTH, NOMPC, NODEPTH, NOMPC},
    {13, MPC_81, 27, MPC_95, 23, MPC_98, 21, NOMPC, NODEPTH, NOMPC, NODEPTH, NOMPC},
    {14, MPC_81, 27, MPC_95, 23, MPC_98, 21, NOMPC, NODEPTH, NOMPC, NODEPTH, NOMPC},

    {15, MPC_81, 27, MPC_98, 24, MPC_99, 22, NOMPC, NODEPTH, NOMPC, NODEPTH, NOMPC},
    {16, MPC_81, 27, MPC_98, 24, MPC_99, 22, NOMPC, NODEPTH, NOMPC, NODEPTH, NOMPC},
    {17, MPC_81, 27, MPC_98, 24, MPC_99, 22, NOMPC, NODEPTH, NOMPC, NODEPTH, NOMPC},
    {18, MPC_81, 28, MPC_95, 26, MPC_98, 24, MPC_99, 22, NOMPC, NODEPTH, NOMPC},
    {19, MPC_81, 28, MPC_95, 26, MPC_98, 24, MPC_99, 22, NOMPC, NODEPTH, NOMPC},

    {20, MPC_81, 30, MPC_95, 28, MPC_98, 26, MPC_99, 24, NOMPC, NODEPTH, NOMPC},
    {21, MPC_81, 30, MPC_95, 28, MPC_98, 26, MPC_99, 24, NOMPC, NODEPTH, NOMPC},
    {22, MPC_81, 30, MPC_95, 38, MPC_98, 26, MPC_99, 24, NOMPC, NODEPTH, NOMPC},
    {23, MPC_81, 32, MPC_95, 30, MPC_98, 28, MPC_99, 26, NOMPC, NODEPTH, NOMPC},
    {24, MPC_81, 32, MPC_95, 30, MPC_98, 28, MPC_99, 26, NOMPC, NODEPTH, NOMPC},
    
    {25, MPC_81, 32, MPC_95, 30, MPC_98, 28, MPC_99, 26, NOMPC, NODEPTH, NOMPC},
    {26, MPC_81, 32, MPC_95, 30, MPC_98, 28, MPC_99, 26, NOMPC, NODEPTH, NOMPC},
    {27, MPC_81, 34, MPC_95, 32, MPC_98, 30, MPC_99, 28, NOMPC, NODEPTH, NOMPC},
    {28, MPC_81, 34, MPC_95, 32, MPC_98, 30, MPC_99, 28, NOMPC, NODEPTH, NOMPC},
    {29, MPC_81, 36, MPC_95, 34, MPC_98, 32, MPC_99, 30, NOMPC, NODEPTH, NOMPC},

    {30, MPC_81, 36, MPC_95, 34, MPC_98, 32, MPC_99, 30, NOMPC, NODEPTH, NOMPC},
    {31, MPC_81, 38, MPC_81, 36, MPC_95, 34, MPC_98, 32, MPC_99, 30, NOMPC},
    {32, MPC_81, 38, MPC_81, 36, MPC_95, 34, MPC_98, 32, MPC_99, 30, NOMPC},
    {33, MPC_81, 38, MPC_95, 36, MPC_98, 33, MPC_99, 30, NOMPC, NODEPTH, NOMPC},
    {34, MPC_81, 38, MPC_95, 36, MPC_98, 33, MPC_99, 30, NOMPC, NODEPTH, NOMPC},

    {35, MPC_81, 42, MPC_81, 39, MPC_95, 36, MPC_98, 33, MPC_99, 30, NOMPC},
    {36, MPC_81, 43, MPC_81, 40, MPC_95, 37, MPC_98, 34, MPC_99, 31, NOMPC},
    {37, MPC_81, 44, MPC_81, 41, MPC_95, 38, MPC_98, 35, MPC_99, 32, NOMPC},
    {38, MPC_81, 45, MPC_81, 42, MPC_95, 39, MPC_98, 36, MPC_99, 33, NOMPC},
    {39, MPC_81, 46, MPC_81, 43, MPC_95, 40, MPC_98, 37, MPC_99, 34, NOMPC},

    {40, MPC_81, 47, MPC_81, 44, MPC_95, 41, MPC_98, 38, MPC_99, 35, NOMPC},
    {41, MPC_81, 48, MPC_81, 45, MPC_95, 42, MPC_98, 39, MPC_99, 36, NOMPC},
    {42, MPC_81, 49, MPC_81, 46, MPC_95, 43, MPC_98, 40, MPC_99, 37, NOMPC},
    {43, MPC_81, 50, MPC_81, 47, MPC_95, 44, MPC_98, 41, MPC_99, 38, NOMPC},
    {44, MPC_81, 51, MPC_81, 48, MPC_95, 45, MPC_98, 42, MPC_99, 39, NOMPC},

    {45, MPC_81, 52, MPC_81, 49, MPC_95, 46, MPC_98, 43, MPC_99, 40, NOMPC},
    {46, MPC_81, 53, MPC_81, 50, MPC_95, 47, MPC_98, 44, MPC_99, 41, NOMPC},
    {47, MPC_81, 54, MPC_81, 51, MPC_95, 48, MPC_98, 45, MPC_99, 42, NOMPC},
    {48, MPC_81, 55, MPC_81, 52, MPC_95, 49, MPC_98, 46, MPC_99, 43, NOMPC},
    {49, MPC_81, 56, MPC_81, 53, MPC_95, 50, MPC_98, 47, MPC_99, 44, NOMPC},

    {50, MPC_81, 57, MPC_81, 54, MPC_95, 51, MPC_98, 48, MPC_99, 45, NOMPC},
    {51, MPC_81, 58, MPC_81, 55, MPC_95, 52, MPC_98, 49, MPC_99, 46, NOMPC},
    {52, MPC_81, 59, MPC_81, 56, MPC_95, 53, MPC_98, 50, MPC_99, 47, NOMPC},
    {53, MPC_81, 60, MPC_81, 57, MPC_95, 54, MPC_98, 51, MPC_99, 48, NOMPC},
    {54, MPC_81, 60, MPC_81, 58, MPC_95, 55, MPC_98, 52, MPC_99, 49, NOMPC},
    
    {55, MPC_81, 60, MPC_81, 58, MPC_95, 56, MPC_98, 53, MPC_99, 50, NOMPC},
    {56, MPC_81, 60, MPC_81, 58, MPC_95, 56, MPC_98, 53, MPC_99, 51, NOMPC},
    {57, MPC_81, 60, MPC_95, 58, MPC_98, 54, MPC_99, 52, NOMPC, NODEPTH, NOMPC},
    {58, MPC_81, 60, MPC_95, 58, MPC_98, 55, MPC_99, 53, NOMPC, NODEPTH, NOMPC},
    {59, MPC_81, 60, MPC_95, 58, MPC_98, 56, MPC_99, 54, NOMPC, NODEPTH, NOMPC},

    {60, MPC_81, 60, NOMPC, NODEPTH, NOMPC, NODEPTH, NOMPC, NODEPTH, NOMPC, NODEPTH, NOMPC}
};

/*
    @brief Get level information

    get all information

    @param level                level to search
    @param n_moves              ply
    @param is_mid_search        flag to store this search is midgame search or not
    @param depth                integer to store the depth
    @param use_mpc              flag to store this search uses MPC (Multi-ProbCut)
    @param mpct                 value for MPC (Multi-ProbCut) constant
*/
void get_level(int level, int n_moves, bool *is_mid_search, int *depth, bool *use_mpc, double *mpct){
    if (level <= 0){
        *is_mid_search = true;
        *depth = 0;
        *use_mpc = false;
        *mpct = NOMPC;
        return;
    } else if (level > 60)
        level = 60;
    Level level_status = level_definition[level];
    int n_empties = 60 - n_moves;
    if (n_empties > level_status.complete0){
        *is_mid_search = true;
        *depth = level_status.mid_lookahead;
        *use_mpc = level_status.mid_mpct != NOMPC;
        *mpct = level_status.mid_mpct;
    } else {
        *is_mid_search = false;
        *depth = n_empties;
        if (n_empties > level_status.complete1){
            *use_mpc = level_status.complete0_mpct != NOMPC;
            *mpct = level_status.complete0_mpct;
        } else if (n_empties > level_status.complete2){
            *use_mpc = level_status.complete1_mpct != NOMPC;
            *mpct = level_status.complete1_mpct;
        } else if (n_empties > level_status.complete3){
            *use_mpc = level_status.complete2_mpct != NOMPC;
            *mpct = level_status.complete2_mpct;
        } else if (n_empties > level_status.complete4){
            *use_mpc = level_status.complete3_mpct != NOMPC;
            *mpct = level_status.complete3_mpct;
        } else{
            *use_mpc = level_status.complete4_mpct != NOMPC;
            *mpct = level_status.complete4_mpct;
        }
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
    Level level_status = level_definition[level];
    int n_empties = 60 - n_moves;
    if (n_empties < level_status.complete0){
        return level_status.mid_mpct != NOMPC;
    } else {
        if (n_empties > level_status.complete1){
            return level_status.complete0_mpct != NOMPC;
        } else if (n_empties > level_status.complete2){
            return level_status.complete1_mpct != NOMPC;
        } else if (n_empties > level_status.complete3){
            return level_status.complete2_mpct != NOMPC;
        } else if (n_empties > level_status.complete4){
            return level_status.complete3_mpct != NOMPC;
        } else{
            return level_status.complete4_mpct != NOMPC;
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
    level = std::max(0, std::min(60, level));
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
    if (level <= 0){
        return true;
    } else if (level > 60)
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
    if (level_definition[level].complete0_mpct == NOMPC)
        return level_definition[level].complete0;
    if (level_definition[level].complete1_mpct == NOMPC)
        return level_definition[level].complete1;
    if (level_definition[level].complete2_mpct == NOMPC)
        return level_definition[level].complete2;
    if (level_definition[level].complete3_mpct == NOMPC)
        return level_definition[level].complete3;
    return level_definition[level].complete4;
}

/*
    @brief check 2 values are almost same

    @param a                    value 1/2
    @param b                    value 2/2
    @return a is near to b?
*/
bool double_near(double a, double b){
    return fabs(a - b) < DOUBLE_NEAR_THRESHOLD;
}

/*
    @brief find the probability of search

    @param mpct                 MPC (Multi-ProbCut) constant
    @return the probability in [%]
*/
int calc_probability(double mpct){
    if (double_near(MPC_81, mpct)){
        return 81;
    }
    if (double_near(MPC_95, mpct)){
        return 95;
    }
    if (double_near(MPC_98, mpct)){
        return 98;
    }
    if (double_near(MPC_99, mpct)){
        return 99;
    }
    if (double_near(NOMPC, mpct)){
        return 100;
    }
    return -1;
}