/*
    Egaroucid Project

    @file level.hpp
        definition of Egaroucid's level
    @date 2021-2025
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once
#include <iostream>
#include <cmath>
#include "common.hpp"

/*
    @brief definition of level categories

    0                   - LIGHT_LEVEL           : light
    LIGHT_LEVEL         - STANDARD_MAX_LEVEL    : standard
    STANDARD_MAX_LEVEL  - PRAGMATIC_MAX_LEVEL   : pragmatic
    PRAGMATIC_MAX_LEVEL - ACCURATE_MAX_LEVEL    : accurate
    ACCURATE_MAX_LEVEL  - 60                    : danger
*/
constexpr int ACCURATE_MAX_LEVEL = 40;
constexpr int PRAGMATIC_MAX_LEVEL = 25;
constexpr int STANDARD_MAX_LEVEL = 21;
constexpr int LIGHT_LEVEL = 15;

constexpr int DEFAULT_LEVEL = 21;

/*
    @brief constants for level definition
*/
constexpr int N_LEVEL = 61; // [0, 60] (0 is not used)
constexpr int NODEPTH = 100;

constexpr int N_SELECTIVITY_LEVEL = 7;
constexpr int MPC_74_LEVEL = 0;
constexpr int MPC_88_LEVEL = 1;
constexpr int MPC_93_LEVEL = 2;
constexpr int MPC_98_LEVEL = 3;
constexpr int MPC_99_LEVEL = 4;
constexpr int MPC_999_LEVEL = 5;
constexpr int MPC_100_LEVEL = 6;
constexpr double SELECTIVITY_PERCENTAGE[N_SELECTIVITY_LEVEL] = {74, 88, 93, 98, 99, 99.9, 100}; // percent

constexpr int MAX_LEVEL = (N_LEVEL - 1);
constexpr int LEVEL_TYPE_BOOK = 1000;

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
    {11, MPC_74_LEVEL, 25, MPC_98_LEVEL, 23, MPC_99_LEVEL, 21, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL},
    {12, MPC_74_LEVEL, 25, MPC_98_LEVEL, 23, MPC_99_LEVEL, 21, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL},
    {13, MPC_74_LEVEL, 27, MPC_93_LEVEL, 25, MPC_98_LEVEL, 23, MPC_99_LEVEL, 21, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL},
    {14, MPC_74_LEVEL, 27, MPC_93_LEVEL, 25, MPC_98_LEVEL, 23, MPC_99_LEVEL, 21, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL},

    {15, MPC_74_LEVEL, 27, MPC_98_LEVEL, 24, MPC_99_LEVEL, 22, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL},
    {16, MPC_74_LEVEL, 27, MPC_98_LEVEL, 24, MPC_99_LEVEL, 22, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL},
    {17, MPC_74_LEVEL, 27, MPC_98_LEVEL, 24, MPC_99_LEVEL, 22, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL},
    {18, MPC_74_LEVEL, 29, MPC_93_LEVEL, 27, MPC_98_LEVEL, 24, MPC_99_LEVEL, 22, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL},
    {19, MPC_74_LEVEL, 29, MPC_93_LEVEL, 27, MPC_98_LEVEL, 24, MPC_99_LEVEL, 22, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL},

    {20, MPC_74_LEVEL, 30, MPC_93_LEVEL, 28, MPC_98_LEVEL, 26, MPC_99_LEVEL, 24, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL},
    {21, MPC_74_LEVEL, 30, MPC_93_LEVEL, 28, MPC_98_LEVEL, 26, MPC_99_LEVEL, 24, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL},
    {22, MPC_74_LEVEL, 30, MPC_93_LEVEL, 38, MPC_98_LEVEL, 26, MPC_99_LEVEL, 24, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL},
    {23, MPC_74_LEVEL, 32, MPC_93_LEVEL, 30, MPC_98_LEVEL, 28, MPC_99_LEVEL, 26, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL},
    {24, MPC_74_LEVEL, 32, MPC_93_LEVEL, 30, MPC_98_LEVEL, 28, MPC_99_LEVEL, 26, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL},
    
    {25, MPC_74_LEVEL, 34, MPC_88_LEVEL, 32, MPC_93_LEVEL, 30, MPC_98_LEVEL, 28, MPC_99_LEVEL, 26, MPC_100_LEVEL},
    {26, MPC_74_LEVEL, 34, MPC_88_LEVEL, 32, MPC_93_LEVEL, 30, MPC_98_LEVEL, 28, MPC_99_LEVEL, 26, MPC_100_LEVEL},
    {27, MPC_74_LEVEL, 36, MPC_88_LEVEL, 34, MPC_93_LEVEL, 32, MPC_98_LEVEL, 30, MPC_99_LEVEL, 28, MPC_100_LEVEL},
    {28, MPC_74_LEVEL, 36, MPC_88_LEVEL, 34, MPC_93_LEVEL, 32, MPC_98_LEVEL, 30, MPC_99_LEVEL, 28, MPC_100_LEVEL},
    {29, MPC_74_LEVEL, 36, MPC_93_LEVEL, 34, MPC_98_LEVEL, 32, MPC_99_LEVEL, 30, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL},

    {30, MPC_74_LEVEL, 36, MPC_93_LEVEL, 34, MPC_98_LEVEL, 32, MPC_99_LEVEL, 30, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL},
    {31, MPC_74_LEVEL, 38, MPC_88_LEVEL, 36, MPC_93_LEVEL, 34, MPC_98_LEVEL, 32, MPC_99_LEVEL, 30, MPC_100_LEVEL},
    {32, MPC_74_LEVEL, 38, MPC_88_LEVEL, 36, MPC_93_LEVEL, 34, MPC_98_LEVEL, 32, MPC_99_LEVEL, 30, MPC_100_LEVEL},
    {33, MPC_74_LEVEL, 38, MPC_93_LEVEL, 36, MPC_98_LEVEL, 34, MPC_99_LEVEL, 32, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL},
    {34, MPC_74_LEVEL, 38, MPC_93_LEVEL, 36, MPC_98_LEVEL, 34, MPC_99_LEVEL, 32, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL},

    {35, MPC_74_LEVEL, 40, MPC_88_LEVEL, 38, MPC_93_LEVEL, 36, MPC_98_LEVEL, 34, MPC_99_LEVEL, 32, MPC_100_LEVEL},
    {36, MPC_74_LEVEL, 40, MPC_88_LEVEL, 38, MPC_93_LEVEL, 36, MPC_98_LEVEL, 34, MPC_99_LEVEL, 32, MPC_100_LEVEL},
    {37, MPC_74_LEVEL, 40, MPC_93_LEVEL, 36, MPC_98_LEVEL, 36, MPC_99_LEVEL, 34, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL},
    {38, MPC_74_LEVEL, 40, MPC_93_LEVEL, 38, MPC_98_LEVEL, 36, MPC_99_LEVEL, 34, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL},
    {39, MPC_74_LEVEL, 42, MPC_88_LEVEL, 40, MPC_93_LEVEL, 38, MPC_98_LEVEL, 36, MPC_99_LEVEL, 34, MPC_100_LEVEL},

    {40, MPC_74_LEVEL, 42, MPC_88_LEVEL, 40, MPC_93_LEVEL, 38, MPC_98_LEVEL, 36, MPC_99_LEVEL, 34, MPC_100_LEVEL},
    {41, MPC_74_LEVEL, 42, MPC_93_LEVEL, 40, MPC_98_LEVEL, 38, MPC_99_LEVEL, 36, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL},
    {42, MPC_74_LEVEL, 42, MPC_93_LEVEL, 40, MPC_98_LEVEL, 38, MPC_99_LEVEL, 36, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL},
    {43, MPC_74_LEVEL, 44, MPC_88_LEVEL, 42, MPC_93_LEVEL, 40, MPC_98_LEVEL, 38, MPC_99_LEVEL, 36, MPC_100_LEVEL},
    {44, MPC_74_LEVEL, 44, MPC_88_LEVEL, 42, MPC_93_LEVEL, 40, MPC_98_LEVEL, 38, MPC_99_LEVEL, 36, MPC_100_LEVEL},

    {45, MPC_74_LEVEL, 44, MPC_93_LEVEL, 42, MPC_98_LEVEL, 40, MPC_99_LEVEL, 38, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL},
    {46, MPC_74_LEVEL, 44, MPC_93_LEVEL, 42, MPC_98_LEVEL, 40, MPC_99_LEVEL, 38, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL},
    {47, MPC_74_LEVEL, 46, MPC_88_LEVEL, 44, MPC_93_LEVEL, 42, MPC_98_LEVEL, 40, MPC_99_LEVEL, 38, MPC_100_LEVEL},
    {48, MPC_74_LEVEL, 46, MPC_88_LEVEL, 44, MPC_93_LEVEL, 42, MPC_98_LEVEL, 40, MPC_99_LEVEL, 38, MPC_100_LEVEL},
    {49, MPC_74_LEVEL, 46, MPC_93_LEVEL, 44, MPC_98_LEVEL, 42, MPC_99_LEVEL, 40, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL},

    {50, MPC_74_LEVEL, 46, MPC_93_LEVEL, 44, MPC_98_LEVEL, 42, MPC_99_LEVEL, 40, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL},
    {51, MPC_74_LEVEL, 48, MPC_88_LEVEL, 46, MPC_93_LEVEL, 44, MPC_98_LEVEL, 42, MPC_99_LEVEL, 40, MPC_100_LEVEL},
    {52, MPC_74_LEVEL, 50, MPC_88_LEVEL, 48, MPC_93_LEVEL, 46, MPC_98_LEVEL, 44, MPC_99_LEVEL, 42, MPC_100_LEVEL},
    {53, MPC_74_LEVEL, 54, MPC_88_LEVEL, 52, MPC_93_LEVEL, 50, MPC_98_LEVEL, 48, MPC_99_LEVEL, 46, MPC_100_LEVEL},
    {54, MPC_74_LEVEL, 56, MPC_88_LEVEL, 54, MPC_93_LEVEL, 52, MPC_98_LEVEL, 50, MPC_99_LEVEL, 48, MPC_100_LEVEL},

    {55, MPC_74_LEVEL, 58, MPC_88_LEVEL, 56, MPC_93_LEVEL, 54, MPC_98_LEVEL, 52, MPC_99_LEVEL, 50, MPC_100_LEVEL},
    {56, MPC_88_LEVEL, 60, MPC_88_LEVEL, 58, MPC_93_LEVEL, 56, MPC_98_LEVEL, 54, MPC_99_LEVEL, 52, MPC_100_LEVEL},
    {57, MPC_88_LEVEL, 60, MPC_93_LEVEL, 58, MPC_98_LEVEL, 56, MPC_99_LEVEL, 54, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL},
    {58, MPC_93_LEVEL, 60, MPC_98_LEVEL, 58, MPC_99_LEVEL, 56, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL},
    {59, MPC_93_LEVEL, 60, MPC_99_LEVEL, 58, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL, NODEPTH, MPC_100_LEVEL},

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
void get_level(int level, int n_moves, bool *is_mid_search, int *depth, uint_fast8_t *mpc_level) {
    level = std::clamp(level, 1, 60);
    Level level_status = level_definition[level];
    int n_empties = 60 - n_moves;
    if (n_empties > level_status.complete0) {
        *is_mid_search = true;
        *depth = level_status.mid_lookahead;
        *mpc_level = level_status.mid_mpc_level;
    } else {
        *is_mid_search = false;
        *depth = n_empties;
        if (n_empties > level_status.complete1) {
            *mpc_level = level_status.complete0_mpc_level;
        } else if (n_empties > level_status.complete2) {
            *mpc_level = level_status.complete1_mpc_level;
        } else if (n_empties > level_status.complete3) {
            *mpc_level = level_status.complete2_mpc_level;
        } else if (n_empties > level_status.complete4) {
            *mpc_level = level_status.complete3_mpc_level;
        } else {
            *mpc_level = level_status.complete4_mpc_level;
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
bool get_level_use_mpc(int level, int n_moves) {
    level = std::clamp(level, 1, 60);
    Level level_status = level_definition[level];
    int n_empties = 60 - n_moves;
    if (n_empties < level_status.complete0) {
        return level_status.mid_mpc_level != MPC_100_LEVEL;
    } else {
        if (n_empties > level_status.complete1) {
            return level_status.complete0_mpc_level != MPC_100_LEVEL;
        } else if (n_empties > level_status.complete2) {
            return level_status.complete1_mpc_level != MPC_100_LEVEL;
        } else if (n_empties > level_status.complete3) {
            return level_status.complete2_mpc_level != MPC_100_LEVEL;
        } else if (n_empties > level_status.complete4) {
            return level_status.complete3_mpc_level != MPC_100_LEVEL;
        } else {
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
void get_level_depth(int level, int *mid_depth, int *end_depth) {
    level = std::clamp(level, 1, 60);
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
bool get_level_midsearch(int level, int n_moves) {
    level = std::clamp(level, 1, 60);
    Level level_status = level_definition[level];
    int n_empties = 60 - n_moves;
    if (n_empties > level_status.complete0) {
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
int get_level_endsearch_depth(int level) {
    return level_definition[level].complete0;
}

/*
    @brief Get level information

    get endgame complete (100%) search depth

    @param level                level to search
    @return endgame complete search depth
*/
int get_level_complete_depth(int level) {
    if (level_definition[level].complete0_mpc_level == MPC_100_LEVEL) {
        return level_definition[level].complete0;
    }
    if (level_definition[level].complete1_mpc_level == MPC_100_LEVEL) {
        return level_definition[level].complete1;
    }
    if (level_definition[level].complete2_mpc_level == MPC_100_LEVEL) {
        return level_definition[level].complete2;
    }
    if (level_definition[level].complete3_mpc_level == MPC_100_LEVEL) {
        return level_definition[level].complete3;
    }
    return level_definition[level].complete4;
}

int get_level_from_depth_mpc_level(int n_discs, int depth, int mpc_level) {
    int n_empties = HW2 - n_discs;
    if (depth >= n_empties) { // endgame search
        for (int level = 0; level < N_LEVEL; ++level) {
            if (level_definition[level].complete0 != NODEPTH && level_definition[level].complete0 >= depth && level_definition[level].complete0_mpc_level == mpc_level) {
                return level;
            }
            if (level_definition[level].complete1 != NODEPTH && level_definition[level].complete1 >= depth && level_definition[level].complete1_mpc_level == mpc_level) {
                return level;
            }
            if (level_definition[level].complete2 != NODEPTH && level_definition[level].complete2 >= depth && level_definition[level].complete2_mpc_level == mpc_level) {
                return level;
            }
            if (level_definition[level].complete3 != NODEPTH && level_definition[level].complete3 >= depth && level_definition[level].complete3_mpc_level == mpc_level) {
                return level;
            }
            if (level_definition[level].complete4 != NODEPTH && level_definition[level].complete4 >= depth && level_definition[level].complete4_mpc_level == mpc_level) {
                return level;
            }
        }
    }
    return depth; // level == midgame lookahead depth
}