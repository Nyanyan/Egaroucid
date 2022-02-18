#pragma once
#include <iostream>

using namespace std;

#define N_LEVEL 61
#define N_MOVES 60
#define MPC_88 1.18
#define MPC_95 1.64
#define MPC_98 2.05
#define MPC_99 2.33
#define NOMPC 10000.0
#define NODEPTH 100

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

    {10, MPC_88, 20, MPC_98, 18, NOMPC, NODEPTH, NOMPC, NODEPTH, NOMPC, NODEPTH, NOMPC},
    {11, MPC_88, 20, MPC_98, 18, NOMPC, NODEPTH, NOMPC, NODEPTH, NOMPC, NODEPTH, NOMPC},
    {12, MPC_88, 20, MPC_99, 18, NOMPC, NODEPTH, NOMPC, NODEPTH, NOMPC, NODEPTH, NOMPC},
    {13, MPC_88, 20, MPC_99, 18, NOMPC, NODEPTH, NOMPC, NODEPTH, NOMPC, NODEPTH, NOMPC},
    {14, MPC_88, 22, MPC_98, 20, NOMPC, NODEPTH, NOMPC, NODEPTH, NOMPC, NODEPTH, NOMPC},

    {15, MPC_88, 22, MPC_98, 20, NOMPC, NODEPTH, NOMPC, NODEPTH, NOMPC, NODEPTH, NOMPC},
    {16, MPC_88, 22, MPC_99, 20, NOMPC, NODEPTH, NOMPC, NODEPTH, NOMPC, NODEPTH, NOMPC},
    {17, MPC_88, 22, MPC_99, 20, NOMPC, NODEPTH, NOMPC, NODEPTH, NOMPC, NODEPTH, NOMPC},
    {18, MPC_88, 24, MPC_98, 22, MPC_99, 20, NOMPC, NODEPTH, NOMPC, NODEPTH, NOMPC},
    {19, MPC_88, 24, MPC_98, 22, MPC_99, 20, NOMPC, NODEPTH, NOMPC, NODEPTH, NOMPC},

    {20, MPC_88, 26, MPC_95, 24, MPC_98, 22, MPC_99, 20, NOMPC, NODEPTH, NOMPC},
    {21, MPC_88, 26, MPC_95, 24, MPC_98, 22, MPC_99, 20, NOMPC, NODEPTH, NOMPC},
    {22, MPC_88, 28, MPC_95, 26, MPC_98, 24, MPC_99, 22, NOMPC, NODEPTH, NOMPC},
    {23, MPC_88, 28, MPC_95, 26, MPC_98, 24, MPC_99, 22, NOMPC, NODEPTH, NOMPC},
    {24, MPC_88, 30, MPC_88, 28, MPC_95, 26, MPC_98, 24, MPC_99, 22, NOMPC},
    
    {25, MPC_88, 30, MPC_95, 28, MPC_98, 26, MPC_99, 24, NOMPC, NODEPTH, NOMPC},
    {26, MPC_88, 32, MPC_88, 30, MPC_95, 28, MPC_98, 26, MPC_99, 24, NOMPC},
    {27, MPC_88, 32, MPC_95, 29, MPC_98, 26, MPC_98, 24, NOMPC, NODEPTH, NOMPC},
    {28, MPC_88, 34, MPC_88, 32, MPC_95, 29, MPC_98, 26, MPC_98, 24, NOMPC},
    {29, MPC_88, 34, MPC_98, 31, MPC_99, 28, MPC_98, 26, NOMPC, NODEPTH, NOMPC},

    {30, MPC_88, 36, MPC_88, 34, MPC_95, 31, MPC_98, 28, MPC_99, 26, NOMPC},
    {31, MPC_88, 36, MPC_98, 34, MPC_99, 30, MPC_99, 26, NOMPC, NODEPTH, NOMPC},
    {32, MPC_88, 38, MPC_88, 36, MPC_98, 34, MPC_99, 30, MPC_99, 26, NOMPC},
    {33, MPC_88, 38, MPC_95, 36, MPC_98, 32, MPC_99, 28, NOMPC, NODEPTH, NOMPC},
    {34, MPC_88, 40, MPC_88, 38, MPC_95, 36, MPC_98, 32, MPC_99, 28, NOMPC},

    {35, MPC_88, 40, MPC_95, 36, MPC_98, 32, MPC_99, 28, NOMPC, NODEPTH, NOMPC},
    {36, MPC_88, 42, MPC_88, 40, MPC_95, 36, MPC_98, 32, MPC_99, 28, NOMPC},
    {37, MPC_88, 42, MPC_95, 38, MPC_98, 34, MPC_99, 30, NOMPC, NODEPTH, NOMPC},
    {38, MPC_88, 44, MPC_88, 42, MPC_95, 38, MPC_98, 34, MPC_99, 30, NOMPC},
    {39, MPC_88, 44, MPC_95, 40, MPC_98, 35, MPC_99, 30, NOMPC, NODEPTH, NOMPC},

    {40, MPC_88, 46, MPC_88, 44, MPC_95, 40, MPC_98, 35, MPC_99, 30, NOMPC},
    {41, MPC_88, 46, MPC_95, 41, MPC_98, 36, MPC_99, 31, NOMPC, NODEPTH, NOMPC},
    {42, MPC_88, 48, MPC_88, 46, MPC_95, 41, MPC_98, 36, MPC_99, 31, NOMPC},
    {43, MPC_88, 48, MPC_95, 43, MPC_98, 38, MPC_99, 33, NOMPC, NODEPTH, NOMPC},
    {44, MPC_88, 50, MPC_88, 48, MPC_95, 43, MPC_98, 38, MPC_99, 33, NOMPC},

    {45, MPC_88, 52, MPC_88, 50, MPC_95, 45, MPC_98, 40, MPC_99, 35, NOMPC},
    {46, MPC_88, 54, MPC_88, 50, MPC_95, 45, MPC_98, 40, MPC_99, 35, NOMPC},
    {47, MPC_88, 56, MPC_88, 52, MPC_95, 47, MPC_98, 42, MPC_99, 37, NOMPC},
    {48, MPC_88, 58, MPC_88, 54, MPC_95, 49, MPC_98, 44, MPC_99, 39, NOMPC},
    {49, MPC_88, 60, MPC_88, 56, MPC_95, 51, MPC_98, 46, MPC_99, 41, NOMPC},

    {50, MPC_88, 60, MPC_88, 58, MPC_95, 53, MPC_98, 48, MPC_99, 43, NOMPC},
    {51, MPC_88, 60, MPC_95, 55, MPC_98, 50, MPC_99, 45, NOMPC, NODEPTH, NOMPC},
    {52, MPC_88, 60, MPC_95, 57, MPC_98, 52, MPC_99, 47, NOMPC, NODEPTH, NOMPC},
    {53, MPC_88, 60, MPC_95, 59, MPC_98, 54, MPC_99, 49, NOMPC, NODEPTH, NOMPC},
    {54, MPC_88, 60, MPC_98, 56, MPC_99, 51, NOMPC, NODEPTH, NOMPC, NODEPTH, NOMPC},
    
    {55, MPC_88, 60, MPC_98, 58, MPC_99, 53, NOMPC, NODEPTH, NOMPC, NODEPTH, NOMPC},
    {56, MPC_88, 60, MPC_99, 55, NOMPC, NODEPTH, NOMPC, NODEPTH, NOMPC, NODEPTH, NOMPC},
    {57, MPC_88, 60, MPC_99, 57, NOMPC, NODEPTH, NOMPC, NODEPTH, NOMPC, NODEPTH, NOMPC},
    {58, MPC_88, 60, MPC_99, 58, NOMPC, NODEPTH, NOMPC, NODEPTH, NOMPC, NODEPTH, NOMPC},
    {59, MPC_88, 60, MPC_99, 59, NOMPC, NODEPTH, NOMPC, NODEPTH, NOMPC, NODEPTH, NOMPC},

    {60, MPC_88, 60, NOMPC, NODEPTH, NOMPC, NODEPTH, NOMPC, NODEPTH, NOMPC, NODEPTH, NOMPC}
};

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

void get_level_depth(int level, int *mid_depth, int *end_depth){
    level = max(0, min(60, level));
    *mid_depth = level_definition[level].mid_lookahead;
    *end_depth = level_definition[level].complete0;
}