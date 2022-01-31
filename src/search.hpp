#pragma once
#include <iostream>
#include <fstream>
#include <math.h>
#include "setting.hpp"
#include "common.hpp"
#include "board.hpp"
#include "evaluate.hpp"
#include "transpose_table.hpp"
#if USE_MULTI_THREAD
    #include <mutex>
#endif

using namespace std;

#define search_epsilon 1
constexpr int cache_hit = 100;
constexpr int cache_both = 100;
constexpr int cache_now = 100;
constexpr int parity_vacant_bonus = 5;
constexpr int canput_bonus = 10;

#define mpc_min_depth 3
#define mpc_max_depth 20
#define mpc_min_depth_final 7
#define mpc_max_depth_final 40

#define simple_mid_threshold 3
#define simple_end_threshold 7
#define simple_end_threshold2 15

#define po_max_depth 15

#define enhanced_mtd_weight 2

const int cell_weight[hw2] = {
    18,  4,  16, 12, 12, 16,  4, 18,
     4,  2,   6,  8,  8,  6,  2,  4,
    16,  6,  14, 10, 10, 14,  6, 16,
    12,  8,  10,  0,  0, 10,  8, 12,
    12,  8,  10,  0,  0, 10,  8, 12,
    16,  6,  14, 10, 10, 14,  6, 16,
     4,  2,   6,  8,  8,  6,  2,  4,
    18,  4,  16, 12, 12, 16,  4, 18
};

const int mpcd[41] = {
    0, 1, 0, 1, 2, 3, 2, 3, 4, 3, 
    4, 3, 4, 5, 4, 5, 6, 5, 6, 7, 
    6, 7, 6, 7, 8, 7, 8, 9, 8, 9, 
    8, 9, 10, 9, 10, 11, 10, 11, 10, 11,
    12
};
#if USE_MID_SMOOTH
    double mpcsd[n_phases][mpc_max_depth - mpc_min_depth + 1]={

    };
    double mpcsd_final[mpc_max_depth_final - mpc_min_depth_final + 1] = {

    };
#else
    double mpcsd[n_phases][mpc_max_depth - mpc_min_depth + 1]={
        {0.979, 0.612, 0.962, 0.861, 0.932, 1.335, 1.286, 1.506, 0.497, 0.508, 0.576, 0.508, 0.678, 0.476, 0.69, 1.013, 0.875, 1.182},
        {1.197, 1.789, 1.931, 1.492, 1.847, 1.219, 2.335, 3.859, 0.752, 1.133, 1.647, 1.166, 1.388, 1.945, 1.666, 1.972, 1.726, 3.039},
        {2.203, 2.161, 2.287, 1.533, 2.061, 1.227, 2.252, 1.922, 1.343, 1.289, 1.972, 1.188, 1.29, 1.056, 1.192, 1.143, 1.201, 1.442},
        {2.587, 2.228, 1.585, 2.347, 1.663, 1.756, 2.667, 2.378, 1.257, 0.997, 0.922, 1.362, 1.152, 1.21, 2.015, 1.816, 1.44, 1.343},
        {2.404, 2.117, 1.631, 2.444, 1.895, 1.424, 3.231, 1.679, 1.362, 1.287, 1.362, 1.641, 1.434, 1.285, 1.393, 2.086, 1.5, 1.994},
        {2.699, 2.923, 2.127, 1.887, 2.788, 1.864, 5.271, 2.007, 2.24, 1.688, 1.72, 1.993, 1.219, 1.442, 1.982, 2.14, 1.323, 1.786},
        {3.564, 2.84, 2.313, 2.487, 2.813, 2.137, 2.807, 2.736, 1.686, 1.994, 1.458, 2.432, 2.091, 1.453, 1.816, 2.188, 1.812, 1.971},
        {2.629, 2.45, 1.572, 3.436, 2.031, 2.052, 2.178, 2.129, 2.134, 1.835, 1.166, 2.284, 1.735, 1.685, 1.701, 2.236, 1.856, 2.657},
        {2.439, 3.049, 2.501, 4.095, 2.923, 2.289, 2.993, 2.306, 1.812, 1.847, 1.451, 2.487, 1.562, 1.83, 1.799, 2.495, 2.264, 2.94},
        {2.326, 2.661, 1.853, 2.907, 3.141, 2.188, 3.709, 3.396, 2.059, 2.007, 1.812, 1.795, 1.758, 1.644, 2.21, 2.373, 2.765, 2.923},
        {3.825, 3.655, 2.299, 2.486, 3.08, 2.064, 2.731, 3.808, 1.679, 2.008, 2.195, 2.264, 1.771, 1.687, 2.942, 2.649, 1.952, 0.0},
        {3.302, 2.699, 2.276, 2.979, 3.951, 3.049, 4.907, 5.179, 2.529, 2.161, 2.239, 3.11, 1.676, 0.0, 0.0, 0.0, 0.0, 0.0},
        {3.119, 2.489, 3.658, 3.911, 4.529, 3.338, 5.929, 4.196, 2.69, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {4.603, 3.06, 4.243, 5.954, 5.282, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {3.976, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
    };
    double mpcsd_final[mpc_max_depth_final - mpc_min_depth_final + 1] = {
        4.242, 4.112, 5.037, 4.516, 5.223, 5.066, 5.01, 5.275, 4.738, 4.468, 4.661, 4.208, 4.21, 4.171, 4.102, 4.482, 4.314, 4.144, 3.92, 3.623, 3.452, 3.785, 3.659, 3.972, 4.983, 5.686, 6.445, 7.514, 7.989, 8.507, 9.284, 9.866, 10.34, 10.877
    };
#endif
unsigned long long can_be_flipped[hw2];
vector<int> vacant_lst;

struct search_result{
    int policy;
    int value;
    int depth;
    int nps;
};

struct enhanced_mtd{
    int policy;
    int error;
    int l;
    int u;
    board b;
};

bool operator< (const enhanced_mtd &elem1, const enhanced_mtd &elem2){
    return elem1.b.v + elem1.error * enhanced_mtd_weight < elem2.b.v + elem2.error * enhanced_mtd_weight;
};

int cmp_vacant(int p, int q){
    return cell_weight[p] > cell_weight[q];
}

inline int move_ordering(board *b, const int hash, const int policy){
    int v = transpose_table.child_get_now(b, hash, policy);
    if (v == -child_inf){
        v = transpose_table.child_get_prev(b, hash, policy);
        if (v == -child_inf){
            mobility mob;
            calc_flip(&mob, b, policy);
            b->move(&mob);
            v = -mid_evaluate(b);
            b->undo(&mob);
        } else
            v += cache_hit;
    } else
        v += cache_hit + cache_both;
    //v += cell_weight[policy];
    return v;
}

inline void move_ordering_eval(board *b){
    b->v = -mid_evaluate(b);
}

inline bool stability_cut(board *b, int *alpha, int *beta){
    int stab[2];
    calc_stability(b, &stab[0], &stab[1]);
    *alpha = max(*alpha, 2 * stab[b->p] - hw2);
    *beta = min(*beta, hw2 - 2 * stab[1 - b->p]);
    return *alpha >= *beta;
}

inline int calc_canput_exact(board *b){
    return pop_count_ull(b->mobility_ull());
}

bool move_ordering_sort(pair<int, board> &a, pair<int, board> &b){
    return a.second.v > b.second.v;
}

bool move_ordering_sort_int_int(pair<int, int> &a, pair<int, int> &b){
    return a.second > b.second;
}
