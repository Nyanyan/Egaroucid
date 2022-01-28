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
#define mpc_max_depth 30
#define mpc_min_depth_final 7
#define mpc_max_depth_final 40

#define simple_mid_threshold 3
#define simple_end_threshold 8

#define po_max_depth 15

const int cell_weight[hw2] = {
    10, 3, 9, 7, 7, 9, 3, 10, 
    3, 2, 4, 5, 5, 4, 2, 3, 
    9, 4, 8, 6, 6, 8, 4, 9, 
    7, 5, 6, 0, 0, 6, 5, 7, 
    7, 5, 6, 0, 0, 6, 5, 7, 
    9, 4, 8, 6, 6, 8, 4, 9, 
    3, 2, 4, 5, 5, 4, 2, 3, 
    10, 3, 9, 7, 7, 9, 3, 10
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
        {0.845, 0.747, 0.951, 0.609, 0.786, 0.793, 0.893, 0.478, 0.504, 0.56, 0.692, 0.655, 0.649, 0.697, 0.507, 0.439, 0.607, 1.091, 1.204, 1.204, 1.317, 0.967, 0.98, 1.214, 1.117, 1.695, 1.072, 1.376},
        {1.326, 1.265, 1.056, 1.376, 0.99, 2.331, 1.397, 2.311, 0.98, 0.639, 1.025, 1.222, 1.131, 1.309, 1.298, 1.319, 1.039, 1.302, 1.351, 1.743, 2.155, 1.076, 1.158, 1.15, 1.045, 1.273, 0.971, 1.017},
        {2.324, 2.124, 1.36, 2.216, 2.408, 2.136, 2.563, 1.75, 1.253, 1.45, 1.436, 1.856, 1.41, 1.526, 1.174, 1.225, 1.226, 1.33, 1.167, 1.597, 1.18, 1.175, 1.025, 0.867, 0.95, 1.167, 0.893, 0.924},
        {2.322, 2.605, 1.759, 2.197, 1.827, 2.062, 1.838, 1.964, 1.69, 1.095, 1.061, 1.704, 1.723, 1.054, 1.409, 1.248, 1.207, 1.156, 1.19, 1.912, 1.194, 1.107, 1.214, 1.198, 1.069, 1.215, 1.296, 1.436},
        {2.548, 1.869, 1.701, 1.851, 1.91, 1.502, 1.903, 1.732, 1.656, 1.141, 1.066, 1.376, 1.593, 1.298, 1.471, 0.956, 1.305, 1.246, 1.091, 1.191, 1.523, 1.228, 1.859, 1.313, 0.937, 1.424, 1.283, 1.395},
        {2.247, 2.679, 1.576, 2.1, 3.041, 3.174, 2.368, 2.513, 1.903, 1.733, 1.492, 1.406, 1.788, 1.158, 1.567, 1.222, 1.296, 1.389, 1.121, 1.063, 1.54, 1.624, 1.502, 1.493, 1.298, 1.706, 1.35, 1.324},
        {2.876, 2.933, 1.882, 2.71, 1.918, 2.234, 2.85, 2.351, 2.256, 1.726, 1.251, 1.698, 1.376, 1.183, 2.032, 1.373, 1.577, 1.588, 1.556, 1.15, 1.228, 1.339, 1.663, 1.46, 1.442, 1.518, 1.338, 1.647},
        {2.792, 1.826, 2.049, 2.524, 2.509, 1.844, 2.739, 2.131, 1.882, 2.037, 1.511, 1.822, 1.502, 1.53, 1.43, 1.225, 1.344, 1.301, 1.585, 2.535, 1.689, 1.121, 1.75, 1.568, 1.597, 1.594, 1.829, 2.27},
        {2.818, 2.36, 2.451, 3.092, 2.454, 1.798, 2.123, 2.236, 1.964, 1.795, 1.933, 1.715, 1.869, 1.489, 1.912, 1.925, 1.582, 1.483, 1.404, 2.138, 2.162, 2.135, 1.359, 1.875, 1.598, 0.0, 0.0, 0.0},
        {2.27, 2.561, 1.859, 3.359, 2.769, 2.67, 2.882, 3.129, 1.853, 1.871, 1.594, 1.661, 1.781, 1.648, 1.879, 2.104, 1.68, 2.219, 1.941, 2.246, 1.641, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {4.532, 3.443, 2.461, 2.749, 2.483, 3.094, 2.792, 3.681, 1.9, 2.021, 2.307, 2.104, 2.442, 1.92, 2.172, 2.166, 1.333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {3.202, 2.496, 2.459, 5.129, 3.162, 3.873, 3.146, 6.775, 2.348, 3.352, 2.316, 2.587, 2.204, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {2.851, 2.892, 3.18, 4.399, 3.753, 5.331, 4.09, 9.566, 2.651, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {3.847, 3.465, 3.881, 5.627, 2.759, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {3.78, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
    };
    double mpcsd_final[mpc_max_depth_final - mpc_min_depth_final + 1] = {
        4.254, 4.028, 4.981, 4.717, 5.293, 5.058, 4.926, 5.025, 4.779, 4.459, 4.88, 4.643, 4.439, 4.336, 4.156, 4.421, 4.257, 4.022, 3.838, 3.679, 3.572, 3.933, 3.848, 3.949, 5.056, 5.821, 6.992, 7.572, 8.556, 8.854, 9.898, 10.263, 10.886, 11.245
    };
#endif
unsigned long long can_be_flipped[hw2];

class search_statistics{
    public:
        unsigned long long searched_nodes;
    #if USE_MULTI_THREAD
        private:
            mutex mtx;
    #endif
    public:
        inline void nodes_increment(){
            #if STATISTICS_MODE
                #if USE_MULTI_THREAD
                    lock_guard<mutex> lock(mtx);
                #endif
                ++searched_nodes;
            #endif
        }
};
search_statistics search_statistics;
vector<int> vacant_lst;

struct search_result{
    int policy;
    int value;
    int depth;
    int nps;
};

int cmp_vacant(int p, int q){
    return cell_weight[p] > cell_weight[q];
}

inline void move_ordering(board *b){
    int l, u;
    int hash = (int)(b->hash() & search_hash_mask);
    transpose_table.get_now(b, hash, &l, &u);
    if (u != inf && l != -inf)
        b->v = -(u + l) / 2 + cache_hit + cache_both + cache_now;
    else if (u != inf)
        b->v = -u + cache_hit + cache_now;
    else if (l != -inf)
        b->v = -l + cache_hit + cache_now;
    else{
        transpose_table.get_prev(b, hash, &l, &u);
        if (u != inf && l != -inf)
            b->v = -(u + l) / 2 + cache_hit + cache_both;
        else if (u != inf)
            b->v = -u + cache_hit;
        else if (l != -inf)
            b->v = -l + cache_hit;
        else
            b->v = -mid_evaluate(b);
    }
}

inline void move_ordering_eval(board *b){
    b->v = -mid_evaluate(b);
}

inline bool stability_cut(board *b, int *alpha, int *beta){
    int b_arr[hw2], stab[2];
    b->translate_to_arr(b_arr);
    calc_stability(b, b_arr, &stab[0], &stab[1]);
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