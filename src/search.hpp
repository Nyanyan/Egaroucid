#pragma once
#include <iostream>
#include <fstream>
#include <math.h>
#include "setting.hpp"
#include "common.hpp"
#include "board.hpp"
#include "evaluate.hpp"
#include "transpose_table.hpp"

using namespace std;

#define search_epsilon 1
constexpr int cache_hit = 100;
constexpr int cache_both = 100;
constexpr int parity_vacant_bonus = 10;
constexpr int canput_bonus = 4;

#define mpc_min_depth 3
#define mpc_max_depth 25
#define mpc_min_depth_final 9
#define mpc_max_depth_final 30

#define simple_mid_threshold 3
#define simple_end_threshold 9

#define po_max_depth 8

#define extra_stability_threshold 58

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

const int mpcd[38] = {0, 1, 0, 1, 2, 3, 2, 3, 4, 3, 4, 3, 4, 5, 4, 5, 6, 5, 6, 7, 6, 7, 6, 7, 8, 7, 8, 9, 8, 9, 8, 9, 10, 9, 10, 11, 10, 11};
#if USE_MID_SMOOTH
    double mpcsd[n_phases][mpc_max_depth - mpc_min_depth + 1]={

    };
    double mpcsd_final[mpc_max_depth_final - mpc_min_depth_final + 1] = {

    };
#else
    double mpcsd[n_phases][mpc_max_depth - mpc_min_depth + 1]={
        {0.845, 0.747, 0.951, 0.622, 0.741, 0.826, 0.893, 0.478, 0.652, 0.567, 0.604, 0.697, 0.845, 0.844, 0.687, 0.931, 0.959, 1.096, 1.117, 1.056, 1.014, 1.417, 1.402},
        {1.326, 1.265, 1.05, 1.308, 1.195, 2.273, 1.391, 1.54, 1.073, 0.851, 1.237, 1.624, 1.204, 1.256, 1.383, 1.238, 1.106, 1.116, 1.563, 1.786, 1.521, 1.556, 1.461},
        {2.324, 2.124, 1.27, 2.247, 2.38, 2.161, 1.731, 2.376, 1.854, 1.533, 1.903, 1.871, 1.481, 1.645, 1.267, 1.402, 1.212, 1.807, 1.334, 1.695, 1.473, 1.57, 1.199},
        {2.322, 2.605, 1.615, 2.17, 1.892, 2.042, 2.188, 1.682, 2.042, 1.332, 1.218, 1.68, 2.116, 1.45, 1.362, 1.183, 1.23, 1.369, 0.856, 2.155, 1.424, 1.454, 1.183},
        {2.548, 1.869, 1.626, 2.0, 1.653, 2.24, 1.799, 1.704, 1.862, 1.653, 1.238, 1.757, 2.113, 1.545, 1.864, 1.251, 1.424, 1.864, 1.256, 1.136, 1.402, 1.781, 2.26},
        {2.247, 2.679, 1.545, 2.14, 2.348, 3.35, 1.63, 2.471, 2.7, 2.429, 1.647, 1.714, 1.915, 1.533, 1.858, 1.539, 1.616, 1.812, 1.582, 1.905, 1.833, 1.464, 1.737},
        {2.876, 3.004, 1.963, 2.87, 1.978, 2.375, 2.222, 2.729, 2.765, 2.366, 1.673, 2.094, 1.706, 1.933, 2.26, 1.996, 1.482, 2.091, 1.511, 1.759, 1.424, 1.624, 2.2},
        {2.792, 1.794, 2.097, 2.259, 2.62, 1.704, 2.839, 2.5, 2.146, 2.045, 1.595, 2.119, 1.92, 1.793, 1.521, 1.645, 1.697, 1.914, 1.502, 3.001, 2.188, 1.246, 2.393},
        {2.818, 2.31, 2.506, 2.97, 2.479, 1.805, 2.395, 2.176, 2.241, 2.183, 2.482, 2.184, 2.341, 1.977, 2.07, 2.305, 1.531, 1.897, 2.131, 2.35, 2.442, 2.443, 2.275},
        {2.27, 2.595, 1.971, 3.258, 2.571, 2.634, 2.802, 2.913, 2.516, 2.051, 1.842, 1.736, 1.873, 2.034, 2.505, 2.462, 1.857, 2.384, 2.806, 2.699, 2.41, 2.699, 3.273},
        {4.392, 3.414, 2.58, 2.723, 2.533, 2.903, 3.223, 3.157, 2.49, 2.903, 2.379, 2.989, 2.51, 2.646, 2.455, 2.89, 3.12, 2.83, 3.255, 3.908, 3.025, 3.484, 4.094},
        {3.203, 2.5, 2.468, 4.99, 2.938, 4.146, 3.888, 4.071, 3.073, 3.15, 2.549, 3.738, 3.818, 3.247, 2.624, 2.235, 2.5, 2.745, 2.733, 3.703, 3.219, 3.912, 3.669},
        {2.837, 2.955, 3.21, 4.638, 3.629, 5.385, 3.773, 6.886, 4.251, 4.024, 2.972, 3.27, 3.059, 2.97, 2.299, 2.104, 2.026, 2.707, 1.868, 2.565, 2.558, 2.361, 2.891},
        {3.672, 3.492, 3.675, 5.741, 3.427, 3.901, 3.243, 3.525, 3.112, 2.62, 1.17, 1.903, 1.865, 0.91, 1.338, 1.082, 1.348, 1.86, 0.535, 1.237, 0.627, 0.343, 0.77},
        {2.077, 2.785, 0.649, 1.646, 0.377, 0.4, 0.729, 0.5, 0.439, 0.61, 0.537, 0.723, 0.177, 0.2, 0.246, 0.2, 1.018, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2}
    };
    double mpcsd_final[mpc_max_depth_final - mpc_min_depth_final + 1] = {
        4.981, 4.717, 5.293, 5.058, 4.926, 5.025, 4.779, 4.459, 4.88, 4.643, 4.439, 4.336, 4.156, 4.421, 4.257, 4.022, 3.838, 3.679, 3.572, 3.933, 3.848, 3.949
    };
#endif
unsigned long long can_be_flipped[hw2];

unsigned long long searched_nodes;
vector<int> vacant_lst;

struct search_result{
    int policy;
    int value;
    int depth;
    int nps;
};

inline void search_init(){
    int i;
    for (int cell = 0; cell < hw2; ++cell){
        can_be_flipped[cell] = 0b1111111110000001100000011000000110000001100000011000000111111111;
        for (i = 0; i < hw; ++i){
            if (global_place[place_included[cell][0]][i] != -1)
                can_be_flipped[cell] |= 1ULL << global_place[place_included[cell][0]][i];
        }
        for (i = 0; i < hw; ++i){
            if (global_place[place_included[cell][1]][i] != -1)
                can_be_flipped[cell] |= 1ULL << global_place[place_included[cell][1]][i];
        }
        for (i = 0; i < hw; ++i){
            if (global_place[place_included[cell][2]][i] != -1)
                can_be_flipped[cell] |= 1ULL << global_place[place_included[cell][2]][i];
        }
        if (place_included[cell][3] != -1){
            for (i = 0; i < hw; ++i){
                if (global_place[place_included[cell][3]][i] != -1)
                    can_be_flipped[cell] |= 1ULL << global_place[place_included[cell][3]][i];
            }
        }
    }
    cerr << "search initialized" << endl;
}

int cmp_vacant(int p, int q){
    return cell_weight[p] > cell_weight[q];
}

inline void move_ordering(board *b){
    int l, u;
    transpose_table.get_prev(b, b->hash() & search_hash_mask, &l, &u);
    if (u != inf && l != -inf)
        b->v = -(u + l) / 2 + cache_hit + cache_both;
    else if (u != inf)
        b->v = -mid_evaluate(b) + cache_hit;
    else if (l != -inf)
        b->v = -mid_evaluate(b) + cache_hit;
    else
        b->v = -mid_evaluate(b);
}

inline void move_ordering_eval(board *b){
    b->v = -mid_evaluate(b);
}

inline void calc_extra_stability(board *b, int p, unsigned long long extra_stability, int *pres, int *ores){
    *pres = 0;
    *ores = 0;
    int y, x;
    extra_stability >>= hw;
    for (y = 1; y < hw_m1; ++y){
        extra_stability >>= 1;
        for (x = 1; x < hw_m1; ++x){
            if ((extra_stability & 1) == 0){
                if (pop_digit[b->b[y]][x] == p)
                    ++*pres;
                else if (pop_digit[b->b[y]][x] == 1 - p)
                    ++*ores;
            }
            extra_stability >>= 1;
        }
        extra_stability >>= 1;
    }
}

inline unsigned long long calc_extra_stability_ull(board *b){
    unsigned long long extra_stability = 0b1111111110000001100000011000000110000001100000011000000111111111;
    for (const int &cell: vacant_lst){
        if (pop_digit[b->b[cell / hw]][cell % hw] == vacant)
            extra_stability |= can_be_flipped[cell];
    }
    return extra_stability;
}

inline bool stability_cut(board *b, int *alpha, int *beta){
    //if (b->n >= extra_stability_threshold){
    //    int ps, os;
    //    calc_extra_stability(b, b->p, calc_extra_stability_ull(b), &ps, &os);
    //    *alpha = max(*alpha, (2 * (calc_stability(b, b->p) + ps) - hw2));
    //    *beta = min(*beta, (hw2 - 2 * (calc_stability(b, 1 - b->p) + os)));
    //} else{
    *alpha = max(*alpha, (2 * calc_stability(b, b->p) - hw2));
    *beta = min(*beta, (hw2 - 2 * calc_stability(b, 1 - b->p)));
    //}
    return *alpha >= *beta;
}

inline int calc_canput_exact(board *b){
    int res = 0;
    for (const int &cell: vacant_lst)
        res += b->legal(cell);
    return res;
}
