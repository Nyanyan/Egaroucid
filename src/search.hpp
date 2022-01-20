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
constexpr int canput_bonus = 0;

#define mpc_min_depth 3
#define mpc_max_depth 20
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

const int mpcd[32] = {0, 1, 0, 1, 2, 3, 2, 3, 4, 3, 4, 3, 4, 5, 4, 5, 6, 5, 6, 7, 6, 7, 6, 7, 8, 7, 8, 9, 8, 9, 8, 9};
#if USE_MID_SMOOTH
    double mpcsd[n_phases][mpc_max_depth - mpc_min_depth + 1]={

    };
    double mpcsd_final[mpc_max_depth_final - mpc_min_depth_final + 1] = {

    };
#else
    double mpcsd[n_phases][mpc_max_depth - mpc_min_depth + 1]={
        {1.449, 1.076, 0.838, 0.875, 0.887, 1.162, 0.839, 1.423, 0.612, 0.508, 0.0, 0.508, 0.508, 0.663, 0.637, 1.261, 1.283, 1.675},
        {1.15, 1.339, 1.069, 1.779, 1.453, 1.224, 2.107, 1.971, 1.155, 1.474, 1.604, 1.353, 1.588, 1.753, 1.585, 2.123, 1.524, 2.215},
        {1.722, 2.347, 1.572, 1.663, 2.074, 2.559, 1.729, 2.987, 2.222, 1.701, 1.378, 1.575, 1.449, 1.228, 1.63, 1.737, 1.162, 1.982},
        {2.681, 3.297, 1.844, 1.972, 2.097, 1.732, 2.235, 2.227, 1.359, 1.096, 0.979, 1.847, 1.228, 1.208, 2.185, 2.278, 1.474, 1.55},
        {2.504, 1.963, 1.771, 2.287, 1.688, 1.671, 2.872, 1.995, 1.945, 1.56, 1.503, 2.188, 1.675, 1.46, 1.551, 2.845, 1.343, 2.043},
        {2.326, 3.089, 1.827, 1.902, 2.358, 1.609, 5.354, 2.127, 2.243, 1.647, 1.925, 2.596, 1.83, 1.319, 2.019, 2.052, 1.453, 2.141},
        {3.061, 3.161, 1.952, 2.425, 2.426, 2.47, 2.572, 2.516, 1.945, 2.288, 1.683, 3.26, 2.445, 1.317, 2.315, 2.031, 2.009, 2.355},
        {3.047, 2.114, 2.178, 3.35, 2.644, 2.38, 2.025, 2.445, 2.518, 2.481, 1.38, 2.944, 1.856, 1.982, 2.315, 2.433, 1.698, 2.752},
        {2.918, 2.597, 2.056, 3.696, 2.542, 2.307, 2.945, 2.281, 2.171, 1.923, 1.96, 2.794, 1.84, 2.08, 1.995, 2.628, 2.68, 2.681},
        {2.487, 2.639, 1.9, 3.198, 3.035, 1.813, 3.641, 3.616, 2.755, 2.283, 2.389, 1.956, 2.119, 1.875, 2.31, 2.685, 3.436, 3.253},
        {4.587, 3.431, 2.299, 2.487, 2.659, 1.668, 2.588, 3.595, 2.82, 2.701, 2.617, 2.52, 2.321, 2.084, 3.17, 3.067, 2.851, 1.936},
        {3.426, 2.782, 2.439, 3.105, 4.443, 3.027, 4.707, 4.8, 3.002, 2.937, 3.071, 3.134, 3.166, 3.111, 2.625, 3.658, 2.581, 3.161},
        {2.957, 2.499, 3.436, 3.453, 4.35, 3.223, 5.827, 4.276, 3.516, 4.28, 2.681, 3.384, 3.542, 2.803, 2.438, 3.42, 2.757, 2.923},
        {3.736, 2.996, 3.06, 5.243, 5.28, 2.973, 4.654, 3.224, 2.829, 2.442, 1.416, 2.347, 1.964, 1.138, 1.41, 1.06, 0.416, 0.766},
        {2.257, 3.004, 0.744, 1.795, 0.752, 0.0, 1.016, 0.0, 0.659, 1.09, 0.0, 1.188, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
    };
    double mpcsd_final[mpc_max_depth_final - mpc_min_depth_final + 1] = {
        4.939, 4.656, 5.307, 5.063, 5.034, 5.03, 4.815, 4.629, 5.158, 4.93, 4.841, 4.531, 4.387, 4.749, 4.731, 4.539, 4.336, 4.18, 4.067, 4.36, 4.325, 4.341
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
    if (b->n >= extra_stability_threshold){
        int ps, os;
        calc_extra_stability(b, b->p, calc_extra_stability_ull(b), &ps, &os);
        *alpha = max(*alpha, (2 * (calc_stability(b, b->p) + ps) - hw2));
        *beta = min(*beta, (hw2 - 2 * (calc_stability(b, 1 - b->p) + os)));
    } else{
        *alpha = max(*alpha, (2 * calc_stability(b, b->p) - hw2));
        *beta = min(*beta, (hw2 - 2 * calc_stability(b, 1 - b->p)));
    }
    return *alpha >= *beta;
}

inline int calc_canput_exact(board *b){
    int res = 0;
    for (const int &cell: vacant_lst)
        res += b->legal(cell);
    return res;
}
