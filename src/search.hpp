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
#define mpc_max_depth 25
#define mpc_min_depth_final 9
#define mpc_max_depth_final 30

#define simple_mid_threshold 3
#define simple_end_threshold 7

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
        {1.414, 0.849, 0.908, 0.655, 0.82, 1.041, 1.225, 1.0, 0.766, 0.745, 0.507, 0.943, 1.1, 0.838, 0.0, 0.186, 0.418, 0.819, 0.572, 0.458, 0.504, 0.483, 0.586},
        {1.183, 0.905, 0.929, 1.532, 1.32, 0.978, 1.183, 1.2, 0.87, 0.95, 0.86, 1.3, 1.113, 0.847, 0.871, 0.463, 1.008, 1.361, 0.454, 0.61, 0.673, 1.164, 1.366},
        {2.57, 1.16, 1.433, 2.61, 2.615, 2.411, 1.795, 1.823, 1.293, 1.581, 1.633, 1.356, 1.185, 0.845, 1.254, 0.869, 0.921, 2.137, 0.692, 0.797, 0.955, 1.293, 1.366},
        {2.751, 2.613, 1.749, 1.806, 2.045, 1.459, 1.694, 1.821, 1.806, 1.796, 1.496, 1.316, 1.641, 0.647, 1.091, 1.018, 1.044, 1.283, 0.727, 0.819, 1.491, 1.231, 1.414},
        {3.124, 2.551, 1.718, 2.362, 2.849, 1.033, 2.308, 1.759, 1.921, 1.915, 1.51, 1.899, 1.525, 0.728, 1.359, 1.39, 1.406, 2.221, 0.59, 0.793, 1.073, 1.305, 1.258},
        {1.225, 1.981, 2.449, 2.06, 2.712, 2.233, 2.314, 3.516, 2.01, 2.072, 1.46, 2.056, 2.027, 2.223, 1.391, 1.37, 1.301, 1.608, 1.079, 0.921, 1.673, 0.802, 1.445},
        {2.463, 2.404, 2.521, 2.366, 2.551, 2.243, 3.024, 1.815, 1.917, 1.426, 1.69, 2.145, 1.763, 0.859, 0.954, 1.393, 1.334, 1.829, 0.649, 0.775, 1.179, 1.422, 1.499},
        {1.593, 1.457, 1.505, 2.644, 2.032, 2.406, 2.782, 3.247, 2.111, 1.816, 2.189, 2.374, 2.591, 0.893, 1.641, 1.224, 1.224, 2.559, 0.994, 0.779, 0.948, 1.697, 1.614},
        {2.29, 2.906, 1.929, 3.072, 2.438, 2.463, 2.689, 3.754, 3.041, 2.384, 1.223, 1.81, 2.479, 0.833, 1.621, 1.295, 1.248, 2.093, 0.702, 1.254, 1.434, 1.315, 2.286},
        {4.161, 0.843, 2.166, 4.259, 3.289, 2.575, 3.414, 3.43, 2.288, 2.54, 2.755, 2.817, 2.504, 0.795, 1.541, 1.575, 2.177, 3.516, 0.976, 1.543, 1.272, 1.542, 2.057},
        {2.898, 2.408, 2.453, 4.039, 2.722, 2.568, 4.551, 4.081, 3.905, 2.628, 2.833, 4.193, 2.209, 1.495, 2.0, 1.528, 1.706, 2.662, 0.994, 1.601, 1.218, 1.384, 1.834},
        {3.005, 3.078, 2.434, 4.243, 3.133, 4.262, 5.787, 4.689, 3.444, 3.766, 2.643, 4.498, 4.048, 1.386, 2.061, 2.068, 2.758, 2.39, 1.121, 1.581, 1.261, 1.417, 2.667},
        {3.528, 2.897, 2.617, 4.674, 3.092, 5.272, 6.154, 5.471, 3.594, 3.934, 2.264, 5.074, 4.368, 1.361, 1.953, 1.956, 2.078, 3.144, 1.337, 1.42, 1.739, 1.103, 2.514},
        {3.407, 4.47, 3.207, 4.782, 3.768, 2.371, 3.592, 3.744, 3.345, 3.215, 1.62, 3.328, 2.034, 0.724, 1.699, 0.861, 0.884, 1.282, 1.906, 1.022, 1.405, 2.689, 1.048},
        {2.151, 2.321, 0.985, 1.613, 0.539, 0.6, 0.745, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6}
    };
    double mpcsd_final[mpc_max_depth_final - mpc_min_depth_final + 1] = {
        4.957, 4.665, 5.19, 4.926, 4.836, 4.952, 4.677, 4.489, 4.874, 4.59, 4.551, 4.392, 4.353, 4.513, 4.483, 4.288, 4.189, 4.042, 4.09, 4.125, 4.128, 4.267
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
