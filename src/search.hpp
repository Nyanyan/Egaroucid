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
#define cache_hit 10000
#define cache_now 10000
#define cache_both 1000
#define cache_low 100
#define parity_vacant_bonus 5
#define canput_bonus 10
#define w_former_search 20
#define w_stability 5
#define w_evaluate 10
#define w_surround 5
#define w_mobility 30
#define w_parity 10

#define mpc_min_depth 2
#define mpc_max_depth 30
#define mpc_min_depth_final 5
#define mpc_max_depth_final 40

#define simple_mid_threshold 2
#define simple_end_threshold 7
#define simple_end_mpc_threshold 5
#define simple_end_threshold2 13
#define end_mpc_depth_threshold 5

#define po_max_depth 15

#define mid_first_threshold_div 5
#define end_first_threshold_div 6

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
    10, 9, 10, 11, 12, 11, 12, 13, 14, 13,
    14
};
#if USE_MID_SMOOTH
    double mpcsd[n_phases][mpc_max_depth - mpc_min_depth + 1]={

    };
    double mpcsd_final[mpc_max_depth_final - mpc_min_depth_final + 1] = {

    };
#else
    constexpr double mpcsd[n_phases][mpc_max_depth - mpc_min_depth + 1]={
        {0.722, 0.776, 0.624, 0.852, 0.659, 0.751, 0.859, 0.444, 0.588, 0.442, 0.511, 0.587, 0.511, 0.464, 0.442, 0.444, 0.442, 0.282, 0.511, 0.576, 0.875, 0.806, 1.06, 0.875, 1.191, 1.083, 1.781, 1.26, 1.432},
        {1.414, 1.25, 1.042, 1.146, 0.565, 0.833, 0.721, 0.851, 0.834, 1.16, 0.83, 0.945, 0.77, 0.833, 0.658, 1.361, 1.062, 1.122, 1.09, 1.268, 1.293, 1.404, 1.429, 1.327, 1.905, 1.465, 1.56, 1.283, 1.82},
        {2.81, 2.239, 2.359, 1.099, 0.806, 1.191, 0.751, 1.436, 0.833, 1.381, 0.955, 0.945, 0.761, 0.676, 0.751, 1.395, 1.035, 1.398, 1.351, 1.501, 1.447, 1.382, 1.301, 1.197, 1.124, 2.053, 1.465, 1.474, 1.777},
        {2.953, 1.248, 1.749, 1.278, 1.424, 1.316, 1.329, 1.057, 1.327, 1.313, 0.703, 0.785, 1.197, 0.929, 0.902, 1.193, 1.096, 1.351, 1.504, 1.313, 1.453, 1.129, 1.122, 1.244, 1.196, 1.435, 1.719, 1.014, 1.243},
        {3.569, 1.756, 1.373, 1.452, 1.587, 1.412, 1.071, 1.539, 1.062, 1.373, 1.41, 1.408, 1.25, 1.551, 1.356, 1.26, 1.391, 1.103, 1.252, 1.213, 1.484, 1.579, 1.319, 1.694, 1.465, 1.765, 1.732, 1.081, 0.868},
        {3.895, 2.22, 1.318, 1.641, 1.414, 1.622, 1.424, 1.597, 1.404, 1.474, 1.654, 1.429, 1.404, 1.567, 1.455, 1.381, 1.283, 1.308, 1.188, 1.351, 1.482, 1.318, 1.414, 1.318, 1.274, 1.296, 1.841, 2.221, 1.422},
        {3.148, 1.726, 1.525, 1.558, 1.474, 1.555, 1.569, 1.301, 1.439, 1.213, 1.309, 1.474, 1.587, 1.794, 1.451, 1.349, 1.022, 1.337, 1.396, 1.361, 1.442, 1.439, 1.628, 1.429, 1.817, 1.606, 2.548, 2.016, 1.956},
        {3.078, 2.069, 1.269, 1.523, 1.53, 1.48, 1.363, 1.706, 1.532, 1.327, 1.536, 1.239, 1.152, 1.635, 1.345, 1.459, 1.439, 1.455, 1.301, 1.294, 1.465, 1.296, 1.806, 1.557, 1.558, 2.284, 2.373, 1.731, 1.403},
        {2.687, 2.3, 1.501, 1.53, 1.978, 1.954, 1.633, 1.622, 1.504, 1.916, 1.382, 1.285, 1.474, 1.531, 1.501, 1.435, 1.579, 1.392, 1.361, 1.666, 1.56, 1.501, 1.619, 1.924, 1.382, 0.983, 0.0, 0.0, 0.0},
        {2.658, 2.668, 1.408, 2.146, 1.574, 1.496, 1.583, 1.719, 1.373, 1.618, 1.245, 1.318, 1.341, 1.348, 1.351, 1.301, 1.424, 1.461, 1.341, 1.65, 2.575, 2.563, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {2.748, 4.344, 1.954, 1.345, 1.702, 1.318, 1.523, 1.956, 1.445, 1.663, 1.501, 1.558, 1.608, 1.682, 1.377, 1.572, 0.953, 0.548, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {3.618, 2.622, 2.75, 1.654, 1.778, 2.014, 1.586, 2.206, 2.183, 2.022, 1.628, 1.339, 1.549, 1.761, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {2.093, 2.808, 2.408, 2.599, 2.513, 2.207, 2.303, 2.828, 3.44, 4.07, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {3.796, 4.198, 3.46, 3.236, 2.685, 2.805, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {3.615, 2.16, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
    };
    constexpr double mpcsd_final[mpc_max_depth_final - mpc_min_depth_final + 1] = {
        4.658, 5.038, 5.311, 5.371, 5.959, 5.937, 5.513, 5.937, 5.528, 5.689, 5.778, 5.909, 5.457, 5.075, 4.93, 4.943, 4.896, 5.115, 4.997, 5.002, 4.827, 5.202, 4.93, 5.172, 5.043, 5.747, 6.261, 7.352, 7.871, 8.315, 8.755, 9.814, 9.679, 10.362, 10.647, 11.514
    };
#endif
unsigned long long can_be_flipped[hw2];
bool search_completed = false;

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
    if (elem1.b.v == elem2.b.v){
        if (elem1.error < 0 && elem2.error > 0)
            return true;
        if (elem2.error < 0 && elem1.error > 0)
            return false;
        if (elem1.error < 0 && elem2.error < 0)
            return elem1.error < elem2.error;
        if (elem1.error > 0 && elem2.error > 0)
            return elem1.error > elem2.error;
        return elem2.error == 0;
    }
    return elem1.b.v < elem2.b.v;
};

int cmp_vacant(int p, int q){
    return cell_weight[p] > cell_weight[q];
}

//int nega_alpha(board *b, bool skipped, int depth, int alpha, int beta, int *n_nodes);

inline int move_ordering(board *b, board *nb, const int hash, const int policy){
    /*
    int v = transpose_table.child_get_now(b, hash, policy) * w_former_search;
    if (v == -child_inf * w_former_search){
        v = transpose_table.child_get_prev(b, hash, policy) * w_former_search;
        if (v == -child_inf * w_former_search){
            //v = -mid_evaluate(nb) * w_former_search;
            v = 0;
        }else
            v += cache_hit;
    } else
        v += cache_hit + cache_now;
    */
    int v = 0;
    int l, u, child_hash = nb->hash() & search_hash_mask;
    transpose_table.get_now(nb, child_hash, &l, &u);
    if (l == -inf && u == inf){
        transpose_table.get_prev(nb, child_hash, &l, &u);
        if (l == -inf && u == inf){
            v = 0;
        } else if (l != -inf && u != inf){
            v = -(l + u) / 2 + cache_hit + cache_both;
        } else if (u != inf){
            v = -u * w_former_search + cache_hit;
        } else if (l != -inf){
            v = -l * w_former_search + cache_hit;
        }
    } else if (l != -inf && u != inf){
        v = -(l + u) / 2 + cache_hit + cache_both + cache_now;
    } else if (u != inf){
        v = -u * w_former_search + cache_hit + cache_now;
    } else if (l != -inf){
        v = -l * w_former_search + cache_hit + cache_now;
    }
    v += cell_weight[policy];
    v += -mid_evaluate(nb) * w_evaluate;
    //int n_nodes = 0;
    //v += -nega_alpha(nb, false, 2, -hw2, hw2, &n_nodes);
    int stab0, stab1;
    calc_stability_fast(nb, &stab0, &stab1);
    unsigned long long n_empties = ~(nb->b | nb->w);
    if (b->p == black){
        v += (stab0 - stab1) * w_stability;
        v += (calc_surround(nb->w, n_empties) - calc_surround(nb->b, n_empties)) * w_surround;
    } else{
        v += (stab1 - stab0) * w_stability;
        v += (calc_surround(nb->b, n_empties) - calc_surround(nb->w, n_empties)) * w_surround;
    }
    v -= pop_count_ull(nb->mobility_ull()) * w_mobility;
    if (b->parity & cell_div4[policy])
        v += w_parity;
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
