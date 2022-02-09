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
#define cache_both 2000
#define cache_high 10000
#define parity_vacant_bonus 5
#define canput_bonus 10
#define w_former_search 30
#define w_stability 5
//#define w_surround 5
#define w_evaluate 10
#define w_mobility 60
#define w_parity 4

#define mpc_min_depth 2
#define mpc_max_depth 30
#define mpc_min_depth_final 5
#define mpc_max_depth_final 40

#define simple_mid_threshold 2
#define simple_end_threshold 7
#define simple_end_threshold2 11
#define simple_end_mpc_threshold 5
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
        {0.722, 0.776, 0.624, 0.852, 0.659, 0.751, 0.859, 0.444, 0.588, 0.442, 0.511, 0.587, 0.511, 0.464, 0.442, 0.444, 0.511, 0.482, 0.442, 0.576, 0.523, 0.588, 0.761, 0.509, 0.923, 1.083, 1.781, 1.586, 1.553},
        {1.414, 2.386, 1.042, 1.146, 0.565, 0.78, 0.721, 0.851, 0.806, 1.122, 0.83, 0.945, 1.215, 1.233, 0.794, 1.24, 0.908, 0.794, 1.042, 0.9, 1.218, 1.327, 1.412, 1.367, 1.864, 1.442, 1.56, 1.349, 1.599},      
        {2.81, 2.239, 2.359, 1.099, 0.806, 1.191, 0.751, 1.436, 0.833, 1.381, 0.737, 0.945, 1.049, 1.139, 1.018, 1.572, 1.116, 1.285, 1.283, 1.484, 1.504, 1.391, 1.398, 1.173, 1.165, 2.111, 1.442, 1.482, 2.315}, 
        {2.953, 1.248, 1.749, 1.278, 1.424, 1.316, 1.329, 1.098, 1.327, 1.341, 0.795, 0.899, 1.122, 1.445, 1.024, 1.043, 1.135, 1.316, 1.504, 1.275, 1.579, 1.063, 1.139, 1.359, 1.234, 1.435, 1.588, 1.153, 1.557},
        {3.569, 1.756, 1.373, 1.488, 1.56, 1.412, 1.071, 1.465, 1.122, 1.435, 1.518, 1.439, 1.25, 1.64, 1.465, 1.268, 1.503, 1.16, 1.373, 1.16, 1.472, 1.56, 1.341, 1.599, 1.579, 1.841, 1.622, 1.294, 1.367},      
        {3.895, 1.849, 1.276, 1.558, 1.285, 1.702, 1.348, 1.606, 1.398, 1.816, 1.599, 1.262, 1.412, 1.539, 1.496, 1.308, 1.233, 1.239, 1.226, 1.285, 1.382, 1.351, 1.504, 1.317, 1.209, 1.239, 1.993, 2.521, 2.018},
        {3.217, 2.226, 1.461, 1.628, 1.319, 1.511, 1.348, 1.285, 1.424, 1.285, 0.821, 1.103, 1.527, 1.732, 1.376, 1.469, 1.173, 1.222, 1.244, 1.316, 1.503, 1.53, 1.341, 1.348, 1.976, 1.926, 2.514, 1.72, 2.345},
        {2.919, 1.877, 1.207, 1.654, 1.933, 1.389, 1.476, 1.628, 1.398, 1.389, 1.526, 1.248, 1.073, 1.689, 1.279, 1.398, 0.955, 1.399, 1.341, 1.429, 1.494, 1.435, 1.627, 1.624, 1.53, 2.126, 2.197, 1.609, 1.775},
        {2.7, 2.305, 1.532, 1.53, 1.956, 2.067, 1.633, 1.622, 1.504, 1.916, 1.442, 1.285, 1.474, 1.473, 1.501, 1.444, 1.579, 1.432, 1.412, 1.666, 1.595, 1.706, 1.663, 1.967, 1.382, 0.983, 0.0, 0.0, 0.0},
        {2.658, 2.668, 1.408, 2.146, 1.574, 1.496, 1.583, 1.719, 1.373, 1.618, 1.245, 1.318, 1.341, 1.294, 1.381, 1.327, 1.268, 1.41, 1.341, 1.65, 2.575, 2.563, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {2.748, 4.344, 1.954, 1.345, 1.702, 1.395, 1.523, 1.956, 1.408, 1.75, 1.435, 1.501, 1.55, 1.758, 1.349, 2.086, 0.996, 0.548, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {3.618, 2.622, 2.75, 1.654, 1.778, 1.806, 1.527, 1.984, 1.589, 2.021, 1.998, 1.543, 1.663, 1.506, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {2.093, 2.808, 2.408, 2.16, 1.761, 1.789, 1.711, 2.415, 3.268, 3.521, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {3.796, 4.198, 3.46, 3.451, 2.685, 2.098, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {3.615, 2.16, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
    };
    constexpr double mpcsd_final[mpc_max_depth_final - mpc_min_depth_final + 1] = {
        4.658, 5.038, 5.311, 5.371, 5.959, 5.937, 5.513, 5.937, 5.528, 5.689, 5.778, 5.909, 5.457, 5.075, 4.93, 4.943, 4.896, 5.115, 4.997, 5.002, 4.747, 5.198, 4.899, 5.169, 4.832, 5.614, 6.121, 7.195, 7.871, 8.315, 8.755, 9.814, 9.679, 10.362, 10.647, 11.514
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

struct ybwc_result{
    int value;
    unsigned long long n_nodes;
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

inline int move_ordering(board *b, board *nb, const int hash, const int policy, int value){
    int v = 0;
    if ((b->p == 0 && nb->w == 0) || (b->p == 1 && nb->b == 0))
        v = inf;
    else{
        int l, u, child_hash = nb->hash() & search_hash_mask;
        transpose_table.get_now(nb, child_hash, &l, &u);
        if (l == -inf && u == inf){
            transpose_table.get_prev(nb, child_hash, &l, &u);
            if (l == -inf && u == inf){
                v = 0;
            } else if (l != -inf && u != inf){
                v = -(l + u) * w_former_search / 2 + cache_hit + cache_both;
            } else if (u != inf){
                v = -u * w_former_search + cache_hit + cache_high;
            } else if (l != -inf){
                v = -l * w_former_search + cache_hit;
            }
        } else if (l != -inf && u != inf){
            v = -(l + u) * w_former_search / 2 + cache_hit + cache_both + cache_now;
        } else if (u != inf){
            v = -u * w_former_search + cache_hit + cache_high + cache_now;
        } else if (l != -inf){
            v = -l * w_former_search + cache_hit + cache_now;
        }
        v += cell_weight[policy];
        v += value * w_evaluate;
        //int n_nodes = 0;
        //v += -nega_alpha(nb, false, 2, -hw2, hw2, &n_nodes);
        int stab0, stab1;
        calc_stability_fast(nb, &stab0, &stab1);
        if (b->p == black){
            v += stab0 * w_stability;
            //v += calc_surround(nb->w, ~(nb->b | nb->w)) * w_surround;
        } else{
            v += stab1 * w_stability;
            //v += calc_surround(nb->b, ~(nb->b | nb->w)) * w_surround;
        }
        v -= pop_count_ull(nb->mobility_ull()) * w_mobility;
        if (b->parity & cell_div4[policy])
            v += w_parity;
    }
    return v;
}

inline int move_ordering(board *b, board *nb, const int hash, const int policy){
    return move_ordering(b, nb, hash, policy, -mid_evaluate(nb));
}

inline void move_ordering_eval(board *b){
    b->v = -mid_evaluate(b);
}

inline int stability_cut(board *b, int *alpha, int *beta){
    int stab[2];
    calc_stability(b, &stab[0], &stab[1]);
    int n_alpha = 2 * stab[b->p] - hw2;
    int n_beta = hw2 - 2 * stab[1 - b->p];
    if (*beta <= n_alpha)
        return n_alpha;
    if (n_beta <= *alpha)
        return n_beta;
    if (n_beta <= n_alpha)
        return n_alpha;
    *alpha = max(*alpha, n_alpha);
    *beta = min(*beta, n_beta);
    return -inf;
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
