#pragma once
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
#define mpc_max_depth 24
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
        {0.459, 0.572, 0.514, 0.836, 1.059, 1.041, 1.477, 1.384, 1.16, 0.806, 0.781, 0.765, 0.69, 0.353, 0.702, 0.677, 0.874, 0.891, 1.007, 0.749, 0.784, 0.917},
        {1.338, 1.205, 1.246, 1.568, 1.564, 1.576, 1.856, 2.047, 1.569, 1.249, 1.124, 1.296, 1.221, 1.077, 1.066, 1.316, 0.948, 1.423, 1.331, 1.538, 1.111, 1.122},
        {2.198, 2.221, 2.083, 2.958, 2.823, 2.553, 2.21, 2.631, 2.13, 1.932, 1.529, 1.728, 1.546, 1.785, 1.905, 1.445, 1.403, 1.692, 1.348, 2.058, 1.387, 1.244},
        {2.426, 2.397, 2.007, 2.572, 2.582, 2.656, 2.895, 2.046, 2.541, 2.018, 1.748, 2.241, 1.889, 1.542, 2.22, 1.736, 1.783, 1.846, 1.743, 1.845, 1.721, 1.61},
        {3.002, 2.118, 2.402, 2.749, 2.765, 2.772, 3.137, 2.827, 3.155, 2.611, 2.192, 2.344, 1.911, 1.926, 2.373, 2.039, 1.592, 2.235, 1.866, 2.603, 1.872, 1.782},
        {2.943, 2.779, 2.54, 3.877, 2.856, 2.44, 3.535, 3.452, 2.924, 2.504, 2.116, 2.62, 2.087, 1.844, 2.542, 2.221, 2.045, 2.528, 1.711, 2.428, 2.214, 2.143},
        {3.201, 2.67, 2.047, 3.76, 3.374, 2.79, 3.122, 3.587, 3.57, 2.815, 2.734, 3.16, 2.72, 2.358, 2.509, 2.215, 2.427, 2.734, 2.275, 3.199, 2.171, 2.855},
        {2.998, 2.909, 2.501, 3.521, 3.146, 3.348, 4.123, 4.546, 3.627, 3.701, 3.11, 3.809, 2.737, 3.0, 3.119, 3.121, 2.668, 2.603, 2.533, 3.34, 2.341, 2.709},
        {3.37, 2.938, 3.91, 3.654, 4.431, 3.722, 5.464, 4.204, 3.602, 3.595, 2.373, 3.094, 2.93, 2.423, 3.067, 2.411, 2.047, 1.899, 1.842, 2.768, 2.07, 2.345},
        {2.568, 2.555, 2.451, 2.887, 2.564, 1.49, 2.518, 1.941, 1.956, 0.881, 0.857, 1.245, 0.762, 0.757, 0.744, 0.594, 0.746, 0.758, 0.0, 0.474, 0.0, 0.0}
    };
    double mpcsd_final[mpc_max_depth_final - mpc_min_depth_final + 1] = {
        5.026, 4.751, 5.311, 5.029, 4.967, 5.186, 4.939, 4.718, 4.916, 4.586, 4.555, 4.562, 4.278, 4.589, 4.519, 4.312, 4.188, 4.02, 3.951, 3.899, 3.858, 4.438
    };
#else
    double mpcsd[n_phases][mpc_max_depth - mpc_min_depth + 1]={
    };
    double mpcsd_final[mpc_max_depth_final - mpc_min_depth_final + 1] = {
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

inline void mpc_init(){
    int i, j;
    for (i = 0; i < n_phases; ++i){
        for (j = 0; j < mpc_max_depth - mpc_min_depth + 1; ++j)
            mpcsd[i][j] /= step;
    }
    for (i = 0; i < mpc_max_depth_final - mpc_min_depth_final + 1; ++i)
        mpcsd_final[i] /= step;
}

inline void search_init(){
    //mpc_init();
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
