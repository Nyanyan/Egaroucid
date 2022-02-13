#pragma once
#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <math.h>
#include <atomic>
#include "setting.hpp"
#include "common.hpp"
#include "board.hpp"
#include "evaluate.hpp"
#include "transpose_table.hpp"

using namespace std;

#define W_BEST1_MOVE 900000000
#define W_BEST2_MOVE 800000000
#define W_BEST3_MOVE 700000000
#define W_CACHE_HIT 1000000
#define W_CACHE_HIGH 10000
#define W_CACHE_VALUE 70
#define W_CELL_WEIGHT 1
#define W_EVALUATE 10
#define W_MOBILITY 61
//#define W_STABILITY 5
#define W_SURROUND 7
#define W_PARITY 4
//#define W_END_CELL_WEIGHT 1
//#define W_END_EVALUATE 5
#define W_END_MOBILITY 30
#define W_END_SURROUND 10
//#define W_END_STABILITY 5
#define W_END_PARITY 10

#define MID_MPC_MIN_DEPTH 2
#define MID_MPC_MAX_DEPTH 30
#define END_MPC_MIN_DEPTH 5
#define END_MPC_MAX_DEPTH 40
#define N_END_MPC_SCORE_DIV 22

#define MID_FAST_DEPTH 3
#define END_FAST_DEPTH 7
#define MID_TO_END_DEPTH 11

#define SCORE_UNDEFINED -INF

constexpr int cell_weight[HW2] = {
    18,  4,  16, 12, 12, 16,  4, 18,
     4,  2,   6,  8,  8,  6,  2,  4,
    16,  6,  14, 10, 10, 14,  6, 16,
    12,  8,  10,  0,  0, 10,  8, 12,
    12,  8,  10,  0,  0, 10,  8, 12,
    16,  6,  14, 10, 10, 14,  6, 16,
     4,  2,   6,  8,  8,  6,  2,  4,
    18,  4,  16, 12, 12, 16,  4, 18
};

constexpr int mpcd[41] = {
    0, 1, 0, 1, 2, 3, 2, 3, 4, 3, 
    4, 3, 4, 5, 4, 5, 6, 5, 6, 7, 
    6, 7, 6, 7, 8, 7, 8, 9, 8, 9, 
    10, 9, 10, 11, 12, 11, 12, 13, 14, 13,
    14
};

constexpr double mpcsd[N_PHASES][MID_MPC_MAX_DEPTH - MID_MPC_MIN_DEPTH + 1]={
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
constexpr double mpcsd_final[(END_MPC_MAX_DEPTH - END_MPC_MIN_DEPTH + 1) / 5 + 1][N_END_MPC_SCORE_DIV] = {
    {1.773, 1.329, 1.291, 7.219, 5.044, 4.877, 4.308, 4.915, 5.259, 4.754, 4.649, 4.728, 5.507, 5.474, 5.044, 5.303, 5.683, 5.925, 3.926, 5.68, 2.248, 0.855},
    {2.828, 2.427, 65, 4.082, 5.502, 5.295, 4.797, 5.159, 5.722, 5.302, 5.157, 5.191, 5.89, 5.608, 5.969, 6.36, 7.059, 7.376, 6.191, 4.154, 3.708, 0.949},
    {4.031, 2.927, 4.19, 4.692, 5.618, 4.551, 5.14, 5.726, 5.38, 5.203, 4.554, 5.012, 5.877, 5.618, 5.569, 6.331, 5.918, 7.134, 5.54, 5.293, 4.6, 2.138},
    {3.215, 4.907, 5.873, 1.764, 7.288, 5.369, 4.929, 4.906, 5.414, 4.755, 4.416, 4.671, 5.298, 5.503, 5.772, 7.518, 6.814, 7.033, 8.864, 7.011, 6.708, 7.537},
    {2.302, 10.066, 5.438, 6.175, 7.09, 4.706, 5.258, 5.326, 5.187, 4.653, 4.483, 4.786, 5.436, 5.486, 6.221, 6.902, 6.664, 7.9, 6.676, 4.994, 6.87, 8.649},
    {65, 65, 6.362, 9.234, 9.938, 9.28, 7.778, 6.484, 6.446, 6.008, 5.736, 6.038, 7.209, 7.377, 7.951, 9.204, 9.273, 11.793, 6.918, 6.848, 6.908, 65},
    {65, 65, 2.646, 3.202, 6.132, 10.728, 10.236, 8.911, 8.484, 7.891, 7.157, 7.888, 9.821, 10.713, 10.098, 10.021, 11.709, 9.681, 10.881, 7.7, 1.0, 65},
    {65, 65, 65, 2.887, 65, 8.706, 6.452, 10.436, 11.052, 8.855, 8.671, 8.67, 9.599, 12.627, 12.814, 9.541, 6.834, 16.688, 13.503, 65, 65, 65}
};
unsigned long long can_be_flipped[HW2];

struct Search_result{
    int policy;
    int value;
    int depth;
    int nps;
};

class Search{
    public:
        Board board;
        bool skipped;
        bool use_mpc;
        double mpct;
        vector<int> vacant_list;
        unsigned long long n_nodes;

    public:
        inline void pass(){
            board.p = 1 - board.p;
            skipped = true;
        }

        inline void undo_pass(){
            board.p = 1 - board.p;
            skipped = false;
        }
};

int cmp_vacant(int p, int q){
    return cell_weight[p] > cell_weight[q];
}

bool cmp_move_ordering(Mobility &a, Mobility &b){
    return a.value > b.value;
}

inline void move_evaluate(Search *search, Mobility *mob, const int best_moves[]){
    mob->value = 0;
    if (mob->pos == best_moves[0])
        mob->value = W_BEST1_MOVE;
    else if (mob->pos == best_moves[1])
        mob->value = W_BEST2_MOVE;
    else if (mob->pos == best_moves[2])
        mob->value = W_BEST3_MOVE;
    else{
        mob->value += cell_weight[mob->pos] * W_CELL_WEIGHT;
        if (search->board.parity & cell_div4[mob->pos])
            mob->value += W_PARITY;
        search->board.move(mob);
            int l, u;
            parent_transpose_table.get_prev(&search->board, search->board.hash() & TRANSPOSE_TABLE_MASK, &l, &u);
            if (u != INF)
                mob->value += W_CACHE_HIT + W_CACHE_HIGH - u * W_CACHE_VALUE;
            else if (l != -INF)
                mob->value += W_CACHE_HIT - l * W_CACHE_VALUE;
            mob->value += -mid_evaluate(&search->board) * W_EVALUATE;
            if (search->board.p == BLACK)
                mob->value += calc_surround(search->board.b, ~(search->board.b | search->board.w)) * W_SURROUND;
            else
                mob->value += calc_surround(search->board.w, ~(search->board.b | search->board.w)) * W_SURROUND;
            /*
            int stab0, stab1;
            calc_stability_fast(&search->board, &stab0, &stab1);
            if (search->board.p == BLACK)
                mob->value += stab1 * W_STABILITY;
            else
                mob->value += stab0 * W_STABILITY;
            */
            mob->value -= pop_count_ull(search->board.mobility_ull()) * W_MOBILITY;
        search->board.undo(mob);
    }
}

inline void move_ordering(Search *search, vector<Mobility> &move_list){
    if (move_list.size() < 2)
        return;
    int best_moves[N_BEST_MOVES];
    int hash_code = search->board.hash() & TRANSPOSE_TABLE_MASK;
    child_transpose_table.get_prev(&search->board, hash_code, best_moves);
    for (Mobility &mob: move_list)
        move_evaluate(search, &mob, best_moves);
    sort(move_list.begin(), move_list.end(), cmp_move_ordering);
}

inline void move_evaluate_fast_first(Search *search, Mobility *mob, const int best_moves[]){
    mob->value = 0;
    if (mob->pos == best_moves[0])
        mob->value = W_BEST1_MOVE;
    else if (mob->pos == best_moves[1])
        mob->value = W_BEST2_MOVE;
    else if (mob->pos == best_moves[2])
        mob->value = W_BEST3_MOVE;
    else{
        //mob->value += cell_weight[mob->pos] * W_END_CELL_WEIGHT;
        if (search->board.parity & cell_div4[mob->pos])
            mob->value += W_END_PARITY;
        search->board.move(mob);
            //mob->value += -mid_evaluate(&search->board) * W_END_EVALUATE;
            if (search->board.p == BLACK)
                mob->value += calc_surround(search->board.b, ~(search->board.b | search->board.w)) * W_END_SURROUND;
            else
                mob->value += calc_surround(search->board.w, ~(search->board.b | search->board.w)) * W_END_SURROUND;
            /*
            int stab0, stab1;
            calc_stability_fast(&search->board, &stab0, &stab1);
            if (search->board.p == BLACK)
                mob->value += stab1 * W_END_STABILITY;
            else
                mob->value += stab0 * W_END_STABILITY;
            */
            mob->value -= pop_count_ull(search->board.mobility_ull()) * W_END_MOBILITY;
        search->board.undo(mob);
    }
}

inline void move_ordering_fast_first(Search *search, vector<Mobility> &move_list){
    if (move_list.size() < 2)
        return;
    int best_moves[N_BEST_MOVES];
    int hash_code = search->board.hash() & TRANSPOSE_TABLE_MASK;
    child_transpose_table.get_prev(&search->board, hash_code, best_moves);
    for (Mobility &mob: move_list)
        move_evaluate_fast_first(search, &mob, best_moves);
    sort(move_list.begin(), move_list.end(), cmp_move_ordering);
}

inline void move_evaluate_fast_first_fast(Search *search, Mobility *mob, const int best_moves[]){
    mob->value = 0;
    if (mob->pos == best_moves[0])
        mob->value = W_BEST1_MOVE;
    else if (mob->pos == best_moves[1])
        mob->value = W_BEST2_MOVE;
    else if (mob->pos == best_moves[2])
        mob->value = W_BEST3_MOVE;
    else if (search->board.parity & cell_div4[mob->pos])
            mob->value += W_END_PARITY;
}

inline void move_ordering_fast_first_fast(Search *search, vector<Mobility> &move_list){
    if (move_list.size() < 2)
        return;
    int best_moves[N_BEST_MOVES];
    int hash_code = search->board.hash() & TRANSPOSE_TABLE_MASK;
    if (!child_transpose_table.get_now(&search->board, hash_code, best_moves))
        child_transpose_table.get_prev(&search->board, hash_code, best_moves);
    for (Mobility &mob: move_list)
        move_evaluate_fast_first_fast(search, &mob, best_moves);
    sort(move_list.begin(), move_list.end(), cmp_move_ordering);
}

inline void move_ordering_value(vector<Mobility> &move_list){
    if (move_list.size() < 2)
        return;
    sort(move_list.begin(), move_list.end(), cmp_move_ordering);
}

inline int stability_cut(Search *search, int *alpha, int *beta){
    int stab[2];
    calc_stability(&search->board, &stab[0], &stab[1]);
    int n_alpha = 2 * stab[search->board.p] - HW2;
    int n_beta = HW2 - 2 * stab[1 - search->board.p];
    if (*beta <= n_alpha)
        return n_alpha;
    if (n_beta <= *alpha)
        return n_beta;
    if (n_beta <= n_alpha)
        return n_alpha;
    *alpha = max(*alpha, n_alpha);
    *beta = min(*beta, n_beta);
    return SCORE_UNDEFINED;
}
