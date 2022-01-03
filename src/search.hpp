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
constexpr int canput_bonus = 1;
//constexpr int mtd_threshold = 0;
constexpr int mtd_end_threshold = 5;

#define mpc_min_depth 3
#define mpc_max_depth 10
#define mpc_min_depth_final 9
#define mpc_max_depth_final 28

#define simple_mid_threshold 3
#define simple_end_threshold 7

#define po_max_depth 8

#define extra_stability_threshold 58

#define ybwc_mid_first_num 1
#define ybwc_end_first_num 2
#define multi_thread_depth 1

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

const int mpcd[30] = {0, 1, 0, 1, 2, 3, 2, 3, 4, 3, 4, 3, 4, 5, 4, 5, 6, 5, 6, 7, 6, 7, 6, 7, 8, 7, 8, 9, 8, 9};
#if USE_MID_SMOOTH
    double mpcsd[n_phases][mpc_max_depth - mpc_min_depth + 1]={
        {32, 23, 19, 32, 43, 70, 105, 103},
        {127, 194, 125, 184, 176, 142, 190, 221},
        {235, 207, 172, 284, 220, 206, 285, 215},
        {256, 238, 213, 261, 275, 235, 295, 245},
        {236, 251, 225, 402, 306, 288, 343, 350},
        {364, 246, 279, 433, 345, 377, 404, 387},
        {296, 371, 358, 506, 532, 321, 450, 374},
        {369, 320, 308, 554, 550, 477, 608, 604},
        {468, 362, 442, 597, 538, 510, 729, 655},
        {372, 359, 242, 441, 329, 215, 283, 197}
    };
    double mpcsd_final[mpc_max_depth_final - mpc_min_depth_final + 1] = {
        527, 506, 557, 541, 535, 587, 558, 532, 522, 512, 489, 488, 477, 529, 529, 523, 534, 536, 542, 571
    };
#else
    double mpcsd[n_phases][mpc_max_depth - mpc_min_depth + 1]={
        {37, 30, 30, 30, 46, 96, 120, 132},
        {174, 159, 153, 188, 177, 167, 240, 158},
        {229, 211, 177, 273, 202, 203, 297, 233},
        {260, 220, 204, 380, 261, 242, 333, 236},
        {315, 297, 237, 326, 332, 339, 358, 309},
        {392, 288, 314, 426, 362, 323, 363, 336},
        {406, 315, 406, 486, 306, 372, 453, 349},
        {435, 376, 352, 461, 409, 382, 588, 537},
        {460, 364, 392, 618, 498, 500, 741, 555},
        {373, 281, 285, 391, 410, 209, 282, 264}
    };
    double mpcsd_final[mpc_max_depth_final - mpc_min_depth_final + 1] = {
        558, 535, 555, 534, 523, 570, 541, 516, 546, 543, 526, 483, 463, 519, 512, 509, 562, 563, 572, 565
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
    mpc_init();
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
        b->v = (u + l) / 2 + cache_hit + cache_both;
    else if (u != inf)
        b->v += u + cache_hit;
    else if (l != -inf)
        b->v += l + cache_hit;
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
