#pragma once
#include "setting.hpp"
#include "common.hpp"
#include "board.hpp"
#include "evaluate.hpp"
#include "transpose_table.hpp"

using namespace std;

#define search_epsilon 1
constexpr int cache_hit = step * 100;
constexpr int cache_both = step * 100;
constexpr int parity_vacant_bonus = step * 10;
constexpr int canput_bonus = step / 10;
constexpr int mtd_threshold = step * 4;
constexpr int mtd_end_threshold = step * 5;

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
    const double mpcsd[n_phases][mpc_max_depth - mpc_min_depth + 1]={
        {46, 64, 55, 71, 110, 114, 137, 151},
        {180, 185, 148, 203, 148, 155, 184, 170},
        {191, 181, 169, 210, 199, 200, 255, 227},
        {233, 194, 217, 281, 272, 235, 284, 265},
        {299, 258, 236, 300, 286, 292, 365, 318},
        {320, 257, 237, 403, 297, 344, 409, 431},
        {309, 331, 265, 391, 418, 367, 588, 501},
        {433, 476, 309, 590, 516, 525, 621, 651},
        {400, 455, 347, 578, 569, 607, 838, 795},
        {414, 396, 359, 553, 373, 300, 414, 336}
    };
    const double mpcsd_final[mpc_max_depth_final - mpc_min_depth_final + 1] = {
        662, 639, 693, 683, 673, 673, 647, 618, 687, 678, 654, 723, 711, 733, 724, 722, 758, 759, 753, 791
    };
#else
    const double mpcsd[n_phases][mpc_max_depth - mpc_min_depth + 1]={
        {78, 75, 90, 148, 133, 139, 184, 186},
        {180, 172, 161, 213, 155, 146, 171, 172},
        {238, 211, 187, 228, 252, 217, 244, 218},
        {279, 212, 260, 320, 249, 246, 324, 357},
        {315, 251, 257, 346, 357, 333, 386, 318},
        {317, 255, 257, 331, 401, 258, 443, 387},
        {448, 406, 345, 575, 448, 484, 602, 477},
        {348, 434, 357, 619, 436, 586, 586, 596},
        {479, 338, 428, 649, 648, 540, 822, 747},
        {469, 455, 323, 507, 273, 223, 392, 245}
    };
    const double mpcsd_final[mpc_max_depth_final - mpc_min_depth_final + 1] = {
        639, 609, 748, 727, 723, 652, 628, 597, 658, 656, 631, 709, 695, 745, 734, 731, 747, 748, 746, 797
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
        *alpha = max(*alpha, step * (2 * (calc_stability(b, b->p) + ps) - hw2));
        *beta = min(*beta, step * (hw2 - 2 * (calc_stability(b, 1 - b->p) + os)));
    } else{
        *alpha = max(*alpha, step * (2 * calc_stability(b, b->p) - hw2));
        *beta = min(*beta, step * (hw2 - 2 * calc_stability(b, 1 - b->p)));
    }
    return *alpha >= *beta;
}

inline int calc_canput_exact(board *b){
    int res = 0;
    for (const int &cell: vacant_lst)
        res += b->legal(cell);
    return res;
}
