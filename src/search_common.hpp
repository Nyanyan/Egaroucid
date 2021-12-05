#pragma once
#include "setting.hpp"
#include "common.hpp"
#include "board.hpp"
#include "evaluate.hpp"

using namespace std;

#define search_epsilon 1
#define cache_hit 10000
#define cache_both 1000
#define parity_vacant_bonus 50
#define canput_bonus 100
#define mtd_threshold 400
#define mtd_threshold_final 100

#define mpc_min_depth 3
#define mpc_max_depth 10
#define mpc_min_depth_final 9
#define mpc_max_depth_final 28
#define mpct_final 4.0

#define simple_mid_threshold 3
#define simple_end_threshold 5

#define po_max_depth 10

#define extra_stability_threshold 58

#define search_hash_table_size 1048576
constexpr int search_hash_mask = search_hash_table_size - 1;

const int cell_weight[hw2] = {
    120, -20, 20, 5, 5, 20, -20, 120,
    -20, -40, -5, -5, -5, -5, -40, -20,
    20, -5, 15, 3, 3, 15, -5, 20,
    5, -5, 3, 3, 3, 3, -5, 5,
    5, -5, 3, 3, 3, 3, -5, 5,
    20, -5, 15, 3, 3, 15, -5, 20,
    -20, -40, -5, -5, -5, -5, -40, -20,
    120, -20, 20, 5, 5, 20, -20, 120
};

const int mpcd[30]={0, 1, 0, 1, 2, 3, 2, 3, 4, 3, 4, 3, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5};
const double mpct[n_phases]={1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5};
const double mpcsd[n_phases][mpc_max_depth - mpc_min_depth + 1]={
    {354, 345, 362, 393, 384, 220, 309, 158},
    {293, 277, 257, 318, 261, 271, 243, 259},
    {347, 276, 293, 333, 356, 235, 335, 264},
    {376, 364, 271, 351, 368, 274, 331, 316},
    {335, 323, 282, 389, 343, 321, 354, 363},
    {441, 348, 352, 452, 428, 350, 411, 426},
    {420, 384, 342, 464, 405, 386, 484, 391},
    {399, 363, 346, 556, 481, 479, 571, 490},
    {493, 487, 390, 626, 529, 497, 651, 496},
    {505, 470, 422, 725, 613, 608, 797, 654},
    {562, 605, 614, 680, 687, 594, 701, 617},
    {473, 343, 286, 360, 267, 244, 333, 271}
};
const double mpcsd_final[mpc_max_depth_final - mpc_min_depth_final + 1] = {
    669, 743, 614, 557, 542, 636, 605, 589, 551, 564, 581, 521, 533, 492, 526, 516, 483, 524, 450, 497
};
int mpctsd[n_phases][mpc_max_depth + 1];
int mpctsd_final[mpc_max_depth_final + 1];
unsigned long long can_be_flipped[hw2];

int searched_nodes;
vector<int> vacant_lst;

struct search_node{
    bool reg;
    int p;
    int k[hw];
    int l;
    int u;
};

struct search_result{
    int policy;
    int value;
    int depth;
    int nps;
};

class transpose_table{
    public:
        search_node table[2][search_hash_table_size];
        int prev;
        int now;
        int hash_reg;
        int hash_get;

    public:
        inline void init_prev(){
            for(int i = 0; i < search_hash_table_size; ++i)
                this->table[this->prev][i].reg = false;
        }

        inline void init_now(){
            for(int i = 0; i < search_hash_table_size; ++i)
                this->table[this->now][i].reg = false;
        }

        inline void reg(board *key, int hash, int l, int u){
            ++this->hash_reg;
            this->table[this->now][hash].reg = true;
            this->table[this->now][hash].p = key->p;
            for (int i = 0; i < hw; ++i)
                this->table[this->now][hash].k[i] = key->b[i];
            this->table[this->now][hash].l = l;
            this->table[this->now][hash].u = u;
        }

        inline void get_now(board *key, const int hash, int *l, int *u){
            if (table[this->now][hash].reg){
                if (compare_key(key, &table[this->now][hash])){
                    *l = table[this->now][hash].l;
                    *u = table[this->now][hash].u;
                    ++this->hash_get;
                    return;
                }
            }
            *l = -inf;
            *u = inf;
        }

        inline void get_prev(board *key, const int hash, int *l, int *u){
            if (table[this->prev][hash].reg){
                if (compare_key(key, &table[this->prev][hash])){
                    *l = table[this->prev][hash].l;
                    *u = table[this->prev][hash].u;
                    ++this->hash_get;
                    return;
                }
            }
            *l = -inf;
            *u = inf;
        }
    
    private:
        inline bool compare_key(board *a, search_node *b){
            return a->p == b->p && 
                a->b[0] == b->k[0] && a->b[1] == b->k[1] && a->b[2] == b->k[2] && a->b[3] == b->k[3] && 
                a->b[4] == b->k[4] && a->b[5] == b->k[5] && a->b[6] == b->k[6] && a->b[7] == b->k[7];
        }
};

transpose_table transpose_table;

inline void search_init(){
    int i, j;
    for (i = 0; i < n_phases; ++i){
        for (j = 0; j <= mpc_max_depth - mpc_min_depth; ++j)
            mpctsd[i][mpc_min_depth + j] = (int)(mpct[i] * mpcsd[i][j]);
    }
    for (i = 0; i <= mpc_max_depth_final - mpc_min_depth_final; ++i)
        mpctsd_final[mpc_min_depth_final + i] = (int)(mpct_final * mpcsd_final[i]);
    transpose_table.now = 0;
    transpose_table.prev = 1;
    transpose_table.init_now();
    transpose_table.init_prev();
    for (int cell = 0; cell < hw2; ++cell){
        can_be_flipped[cell] = 0b1111111111000011100000011000000110000001100000011100001111111111;
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
        b->v = u + cache_both + cache_hit;
    else if (u != inf)
        b->v += u + cache_hit;
    else if (l != -inf)
        b->v += l + cache_hit;
    else
        b->v = -mid_evaluate(b);
}

inline int calc_xx_stability(board *b, int p){
    return
        (pop_digit[b->b[1]][2] == p && pop_digit[b->b[0]][2] == p && pop_digit[b->b[0]][1] == p && pop_digit[b->b[1]][0] == p && pop_digit[b->b[1]][1] == p && (pop_digit[b->b[0]][3] == p || (pop_digit[b->b[2]][1] == p && pop_digit[b->b[3]][0] == p) || (pop_digit[b->b[0]][3] != vacant && pop_digit[b->b[2]][1] != vacant && pop_digit[b->b[3]][0] != vacant))) + 
        (pop_digit[b->b[1]][5] == p && pop_digit[b->b[0]][5] == p && pop_digit[b->b[0]][6] == p && pop_digit[b->b[1]][7] == p && pop_digit[b->b[1]][6] == p && (pop_digit[b->b[0]][4] == p || (pop_digit[b->b[2]][6] == p && pop_digit[b->b[3]][7] == p) || (pop_digit[b->b[0]][4] != vacant && pop_digit[b->b[2]][6] != vacant && pop_digit[b->b[3]][7] != vacant))) + 
        (pop_digit[b->b[6]][2] == p && pop_digit[b->b[7]][2] == p && pop_digit[b->b[7]][1] == p && pop_digit[b->b[6]][0] == p && pop_digit[b->b[6]][1] == p && (pop_digit[b->b[7]][3] == p || (pop_digit[b->b[5]][1] == p && pop_digit[b->b[4]][0] == p) || (pop_digit[b->b[7]][3] != vacant && pop_digit[b->b[5]][1] != vacant && pop_digit[b->b[4]][0] != vacant))) + 
        (pop_digit[b->b[6]][5] == p && pop_digit[b->b[7]][5] == p && pop_digit[b->b[7]][6] == p && pop_digit[b->b[6]][7] == p && pop_digit[b->b[6]][6] == p && (pop_digit[b->b[7]][4] == p || (pop_digit[b->b[5]][6] == p && pop_digit[b->b[4]][7] == p) || (pop_digit[b->b[7]][4] != vacant && pop_digit[b->b[5]][6] != vacant && pop_digit[b->b[4]][7] != vacant))) + 
        (pop_digit[b->b[9]][2] == p && pop_digit[b->b[8]][2] == p && pop_digit[b->b[8]][1] == p && pop_digit[b->b[9]][0] == p && pop_digit[b->b[9]][1] == p && (pop_digit[b->b[8]][3] == p || (pop_digit[b->b[10]][1] == p && pop_digit[b->b[11]][0] == p) || (pop_digit[b->b[8]][3] != vacant && pop_digit[b->b[10]][1] != vacant && pop_digit[b->b[11]][0] != vacant))) + 
        (pop_digit[b->b[9]][5] == p && pop_digit[b->b[8]][5] == p && pop_digit[b->b[8]][6] == p && pop_digit[b->b[9]][7] == p && pop_digit[b->b[9]][6] == p && (pop_digit[b->b[8]][4] == p || (pop_digit[b->b[10]][6] == p && pop_digit[b->b[11]][7] == p) || (pop_digit[b->b[8]][4] != vacant && pop_digit[b->b[10]][6] != vacant && pop_digit[b->b[11]][7] != vacant))) + 
        (pop_digit[b->b[14]][2] == p && pop_digit[b->b[15]][2] == p && pop_digit[b->b[15]][1] == p && pop_digit[b->b[14]][0] == p && pop_digit[b->b[14]][1] == p && (pop_digit[b->b[15]][3] == p || (pop_digit[b->b[13]][1] == p && pop_digit[b->b[12]][0] == p) || (pop_digit[b->b[15]][3] != vacant && pop_digit[b->b[13]][1] != vacant && pop_digit[b->b[12]][0] != vacant))) + 
        (pop_digit[b->b[14]][5] == p && pop_digit[b->b[15]][5] == p && pop_digit[b->b[15]][6] == p && pop_digit[b->b[14]][7] == p && pop_digit[b->b[14]][6] == p && (pop_digit[b->b[15]][4] == p || (pop_digit[b->b[13]][6] == p && pop_digit[b->b[12]][7] == p) || (pop_digit[b->b[15]][4] != vacant && pop_digit[b->b[13]][6] != vacant && pop_digit[b->b[12]][7] != vacant)));
}

inline int calc_x_stability(board *b, int p){
    return
        (pop_digit[b->b[1]][1] == p && (pop_digit[b->b[0]][2] == p || pop_digit[b->b[2]][0] == p || (pop_digit[b->b[0]][2] != vacant || pop_digit[b->b[2]][0] != vacant)) && pop_digit[b->b[0]][1] == p && pop_digit[b->b[1]][0] == p && pop_digit[b->b[0]][0] == p) + 
        (pop_digit[b->b[1]][6] == p && (pop_digit[b->b[0]][5] == p || pop_digit[b->b[2]][7] == p || (pop_digit[b->b[0]][5] != vacant || pop_digit[b->b[2]][7] != vacant)) && pop_digit[b->b[0]][6] == p && pop_digit[b->b[1]][7] == p && pop_digit[b->b[0]][7] == p) + 
        (pop_digit[b->b[6]][1] == p && (pop_digit[b->b[7]][2] == p || pop_digit[b->b[5]][0] == p || (pop_digit[b->b[7]][2] != vacant || pop_digit[b->b[5]][0] != vacant)) && pop_digit[b->b[7]][1] == p && pop_digit[b->b[6]][0] == p && pop_digit[b->b[7]][0] == p) + 
        (pop_digit[b->b[6]][6] == p && (pop_digit[b->b[7]][5] == p || pop_digit[b->b[5]][7] == p || (pop_digit[b->b[7]][5] != vacant || pop_digit[b->b[5]][7] != vacant)) && pop_digit[b->b[7]][6] == p && pop_digit[b->b[6]][7] == p && pop_digit[b->b[7]][7] == p);
}

inline int calc_stability(board *b, int p, unsigned long long extra_stability){
    int res = 
        stability_edge_arr[p][b->b[0]] + stability_edge_arr[p][b->b[7]] + stability_edge_arr[p][b->b[8]] + stability_edge_arr[p][b->b[15]] + 
        stability_corner_arr[p][b->b[0]] + stability_corner_arr[p][b->b[7]] + 
        calc_x_stability(b, p); // + calc_xx_stability(b, p);
    int y, x;
    extra_stability >>= hw;
    for (y = 1; y < hw_m1; ++y){
        extra_stability >>= 1;
        for (x = 1; x < hw_m1; ++x){
            if ((extra_stability & 1) == 0 && pop_digit[b->b[y]][x] == p)
                ++res;
            extra_stability >>= 1;
        }
        extra_stability >>= 1;
    }
    return res;
}

inline int calc_stability(board *b, int p){
    return
        stability_edge_arr[p][b->b[0]] + stability_edge_arr[p][b->b[7]] + stability_edge_arr[p][b->b[8]] + stability_edge_arr[p][b->b[15]] + 
        stability_corner_arr[p][b->b[0]] + stability_corner_arr[p][b->b[7]] + 
        calc_x_stability(b, p); // + calc_xx_stability(b, p);
}

inline unsigned long long calc_extra_stability(board *b){
    unsigned long long extra_stability = 0b1111111111000011100000011000000110000001100000011100001111111111;
    for (const &cell: vacant_lst){
        if (pop_digit[b->b[cell / hw]][cell % hw] == vacant)
            extra_stability |= can_be_flipped[cell];
    }
    return extra_stability;
}

inline bool stability_cut(board *b, int *alpha, int *beta){
    if (b->n >= extra_stability_threshold){
        unsigned long long extra_stability = calc_extra_stability(b);
        *alpha = max(*alpha, step * (2 * calc_stability(b, 1 - b->p, extra_stability) - hw2));
        *beta = min(*beta, step * (hw2 - 2 * calc_stability(b, b->p, extra_stability)));
    } else{
        *alpha = max(*alpha, step * (2 * calc_stability(b, 1 - b->p) - hw2));
        *beta = min(*beta, step * (hw2 - 2 * calc_stability(b, b->p)));
    }
    return *alpha >= *beta;
}

inline int calc_canput_exact(board *b){
    int res = 0;
    for (const int &cell: vacant_lst)
        res += b->legal(cell);
    return res;
}
