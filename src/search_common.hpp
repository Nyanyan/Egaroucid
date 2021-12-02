#pragma once
#include "common.hpp"
#include "board.hpp"
#include "evaluate.hpp"

using namespace std;

#define search_epsilon 1
#define cache_hit 10000
#define cache_both 1000
#define mtd_threshold 400

#define mpc_min_depth 1
#define mpc_max_depth 10
#define mpc_min_depth_final 10
#define mpc_max_depth_final 28
#define mpct_final 1.1

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

const int mpcd[30]={0,0,0,1,2,1,2,3,4,3,4,3,4,5,6,5,6,5,6,7,6,7,8,9,8,9,10,11,10,11};
const double mpct[6]={1.6,1.6,1.6,1.5,1.5,1.4};
const double mpcsd[6][mpc_max_depth-mpc_min_depth+1]={
    {482,512,352,298,474,372,349,323,463,335},
    {312,381,310,261,354,322,291,313,389,371},
    {389,463,392,352,548,402,422,441,530,538},
    {417,490,436,405,570,494,452,438,527,524},
    {486,554,519,463,635,665,555,550,635,581},
    {433,517,430,391,560,556,383,345,567,332}
};
const double mpcsd_final[mpc_max_depth_final - mpc_min_depth_final + 1] = {
    1564, 1309, 1440, 1453, 1152, 1580, 1394, 1655, 1381, 1578, 1660, 1452, 1574, 1725, 1655, 1514, 1378, 1622, 1489
};
int mpctsd[6][mpc_max_depth + 1];
int mpctsd_final[mpc_max_depth_final + 1];

int searched_nodes;
vector<int> vacant_lst;

struct search_node{
    bool reg;
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

        inline void reg(const int key[], int hash, int l, int u){
            ++this->hash_reg;
            this->table[this->now][hash].reg = true;
            for (int i = 0; i < hw; ++i)
                this->table[this->now][hash].k[i] = key[i];
            this->table[this->now][hash].l = l;
            this->table[this->now][hash].u = u;
        }

        inline void get_now(const int key[], const int hash, int *l, int *u){
            if (table[this->now][hash].reg){
                if (compare_key(key, table[this->now][hash].k)){
                    *l = table[this->now][hash].l;
                    *u = table[this->now][hash].u;
                    ++this->hash_get;
                    return;
                }
            }
            *l = -inf;
            *u = -inf;
        }

        inline void get_prev(const int key[], const int hash, int *l, int *u){
            if (table[this->prev][hash].reg){
                if (compare_key(key, table[this->prev][hash].k)){
                    *l = table[this->prev][hash].l;
                    *u = table[this->prev][hash].u;
                    ++this->hash_get;
                    return;
                }
            }
            *l = -inf;
            *u = -inf;
        }
    
    private:
        inline bool compare_key(const int a[], const int b[]){
            return
                a[0] == b[0] && a[1] == b[1] && a[2] == b[2] && a[3] == b[3] && 
                a[4] == b[4] && a[5] == b[5] && a[6] == b[6] && a[7] == b[7];
        }
};

transpose_table transpose_table;

int cmp_vacant(int p, int q){
    return cell_weight[p] > cell_weight[q];
}

inline void move_ordering(board *b){
    int l, u;
    transpose_table.get_prev(b->b, b->hash() & search_hash_mask, &l, &u);
    b->v = -max(l, u);
    if (u != -inf && l != -inf)
        b->v += cache_both;
    if (u != -inf || l != -inf)
        b->v += cache_hit;
    else
        b->v = -mid_evaluate(b);
}

inline void search_common_init(){
    int i, j;
    for (i = 0; i < 6; ++i){
        for (j = 0; j < mpc_max_depth - mpc_min_depth + 1; ++j)
            mpctsd[i][mpc_min_depth + j] = (int)(mpct[i] * mpcsd[i][j]);
    }
    for (i = 0; i < mpc_max_depth_final - mpc_min_depth_final + 1; ++i)
        mpctsd_final[mpc_min_depth_final + i] = (int)(mpct_final * mpcsd_final[i]);
    transpose_table.now = 0;
    transpose_table.prev = 1;
    transpose_table.init_now();
    transpose_table.init_prev();
    cerr << "search common initialized" << endl;
}

