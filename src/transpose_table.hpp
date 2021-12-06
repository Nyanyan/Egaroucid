#pragma once
#include "setting.hpp"
#include "common.hpp"
#include "board.hpp"
#if USE_MULTI_THREAD
    #include <mutex>
#endif

#define search_hash_table_size 1048576
constexpr int search_hash_mask = search_hash_table_size - 1;

struct search_node{
    bool reg;
    int p;
    int k[hw];
    int l;
    int u;
};

class transpose_table{
    public:
        search_node table[2][search_hash_table_size];
        int prev;
        int now;
        int hash_reg;
        int hash_get;
    
    #if USE_MULTI_THREAD
        private:
            mutex mtx;
    #endif

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
            #if USE_MULTI_THREAD
                this->mtx.lock();
            #endif
            ++this->hash_reg;
            this->table[this->now][hash].reg = true;
            this->table[this->now][hash].p = key->p;
            for (int i = 0; i < hw; ++i)
                this->table[this->now][hash].k[i] = key->b[i];
            this->table[this->now][hash].l = l;
            this->table[this->now][hash].u = u;
            #if USE_MULTI_THREAD
                this->mtx.unlock();
            #endif
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

inline void transpose_table_init(){
    transpose_table.now = 0;
    transpose_table.prev = 1;
    transpose_table.init_now();
    transpose_table.init_prev();
    cerr << "transpose table initialized" << endl;
}