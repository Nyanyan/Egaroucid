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
        int prev;
        int now;
        int hash_reg;
        int hash_get;
    
    private:
        search_node table[2][search_hash_table_size];
        #if USE_MULTI_THREAD
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
            if (this->table[this->now][hash].reg){
                if (compare_key(key->b, this->table[this->now][hash].k)){
                    *l = this->table[this->now][hash].l;
                    *u = this->table[this->now][hash].u;
                    ++this->hash_get;
                } else{
                    *l = -inf;
                    *u = inf;
                }
            } else{
                *l = -inf;
                *u = inf;
            }
        }

        inline void get_prev(board *key, const int hash, int *l, int *u){
            if (this->table[this->prev][hash].reg){
                if (compare_key(key->b, this->table[this->prev][hash].k)){
                    *l = this->table[this->prev][hash].l;
                    *u = this->table[this->prev][hash].u;
                    ++this->hash_get;
                } else{
                    *l = -inf;
                    *u = inf;
                }
            } else{
                *l = -inf;
                *u = inf;
            }
        }
    
    private:
        inline bool compare_key(int b[], int k[]){
            return 
                b[0] == k[0] && b[1] == k[1] && b[2] == k[2] && b[3] == k[3] && 
                b[4] == k[4] && b[5] == k[5] && b[6] == k[6] && b[7] == k[7];
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