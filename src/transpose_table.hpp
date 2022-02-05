#pragma once
#include "setting.hpp"
#include "common.hpp"
#include "board.hpp"
#if USE_MULTI_THREAD
    #include <mutex>
#endif

#define search_hash_table_size 1048576
constexpr int search_hash_mask = search_hash_table_size - 1;
#define child_inf 100

class search_node{
    public:
        bool reg;
        unsigned long long b;
        unsigned long long w;
        int p;
        int l;
        int u;
    #if USE_MULTI_THREAD
        private:
            mutex mtx;
    #endif
    public:
        inline void register_value(board *bd, const int ll, const int uu){
            #if USE_MULTI_THREAD
                lock_guard<mutex> lock(mtx);
            #endif
            reg = true;
            b = bd->b;
            w = bd->w;
            p = bd->p;
            l = ll;
            u = uu;
        }

        inline void register_value(const int ll, const int uu){
            #if USE_MULTI_THREAD
                lock_guard<mutex> lock(mtx);
            #endif
            l = ll;
            u = uu;
        }

        inline void get(int *ll, int *uu){
            *ll = l;
            *uu = u;
        }
};

class transpose_table{
    public:
        int prev;
        int now;
        int hash_reg;
        int hash_get;
    
    private:
        search_node table[2][search_hash_table_size];

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
            //++this->hash_reg;
            if (!this->table[this->now][hash].reg)
                this->table[this->now][hash].register_value(key, l, u);
            else if (!compare_key(key, &this->table[this->now][hash]))
                this->table[this->now][hash].register_value(key, l, u);
            else
                this->table[this->now][hash].register_value(l, u);
        }

        inline void get_now(board *key, const int hash, int *l, int *u){
            if (this->table[this->now][hash].reg){
                if (compare_key(key, &this->table[this->now][hash])){
					this->table[this->now][hash].get(l, u);
                    //++this->hash_get;
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
                if (compare_key(key, &this->table[this->prev][hash])){
                    this->table[this->prev][hash].get(l, u);
                    //++this->hash_get;
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
        inline bool compare_key(board *a, search_node *b){
            return a->p == b->p && a->b == b->b && a->w == b->w;
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