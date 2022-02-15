#pragma once
#include "setting.hpp"
#include "common.hpp"
#include "Board.hpp"
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
    #if USE_MULTI_THREAD && false
        private:
            mutex mtx;
    #endif
    public:
        inline void register_value(Board *bd, const int ll, const int uu){
            #if USE_MULTI_THREAD && false
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
            #if USE_MULTI_THREAD && false
                lock_guard<mutex> lock(mtx);
            #endif
            l = ll;
            u = uu;
        }

        inline void get(int *ll, int *uu){
            #if USE_MULTI_THREAD && false
                lock_guard<mutex> lock(mtx);
            #endif
            *ll = l;
            *uu = u;
        }

        inline void copy(search_node *to_node){
            to_node->reg = reg;
            to_node->b = b;
            to_node->w = w;
            to_node->p = p;
            to_node->l = l;
            to_node->u = u;
        }

        inline void merge(search_node *n_node){
            reg = n_node->reg;
            b = n_node->b;
            w = n_node->w;
            p = n_node->p;
            l = n_node->l;
            u = n_node->u;
        }
};

class Transpose_table{
    public:
        int prev;
        int now;
        int hash_reg;
        int hash_get;
        search_node table[2][search_hash_table_size];
        #if USE_MULTI_THREAD && false
            mutex mtx;
        #endif

    public:
        inline void init_prev(){
            #if USE_MULTI_THREAD && false
                lock_guard<mutex> lock(mtx);
            #endif
            for(int i = 0; i < search_hash_table_size; ++i)
                this->table[this->prev][i].reg = false;
        }

        inline void init_now(){
            #if USE_MULTI_THREAD && false
                lock_guard<mutex> lock(mtx);
            #endif
            for(int i = 0; i < search_hash_table_size; ++i)
                this->table[this->now][i].reg = false;
        }

        inline void copy_prev(Transpose_table *to_table){
            for(int i = 0; i < search_hash_table_size; ++i)
                this->table[this->prev][i].copy(&to_table->table[this->prev][i]);
        }

        inline void copy_now(Transpose_table *to_table){
            for(int i = 0; i < search_hash_table_size; ++i)
                this->table[this->now][i].copy(&to_table->table[this->now][i]);
        }

        inline void merge(Transpose_table *n_table){
            for(int i = 0; i < search_hash_table_size; ++i)
                this->table[this->now][i].merge(&n_table->table[this->now][i]);
        }

        inline void reg(Board *key, int hash, int l, int u){
            #if USE_MULTI_THREAD && false
                lock_guard<mutex> lock(mtx);
            #endif
            //++this->hash_reg;
            if (!this->table[this->now][hash].reg)
                this->table[this->now][hash].register_value(key, l, u);
            else if (!compare_key(key, &this->table[this->now][hash]))
                this->table[this->now][hash].register_value(key, l, u);
            else
                this->table[this->now][hash].register_value(l, u);
        }

        inline void get_now(Board *key, const int hash, int *l, int *u){
            #if USE_MULTI_THREAD && false
                lock_guard<mutex> lock(mtx);
            #endif
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

        inline void get_prev(Board *key, const int hash, int *l, int *u){
            #if USE_MULTI_THREAD && false
                lock_guard<mutex> lock(mtx);
            #endif
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
        inline bool compare_key(Board *a, search_node *b){
            return a->p == b->p && a->b == b->b && a->w == b->w;
        }
};

Transpose_table transpose_table;

inline void transpose_table_init(){
    transpose_table.now = 0;
    transpose_table.prev = 1;
    transpose_table.init_now();
    transpose_table.init_prev();
    cerr << "transpose table initialized" << endl;
}