#pragma once
#include "setting.hpp"
#include "common.hpp"
#include "board.hpp"
#if USE_MULTI_THREAD
    #include <mutex>
#endif

#define search_hash_table_size 1048576
constexpr int search_hash_mask = search_hash_table_size - 1;

class search_node{
    public:
        bool reg;
        uint_fast16_t k[hw];
        int p;
        int l;
        int u;
    #if USE_MULTI_THREAD
        private:
            mutex mtx;
    #endif
    public:
        inline void register_value(uint_fast16_t key[], int pp, int ll, int uu){
            #if USE_MULTI_THREAD
                lock_guard<mutex> lock(mtx);
            #endif
            reg = true;
            for (int i = 0; i < hw; ++i)
                k[i] = key[i];
            p = pp;
            l = ll;
            u = uu;
        }

        inline void register_value(int ll, int uu){
            #if USE_MULTI_THREAD
                lock_guard<mutex> lock(mtx);
            #endif
            l = ll;
            u = uu;
        }

        inline void get(int *ll, int *uu){
            #if USE_MULTI_THREAD
                lock_guard<mutex> lock(mtx);
            #endif
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
        #if USE_MULTI_THREAD
            mutex mtx;
        #endif

    public:
        inline void init_prev(){
            #if USE_MULTI_THREAD
                lock_guard<mutex> lock(mtx);
            #endif
            for(int i = 0; i < search_hash_table_size; ++i)
                this->table[this->prev][i].reg = false;
        }

        inline void init_now(){
            #if USE_MULTI_THREAD
                lock_guard<mutex> lock(mtx);
            #endif
            for(int i = 0; i < search_hash_table_size; ++i)
                this->table[this->now][i].reg = false;
        }

        inline void reg(board *key, int hash, int l, int u){
            #if USE_MULTI_THREAD
                lock_guard<mutex> lock(mtx);
            #endif
            //++this->hash_reg;
            if (!this->table[this->now][hash].reg)
                this->table[this->now][hash].register_value(key->b, key->p, l, u);
            else if (key->p != this->table[this->now][hash].p || !compare_key(key->b, this->table[this->now][hash].k))
                this->table[this->now][hash].register_value(key->b, key->p, l, u);
            else
                this->table[this->now][hash].register_value(l, u);
        }

        inline void get_now(board *key, const int hash, int *l, int *u){
            //#if USE_MULTI_THREAD
            //    lock_guard<mutex> lock(mtx);
            //#endif
            if (this->table[this->now][hash].reg){
                if (key->p == this->table[this->now][hash].p && compare_key(key->b, this->table[this->now][hash].k)){
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
            //#if USE_MULTI_THREAD
            //    lock_guard<mutex> lock(mtx);
            //#endif
            if (this->table[this->prev][hash].reg){
                if (key->p == this->table[this->prev][hash].p && compare_key(key->b, this->table[this->prev][hash].k)){
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
        inline bool compare_key(uint_fast16_t b[], uint_fast16_t k[]){
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