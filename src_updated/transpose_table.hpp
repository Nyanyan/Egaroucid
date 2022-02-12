#pragma once
#include <atomic>
#include "setting.hpp"
#include "common.hpp"
#include "board.hpp"
#if USE_MULTI_THREAD
    #include "thread_pool.hpp"
    #include <math.h>
    #include <future>
    #include <vector>
#endif

#define TRANSPOSE_TABLE_SIZE 1048576
#define TRANSPOSE_TABLE_MASK 1048575

#define TRANSPOSE_TABLE_UNDEFINED -INF

#define N_BEST_MOVES 3

class Node_child_transpose_table{
    public:
        atomic<bool> reg;
        atomic<unsigned long long> b;
        atomic<unsigned long long> w;
        atomic<int> p;
        atomic<int> best_moves[N_BEST_MOVES];
        atomic<int> best_value;

    public:
        inline void init(){
            reg = false;
        }

        inline void register_value(const Board *board, const int policy, const int value){
            reg = true;
            b = board->b;
            w = board->w;
            p = board->p;
            best_moves[0] = policy;
            best_moves[1] = TRANSPOSE_TABLE_UNDEFINED;
            best_moves[2] = TRANSPOSE_TABLE_UNDEFINED;
            best_value = value;
        }

        inline void register_value(const int policy, const int value){
            if (best_value < value){
                best_value = value;
                best_moves[2] = best_moves[1].load();
                best_moves[1] = best_moves[0].load();
                best_moves[0] = policy;
            }
        }

        inline void get(int b[]) const{
            b[0] = best_moves[0].load();
            b[1] = best_moves[1].load();
            b[2] = best_moves[2].load();
        }
};

#if USE_MULTI_THREAD
    void child_init_p(int id, Node_child_transpose_table table[2][TRANSPOSE_TABLE_SIZE], const int idx, const int s, const int e){
        for(int i = s; i < e; ++i)
            table[idx][i].init();
    }
#endif

class Child_transpose_table{
    private:
        int prev;
        int now;
        Node_child_transpose_table table[2][TRANSPOSE_TABLE_SIZE];
        atomic<int> n_reg;

    public:
        inline void init(){
            now = 0;
            prev = 1;
            init_now();
            init_prev();
        }

        inline void ready_next_search(){
            swap(now, prev);
            init_now();
        }

        #if USE_MULTI_THREAD
            inline void init_prev(){
                const int thread_size = thread_pool.size();
                const int delta = ceil((double)TRANSPOSE_TABLE_SIZE / thread_size);
                int s = 0;
                vector<future<void>> init_future;
                for (int i = 0; i < thread_size; ++i){
                    init_future.emplace_back(thread_pool.push(child_init_p, table, prev, s, min(TRANSPOSE_TABLE_SIZE, s + delta)));
                    s += delta;
                }
                for (int i = 0; i < thread_size; ++i)
                    init_future[i].get();
            }

            inline void init_now(){
                const int thread_size = thread_pool.size();
                const int delta = ceil((double)TRANSPOSE_TABLE_SIZE / thread_size);
                int s = 0;
                vector<future<void>> init_future;
                for (int i = 0; i < thread_size; ++i){
                    init_future.emplace_back(thread_pool.push(child_init_p, table, now, s, min(TRANSPOSE_TABLE_SIZE, s + delta)));
                    s += delta;
                }
                for (int i = 0; i < thread_size; ++i)
                    init_future[i].get();
            }
        #else
            inline void init_prev(){
                for(int i = 0; i < TRANSPOSE_TABLE_SIZE; ++i)
                    table[prev][i].init();
            }

            inline void init_now(){
                for(int i = 0; i < TRANSPOSE_TABLE_SIZE; ++i)
                    table[now][i].init();
            }
        #endif

        inline int now_idx() const{
            return now;
        }

        inline int prev_idx() const{
            return prev;
        }

        inline void reg(const Board *board, const int hash, const int policy, const int value){
            if (global_searching){
                if (!table[now][hash].reg){
                    table[now][hash].register_value(board, policy, value);
                    //++n_reg;
                } else if (!compare_key(board, &table[now][hash])){
                    table[now][hash].register_value(board, policy, value);
                    //++n_reg;
                } else
                    table[now][hash].register_value(policy, value);
            }
        }

        inline bool get_now(Board *board, const int hash, int best_moves[]) const{
            if (table[now][hash].reg){
                if (compare_key(board, &table[now][hash])){
					table[now][hash].get(best_moves);
                    return true;
                }
            }
            best_moves[0] = TRANSPOSE_TABLE_UNDEFINED;
            best_moves[1] = TRANSPOSE_TABLE_UNDEFINED;
            best_moves[2] = TRANSPOSE_TABLE_UNDEFINED;
            return false;
        }

        inline bool get_prev(Board *board, const int hash, int best_moves[]) const{
            if (table[prev][hash].reg){
                if (compare_key(board, &table[prev][hash])){
					table[prev][hash].get(best_moves);
                    return true;
                }
            }
            best_moves[0] = TRANSPOSE_TABLE_UNDEFINED;
            best_moves[1] = TRANSPOSE_TABLE_UNDEFINED;
            best_moves[2] = TRANSPOSE_TABLE_UNDEFINED;
            return false;
        }

        inline int get_n_reg() const{
            return n_reg.load();
        }

    private:
        inline bool compare_key(const Board *a, const Node_child_transpose_table *b) const{
            return a->p == b->p && a->b == b->b && a->w == b->w;
        }
};


class Node_parent_transpose_table{
    public:
        atomic<bool> reg;
        atomic<unsigned long long> b;
        atomic<unsigned long long> w;
        atomic<int> p;
        atomic<int> lower;
        atomic<int> upper;

    public:
        inline void init(){
            reg = false;
        }

        inline void register_value(const Board *board, const int l, const int u){
            reg = true;
            b = board->b;
            w = board->w;
            p = board->p;
            lower = l;
            upper = u;
        }

        inline void register_value(const int l, const int u){
            if (lower < l)
                lower = l;
            if (u < upper)
                upper = u;
        }

        inline void get(int *l, int *u) const{
            *l = lower;
            *u = upper;
        }
};

#if USE_MULTI_THREAD
    void parent_init_p(int id, Node_parent_transpose_table table[], const int s, const int e){
        for(int i = s; i < e; ++i)
            table[i].init();
    }
#endif

class Parent_transpose_table{
    private:
        Node_parent_transpose_table table[TRANSPOSE_TABLE_SIZE];

    public:
        #if USE_MULTI_THREAD
            inline void init(){
                const int thread_size = thread_pool.size();
                const int delta = ceil((double)TRANSPOSE_TABLE_SIZE / thread_size);
                int s = 0;
                vector<future<void>> init_future;
                for (int i = 0; i < thread_size; ++i){
                    init_future.emplace_back(thread_pool.push(parent_init_p, table, s, min(TRANSPOSE_TABLE_SIZE, s + delta)));
                    s += delta;
                }
                for (int i = 0; i < thread_size; ++i)
                    init_future[i].get();
            }
        #else
            inline void init(){
                for(int i = 0; i < TRANSPOSE_TABLE_SIZE; ++i)
                    table[i].init();
            }
        #endif

        inline void reg(const Board *board, const int hash, const int l, const int u){
            if (global_searching){
                if (!table[hash].reg)
                    table[hash].register_value(board, l, u);
                else if (!compare_key(board, &table[hash]))
                    table[hash].register_value(board, l, u);
                else
                    table[hash].register_value(l, u);
            }
        }

        inline void get(Board *board, const int hash, int *l, int *u) const{
            if (table[hash].reg){
                if (compare_key(board, &table[hash])){
					table[hash].get(l, u);
                    return;
                }
            }
            *l = -INF;
            *u = INF;
        }

    private:
        inline bool compare_key(const Board *a, const Node_parent_transpose_table *b) const{
            return a->p == b->p && a->b == b->b && a->w == b->w;
        }
};

Parent_transpose_table parent_transpose_table;
Child_transpose_table child_transpose_table;
