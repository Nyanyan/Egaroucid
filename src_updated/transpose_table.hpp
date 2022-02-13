#pragma once
#include "setting.hpp"
#include "common.hpp"
#include "board.hpp"
#if USE_MULTI_THREAD
    #include "thread_pool.hpp"
    #include <math.h>
    #include <future>
    #include <vector>
#endif
#if USE_BOOST
    #include <boost/atomic/atomic.hpp>
#else
    #include <atomic>
#endif

using namespace std;

#define TRANSPOSE_TABLE_SIZE 1048576
#define TRANSPOSE_TABLE_MASK 1048575

#define TRANSPOSE_TABLE_UNDEFINED -INF

#define N_BEST_MOVES 3

class Node_child_transpose_table{
    public:
        atomic<unsigned long long> b;
        atomic<unsigned long long> w;
        atomic<int> p;
        atomic<int> best_moves[N_BEST_MOVES];
        atomic<int> best_value;

    public:
        inline void init(){
            b.store(0, memory_order_relaxed);
            w.store(0, memory_order_relaxed);
            p.store(-1, memory_order_relaxed);
            best_value.store(-INF, memory_order_relaxed);
        }

        inline void register_value(const Board *board, const int policy, const int value){
            b.store(board->b);
            w.store(board->w);
            p.store(board->p);
            best_moves[0].store(policy);
            best_moves[1].store(TRANSPOSE_TABLE_UNDEFINED);
            best_moves[2].store(TRANSPOSE_TABLE_UNDEFINED);
            best_value.store(value);
        }

        inline void register_value(const int policy, const int value){
            if (best_value.load(memory_order_relaxed) < value){
                best_value.store(value);
                best_moves[2].store(best_moves[1].load(memory_order_relaxed));
                best_moves[1].store(best_moves[0].load(memory_order_relaxed));
                best_moves[0].store(policy);
            }
        }

        inline void register_value(const Board *board, const int policies[], const int value){
            b.store(board->b);
            w.store(board->w);
            p.store(board->p);
            best_value.store(value);
            best_moves[0].store(policies[0]);
            best_moves[1].store(policies[1]);
            best_moves[2].store(policies[2]);
        }

        inline void register_value(const int policies[], const int value){
            best_value.store(value);
            best_moves[0].store(policies[0]);
            best_moves[1].store(policies[1]);
            best_moves[2].store(policies[2]);
        }

        inline void get(int b[]) const{
            b[0] = best_moves[0].load(memory_order_relaxed);
            b[1] = best_moves[1].load(memory_order_relaxed);
            b[2] = best_moves[2].load(memory_order_relaxed);
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
                    s = min(TRANSPOSE_TABLE_SIZE - 1, s + delta);
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
                    s = min(TRANSPOSE_TABLE_SIZE - 1, s + delta);
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
            if (!compare_key(board, &table[now][hash])){
                table[now][hash].register_value(board, policy, value);
                //++n_reg;
            } else
                table[now][hash].register_value(policy, value);
        }

        inline void reg(const Board *board, const int hash, const int policies[], const int value){
            if (!compare_key(board, &table[now][hash])){
                table[now][hash].register_value(board, policies, value);
                //++n_reg;
            } else
                table[now][hash].register_value(policies, value);
        }

        inline bool get_now(Board *board, const int hash, int best_moves[]) const{
            if (compare_key(board, &table[now][hash])){
                table[now][hash].get(best_moves);
                return true;
            }
            best_moves[0] = TRANSPOSE_TABLE_UNDEFINED;
            best_moves[1] = TRANSPOSE_TABLE_UNDEFINED;
            best_moves[2] = TRANSPOSE_TABLE_UNDEFINED;
            return false;
        }

        inline bool get_prev(Board *board, const int hash, int best_moves[]) const{
            if (compare_key(board, &table[prev][hash])){
                table[prev][hash].get(best_moves);
                return true;
            }
            best_moves[0] = TRANSPOSE_TABLE_UNDEFINED;
            best_moves[1] = TRANSPOSE_TABLE_UNDEFINED;
            best_moves[2] = TRANSPOSE_TABLE_UNDEFINED;
            return false;
        }

        inline int get_best_value(Board *board, const int hash){
            if (compare_key(board, &table[now][hash]))
                return table[now][hash].best_value.load(memory_order_relaxed);
            return TRANSPOSE_TABLE_UNDEFINED;
        }

        inline int get_n_reg() const{
            return n_reg.load();
        }

    private:
        inline bool compare_key(const Board *a, const Node_child_transpose_table *b) const{
            return a->p == b->p.load(memory_order_relaxed) && a->b == b->b.load(memory_order_relaxed) && a->w == b->w.load(memory_order_relaxed);
        }
};


class Node_parent_transpose_table{
    public:
        atomic<unsigned long long> b;
        atomic<unsigned long long> w;
        atomic<int> p;
        atomic<int> lower;
        atomic<int> upper;

    public:
        inline void init(){
            b.store(0, memory_order_relaxed);
            w.store(0, memory_order_relaxed);
            p.store(-1, memory_order_relaxed);
        }

        inline void register_value(const Board *board, const int l, const int u){
            b.store(board->b);
            w.store(board->w);
            p.store(board->p);
            lower.store(l);
            upper.store(u);
        }

        inline void register_value(const int l, const int u){
            if (lower.load(memory_order_relaxed) < l)
                lower.store(l);
            if (u < upper.load(memory_order_relaxed))
                upper.store(u);
        }

        inline void get(int *l, int *u) const{
            *l = lower.load(memory_order_relaxed);
            *u = upper.load(memory_order_relaxed);
        }
};

#if USE_MULTI_THREAD
    void parent_init_p(int id, Node_parent_transpose_table table[2][TRANSPOSE_TABLE_SIZE], const int idx, const int s, const int e){
        for(int i = s; i < e; ++i)
            table[idx][i].init();
    }
#endif

class Parent_transpose_table{
    private:
        int prev;
        int now;
        Node_parent_transpose_table table[2][TRANSPOSE_TABLE_SIZE];

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
            inline void init_now(){
                const int thread_size = thread_pool.size();
                const int delta = ceil((double)TRANSPOSE_TABLE_SIZE / thread_size);
                int s = 0;
                vector<future<void>> init_future;
                for (int i = 0; i < thread_size; ++i){
                    init_future.emplace_back(thread_pool.push(parent_init_p, table, now, s, min(TRANSPOSE_TABLE_SIZE, s + delta)));
                    s = min(TRANSPOSE_TABLE_SIZE - 1, s + delta);
                }
                for (int i = 0; i < thread_size; ++i)
                    init_future[i].get();
            }

            inline void init_prev(){
                const int thread_size = thread_pool.size();
                const int delta = ceil((double)TRANSPOSE_TABLE_SIZE / thread_size);
                int s = 0;
                vector<future<void>> init_future;
                for (int i = 0; i < thread_size; ++i){
                    init_future.emplace_back(thread_pool.push(parent_init_p, table, prev, s, min(TRANSPOSE_TABLE_SIZE, s + delta)));
                    s = min(TRANSPOSE_TABLE_SIZE - 1, s + delta);
                }
                for (int i = 0; i < thread_size; ++i)
                    init_future[i].get();
            }
        #else
            inline void init_now(){
                for(int i = 0; i < TRANSPOSE_TABLE_SIZE; ++i)
                    table[now][i].init();
            }

            inline void init_prev(){
                for(int i = 0; i < TRANSPOSE_TABLE_SIZE; ++i)
                    table[prev][i].init();
            }
        #endif

        inline void reg(const Board *board, const int hash, const int l, const int u){
            if (!compare_key(board, &table[now][hash]))
                table[now][hash].register_value(board, l, u);
            else
                table[now][hash].register_value(l, u);
        }

        inline void get_now(Board *board, const int hash, int *l, int *u) const{
            if (compare_key(board, &table[now][hash])){
                table[now][hash].get(l, u);
                return;
            }
            *l = -INF;
            *u = INF;
        }

        inline void get_prev(Board *board, const int hash, int *l, int *u) const{
            if (compare_key(board, &table[prev][hash])){
                table[prev][hash].get(l, u);
                return;
            }
            *l = -INF;
            *u = INF;
        }

    private:
        inline bool compare_key(const Board *a, const Node_parent_transpose_table *b) const{
            return a->p == b->p.load(memory_order_relaxed) && a->b == b->b.load(memory_order_relaxed) && a->w == b->w.load(memory_order_relaxed);
        }
};

Parent_transpose_table parent_transpose_table;
Child_transpose_table child_transpose_table;
