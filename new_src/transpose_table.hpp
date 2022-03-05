#pragma once
#include "setting.hpp"
#include "common.hpp"
#include "board.hpp"
#include <atomic>
#if USE_MULTI_THREAD
    #include "thread_pool.hpp"
    #include <future>
#endif

using namespace std;

#define TRANSPOSE_TABLE_SIZE 33554432
#define TRANSPOSE_TABLE_MASK 33554431

#define TRANSPOSE_TABLE_UNDEFINED -INF

class Node_child_transpose_table{
    private:
        atomic<unsigned long long> player;
        atomic<unsigned long long> opponent;
        atomic<int> best_move;
        atomic<int> best_value;
        //atomic<Node_child_transpose_table*> p_n_node;

    public:

        inline void init(){
            free(this);
        }

        inline void register_value(const int policy, const int value){
            if (best_value.load(memory_order_relaxed) < value){
                best_value.store(value);
                best_move.store(policy);
            }
        }

        inline void register_value(const Board *board, const int policy, const int value){
            player.store(board->player);
            opponent.store(board->opponent);
            best_value.store(value);
            best_move.store(policy);
        }

        inline int get() const{
            return best_move.load();
        }

        inline bool compare(const Board *a) const{
            return a->player == player.load(memory_order_relaxed) && a->opponent == opponent.load(memory_order_relaxed);
        }
};

#if USE_MULTI_THREAD
    void init_child_transpose_table(int id, Node_child_transpose_table *table[], int s, int e){
        for(int i = s; i < e; ++i){
            if (table[i] != NULL){
                table[i]->init();
                table[i] = NULL;
            }
        }
    }
#endif

class Child_transpose_table{
    private:
        Node_child_transpose_table *table[TRANSPOSE_TABLE_SIZE];

    public:
        inline void first_init(){
            for(int i = 0; i < TRANSPOSE_TABLE_SIZE; ++i)
                table[i] = NULL;
        }

        #if USE_MULTI_THREAD
            inline void init(){
                int thread_size = thread_pool.size();
                int delta = (TRANSPOSE_TABLE_SIZE + thread_size - 1) / thread_size;
                int s = 0, e;
                vector<future<void>> tasks;
                for (int i = 0; i < thread_size; ++i){
                    e = min(TRANSPOSE_TABLE_SIZE, s + delta);
                    tasks.emplace_back(thread_pool.push(init_child_transpose_table, table, s, e));
                    s = e;
                }
                for (future<void> &task: tasks)
                    task.get();
            }
        #else
            inline void init(){
                for(int i = 0; i < TRANSPOSE_TABLE_SIZE; ++i){
                    if (table[i] != NULL){
                        table[i]->init();
                        table[i] = NULL;
                    }
                }
            }
        #endif

        inline void reg(const Board *board, const uint32_t hash, const int policy, const int value){
            if (table[hash] != NULL){
                if (table[hash]->compare(board))
                    table[hash]->register_value(policy, value);
                else
                    table[hash]->register_value(board, policy, value);
            } else{
                table[hash] = (Node_child_transpose_table*)malloc(sizeof(Node_child_transpose_table));
                table[hash]->register_value(board, policy, value);
            }
        }

        inline int get(Board *board, const uint32_t hash) const{
            if (table[hash] != NULL){
                if (table[hash]->compare(board)){
                    return table[hash]->get();
                }
            }
            return TRANSPOSE_TABLE_UNDEFINED;
        }
};


class Node_parent_transpose_table{
    private:
        atomic<unsigned long long> player;
        atomic<unsigned long long> opponent;
        atomic<int> lower;
        atomic<int> upper;
        //atomic<Node_child_transpose_table*> p_n_node;

    public:

        inline void init(){
            free(this);
        }

        inline void register_value(const Board *board, const int l, const int u){
            player.store(board->player);
            opponent.store(board->opponent);
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

        inline bool compare(const Board *a) const{
            return a->player == player.load(memory_order_relaxed) && a->opponent == opponent.load(memory_order_relaxed);
        }
};

#if USE_MULTI_THREAD
    void init_parent_transpose_table(int id, Node_parent_transpose_table *table[], int s, int e){
        for(int i = s; i < e; ++i){
            if (table[i] != NULL){
                table[i]->init();
                table[i] = NULL;
            }
        }
    }
#endif

class Parent_transpose_table{
    private:
        Node_parent_transpose_table *table[TRANSPOSE_TABLE_SIZE];

    public:
        inline void first_init(){
            for(int i = 0; i < TRANSPOSE_TABLE_SIZE; ++i)
                table[i] = NULL;
        }

        #if USE_MULTI_THREAD
            inline void init(){
                int thread_size = thread_pool.size();
                int delta = (TRANSPOSE_TABLE_SIZE + thread_size - 1) / thread_size;
                int s = 0, e;
                vector<future<void>> tasks;
                for (int i = 0; i < thread_size; ++i){
                    e = min(TRANSPOSE_TABLE_SIZE, s + delta);
                    tasks.emplace_back(thread_pool.push(init_parent_transpose_table, table, s, e));
                    s = e;
                }
                for (future<void> &task: tasks)
                    task.get();
            }
        #else
            inline void init(){
                for(int i = 0; i < TRANSPOSE_TABLE_SIZE; ++i){
                    if (table[i] != NULL){
                        table[i]->init();
                        table[i] = NULL;
                    }
                }
            }
        #endif

        inline void reg(const Board *board, const uint32_t hash, const int l, const int u){
            if (table[hash] != NULL){
                if (table[hash]->compare(board))
                    table[hash]->register_value(l, u);
                else
                    table[hash]->register_value(board, l, u);
            } else{
                table[hash] = (Node_parent_transpose_table*)malloc(sizeof(Node_parent_transpose_table));
                table[hash]->register_value(board, l, u);
            }
        }

        inline void get(Board *board, const uint32_t hash, int *l, int *u) const{
            if (table[hash] != NULL){
                if (table[hash]->compare(board)){
                    table[hash]->get(l, u);
                    return;
                }
            }
            *l = -INF;
            *u = INF;
        }
};

Parent_transpose_table parent_transpose_table;
Child_transpose_table child_transpose_table;
