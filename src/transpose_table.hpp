#pragma once
#include "setting.hpp"
#include "common.hpp"
#include "board.hpp"
#include <atomic>
#if USE_MULTI_THREAD
    #include "thread_pool.hpp"
    #include <future>
#endif
#include <mutex>

using namespace std;

#define TRANSPOSE_TABLE_SIZE 33554432
#define TRANSPOSE_TABLE_MASK 33554431
#define TRANSPOSE_TABLE_DIVISION 1024
#define TRANSPOSE_MINI_TABLE_SIZE (TRANSPOSE_TABLE_SIZE / TRANSPOSE_TABLE_DIVISION)

#define TRANSPOSE_TABLE_UNDEFINED -INF

class Node_child_transpose_table{
    private:
        atomic<uint64_t> player;
        atomic<uint64_t> opponent;
        atomic<int> best_move;
        //atomic<Node_child_transpose_table*> p_n_node;

    public:

        inline void init(){
            /*
            Node_child_transpose_table *p = p_n_node.load(memory_order_relaxed);
            if (p != NULL){
                p->init();
            }
            */
            free(this);
        }

        inline void register_value(const int policy){
            best_move.store(policy);
        }

        inline void register_value(const Board *board, const int policy){
            player.store(board->player);
            opponent.store(board->opponent);
            best_move.store(policy);
        }

        inline int get() const{
            return best_move.load();
        }

        inline bool compare(const Board *a) const{
            return a->player == player.load(memory_order_relaxed) && a->opponent == opponent.load(memory_order_relaxed);
        }
        /*
        inline Node_child_transpose_table* get_p_n_node(){
            return p_n_node.load(memory_order_relaxed);
        }

        inline void register_p_n_node(Node_child_transpose_table *p){
            return p_n_node.store(p);
        }
        */
};

struct Child_mini_transpose_table{
    Node_child_transpose_table *table[TRANSPOSE_MINI_TABLE_SIZE];
    mutex mtx;
};

#if USE_MULTI_THREAD
    void init_child_transpose_table(int id, Child_mini_transpose_table *table, int s, int e){
        lock_guard<mutex> lock(table->mtx);
        for(int i = s; i < e; ++i){
            if (table->table[i] != NULL){
                table->table[i]->init();
                table->table[i] = NULL;
            }
        }
    }
#endif

class Child_transpose_table{
    private:
        Child_mini_transpose_table table[TRANSPOSE_TABLE_DIVISION];

    public:
        inline void first_init(){
            int i, j;
            for(i = 0; i < TRANSPOSE_TABLE_DIVISION; ++i){
                for (j = 0; j < TRANSPOSE_MINI_TABLE_SIZE; ++j)
                    table[i].table[j] = NULL;
            }
        }

        #if USE_MULTI_THREAD
            inline void init(){
                vector<future<void>> tasks;
                for(int i = 0; i < TRANSPOSE_TABLE_DIVISION; ++i){
                    tasks.emplace_back(thread_pool.push(init_child_transpose_table, &table[i], 0, TRANSPOSE_MINI_TABLE_SIZE));
                }
                for (future<void> &task: tasks)
                    task.get();
            }
        #else
            inline void init(){
                int i, j;
            for(i = 0; i < TRANSPOSE_TABLE_DIVISION; ++i){
                for (j = 0; j < TRANSPOSE_MINI_TABLE_SIZE; ++j)
                    if(table[i].table[j] != NULL){
                        table[i].table[j]->init();
                        table[i].table[j] = NULL;
                    }
            }
            }
        #endif

        inline void reg(const Board *board, const uint32_t hash, const int policy){
            lock_guard<mutex> lock(table[get_mini_idx(hash)].mtx);
            Node_child_transpose_table *p_node = get_node(hash);
            if (p_node != NULL){
                if (p_node != NULL){
                    if (p_node->compare(board))
                        p_node->register_value(policy);
                    else
                        p_node->register_value(board, policy);
                }
            } else{
                p_node = (Node_child_transpose_table*)malloc(sizeof(Node_child_transpose_table));
                p_node->register_value(board, policy);
                table[hash / TRANSPOSE_MINI_TABLE_SIZE].table[hash % TRANSPOSE_MINI_TABLE_SIZE] = p_node;
            }
        }

        inline int get(Board *board, const uint32_t hash) const{
            Node_child_transpose_table *p_node = get_node(hash);
            if (p_node != NULL){
                if (p_node->compare(board))
                    return p_node->get();
            }
            return TRANSPOSE_TABLE_UNDEFINED;
        }
    
    private:
        inline Node_child_transpose_table* get_node(const uint32_t hash) const{
            return table[hash / TRANSPOSE_MINI_TABLE_SIZE].table[hash % TRANSPOSE_MINI_TABLE_SIZE];
        }

        inline uint32_t get_mini_idx(const uint32_t hash) const{
            return hash / TRANSPOSE_MINI_TABLE_SIZE;
        }
};


class Node_parent_transpose_table{
    private:
        atomic<uint64_t> player;
        atomic<uint64_t> opponent;
        atomic<int> lower;
        atomic<int> upper;
        //atomic<Node_parent_transpose_table*> p_n_node;

    public:

        inline void init(){
            //Node_parent_transpose_table *p = p_n_node.load(memory_order_relaxed);
            //if (p != NULL){
            //    p->init();
            //}
            free(this);
        }

        inline void register_value(const Board *board, const int l, const int u){
            player.store(board->player);
            opponent.store(board->opponent);
            lower.store(l);
            upper.store(u);
        }

        inline void register_value(const int l, const int u){
            lower.store(l);
            upper.store(u);
        }

        inline void get(int *l, int *u) const{
            *l = lower.load(memory_order_relaxed);
            *u = upper.load(memory_order_relaxed);
        }

        inline bool compare(const Board *a) const{
            return a->player == player.load(memory_order_relaxed) && a->opponent == opponent.load(memory_order_relaxed);
        }
        /*
        inline Node_parent_transpose_table* get_p_n_node(){
            return p_n_node.load(memory_order_relaxed);
        }

        inline void register_p_n_node(Node_parent_transpose_table *p){
            return p_n_node.store(p);
        }
        */
};

struct Parent_mini_transpose_table{
    Node_parent_transpose_table *table[TRANSPOSE_MINI_TABLE_SIZE];
    mutex mtx;
};

#if USE_MULTI_THREAD
    void init_parent_transpose_table(int id, Parent_mini_transpose_table *table, int s, int e){
        lock_guard<mutex> lock(table->mtx);
        for(int i = s; i < e; ++i){
            if (table->table[i] != NULL){
                table->table[i]->init();
                table->table[i] = NULL;
            }
        }
    }
#endif

class Parent_transpose_table{
    private:
        Parent_mini_transpose_table table[TRANSPOSE_TABLE_DIVISION];

    public:
        inline void first_init(){
            int i, j;
            for(i = 0; i < TRANSPOSE_TABLE_DIVISION; ++i){
                for (j = 0; j < TRANSPOSE_MINI_TABLE_SIZE; ++j){
                    table[i].table[j] = NULL;
                }
            }
        }

        #if USE_MULTI_THREAD
            inline void init(){
                vector<future<void>> tasks;
                for(int i = 0; i < TRANSPOSE_TABLE_DIVISION; ++i){
                    tasks.emplace_back(thread_pool.push(init_parent_transpose_table, &table[i], 0, TRANSPOSE_MINI_TABLE_SIZE));
                }
                for (future<void> &task: tasks)
                    task.get();
            }
        #else
            inline void init(){
                int i, j;
            for(i = 0; i < TRANSPOSE_TABLE_DIVISION; ++i){
                for (j = 0; j < TRANSPOSE_MINI_TABLE_SIZE; ++j)
                    if(table[i].table[j] != NULL){
                        table[i].table[j]->init();
                        table[i].table[j] = NULL;
                    }
            }
            }
        #endif

        inline void reg(const Board *board, const uint32_t hash, const int l, const int u){
            lock_guard<mutex> lock(table[get_mini_idx(hash)].mtx);
            Node_parent_transpose_table *p_node = get_node(hash);
            if (p_node != NULL){
                if (p_node->compare(board))
                    p_node->register_value(l, u);
                else
                    p_node->register_value(board, l, u);
            } else{
                p_node = (Node_parent_transpose_table*)malloc(sizeof(Node_parent_transpose_table));
                p_node->register_value(board, l, u);
                table[hash / TRANSPOSE_MINI_TABLE_SIZE].table[hash % TRANSPOSE_MINI_TABLE_SIZE] = p_node;
            }
        }

        inline void get(Board *board, const uint32_t hash, int *l, int *u) const{
            Node_parent_transpose_table *p_node = get_node(hash);
            if (p_node != NULL){
                if (p_node->compare(board)){
                    p_node->get(l, u);
                    return;
                }
            }
            *l = -INF;
            *u = INF;
        }
    
    private:
        inline Node_parent_transpose_table* get_node(const uint32_t hash) const{
            return table[hash / TRANSPOSE_MINI_TABLE_SIZE].table[hash % TRANSPOSE_MINI_TABLE_SIZE];
        }

        inline uint32_t get_mini_idx(const uint32_t hash) const{
            return hash / TRANSPOSE_MINI_TABLE_SIZE;
        }
};

Parent_transpose_table parent_transpose_table;
Child_transpose_table child_transpose_table;
