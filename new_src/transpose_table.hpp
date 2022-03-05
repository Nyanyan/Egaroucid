#pragma once
#include "setting.hpp"
#include "common.hpp"
#include "board.hpp"
#include <atomic>

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
            //Node_child_transpose_table* next_node = p_n_node.load();
            //if (next_node != NULL)
            //    next_node->init();
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

        //inline Node_child_transpose_table* next_node(){
        //    return p_n_node.load();
        //}
};

class Child_transpose_table{
    private:
        int prev;
        int now;
        Node_child_transpose_table *table[2][TRANSPOSE_TABLE_SIZE];
        atomic<int> n_reg;

    public:
        inline void first_init(){
            now = 0;
            prev = 1;
            for(int i = 0; i < TRANSPOSE_TABLE_SIZE; ++i){
                table[prev][i] = NULL;
                table[now][i] = NULL;
            }
        }

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

        inline void init_prev(){
            for(int i = 0; i < TRANSPOSE_TABLE_SIZE; ++i){
                if (table[prev][i] != NULL){
                    table[prev][i]->init();
                    table[prev][i] = NULL;
                }
            }
        }

        inline void init_now(){
            for(int i = 0; i < TRANSPOSE_TABLE_SIZE; ++i){
                if (table[now][i] != NULL){
                    table[now][i]->init();
                    table[now][i] = NULL;
                }
            }
        }

        inline int now_idx() const{
            return now;
        }

        inline int prev_idx() const{
            return prev;
        }

        inline void reg(const Board *board, const uint32_t hash, const int policy, const int value){
            if (table[now][hash] != NULL){
                if (table[now][hash]->compare(board))
                    table[now][hash]->register_value(policy, value);
                else
                    table[now][hash]->register_value(board, policy, value);
            } else{
                table[now][hash] = (Node_child_transpose_table*)malloc(sizeof(Node_child_transpose_table));
                table[now][hash]->register_value(board, policy, value);
            }
        }

        inline void reg(const int idx, const Board *board, const uint32_t hash, const int policy, const int value){
            if (table[idx][hash] != NULL){
                if (table[idx][hash]->compare(board))
                    table[idx][hash]->register_value(policy, value);
                else
                    table[idx][hash]->register_value(board, policy, value);
            } else{
                table[idx][hash] = (Node_child_transpose_table*)malloc(sizeof(Node_child_transpose_table));
                table[idx][hash]->register_value(board, policy, value);
            }
        }

        inline int get_now(Board *board, const uint32_t hash) const{
            if (table[now][hash] != NULL){
                if (table[now][hash]->compare(board)){
                    return table[now][hash]->get();
                }
            }
            return TRANSPOSE_TABLE_UNDEFINED;
        }

        inline int get_prev(Board *board, const uint32_t hash) const{
            if (table[prev][hash] != NULL){
                if (table[prev][hash]->compare(board)){
                    return table[prev][hash]->get();
                }
            }
            return TRANSPOSE_TABLE_UNDEFINED;
        }

        inline int get_n_reg() const{
            return n_reg.load();
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
            //Node_child_transpose_table* next_node = p_n_node.load();
            //if (next_node != NULL)
            //    next_node->init();
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

class Parent_transpose_table{
    private:
        Node_parent_transpose_table *table[TRANSPOSE_TABLE_SIZE];

    public:
        inline void first_init(){
            for(int i = 0; i < TRANSPOSE_TABLE_SIZE; ++i)
                table[i] = NULL;
        }

        inline void init(){
            for(int i = 0; i < TRANSPOSE_TABLE_SIZE; ++i){
                if (table[i] != NULL){
                    table[i]->init();
                    table[i] = NULL;
                }
            }
        }

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
