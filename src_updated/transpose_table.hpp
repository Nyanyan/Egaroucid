#pragma once
#include <atomic>
#include "setting.hpp"
#include "common.hpp"
#include "board.hpp"

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
            best_moves[0] = TRANSPOSE_TABLE_UNDEFINED;
            best_moves[1] = TRANSPOSE_TABLE_UNDEFINED;
            best_moves[2] = TRANSPOSE_TABLE_UNDEFINED;
            best_value = -INF;
        }

        inline void register_value(const Board *board, const int policy, const int value){
            init();
            reg = true;
            b = board->b;
            w = board->w;
            p = board->p;
            if (best_value < value){
                best_value = value;
                int tmp = best_moves[1];
                best_moves[2] = tmp;
                tmp = best_moves[0];
                best_moves[1] = tmp;
                best_moves[0] = policy;
            } else{
                if (best_moves[1] == TRANSPOSE_TABLE_UNDEFINED)
                    best_moves[1] = policy;
                else if (best_moves[2] == TRANSPOSE_TABLE_UNDEFINED)
                    best_moves[2] = policy;
            }
        }

        inline void register_value(const int policy, const int value){
            if (best_value < value){
                best_value = value;
                int tmp = best_moves[1];
                best_moves[2] = tmp;
                tmp = best_moves[0];
                best_moves[1] = tmp;
                best_moves[0] = policy;
            }
        }

        inline void get(int b[]){
            b[0] = best_moves[0];
            b[1] = best_moves[1];
            b[2] = best_moves[2];
        }
};

class Child_transpose_table{
    private:
        int prev;
        int now;
        Node_child_transpose_table table[2][TRANSPOSE_TABLE_SIZE];

    public:
        inline void init(){
            now = 0;
            prev = 1;
            init_now();
            init_prev();
        }

        inline void ready_next_search(){
            now = 1 - now;
            prev = 1 - prev;
            init_now();
        }

        inline void init_prev(){
            for(int i = 0; i < TRANSPOSE_TABLE_SIZE; ++i)
                table[prev][i].init();
        }

        inline void init_now(){
            for(int i = 0; i < TRANSPOSE_TABLE_SIZE; ++i)
                table[now][i].init();
        }

        inline int now_idx(){
            return now;
        }

        inline int prev_idx(){
            return prev;
        }

        inline void reg(const Board *board, const int hash, const int policy, const int value){
            if (global_searching){
                if (!table[now][hash].reg)
                    table[now][hash].register_value(board, policy, value);
                else if (!compare_key(board, &table[now][hash]))
                    table[now][hash].register_value(board, policy, value);
                else
                    table[now][hash].register_value(policy, value);
            }
        }

        inline bool get_now(Board *board, const int hash, int best_moves[]){
            if (table[now][hash].reg){
                if (compare_key(board, &table[now][hash])){
					table[now][hash].get(best_moves);
                    return true;
                }
            }
            return false;
        }

        inline bool get_prev(Board *board, const int hash, int best_moves[]){
            if (table[prev][hash].reg){
                if (compare_key(board, &table[prev][hash])){
					table[prev][hash].get(best_moves);
                    return true;
                }
            }
            return false;
        }

    private:
        inline bool compare_key(const Board *a, Node_child_transpose_table *b){
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
            lower = -INF;
            upper = INF;
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

        inline void get(int *l, int *u){
            *l = lower;
            *u = upper;
        }
};

class Parent_transpose_table{
    private:
        Node_parent_transpose_table table[TRANSPOSE_TABLE_SIZE];

    public:
        inline void init(){
            for(int i = 0; i < TRANSPOSE_TABLE_SIZE; ++i)
                table[i].init();
        }

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

        inline void get(Board *board, const int hash, int *l, int *u){
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
        inline bool compare_key(const Board *a, Node_parent_transpose_table *b){
            return a->p == b->p && a->b == b->b && a->w == b->w;
        }
};

Parent_transpose_table parent_transpose_table;
Child_transpose_table child_transpose_table;
