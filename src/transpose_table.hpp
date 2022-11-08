/*
    Egaroucid Project

    @date 2021-2022
    @author Takuto Yamana (a.k.a Nyanyan)
    @license GPL-3.0 license
*/

#pragma once
#include "setting.hpp"
#include "common.hpp"
#include "board.hpp"
#include "thread_pool.hpp"
#include <future>
#include <atomic>

using namespace std;

#define TRANSPOSE_TABLE_UNDEFINED -INF

//#define TRANSPOSE_TABLE_STRENGTH_MAGIC_NUMBER 8

inline double data_strength(const double t, const int d){
    //return t * (TRANSPOSE_TABLE_STRENGTH_MAGIC_NUMBER + d);
    return t + 4.0 * d;
}

class Node_child_transpose_table{
    private:
        atomic<uint64_t> player;
        atomic<uint64_t> opponent;
        atomic<int> best_move;

    public:

        inline void init(){
            player.store(0ULL);
            opponent.store(0ULL);
            best_move.store(0);
        }

        inline void register_value_with_board(const Board *board, const int policy){
            player.store(board->player);
            opponent.store(board->opponent);
            best_move.store(policy);
        }

        inline void register_value_with_board(Node_child_transpose_table *from){
            player.store(from->player.load());
            opponent.store(from->opponent.load());
            best_move.store(from->best_move.load());
        }

        inline int get(const Board *board){
            int res;
            if (board->player != player.load(memory_order_relaxed) || board->opponent != opponent.load(memory_order_relaxed))
                res = TRANSPOSE_TABLE_UNDEFINED;
            else{
                res = best_move.load(memory_order_relaxed);
                if (board->player != player.load(memory_order_relaxed) || board->opponent != opponent.load(memory_order_relaxed))
                    res = TRANSPOSE_TABLE_UNDEFINED;
            }
            return res;
        }

        inline int n_stones(){
            return pop_count_ull(player.load(memory_order_relaxed) | opponent.load(memory_order_relaxed));
        }
};


void init_child_transpose_table(Node_child_transpose_table table[], size_t s, size_t e){
    for(size_t i = s; i < e; ++i){
        table[i].init();
    }
}

void copy_child_transpose_table(Node_child_transpose_table from[], Node_child_transpose_table to[], int s, int e){
    for(int i = s; i < e; ++i){
        to[i].register_value_with_board(&from[i]);
    }
}

class Child_transpose_table{
    private:
        Node_child_transpose_table *table;
        size_t table_size;

    public:
        Child_transpose_table(){
            table = NULL;
            table_size = 0;
        }

        inline bool resize(int hash_level){
            size_t n_table_size = hash_sizes[hash_level];
            free(table);
            table = (Node_child_transpose_table*)malloc(sizeof(Node_child_transpose_table) * n_table_size);
            if (table == NULL)
                return false;
            table_size = n_table_size;
            init();
            return true;
        }

        inline void init(){
            if (thread_pool.size() == 0){
                for (size_t i = 0; i < table_size; ++i)
                    table[i].init();
            } else{
                int thread_size = thread_pool.size();
                size_t delta = (table_size + thread_size - 1) / thread_size;
                size_t s = 0, e;
                vector<future<void>> tasks;
                for (int i = 0; i < thread_size; ++i){
                    e = min(table_size, s + delta);
                    tasks.emplace_back(thread_pool.push(bind(&init_child_transpose_table, table, s, e)));
                    s = e;
                }
                for (future<void> &task: tasks)
                    task.get();
            }
        }

        inline void reg(const Board *board, const uint32_t hash, const int policy){
            table[hash].register_value_with_board(board, policy);
        }

        inline int get(const Board *board, const uint32_t hash){
            return table[hash].get(board);
        }
};


class Node_parent_transpose_table{
    private:
        atomic<uint64_t> player;
        atomic<uint64_t> opponent;
        atomic<int> lower;
        atomic<int> upper;
        atomic<double> mpct;
        atomic<int> depth;

    public:

        inline void init(){
            player.store(0ULL);
            opponent.store(0ULL);
            lower.store(-INF);
            upper.store(INF);
            mpct.store(0.0);
            depth.store(0);
        }

        inline void register_value_with_board(const Board *board, const int l, const int u, const double t, const int d){
            if (board->player == player.load(memory_order_relaxed) && board->opponent == opponent.load(memory_order_relaxed) && data_strength(mpct.load(memory_order_relaxed), depth.load(memory_order_relaxed)) > data_strength(t, d))
                return;
            player.store(board->player);
            opponent.store(board->opponent);
            lower.store(l);
            upper.store(u);
            mpct.store(t);
            depth.store(d);
        }

        inline void register_value_with_board(Node_parent_transpose_table *from){
            player.store(from->player);
            opponent.store(from->opponent);
            lower.store(from->lower);
            upper.store(from->upper);
            mpct.store(from->mpct);
            depth.store(from->depth);
        }

        inline void get(const Board *board, int *l, int *u, const double t, const int d){
            if (data_strength(mpct.load(memory_order_relaxed), depth.load(memory_order_relaxed)) < data_strength(t, d)){
                *l = -INF;
                *u = INF;
            } else{
                if (board->player != player.load(memory_order_relaxed) || board->opponent != opponent.load(memory_order_relaxed)){
                    *l = -INF;
                    *u = INF;
                } else{
                    *l = lower.load(memory_order_relaxed);
                    *u = upper.load(memory_order_relaxed);
                    if (board->player != player.load(memory_order_relaxed) || board->opponent != opponent.load(memory_order_relaxed)){
                        *l = -INF;
                        *u = INF;
                    }
                }
            }
        }

        inline bool contain(const Board *board){
            return board->player == player.load(memory_order_relaxed) && board->opponent == opponent.load(memory_order_relaxed);
        }

        inline int n_stones() const{
            return pop_count_ull(player.load(memory_order_relaxed) | opponent.load(memory_order_relaxed));
        }
};

void init_parent_transpose_table(Node_parent_transpose_table table[], size_t s, size_t e){
    for(size_t i = s; i < e; ++i){
        table[i].init();
    }
}

class Parent_transpose_table{
    private:
        Node_parent_transpose_table *table;
        size_t table_size;

    public:
        Parent_transpose_table(){
            table = NULL;
            table_size = 0;
        }

        inline bool resize(int hash_level){
            size_t n_table_size = hash_sizes[hash_level];
            free(table);
            table = (Node_parent_transpose_table*)malloc(sizeof(Node_parent_transpose_table) * n_table_size);
            if (table == NULL)
                return false;
            table_size = n_table_size;
            init();
            return true;
        }

        inline void init(){
            if (thread_pool.size() == 0){
                for (size_t i = 0; i < table_size; ++i)
                    table[i].init();
            } else{
                int thread_size = thread_pool.size();
                size_t delta = (table_size + thread_size - 1) / thread_size;
                size_t s = 0, e;
                vector<future<void>> tasks;
                for (int i = 0; i < thread_size; ++i){
                    e = min(table_size, s + delta);
                    tasks.emplace_back(thread_pool.push(bind(&init_parent_transpose_table, table, s, e)));
                    s = e;
                }
                for (future<void> &task: tasks)
                    task.get();
            }
        }

        inline void reg(const Board *board, const uint32_t hash, const int l, const int u, const double t, const int d){
            table[hash].register_value_with_board(board, l, u, t, d);
        }

        inline void get(const Board *board, const uint32_t hash, int *l, int *u, const double t, const int d){
            table[hash].get(board, l, u, t, d);
        }

        inline bool contain(const Board *board, const uint32_t hash){
            return table[hash].contain(board);
        }
};

Parent_transpose_table parent_transpose_table;
Child_transpose_table child_transpose_table;

bool hash_resize(int hash_level, int n_hash_level){
    if (!parent_transpose_table.resize(n_hash_level)){
        cerr << "hash resize failed" << endl;
        parent_transpose_table.resize(hash_level);
        return false;
    }
    if (!child_transpose_table.resize(n_hash_level)){
        cerr << "hash resize failed" << endl;
        parent_transpose_table.resize(hash_level);
        child_transpose_table.resize(hash_level);
        return false;
    }
    if (!hash_init(n_hash_level)){
        cerr << "can't get hash. you can ignore this error" << endl;
        hash_init_rand(n_hash_level);
    }
    double size_mb = (double)(sizeof(Node_parent_transpose_table) + sizeof(Node_child_transpose_table)) / 1024 / 1024 * hash_sizes[n_hash_level];
    cerr << "hash resized to level " << n_hash_level << " elements " << hash_sizes[n_hash_level] << " size " << size_mb << " MB" << endl;
    return true;
}