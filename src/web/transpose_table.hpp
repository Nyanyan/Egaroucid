/*
    Egaroucid Project

    @date 2021-2023
    @author Takuto Yamana (a.k.a Nyanyan)
    @license GPL-3.0 license
*/

#pragma once
#include "setting.hpp"
#include "common.hpp"
#include "board.hpp"
#include <future>
#include <atomic>

using namespace std;

#define TRANSPOSE_TABLE_SIZE 262144
#define TRANSPOSE_TABLE_MASK 262143

#define CACHE_SAVE_EMPTY 10

#define TRANSPOSE_TABLE_UNDEFINED -INF

#define TRANSPOSE_TABLE_STRENGTH_MAGIC_NUMBER 8

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

class Child_transpose_table{
    private:
        Node_child_transpose_table table[TRANSPOSE_TABLE_SIZE];

    public:
        inline void first_init(){
            init();
        }

        inline void init(){
            for (int i = 0; i < TRANSPOSE_TABLE_SIZE; ++i)
                table[i].init();
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

class Parent_transpose_table{
    private:
        Node_parent_transpose_table table[TRANSPOSE_TABLE_SIZE];

    public:
        inline void first_init(){
            init();
        }

        inline void init(){
            for (int i = 0; i < TRANSPOSE_TABLE_SIZE; ++i)
                table[i].init();
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