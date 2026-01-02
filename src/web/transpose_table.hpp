/*
    Egaroucid Project

    @date 2021-2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once
#include "setting.hpp"
#include "common.hpp"
#include "board.hpp"
#include <future>

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
        uint64_t player;
        uint64_t opponent;
        int best_move;

    public:

        inline void init(){
            player = 0ULL;
            opponent = 0ULL;
            best_move = 0;
        }

        inline void register_value_with_board(const Board *board, const int policy){
            player = board->player;
            opponent = board->opponent;
            best_move = policy;
        }

        inline void register_value_with_board(Node_child_transpose_table *from){
            player = from->player;
            opponent = from->opponent;
            best_move = from->best_move;
        }

        inline int get(const Board *board){
            int res;
            if (board->player != player || board->opponent != opponent)
                res = TRANSPOSE_TABLE_UNDEFINED;
            else{
                res = best_move;
                if (board->player != player || board->opponent != opponent)
                    res = TRANSPOSE_TABLE_UNDEFINED;
            }
            return res;
        }

        inline int n_stones(){
            return pop_count_ull(player | opponent);
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
        uint64_t player;
        uint64_t opponent;
        int lower;
        int upper;
        double mpct;
        int depth;

    public:

        inline void init(){
            player = 0ULL;
            opponent = 0ULL;
            lower = -INF;
            upper = INF;
            mpct = 0.0;
            depth = 0;
        }

        inline void register_value_with_board(const Board *board, const int l, const int u, const double t, const int d){
            if (board->player == player && board->opponent == opponent && data_strength(mpct, depth) > data_strength(t, d))
                return;
            player = board->player;
            opponent = board->opponent;
            lower = l;
            upper = u;
            mpct = t;
            depth = d;
        }

        inline void register_value_with_board(Node_parent_transpose_table *from){
            player = from->player;
            opponent = from->opponent;
            lower = from->lower;
            upper = from->upper;
            mpct = from->mpct;
            depth = from->depth;
        }

        inline void get(const Board *board, int *l, int *u, const double t, const int d){
            if (data_strength(mpct, depth) < data_strength(t, d)){
                *l = -INF;
                *u = INF;
            } else{
                if (board->player != player || board->opponent != opponent){
                    *l = -INF;
                    *u = INF;
                } else{
                    *l = lower;
                    *u = upper;
                    if (board->player != player || board->opponent != opponent){
                        *l = -INF;
                        *u = INF;
                    }
                }
            }
        }

        inline bool contain(const Board *board){
            return board->player == player && board->opponent == opponent;
        }

        inline int n_stones() const{
            return pop_count_ull(player | opponent);
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