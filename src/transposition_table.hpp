/*
    Egaroucid Project

    @file transposition_table.hpp
        Transposition table
    @date 2021-2022
    @author Takuto Yamana (a.k.a. Nyanyan)
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

/*
    @brief constants
*/
#define TRANSPOSITION_TABLE_UNDEFINED -INF
#define TRANSPOSITION_TABLE_STACK_SIZE 16777216

/*
    @brief calculate the reliability

    @param t                    probability of MPC (Multi-ProbCut)
    @param d                    depth of the search
    @return reliability (strength)
*/
inline double data_strength(const double t, const int d){
    return t + 4.0 * d;
}

/*
    @brief Node of best move transposition table

    @param player               a bitboard representing player
    @param opponent             a bitboard representing opponent
    @param best_move            best move
*/
class Node_best_move_transposition_table{
    private:
        atomic<uint64_t> player;
        atomic<uint64_t> opponent;
        atomic<int> best_move;

    public:

        /*
            @brief Initialize a node
        */
        inline void init(){
            player.store(0ULL);
            opponent.store(0ULL);
            best_move.store(0);
        }

        /*
            @brief Register best move

            Always overwrite

            @param board                new board
            @param policy               new best move
        */
        inline void reg(const Board *board, const int policy){
            player.store(board->player);
            opponent.store(board->opponent);
            best_move.store(policy);
        }

        /*
            @brief Get best move

            @param board                new board
            @return TRANSPOSITION_TABLE_UNDEFINED if no data found, else the best move
        */
        inline int get(const Board *board){
            int res;
            if (board->player != player.load(memory_order_relaxed) || board->opponent != opponent.load(memory_order_relaxed))
                res = TRANSPOSITION_TABLE_UNDEFINED;
            else{
                res = best_move.load(memory_order_relaxed);
                if (board->player != player.load(memory_order_relaxed) || board->opponent != opponent.load(memory_order_relaxed))
                    res = TRANSPOSITION_TABLE_UNDEFINED;
            }
            return res;
        }
};

/*
    @brief Initialize best move transposition table in parallel

    @param table                transposition table
    @param s                    start index
    @param e                    end index
*/
void init_best_move_transposition_table(Node_best_move_transposition_table table[], size_t s, size_t e){
    for(size_t i = s; i < e; ++i){
        table[i].init();
    }
}

/*
    @brief Best move transposition table structure

    @param table_stack          transposition table on stack
    @param table_heap           transposition table on heap
    @param table_size           total table size
*/
class Best_move_transposition_table{
    private:
        Node_best_move_transposition_table table_stack[TRANSPOSITION_TABLE_STACK_SIZE];
        Node_best_move_transposition_table *table_heap;
        size_t table_size;

    public:
        /*
            @brief Constructer of best move transposition table
        */
        Best_move_transposition_table(){
            table_heap = NULL;
            table_size = 0;
        }

        /*
            @brief Resize best move transposition table

            @param hash_level           hash level representing the size
            @return table initialized?
        */
        inline bool resize(int hash_level){
            size_t n_table_size = hash_sizes[hash_level];
            free(table_heap);
            if (n_table_size > TRANSPOSITION_TABLE_STACK_SIZE){
                table_heap = (Node_best_move_transposition_table*)malloc(sizeof(Node_best_move_transposition_table) * (n_table_size - TRANSPOSITION_TABLE_STACK_SIZE));
                if (table_heap == NULL)
                    return false;
            }
            table_size = n_table_size;
            init();
            return true;
        }

        /*
            @brief Initialize best move transposition table
        */
        inline void init(){
            if (thread_pool.size() == 0){
                for (size_t i = 0; i < min(table_size, (size_t)TRANSPOSITION_TABLE_STACK_SIZE); ++i)
                    table_stack[i].init();
                for (size_t i = 0; i < table_size - (size_t)TRANSPOSITION_TABLE_STACK_SIZE; ++i)
                    table_heap[i].init();
            } else{
                int thread_size = thread_pool.size();
                size_t delta = (min(table_size, (size_t)TRANSPOSITION_TABLE_STACK_SIZE) + thread_size - 1) / thread_size;
                size_t s = 0, e;
                vector<future<void>> tasks;
                for (int i = 0; i < thread_size; ++i){
                    e = min(min(table_size, (size_t)TRANSPOSITION_TABLE_STACK_SIZE), s + delta);
                    tasks.emplace_back(thread_pool.push(bind(&init_best_move_transposition_table, table_stack, s, e)));
                    s = e;
                }
                if (table_size > TRANSPOSITION_TABLE_STACK_SIZE){
                    delta = (table_size - (size_t)TRANSPOSITION_TABLE_STACK_SIZE + thread_size - 1) / thread_size;
                    s = 0;
                    for (int i = 0; i < thread_size; ++i){
                        e = min(table_size - (size_t)TRANSPOSITION_TABLE_STACK_SIZE, s + delta);
                        tasks.emplace_back(thread_pool.push(bind(&init_best_move_transposition_table, table_heap, s, e)));
                        s = e;
                    }
                }
                for (future<void> &task: tasks)
                    task.get();
            }
        }

        /*
            @brief register a value to best move transposition table

            @param board                board to register
            @param hash                 hash code
            @param policy               best move
        */
        inline void reg(const Board *board, const uint32_t hash, const int policy){
            if (hash < TRANSPOSITION_TABLE_STACK_SIZE)
                table_stack[hash].reg(board, policy);
            else
                table_heap[hash - TRANSPOSITION_TABLE_STACK_SIZE].reg(board, policy);
        }

        /*
            @brief get best move from best move transposition table

            @param board                board to register
            @param hash                 hash code
            @return TRANSPOSITION_TABLE_UNDEFINED if not found, else best move 
        */
        inline int get(const Board *board, const uint32_t hash){
            if (hash < TRANSPOSITION_TABLE_STACK_SIZE)
                return table_stack[hash].get(board);
            return table_heap[hash - TRANSPOSITION_TABLE_STACK_SIZE].get(board);
        }
};


class Node_value_transposition_table{
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

        inline void register_value_with_board(Node_value_transposition_table *from){
            player.store(from->player);
            opponent.store(from->opponent);
            lower.store(from->lower);
            upper.store(from->upper);
            mpct.store(from->mpct);
            depth.store(from->depth);
        }

        inline void get(const Board *board, int *l, int *u, const double t, const int d, bool *mpc_used){
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
                    *mpc_used |= mpct.load(memory_order_relaxed) < NOMPC;
                    if (board->player != player.load(memory_order_relaxed) || board->opponent != opponent.load(memory_order_relaxed)){
                        *l = -INF;
                        *u = INF;
                    }
                }
            }
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

void init_value_transposition_table(Node_value_transposition_table table[], size_t s, size_t e){
    for(size_t i = s; i < e; ++i){
        table[i].init();
    }
}

class Value_transposition_table{
    private:
        Node_value_transposition_table table_stack[TRANSPOSITION_TABLE_STACK_SIZE];
        Node_value_transposition_table *table_heap;
        size_t table_size;

    public:
        Value_transposition_table(){
            table_heap = NULL;
            table_size = 0;
        }

        inline bool resize(int hash_level){
            size_t n_table_size = hash_sizes[hash_level];
            free(table_heap);
            if (n_table_size > TRANSPOSITION_TABLE_STACK_SIZE){
                table_heap = (Node_value_transposition_table*)malloc(sizeof(Node_value_transposition_table) * (n_table_size - TRANSPOSITION_TABLE_STACK_SIZE));
                if (table_heap == NULL)
                    return false;
            }
            table_size = n_table_size;
            init();
            return true;
        }

        inline void init(){
            if (thread_pool.size() == 0){
                for (size_t i = 0; i < min(table_size, (size_t)TRANSPOSITION_TABLE_STACK_SIZE); ++i)
                    table_stack[i].init();
                for (size_t i = 0; i < table_size - (size_t)TRANSPOSITION_TABLE_STACK_SIZE; ++i)
                    table_heap[i].init();
            } else{
                int thread_size = thread_pool.size();
                size_t delta = (min(table_size, (size_t)TRANSPOSITION_TABLE_STACK_SIZE) + thread_size - 1) / thread_size;
                size_t s = 0, e;
                vector<future<void>> tasks;
                for (int i = 0; i < thread_size; ++i){
                    e = min(min(table_size, (size_t)TRANSPOSITION_TABLE_STACK_SIZE), s + delta);
                    tasks.emplace_back(thread_pool.push(bind(&init_value_transposition_table, table_stack, s, e)));
                    s = e;
                }
                if (table_size > TRANSPOSITION_TABLE_STACK_SIZE){
                    delta = (table_size - (size_t)TRANSPOSITION_TABLE_STACK_SIZE + thread_size - 1) / thread_size;
                    s = 0;
                    for (int i = 0; i < thread_size; ++i){
                        e = min(table_size - (size_t)TRANSPOSITION_TABLE_STACK_SIZE, s + delta);
                        tasks.emplace_back(thread_pool.push(bind(&init_value_transposition_table, table_heap, s, e)));
                        s = e;
                    }
                }
                for (future<void> &task: tasks)
                    task.get();
            }
        }

        inline void reg(const Board *board, const uint32_t hash, const int l, const int u, const double t, const int d){
            if (hash < TRANSPOSITION_TABLE_STACK_SIZE)
                table_stack[hash].register_value_with_board(board, l, u, t, d);
            else
                table_heap[hash - TRANSPOSITION_TABLE_STACK_SIZE].register_value_with_board(board, l, u, t, d);
        }

        inline void get(const Board *board, const uint32_t hash, int *l, int *u, const double t, const int d, bool *mpc_used){
            if (hash < TRANSPOSITION_TABLE_STACK_SIZE)
                table_stack[hash].get(board, l, u, t, d, mpc_used);
            else
                table_heap[hash - TRANSPOSITION_TABLE_STACK_SIZE].get(board, l, u, t, d, mpc_used);
        }

        inline void get(const Board *board, const uint32_t hash, int *l, int *u, const double t, const int d){
            if (hash < TRANSPOSITION_TABLE_STACK_SIZE)
                table_stack[hash].get(board, l, u, t, d);
            else
                table_heap[hash - TRANSPOSITION_TABLE_STACK_SIZE].get(board, l, u, t, d);
        }

        inline bool contain(const Board *board, const uint32_t hash){
            if (hash < TRANSPOSITION_TABLE_STACK_SIZE)
                return table_stack[hash].contain(board);
            return table_heap[hash - TRANSPOSITION_TABLE_STACK_SIZE].contain(board);
        }
};

Value_transposition_table value_transposition_table;
Best_move_transposition_table best_move_transposition_table;

bool hash_resize(int hash_level, int n_hash_level){
    if (!value_transposition_table.resize(n_hash_level)){
        cerr << "parent hash table resize failed" << endl;
        value_transposition_table.resize(hash_level);
        return false;
    }
    if (!best_move_transposition_table.resize(n_hash_level)){
        cerr << "child hash table resize failed" << endl;
        value_transposition_table.resize(hash_level);
        best_move_transposition_table.resize(hash_level);
        return false;
    }
    if (!hash_init(n_hash_level)){
        cerr << "can't get hash. you can ignore this error" << endl;
        hash_init_rand(n_hash_level);
    }
    double size_mb = (double)(sizeof(Node_value_transposition_table) + sizeof(Node_best_move_transposition_table)) / 1024 / 1024 * hash_sizes[n_hash_level];
    cerr << "hash resized to level " << n_hash_level << " elements " << hash_sizes[n_hash_level] << " size " << size_mb << " MB" << endl;
    return true;
}