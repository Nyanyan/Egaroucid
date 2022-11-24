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

/*
    @brief constants
*/
#define TRANSPOSITION_TABLE_UNDEFINED SCORE_INF
constexpr size_t TRANSPOSITION_TABLE_STACK_SIZE = hash_sizes[DEFAULT_HASH_LEVEL];

/*
    @brief Calculate the reliability

    @param t                    probability of MPC (Multi-ProbCut)
    @param d                    depth of the search
    @return reliability (strength)
*/
inline int data_strength(const uint_fast8_t mpc_level, const int d){
    return (mpc_level << 7) + d;
}

/*
    @brief Node of best move transposition table

    @param player               a bitboard representing player
    @param opponent             a bitboard representing opponent
    @param best_move            best move
*/
class Node_best_move_transposition_table{
    private:
        std::atomic<uint64_t> player;
        std::atomic<uint64_t> opponent;
        std::atomic<int8_t> best_move;

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
            if (board->player != player.load(std::memory_order_relaxed) || board->opponent != opponent.load(std::memory_order_relaxed))
                res = TRANSPOSITION_TABLE_UNDEFINED;
            else{
                res = best_move.load(std::memory_order_relaxed);
                if (board->player != player.load(std::memory_order_relaxed) || board->opponent != opponent.load(std::memory_order_relaxed))
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
            @brief Constructor of best move transposition table
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
                if (table_heap == nullptr)
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
                for (size_t i = 0; i < std::min(table_size, (size_t)TRANSPOSITION_TABLE_STACK_SIZE); ++i)
                    table_stack[i].init();
                for (size_t i = 0; i < table_size - (size_t)TRANSPOSITION_TABLE_STACK_SIZE; ++i)
                    table_heap[i].init();
            } else{
                int thread_size = thread_pool.size();
                size_t delta = (std::min(table_size, (size_t)TRANSPOSITION_TABLE_STACK_SIZE) + thread_size - 1) / thread_size;
                size_t s = 0, e;
                std::vector<std::future<void>> tasks;
                for (int i = 0; i < thread_size; ++i){
                    e = std::min(std::min(table_size, (size_t)TRANSPOSITION_TABLE_STACK_SIZE), s + delta);
                    tasks.emplace_back(thread_pool.push(std::bind(&init_best_move_transposition_table, table_stack, s, e)));
                    s = e;
                }
                if (table_size > TRANSPOSITION_TABLE_STACK_SIZE){
                    delta = (table_size - (size_t)TRANSPOSITION_TABLE_STACK_SIZE + thread_size - 1) / thread_size;
                    s = 0;
                    for (int i = 0; i < thread_size; ++i){
                        e = std::min(table_size - (size_t)TRANSPOSITION_TABLE_STACK_SIZE, s + delta);
                        tasks.emplace_back(thread_pool.push(std::bind(&init_best_move_transposition_table, table_heap, s, e)));
                        s = e;
                    }
                }
                for (std::future<void> &task: tasks)
                    task.get();
            }
        }

        /*
            @brief Register a board to best move transposition table

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

/*
    @brief Node of value transposition table

    @param player               a bitboard representing player
    @param opponent             a bitboard representing opponent
    @param lower                lower bound
    @param upper                upper bound
    @param mpc_level                 MPC (Multi-ProbCut) probability
    @param depth                search depth
*/
class Node_value_transposition_table{
    private:
        std::atomic<uint64_t> player;
        std::atomic<uint64_t> opponent;
        std::atomic<int8_t> lower;
        std::atomic<int8_t> upper;
        std::atomic<uint8_t> mpc_level;
        std::atomic<int8_t> depth;

    public:
        /*
            @brief Initialize a node
        */
        inline void init(){
            player.store(0ULL);
            opponent.store(0ULL);
            lower.store(-SCORE_INF);
            upper.store(SCORE_INF);
            mpc_level.store(0);
            depth.store(0);
        }

        /*
            @brief Register best move

            Always overwrite

            @param board                new board
            @param l                    new lower bound
            @param u                    new upper bound
            @param mpc_level            MPC (Multi-ProbCut) level
            @param d                    new depth
        */
        inline void reg(const Board *board, const int l, const int u, const uint_fast8_t ml, const int d){
            if (board->player == player.load(std::memory_order_relaxed) && board->opponent == opponent.load(std::memory_order_relaxed) && data_strength(mpc_level.load(std::memory_order_relaxed), depth.load(std::memory_order_relaxed)) > data_strength(ml, d))
                return;
            player.store(board->player);
            opponent.store(board->opponent);
            lower.store(l);
            upper.store(u);
            mpc_level.store(ml);
            depth.store(d);
        }

        /*
            @brief Get lower / upper bound

            Although board found, if the strength is weaker, ignore it.

            @param board                new board
            @param l                    lower bound to store
            @param u                    upper bound to store
            @param mpc_level            requested MPC (Multi-ProbCut) level
            @param d                    requested depth
        */
        inline void get(const Board *board, int *l, int *u, const uint_fast8_t ml, const int d){
            if (data_strength(mpc_level.load(std::memory_order_relaxed), depth.load(std::memory_order_relaxed)) < data_strength(ml, d)){
                *l = -SCORE_INF;
                *u = SCORE_INF;
            } else{
                if (board->player != player.load(std::memory_order_relaxed) || board->opponent != opponent.load(std::memory_order_relaxed)){
                    *l = -SCORE_INF;
                    *u = SCORE_INF;
                } else{
                    *l = lower.load(std::memory_order_relaxed);
                    *u = upper.load(std::memory_order_relaxed);
                    if (board->player != player.load(std::memory_order_relaxed) || board->opponent != opponent.load(std::memory_order_relaxed)){
                        *l = -SCORE_INF;
                        *u = SCORE_INF;
                    }
                }
            }
        }
};

/*
    @brief Initialize value transposition table in parallel

    @param table                transposition table
    @param s                    start index
    @param e                    end index
*/
void init_value_transposition_table(Node_value_transposition_table table[], size_t s, size_t e){
    for(size_t i = s; i < e; ++i){
        table[i].init();
    }
}

/*
    @brief Value transposition table structure

    @param table_stack          transposition table on stack
    @param table_heap           transposition table on heap
    @param table_size           total table size
*/
class Value_transposition_table{
    private:
        Node_value_transposition_table table_stack[TRANSPOSITION_TABLE_STACK_SIZE];
        Node_value_transposition_table *table_heap;
        size_t table_size;

    public:
        /*
            @brief Constructor of best move transposition table
        */
        Value_transposition_table(){
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
                table_heap = (Node_value_transposition_table*)malloc(sizeof(Node_value_transposition_table) * (n_table_size - TRANSPOSITION_TABLE_STACK_SIZE));
                if (table_heap == nullptr)
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
                for (size_t i = 0; i < std::min(table_size, (size_t)TRANSPOSITION_TABLE_STACK_SIZE); ++i)
                    table_stack[i].init();
                for (size_t i = 0; i < table_size - (size_t)TRANSPOSITION_TABLE_STACK_SIZE; ++i)
                    table_heap[i].init();
            } else{
                int thread_size = thread_pool.size();
                size_t delta = (std::min(table_size, (size_t)TRANSPOSITION_TABLE_STACK_SIZE) + thread_size - 1) / thread_size;
                size_t s = 0, e;
                std::vector<std::future<void>> tasks;
                for (int i = 0; i < thread_size; ++i){
                    e = std::min(std::min(table_size, (size_t)TRANSPOSITION_TABLE_STACK_SIZE), s + delta);
                    tasks.emplace_back(thread_pool.push(std::bind(&init_value_transposition_table, table_stack, s, e)));
                    s = e;
                }
                if (table_size > TRANSPOSITION_TABLE_STACK_SIZE){
                    delta = (table_size - (size_t)TRANSPOSITION_TABLE_STACK_SIZE + thread_size - 1) / thread_size;
                    s = 0;
                    for (int i = 0; i < thread_size; ++i){
                        e = std::min(table_size - (size_t)TRANSPOSITION_TABLE_STACK_SIZE, s + delta);
                        tasks.emplace_back(thread_pool.push(std::bind(&init_value_transposition_table, table_heap, s, e)));
                        s = e;
                    }
                }
                for (std::future<void> &task: tasks)
                    task.get();
            }
        }

        /*
            @brief Register a board to value transposition table

            @param board                board to register
            @param hash                 hash code
            @param l                    lower bound
            @param u                    upper bound
            @param mpc_level            MPC (Multi-ProbCut) level
            @param d                    depth
        */
        inline void reg(const Board *board, const uint32_t hash, const int l, const int u, const uint_fast8_t ml, const int d){
            if (hash < TRANSPOSITION_TABLE_STACK_SIZE)
                table_stack[hash].reg(board, l, u, ml, d);
            else
                table_heap[hash - TRANSPOSITION_TABLE_STACK_SIZE].reg(board, l, u, ml, d);
        }

        /*
            @brief Get bounds from value transposition table

            @param board                board to register
            @param hash                 hash code
            @param l                    lower bound to store
            @param u                    upper bound to store
            @param mpc_level            requested MPC (Multi-ProbCut) level
            @param d                    requested depth
        */
        inline void get(const Board *board, const uint32_t hash, int *l, int *u, const uint_fast8_t ml, const int d){
            if (hash < TRANSPOSITION_TABLE_STACK_SIZE)
                table_stack[hash].get(board, l, u, ml, d);
            else
                table_heap[hash - TRANSPOSITION_TABLE_STACK_SIZE].get(board, l, u, ml, d);
        }
};

Value_transposition_table value_transposition_table;
Best_move_transposition_table best_move_transposition_table;

/*
    @brief Resize hash and transposition tables

    @param hash_level_failed    hash level used when failed
    @param hash_level           new hash level
    @return hash resized?
*/
bool hash_resize(int hash_level_failed, int hash_level, bool show_log){
    if (!value_transposition_table.resize(hash_level)){
        std::cerr << "[ERROR] value hash table resize failed. resize to level " << hash_level_failed << std::endl;
        value_transposition_table.resize(hash_level_failed);
        if (!hash_init(hash_level_failed)){
            std::cerr << "[ERROR] can't get hash. you can ignore this error" << std::endl;
            hash_init_rand(hash_level_failed);
        }
        global_hash_level = hash_level_failed;
        return false;
    }
    if (!best_move_transposition_table.resize(hash_level)){
        std::cerr << "[ERROR] best move hash table resize failed. resize to level " << hash_level_failed << std::endl;
        value_transposition_table.resize(hash_level_failed);
        best_move_transposition_table.resize(hash_level_failed);
        if (!hash_init(hash_level_failed)){
            std::cerr << "[ERROR] can't get hash. you can ignore this error" << std::endl;
            hash_init_rand(hash_level_failed);
        }
        global_hash_level = hash_level_failed;
        return false;
    }
    if (!hash_init(hash_level)){
        std::cerr << "[ERROR] can't get hash. you can ignore this error" << std::endl;
        hash_init_rand(hash_level);
    }
    global_hash_level = hash_level;
    if (show_log){
        double size_mb = (double)(sizeof(Node_value_transposition_table) + sizeof(Node_best_move_transposition_table)) / 1024 / 1024 * hash_sizes[hash_level];
        std::cerr << "hash resized to level " << hash_level << " elements " << hash_sizes[hash_level] << " size " << size_mb << " MB" << std::endl;
    }
    return true;
}