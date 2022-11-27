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
#include "spinlock.hpp"
#include "search.hpp"
#include <future>

/*
    @brief constants
*/
#define TRANSPOSITION_TABLE_UNDEFINED SCORE_INF
#define TRANSPOSITION_TABLE_N_LOOP 3
constexpr size_t TRANSPOSITION_TABLE_STACK_SIZE = hash_sizes[DEFAULT_HASH_LEVEL] + TRANSPOSITION_TABLE_N_LOOP - 1;
#define N_TRANSPOSITION_MOVES 2

// date manager
#define MAX_DATE 127
#define INIT_DATE 1

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
    @brief Hash data

    @param level
        @param date                 search date (to rewrite old entry)  more important
        @param depth                depth                                      |
        @param mpc_level            MPC level                                  |
        @param cost                 search cost (log2(nodes))           less important
    @param lower                lower bound
    @param upper                upper bound
    @param moves                best moves
*/

class Hash_data{
    private:
        union{
            struct{ // for little endian
                uint8_t cost;
                uint8_t mpc_level;
                uint8_t depth;
                uint8_t date;
            } level_data;
            uint32_t level;
        } level;
        int8_t lower;
        int8_t upper;
        uint8_t moves[N_TRANSPOSITION_MOVES];

    public:

        /*
            @brief Initialize a node
        */
        inline void init(){
            lower = -SCORE_MAX;
            upper = SCORE_MAX;
            moves[0] = TRANSPOSITION_TABLE_UNDEFINED;
            moves[1] = TRANSPOSITION_TABLE_UNDEFINED;
            level.level = 0;
            //std::cerr << (int)level.level_data.date << " " << (int)level.level_data.depth << " " << (int)level.level_data.mpc_level << " " << (int)level.level_data.cost << std::endl;
        }

        /*
            @brief Register value (same level)

            @param alpha                alpha bound
            @param beta                 beta bound
            @param value                best value
            @param policy               best move
        */
        inline void reg_same_level(const int alpha, const int beta, const int value, const int policy){
            if (value < beta && value < upper)
                upper = (int8_t)value;
            if (alpha < value && lower < value)
                lower = (int8_t)value;
            if ((alpha < value || value == -SCORE_MAX) && moves[0] != policy){
                moves[1] = moves[0];
                moves[0] = (uint8_t)policy;
            }
        }

        /*
            @brief Register value (level increase)

            update best moves, reset other data

            @param d                    depth
            @param ml                   MPC level
            @param c                    cost
            @param dt                   date
            @param alpha                alpha bound
            @param beta                 beta bound
            @param value                best value
            @param policy               best move
        */
        inline void reg_new_level(const int d, const uint_fast8_t ml, const uint8_t c, const uint8_t dt, const int alpha, const int beta, const int value, const int policy){
            if (value < beta)
                upper = (int8_t)value;
            else
                upper = SCORE_MAX;
            if (alpha < value)
                lower = (int8_t)value;
            else
                lower = -SCORE_MAX;
            if ((alpha < value || value == -SCORE_MAX) && moves[0] != policy){
                moves[1] = moves[0];
                moves[0] = policy;
            }
            level.level_data.depth = d;
            level.level_data.mpc_level = ml;
            //level.level_data.cost = std::max(level.level_data.cost, c);
            level.level_data.date = dt;
        }

        /*
            @brief Register value (new board)

            update best moves, reset other data

            @param d                    depth
            @param ml                   MPC level
            @param c                    cost
            @param dt                   date
            @param alpha                alpha bound
            @param beta                 beta bound
            @param value                best value
            @param policy               best move
        */
        inline void reg_new_data(const int d, const uint_fast8_t ml, const uint8_t c, const uint8_t dt, const int alpha, const int beta, const int value, const int policy){
            if (value < beta)
                upper = (int8_t)value;
            else
                upper = SCORE_MAX;
            if (alpha < value)
                lower = (int8_t)value;
            else
                lower = -SCORE_MAX;
            if ((alpha < value || value == -SCORE_MAX) && moves[0] != policy)
                moves[0] = policy;
            else
                moves[0] = TRANSPOSITION_TABLE_UNDEFINED;
            moves[1] = TRANSPOSITION_TABLE_UNDEFINED;
            level.level_data.depth = d;
            level.level_data.mpc_level = ml;
            //level.level_data.cost = c;
            level.level_data.date = dt;
        }

        /*
            @brief Get level of the element

            @return level
        */
        inline uint32_t get_level(){
            //std::cerr << level.level << " " << (int)level.level_data.date << " " << (int)level.level_data.depth << " " << (int)level.level_data.mpc_level << " " << (int)level.level_data.cost << std::endl;
            return level.level;
        }

        /*
            @brief Get moves

            @param res_moves            array to store result
        */
        inline void get_moves(uint_fast8_t res_moves[]){
            res_moves[0] = moves[0];
            res_moves[1] = moves[1];
        }

        /*
            @brief Get bounds

            @param l                    lower bound
            @param u                    upper bound
        */
        inline void get_bounds(int *l, int *u){
            *l = lower;
            *u = upper;
        }

        /*
            @brief Reset date
        */
        inline void reset_date(){
            level.level_data.date = 0;
        }
};

struct Hash_node{
    Board board;
    Hash_data data;
    Spinlock lock;

    void init(){
        board.player = 0ULL;
        board.opponent = 0ULL;
        data.init();
    }
};

/*
    @brief Initialize transposition table in parallel

    @param table                transposition table
    @param s                    start index
    @param e                    end index
*/
void init_transposition_table(Hash_node table[], size_t s, size_t e){
    for(size_t i = s; i < e; ++i){
        table[i].init();
    }
}

/*
    @brief Resetting date in parallel

    @param table                transposition table
    @param s                    start index
    @param e                    end index
*/
void reset_date_transposition_table(Hash_node table[], size_t s, size_t e){
    for(size_t i = s; i < e; ++i){
        table[i].data.reset_date();
    }
}

/*
    @brief Best move transposition table structure

    @param table_stack          transposition table on stack
    @param table_heap           transposition table on heap
    @param table_size           total table size
*/
class Transposition_table{
    private:
        Hash_node table_stack[TRANSPOSITION_TABLE_STACK_SIZE];
        Hash_node *table_heap;
        size_t table_size;

    public:
        /*
            @brief Constructor of transposition table
        */
        Transposition_table(){
            table_heap = nullptr;
            table_size = 0;
        }

        /*
            @brief Resize transposition table

            @param hash_level           hash level representing the size
            @return table initialized?
        */
        inline bool resize(int hash_level){
            size_t n_table_size = hash_sizes[hash_level] + TRANSPOSITION_TABLE_N_LOOP - 1;
            table_size = 0;
			if (table_heap != nullptr) {
				free(table_heap);
				table_heap = nullptr;
			}
            if (n_table_size > TRANSPOSITION_TABLE_STACK_SIZE){
                table_heap = (Hash_node*)malloc(sizeof(Hash_node) * (n_table_size - TRANSPOSITION_TABLE_STACK_SIZE));
                if (table_heap == nullptr)
                    return false;
            }
            table_size = n_table_size;
            init();
            return true;
        }

        /*
            @brief Initialize transposition table
        */
        inline void init(){
            if (thread_pool.size() == 0){
                for (size_t i = 0; i < std::min(table_size, (size_t)TRANSPOSITION_TABLE_STACK_SIZE); ++i)
                    table_stack[i].init();
                if (table_size > TRANSPOSITION_TABLE_STACK_SIZE){
                    for (size_t i = 0; i < table_size - (size_t)TRANSPOSITION_TABLE_STACK_SIZE; ++i)
                        table_heap[i].init();
                }
            } else{
                int thread_size = thread_pool.size();
                size_t delta = (std::min(table_size, (size_t)TRANSPOSITION_TABLE_STACK_SIZE) + thread_size - 1) / thread_size;
                size_t s = 0, e;
                std::vector<std::future<void>> tasks;
                for (int i = 0; i < thread_size; ++i){
                    e = std::min(std::min(table_size, (size_t)TRANSPOSITION_TABLE_STACK_SIZE), s + delta);
                    tasks.emplace_back(thread_pool.push(std::bind(&init_transposition_table, table_stack, s, e)));
                    s = e;
                }
                if (table_size > TRANSPOSITION_TABLE_STACK_SIZE){
                    delta = (table_size - (size_t)TRANSPOSITION_TABLE_STACK_SIZE + thread_size - 1) / thread_size;
                    s = 0;
                    for (int i = 0; i < thread_size; ++i){
                        e = std::min(table_size - (size_t)TRANSPOSITION_TABLE_STACK_SIZE, s + delta);
                        tasks.emplace_back(thread_pool.push(std::bind(&init_transposition_table, table_heap, s, e)));
                        s = e;
                    }
                }
                for (std::future<void> &task: tasks)
                    task.get();
            }
        }

        /*
            @brief set all date to 0
        */
        inline void reset_date(){
            if (thread_pool.size() == 0){
                for (size_t i = 0; i < std::min(table_size, (size_t)TRANSPOSITION_TABLE_STACK_SIZE); ++i)
                    table_stack[i].data.reset_date();
                if (table_size > TRANSPOSITION_TABLE_STACK_SIZE){
                    for (size_t i = 0; i < table_size - (size_t)TRANSPOSITION_TABLE_STACK_SIZE; ++i)
                        table_heap[i].data.reset_date();
                }
            } else{
                int thread_size = thread_pool.size();
                size_t delta = (std::min(table_size, (size_t)TRANSPOSITION_TABLE_STACK_SIZE) + thread_size - 1) / thread_size;
                size_t s = 0, e;
                std::vector<std::future<void>> tasks;
                for (int i = 0; i < thread_size; ++i){
                    e = std::min(std::min(table_size, (size_t)TRANSPOSITION_TABLE_STACK_SIZE), s + delta);
                    tasks.emplace_back(thread_pool.push(std::bind(&reset_date_transposition_table, table_stack, s, e)));
                    s = e;
                }
                if (table_size > TRANSPOSITION_TABLE_STACK_SIZE){
                    delta = (table_size - (size_t)TRANSPOSITION_TABLE_STACK_SIZE + thread_size - 1) / thread_size;
                    s = 0;
                    for (int i = 0; i < thread_size; ++i){
                        e = std::min(table_size - (size_t)TRANSPOSITION_TABLE_STACK_SIZE, s + delta);
                        tasks.emplace_back(thread_pool.push(std::bind(&reset_date_transposition_table, table_heap, s, e)));
                        s = e;
                    }
                }
                for (std::future<void> &task: tasks)
                    task.get();
            }
        }

        /*
            @brief Register items

            @param search               Search information
            @param hash                 hash code
            @param depth                depth
            @param alpha                alpha bound
            @param beta                 beta bound
            @param value                best score
            @param policy               best move
            @param cost                 search cost (log2(nodes))
        */
        inline void reg(const Search *search, uint32_t hash, const int depth, int alpha, int beta, int value, int policy){
            Hash_node *node = get_node(hash);
            Hash_node *worst_node = nullptr;
            //const uint32_t level = ((uint32_t)search->date << 24) | ((uint32_t)cost << 16) | ((uint32_t)search->mpc_level << 8) | depth;
            const uint32_t level = ((uint32_t)search->date << 24) | ((uint32_t)depth << 16) | ((uint32_t)search->mpc_level << 8);
            uint32_t node_level, worst_level = 0xFFFFFFFFU;
            for (uint_fast8_t i = 0; i < TRANSPOSITION_TABLE_N_LOOP; ++i){
                if (node->data.get_level() <= level){
                    node->lock.lock();
                        node_level = node->data.get_level();
                        if (node_level <= level){
                            if (node->board.player == search->board.player && node->board.opponent == search->board.opponent){
                                if (node_level == level)
                                    node->data.reg_same_level(alpha, beta, value, policy);
                                else
                                    node->data.reg_new_level(depth, search->mpc_level, 0, search->date, alpha, beta, value, policy);
                                node->lock.unlock();
                                return;
                            } else if (node_level < worst_level){
                                worst_level = node_level;
                                worst_node = node;
                            }
                        }
                    node->lock.unlock();
                }
                ++hash;
                node = get_node(hash);
            }
            if (worst_node != nullptr){
                worst_node->lock.lock();
                    if (worst_node->data.get_level() <= level){
                        worst_node->board.player = search->board.player;
                        worst_node->board.opponent = search->board.opponent;
                        worst_node->data.reg_new_data(depth, search->mpc_level, 0, search->date, alpha, beta, value, policy);
                    }
                worst_node->lock.unlock();
            }
        }

        /*
            @brief get best move from transposition table

            @param search               Search information
            @param hash                 hash code
            @param depth                depth
            @param lower                lower bound to store
            @param upper                upper bound to store
            @param moves                best moves to store
        */
        inline void get(const Search *search, uint32_t hash, const int depth, int *lower, int *upper, uint_fast8_t moves[]){
            Hash_node *node = get_node(hash);
            const uint32_t level = ((uint32_t)search->date << 24) | ((uint32_t)depth << 16) | ((uint32_t)search->mpc_level << 8);
            for (uint_fast8_t i = 0; i < TRANSPOSITION_TABLE_N_LOOP; ++i){
                if (node->board.player == search->board.player && node->board.opponent == search->board.opponent){
                    node->lock.lock();
                        if (node->board.player == search->board.player && node->board.opponent == search->board.opponent){
                            node->data.get_moves(moves);
                            if (node->data.get_level() >= level)
                                node->data.get_bounds(lower, upper);
                            node->lock.unlock();
                            return;
                        }
                    node->lock.unlock();
                }
                ++hash;
                node = get_node(hash);
            }
        }

        /*
            @brief get best move from transposition table

            @param board                board
            @param hash                 hash code
            @return best move
        */
        inline int get_best_move(const Board *board, uint32_t hash){
            Hash_node *node = get_node(hash);
            for (uint_fast8_t i = 0; i < TRANSPOSITION_TABLE_N_LOOP; ++i){
                if (node->board.player == board->player && node->board.opponent == board->opponent){
                    node->lock.lock();
                        if (node->board.player == board->player && node->board.opponent == board->opponent){
                            uint_fast8_t moves[N_TRANSPOSITION_MOVES];
                            node->data.get_moves(moves);
                            node->lock.unlock();
                            return moves[0];
                        }
                    node->lock.unlock();
                }
                ++hash;
                node = get_node(hash);
            }
            return TRANSPOSITION_TABLE_UNDEFINED;
        }

    private:
        inline Hash_node* get_node(uint32_t hash){
            if (hash < TRANSPOSITION_TABLE_STACK_SIZE)
                return &table_stack[hash];
            return &table_heap[hash - TRANSPOSITION_TABLE_STACK_SIZE];
        }
};

Transposition_table transposition_table;

/*
    @brief Resize hash and transposition tables

    @param hash_level_failed    hash level used when failed
    @param hash_level           new hash level
    @return hash resized?
*/
bool hash_resize(int hash_level_failed, int hash_level, bool show_log){
    if (!transposition_table.resize(hash_level)){
        std::cerr << "[ERROR] hash table resize failed. resize to level " << hash_level_failed << std::endl;
        transposition_table.resize(hash_level_failed);
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
        double size_mb = (double)sizeof(Hash_node) / 1024 / 1024 * hash_sizes[hash_level];
        std::cerr << "hash resized to level " << hash_level << " elements " << hash_sizes[hash_level] << " size " << size_mb << " MB" << std::endl;
    }
    return true;
}

/*
    @brief manage date

    @return new date
*/
uint8_t manage_date(uint8_t date){
    if (date > MAX_DATE){
        transposition_table.reset_date();
        return INIT_DATE;
    }
    return date;
}
