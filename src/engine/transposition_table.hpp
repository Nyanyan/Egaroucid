/*
    Egaroucid Project

    @file transposition_table.hpp
        Transposition table
    @date 2021-2024
    @author Takuto Yamana
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
#include <functional>

//#define USE_TT_DEPTH_THRESHOLD 0

/*
    @brief constants
*/
#define TRANSPOSITION_TABLE_UNDEFINED SCORE_INF
#define TRANSPOSITION_TABLE_N_LOOP 3
#if TT_USE_STACK
    constexpr size_t TRANSPOSITION_TABLE_STACK_SIZE = hash_sizes[DEFAULT_HASH_LEVEL] + TRANSPOSITION_TABLE_N_LOOP - 1;
#endif
#define N_TRANSPOSITION_MOVES 2
#define TT_REGISTER_THRESHOLD_RATE 4.0

inline uint32_t get_level_common(uint8_t depth, uint8_t mpc_level){
    return ((uint32_t)depth << 8) | mpc_level;
}

inline uint32_t get_level_common(int depth, uint_fast8_t mpc_level){
    return ((uint32_t)depth << 8) | mpc_level;
}

class Hash_node{
    private:
        Board key;
        uint8_t mpc_level;
        uint8_t depth;
        uint8_t importance;
        int8_t lower;
        int8_t upper;
        uint8_t moves[N_TRANSPOSITION_MOVES];

    public:
        void init(){
            key.player = 0;
            key.opponent = 0;
            lower = -SCORE_MAX;
            upper = SCORE_MAX;
            moves[0] = TRANSPOSITION_TABLE_UNDEFINED;
            moves[1] = TRANSPOSITION_TABLE_UNDEFINED;
            mpc_level = 0;
            depth = 0;
            importance = 0;
        }

        /*
            @brief Get level of the element

            @return level
        */
        inline uint32_t get_level(){
            if (importance)
                return get_level_common(depth, mpc_level);
            return 0;
        }

        /*
            @brief Get level of the element

            @return level
        */
        inline uint32_t get_level_no_importance(){
            return get_level_common(depth, mpc_level);
        }

        bool is_valid(const Board *board, uint32_t required_level){
            int l = lower;
            int u = upper;
            uint32_t node_level = get_level_no_importance();
            if (l != lower || u != upper){
                return false;
            }
            return key.player ^ l == board->player && key.opponent ^ u == board->opponent && node_level >= required_level;
        }

        bool is_valid_any_level(const Board *board){
            return key.player ^ lower == board->player && key.opponent ^ upper == board->opponent;
        }

        bool try_register(const Board *board, int r_depth, uint_fast8_t r_mpc_level, int alpha, int beta, int value, uint_fast8_t policy){
            int l = lower;
            int u = upper;
            uint8_t ml = mpc_level;
            uint8_t d = depth;
            uint8_t imp = importance;
            uint8_t m0 = moves[0];
            uint8_t m1 = moves[1];
            uint64_t key_p = key.player;
            uint64_t key_o = key.opponent;
            uint32_t node_level = 0;
            if (imp){
                node_level = get_level_common(d, ml);
            }
            uint32_t level = get_level_common(r_depth, r_mpc_level);
            uint64_t mask = get_mask(l, u, d, ml, m0, m1);
            if (node_level <= level){
                if (key_p == board->player ^ mask && key_o == board->opponent ^ mask){ // same board
                    if (node_level == level){ // same level
                        if (value < beta && value < u)
                            u = (int8_t)value;
                        if (alpha < value && l < value)
                            l = (int8_t)value;
                    } else{ // new level
                        ml = r_mpc_level;
                        d = r_depth;
                        if (value < beta)
                            u = (int8_t)value;
                        else
                            u = SCORE_MAX;
                        if (alpha < value)
                            l = (int8_t)value;
                        else
                            l = -SCORE_MAX;
                    }
                    if ((alpha < value || value == -SCORE_MAX) && m0 != policy && policy != TRANSPOSITION_TABLE_UNDEFINED){
                        m1 = m0;
                        m0 = (uint8_t)policy;
                    }
                } else{ // rewrite node
                    ml = r_mpc_level;
                    d = r_depth;
                    if (value < beta)
                        u = (int8_t)value;
                    else
                        u = SCORE_MAX;
                    if (alpha < value)
                        l = (int8_t)value;
                    else
                        l = -SCORE_MAX;
                    if (policy != TRANSPOSITION_TABLE_UNDEFINED){
                        m0 = (uint8_t)policy;
                        m1 = TRANSPOSITION_TABLE_UNDEFINED;
                    }
                }
                // register
                lower = l;
                upper = u;
                mpc_level = ml;
                depth = d;
                importance = 1;
                moves[0] = m0;
                moves[1] = m1;
                mask = get_mask(l, u, d, ml, m0, m1);
                key.player = board->player ^ mask;
                key.opponent = board->opponent ^ mask;
                return true;
            }
            return false;
        }

        bool try_get_bounds(const Board *board, uint32_t required_level, int *res_l, int *res_u){
            int l = lower;
            int u = upper;
            uint8_t ml = mpc_level;
            uint8_t d = depth;
            uint8_t m0 = moves[0];
            uint8_t m1 = moves[1];
            uint64_t key_p = key.player;
            uint64_t key_o = key.opponent;
            uint64_t mask = get_mask(l, u, d, ml, m0, m1);
            uint32_t node_level = get_level_common(d, ml);
            if (key_p == board->player ^ mask && key_o == board->opponent ^ mask && node_level >= required_level){
                *res_l = l;
                *res_u = u;
                return true;
            }
            return false;
        }

        bool try_get_bounds_any_level(const Board *board, int *res_l, int *res_u){
            int l = lower;
            int u = upper;
            uint8_t ml = mpc_level;
            uint8_t d = depth;
            uint8_t m0 = moves[0];
            uint8_t m1 = moves[1];
            uint64_t key_p = key.player;
            uint64_t key_o = key.opponent;
            uint64_t mask = get_mask(l, u, d, ml, m0, m1);
            if (key_p == board->player ^ mask && key_o == board->opponent ^ mask){
                *res_l = l;
                *res_u = u;
                return true;
            }
            return false;
        }

        int try_get_best_move_any_level(const Board *board){
            int l = lower;
            int u = upper;
            uint8_t ml = mpc_level;
            uint8_t d = depth;
            uint8_t m0 = moves[0];
            uint8_t m1 = moves[1];
            uint64_t key_p = key.player;
            uint64_t key_o = key.opponent;
            uint64_t mask = get_mask(l, u, d, ml, m0, m1);
            if (key_p == board->player ^ mask && key_o == board->opponent ^ mask){
                return m0;
            }
            return TRANSPOSITION_TABLE_UNDEFINED;
        }

        bool try_get_moves_any_level(const Board *board, uint_fast8_t res_moves[]){
            int l = lower;
            int u = upper;
            uint8_t ml = mpc_level;
            uint8_t d = depth;
            uint8_t m0 = moves[0];
            uint8_t m1 = moves[1];
            uint64_t key_p = key.player;
            uint64_t key_o = key.opponent;
            uint64_t mask = get_mask(l, u, d, ml, m0, m1);
            if (key_p == board->player ^ mask && key_o == board->opponent ^ mask){
                res_moves[0] = m0;
                res_moves[1] = m1;
                return true;
            }
            return false;
        }

        void set_importance_zero(){
            importance = 0;
        }
    
    private:
        uint64_t get_mask(int8_t l, int8_t u, uint8_t d, uint8_t ml, uint8_t m0, uint8_t m1){
            return ((uint64_t)m0 << 40) | ((uint64_t)m1 << 32) | ((uint64_t)ml << 24) | ((uint64_t)d << 16) | ((uint64_t)(u + SCORE_MAX) << 8) | (l + SCORE_MAX);
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
void set_importance_zero_transposition_table(Hash_node table[], size_t s, size_t e){
    for(size_t i = s; i < e; ++i){
        table[i].set_importance_zero();
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
        std::mutex mtx;
        #if TT_USE_STACK
            Hash_node table_stack[TRANSPOSITION_TABLE_STACK_SIZE];
        #endif
        #if USE_CHANGEABLE_HASH_LEVEL || !TT_USE_STACK
            Hash_node *table_heap;
        #endif
        size_t table_size;
        std::atomic<uint64_t> n_registered;
        uint64_t n_registered_threshold;

    public:
        /*
            @brief Constructor of transposition table
        */
        Transposition_table(){
            #if USE_CHANGEABLE_HASH_LEVEL || !TT_USE_STACK
                table_heap = nullptr;
            #endif
            table_size = 0;
            n_registered = 0;
            n_registered_threshold = 0;
        }

        #if USE_CHANGEABLE_HASH_LEVEL
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
                #if TT_USE_STACK
                    if (n_table_size > TRANSPOSITION_TABLE_STACK_SIZE){
                        table_heap = (Hash_node*)malloc(sizeof(Hash_node) * (n_table_size - TRANSPOSITION_TABLE_STACK_SIZE));
                        if (table_heap == nullptr)
                            return false;
                    }
                #else
                    table_heap = (Hash_node*)malloc(sizeof(Hash_node) * n_table_size);
                    if (table_heap == nullptr)
                        return false;
                #endif
                table_size = n_table_size;
                n_registered_threshold = table_size * TT_REGISTER_THRESHOLD_RATE;
                init();
                return true;
            }
        #else
            inline bool set_size(){
                table_size = TRANSPOSITION_TABLE_STACK_SIZE;
                n_registered_threshold = table_size * TT_REGISTER_THRESHOLD_RATE;
                init();
                return true;
            }
        #endif

        /*
            @brief Initialize transposition table
        */
        inline void init(){
            int thread_size = thread_pool.size();
            if (thread_size == 0){
                #if TT_USE_STACK
                    for (size_t i = 0; i < std::min(table_size, (size_t)TRANSPOSITION_TABLE_STACK_SIZE); ++i)
                        table_stack[i].init();
                    #if USE_CHANGEABLE_HASH_LEVEL
                        if (table_size > TRANSPOSITION_TABLE_STACK_SIZE){
                            for (size_t i = 0; i < table_size - (size_t)TRANSPOSITION_TABLE_STACK_SIZE; ++i)
                                table_heap[i].init();
                        }
                    #endif
                #else
                    for (size_t i = 0; i < table_size; ++i)
                        table_heap[i].init();
                #endif
            } else{
                size_t s, e;
                std::vector<std::future<void>> tasks;
                #if TT_USE_STACK
                    size_t delta = (std::min(table_size, (size_t)TRANSPOSITION_TABLE_STACK_SIZE) + thread_size - 1) / thread_size;
                    s = 0;
                    for (int i = 0; i < thread_size; ++i){
                        e = std::min(std::min(table_size, (size_t)TRANSPOSITION_TABLE_STACK_SIZE), s + delta);
                        bool pushed = false;
                        while (!pushed){
                            tasks.emplace_back(thread_pool.push(&pushed, std::bind(&init_transposition_table, table_stack, s, e)));
                            if (!pushed)
                                tasks.pop_back();
                        }
                        s = e;
                    }
                    #if USE_CHANGEABLE_HASH_LEVEL
                        if (table_size > TRANSPOSITION_TABLE_STACK_SIZE){
                            delta = (table_size - (size_t)TRANSPOSITION_TABLE_STACK_SIZE + thread_size - 1) / thread_size;
                            s = 0;
                            for (int i = 0; i < thread_size; ++i){
                                e = std::min(table_size - (size_t)TRANSPOSITION_TABLE_STACK_SIZE, s + delta);
                                bool pushed = false;
                                while (!pushed){
                                    tasks.emplace_back(thread_pool.push(&pushed, std::bind(&init_transposition_table, table_heap, s, e)));
                                    if (!pushed)
                                        tasks.pop_back();
                                }
                                s = e;
                            }
                        }
                    #endif
                #else
                    size_t delta = (table_size + thread_size - 1) / thread_size;
                    s = 0;
                    for (int i = 0; i < thread_size; ++i){
                        e = std::min(table_size, s + delta);
                        bool pushed = false;
                        while (!pushed){
                            tasks.emplace_back(thread_pool.push(&pushed, std::bind(&init_transposition_table, table_heap, s, e)));
                            if (!pushed)
                                tasks.pop_back();
                        }
                        s = e;
                    }
                #endif
                for (std::future<void> &task: tasks)
                    task.get();
            }
        }

        /*
            @brief set all date to 0
        */
        inline void reset_importance(){
            std::lock_guard lock(mtx);
            reset_importance_proc();
        }

        /*
            @brief set all date to 0

            create new thread
        */
        inline void reset_importance_new_thread(int thread_size){
            size_t s, e;
            std::vector<std::future<void>> tasks;
            #if TT_USE_STACK
                size_t delta = (std::min(table_size, (size_t)TRANSPOSITION_TABLE_STACK_SIZE) + thread_size - 1) / thread_size;
                s = 0;
                for (int i = 0; i < thread_size; ++i){
                    e = std::min(std::min(table_size, (size_t)TRANSPOSITION_TABLE_STACK_SIZE), s + delta);
                    tasks.emplace_back(std::async(std::launch::async, std::bind(&set_importance_zero_transposition_table, table_stack, s, e)));
                    s = e;
                }
                #if USE_CHANGEABLE_HASH_LEVEL
                    if (table_size > TRANSPOSITION_TABLE_STACK_SIZE){
                        delta = (table_size - (size_t)TRANSPOSITION_TABLE_STACK_SIZE + thread_size - 1) / thread_size;
                        s = 0;
                        for (int i = 0; i < thread_size; ++i){
                            e = std::min(table_size - (size_t)TRANSPOSITION_TABLE_STACK_SIZE, s + delta);
                            tasks.emplace_back(std::async(std::launch::async, std::bind(&set_importance_zero_transposition_table, table_heap, s, e)));
                            s = e;
                        }
                    }
                #endif
            #else
                size_t delta = (table_size + thread_size - 1) / thread_size;
                s = 0;
                for (int i = 0; i < thread_size; ++i){
                    e = std::min(table_size, s + delta);
                    tasks.emplace_back(std::async(std::launch::async, std::bind(&set_importance_zero_transposition_table, table_heap, s, e)));
                    s = e;
                }
            #endif
            for (std::future<void> &task: tasks)
                task.get();
            n_registered = 0;
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
            const uint32_t level = get_level_common(depth, search->mpc_level);
            uint32_t node_level;
            for (uint_fast8_t i = 0; i < TRANSPOSITION_TABLE_N_LOOP; ++i){
                if (node->try_register(&search->board, depth, search->mpc_level, alpha, beta, value, policy)){
                    break;
                }
                ++hash;
                node = get_node(hash);
            }
            if (n_registered >= n_registered_threshold){
                std::lock_guard lock(mtx);
                if (n_registered >= n_registered_threshold){
                    //std::cerr << "resetting transposition importance" << std::endl;
                    reset_importance_proc();
                }
            }
        }

        /*
            @brief get bounds and best moves from transposition table

            @param search               Search information
            @param hash                 hash code
            @param depth                depth
            @param lower                lower bound to store
            @param upper                upper bound to store
            @param moves                best moves to store
        */
        inline void get(const Search *search, uint32_t hash, const int depth, int *lower, int *upper, uint_fast8_t moves[]){
            Hash_node *node = get_node(hash);
            const uint32_t level = get_level_common(depth, search->mpc_level);
            for (uint_fast8_t i = 0; i < TRANSPOSITION_TABLE_N_LOOP; ++i){
                if (node->try_get_moves_any_level(&search->board, moves)){
                    return;
                }
                ++hash;
                node = get_node(hash);
            }
        }

        /*
            @brief get bounds from transposition table

            @param search               Search information
            @param hash                 hash code
            @param depth                depth
            @param lower                lower bound to store
            @param upper                upper bound to store
        */
        inline void get_bounds(const Search *search, uint32_t hash, const int depth, int *lower, int *upper){
            Hash_node *node = get_node(hash);
            const uint32_t level = get_level_common(depth, search->mpc_level);
            for (uint_fast8_t i = 0; i < TRANSPOSITION_TABLE_N_LOOP; ++i){
                if (node->try_get_bounds(&search->board, level, lower, upper)){
                    return;
                }
                ++hash;
                node = get_node(hash);
            }
        }

        /*
            @brief get bounds from transposition table (no level check)

            @param search               Search information
            @param hash                 hash code
            @param lower                lower bound to store
            @param upper                upper bound to store
        */
        inline bool get_bounds_any_level(const Search *search, uint32_t hash, int *lower, int *upper){
            Hash_node *node = get_node(hash);
            for (uint_fast8_t i = 0; i < TRANSPOSITION_TABLE_N_LOOP; ++i){
                if (node->try_get_bounds_any_level(&search->board, lower, upper)){
                    return true;
                }
                ++hash;
                node = get_node(hash);
            }
            return false;
        }

        /*
            @brief get bounds from transposition table (no level check)

            @param board                Board
            @param hash                 hash code
            @param lower                lower bound to store
            @param upper                upper bound to store
        */
        inline bool get_bounds_any_level(const Board *board, uint32_t hash, int *lower, int *upper){
            Hash_node *node = get_node(hash);
            for (uint_fast8_t i = 0; i < TRANSPOSITION_TABLE_N_LOOP; ++i){
                if (node->try_get_bounds_any_level(board, lower, upper)){
                    return true;
                }
                ++hash;
                node = get_node(hash);
            }
            return false;
        }

        /*
            @brief get best move from transposition table (no level check)

            @param board                board
            @param hash                 hash code
            @return best move
        */
        inline int get_best_move(const Board *board, uint32_t hash){
            Hash_node *node = get_node(hash);
            int move = TRANSPOSITION_TABLE_UNDEFINED;
            for (uint_fast8_t i = 0; i < TRANSPOSITION_TABLE_N_LOOP; ++i){
                move = node->try_get_best_move_any_level(board);
                if (move != TRANSPOSITION_TABLE_UNDEFINED){
                    return move;
                }
                ++hash;
                node = get_node(hash);
            }
            return TRANSPOSITION_TABLE_UNDEFINED;
        }

        /*
            @brief get best moves from transposition table with any level

            @param board                board
            @param hash                 hash code
            @return best move
        */
        inline bool get_moves_any_level(const Board *board, uint32_t hash, uint_fast8_t moves[]){
            Hash_node *node = get_node(hash);
            for (uint_fast8_t i = 0; i < TRANSPOSITION_TABLE_N_LOOP; ++i){
                if (node->try_get_moves_any_level(board, moves)){
                    return true;
                }
                ++hash;
                node = get_node(hash);
            }
            return false;
        }

        /*
            @brief get best move from transposition table

            @param board                board
            @param hash                 hash code
            @return best move
        */
        /*
        inline bool get_if_perfect(const Board *board, uint32_t hash, int *val, int *best_move){
            Hash_node *node = get_node(hash);
            for (uint_fast8_t i = 0; i < TRANSPOSITION_TABLE_N_LOOP; ++i){
                if (node->try_get_data_if_perfect(board, moves)){
                }
                if (node->board.player == board->player && node->board.opponent == board->opponent){
                    node->lock.lock();
                        if (node->board.player == board->player && node->board.opponent == board->opponent && node->data.get_mpc_level() == MPC_100_LEVEL){
                            uint_fast8_t moves[N_TRANSPOSITION_MOVES];
                            node->data.get_moves(moves);
                            int l, u;
                            node->data.get_bounds(&l, &u);
                            node->lock.unlock();
                            bool is_perfect = l == u;
                            if (is_perfect){
                                *val = l;
                                *best_move = moves[0];
                            }
                            return is_perfect;
                        }
                    node->lock.unlock();
                }
                ++hash;
                node = get_node(hash);
            }
            return false;
        }
        */

        inline bool has_node(const Search *search, uint32_t hash, int depth){
            Hash_node *node = get_node(hash);
            const uint32_t level = get_level_common(depth, search->mpc_level);
            for (uint_fast8_t i = 0; i < TRANSPOSITION_TABLE_N_LOOP; ++i){
                if (node->is_valid(&search->board, level)){
                    return true;
                }
                ++hash;
                node = get_node(hash);
            }
            return false;
        }

        inline bool has_node_any_level(const Search *search, uint32_t hash){
            Hash_node *node = get_node(hash);
            for (uint_fast8_t i = 0; i < TRANSPOSITION_TABLE_N_LOOP; ++i){
                if (node->is_valid_any_level(&search->board)){
                    return true;
                }
                ++hash;
                node = get_node(hash);
            }
            return false;
        }

    private:
        inline Hash_node* get_node(uint32_t hash){
            #if TT_USE_STACK
                #if USE_CHANGEABLE_HASH_LEVEL
                    if (hash < TRANSPOSITION_TABLE_STACK_SIZE)
                        return &table_stack[hash];
                    return &table_heap[hash - TRANSPOSITION_TABLE_STACK_SIZE];
                #else
                    return &table_stack[hash];
                #endif
            #else
                return &table_heap[hash];
            #endif
        }

        inline void reset_importance_proc(){
            //std::cerr << "importance reset n_registered " << n_registered << " threshold " << n_registered_threshold << " table_size " << table_size << std::endl;
            #if TT_USE_STACK
                for (size_t i = 0; i < std::min(table_size, (size_t)TRANSPOSITION_TABLE_STACK_SIZE); ++i)
                    table_stack[i].set_importance_zero();
                #if USE_CHANGEABLE_HASH_LEVEL
                    if (table_size > TRANSPOSITION_TABLE_STACK_SIZE){
                        for (size_t i = 0; i < table_size - (size_t)TRANSPOSITION_TABLE_STACK_SIZE; ++i)
                            table_heap[i].set_importance_zero();
                    }
                #endif
            #else
                for (size_t i = 0; i < table_size; ++i)
                    table_heap[i].set_importance_zero();
            #endif
            n_registered.store(0);
        }
};
//#endif

Transposition_table transposition_table;

void transposition_table_init(){
    transposition_table.init();
}

#if USE_CHANGEABLE_HASH_LEVEL
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
        @brief Resize hash and transposition tables

        @param hash_level_failed    hash level used when failed
        @param hash_level           new hash level
        @param binary_path          path to binary
        @return hash resized?
    */
    bool hash_resize(int hash_level_failed, int hash_level, std::string binary_path, bool show_log){
        if (!transposition_table.resize(hash_level)){
            std::cerr << "[ERROR] hash table resize failed. resize to level " << hash_level_failed << std::endl;
            transposition_table.resize(hash_level_failed);
            if (!hash_init(hash_level_failed, binary_path)){
                std::cerr << "[ERROR] can't get hash. you can ignore this error" << std::endl;
                hash_init_rand(hash_level_failed);
            }
            global_hash_level = hash_level_failed;
            return false;
        }
        if (!hash_init(hash_level, binary_path)){
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
#else
    bool hash_tt_init(std::string binary_path, bool show_log){
        transposition_table.set_size();
        if (!hash_init(DEFAULT_HASH_LEVEL, binary_path)){
            std::cerr << "[ERROR] can't get hash. you can ignore this error" << std::endl;
            hash_init_rand(DEFAULT_HASH_LEVEL);
        }
        return true;
    }

    bool hash_tt_init(bool show_log){
        transposition_table.set_size();
        if (!hash_init(DEFAULT_HASH_LEVEL)){
            std::cerr << "[ERROR] can't get hash. you can ignore this error" << std::endl;
            hash_init_rand(DEFAULT_HASH_LEVEL);
        }
        return true;
    }
#endif

inline bool transposition_cutoff(Search *search, uint32_t hash_code, int depth, int *alpha, int *beta, int *v, uint_fast8_t moves[]){
    //if (depth >= USE_TT_DEPTH_THRESHOLD){
    int lower = -SCORE_MAX, upper = SCORE_MAX;
    transposition_table.get(search, hash_code, depth, &lower, &upper, moves);
    if (upper == lower || upper <= *alpha){
        *v = upper;
        return true;
    }
    if (*beta <= lower){
        *v = lower;
        return true;
    }
    if (*alpha < lower){
        *alpha = lower;
    }
    if(upper < *beta){
        *beta = upper;
    }
    //}
    return false;
}

inline bool transposition_cutoff_bestmove(Search *search, uint32_t hash_code, int depth, int *alpha, int *beta, int *v, int *best_move){
    //if (depth >= USE_TT_DEPTH_THRESHOLD){
    int lower = -SCORE_MAX, upper = SCORE_MAX;
    uint_fast8_t moves[N_TRANSPOSITION_MOVES];
    transposition_table.get(search, hash_code, depth, &lower, &upper, moves);
    if (upper == lower || upper <= *alpha){
        *v = upper;
        *best_move = moves[0];
        return true;
    }
    if (*beta <= lower){
        *v = lower;
        *best_move = moves[0];
        return true;
    }
    if (*alpha < lower){
        *alpha = lower;
    }
    if(upper < *beta){
        *beta = upper;
    }
    //}
    return false;
}

inline bool transposition_cutoff_nws(Search *search, uint32_t hash_code, int depth, int alpha, int *v, uint_fast8_t moves[]){
    //if (depth >= USE_TT_DEPTH_THRESHOLD){
    int lower = -SCORE_MAX, upper = SCORE_MAX;
    transposition_table.get(search, hash_code, depth, &lower, &upper, moves);
    if (upper == lower || upper <= alpha){
        *v = upper;
        return true;
    }
    if (alpha < lower){
        *v = lower;
        return true;
    }
    //}
    return false;
}

inline bool transposition_cutoff_nws(Search *search, uint32_t hash_code, int depth, int alpha, int *v){
    //if (depth >= USE_TT_DEPTH_THRESHOLD){
    int lower = -SCORE_MAX, upper = SCORE_MAX;
    transposition_table.get_bounds(search, hash_code, depth, &lower, &upper);
    if (upper == lower || upper <= alpha){
        *v = upper;
        return true;
    }
    if (alpha < lower){
        *v = lower;
        return true;
    }
    //}
    return false;
}

inline bool transposition_cutoff_nws_bestmove(Search *search, uint32_t hash_code, int depth, int alpha, int *v, int *best_move){
    //if (depth >= USE_TT_DEPTH_THRESHOLD){
    int lower = -SCORE_MAX, upper = SCORE_MAX;
    uint_fast8_t moves[N_TRANSPOSITION_MOVES];
    transposition_table.get(search, hash_code, depth, &lower, &upper, moves);
    if (upper == lower || upper <= alpha){
        *v = upper;
        *best_move = moves[0];
        return true;
    }
    if (alpha < lower){
        *v = lower;
        *best_move = moves[0];
        return true;
    }
    //}
    return false;
}
