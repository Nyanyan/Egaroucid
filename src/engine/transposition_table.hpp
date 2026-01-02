/*
    Egaroucid Project

    @file transposition_table.hpp
        Transposition table
    @date 2021-2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
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
constexpr int TRANSPOSITION_TABLE_N_LOOP = 3;
#if TT_USE_STACK
constexpr size_t TRANSPOSITION_TABLE_STACK_SIZE = hash_sizes[DEFAULT_HASH_LEVEL] + TRANSPOSITION_TABLE_N_LOOP - 1;
#endif
constexpr int N_TRANSPOSITION_MOVES = 2;
constexpr double TT_REGISTER_THRESHOLD_RATE = 0.4;

constexpr int TRANSPOSITION_TABLE_HAS_NODE = 100;
constexpr int TRANSPOSITION_TABLE_NOT_HAS_NODE = -100;

bool transposition_table_auto_reset_importance = true;

inline uint32_t get_level_common(uint8_t depth, uint8_t mpc_level) {
    return ((uint32_t)depth << 8) | mpc_level;
}

inline uint32_t get_level_common(int depth, uint_fast8_t mpc_level) {
    return ((uint32_t)depth << 8) | mpc_level;
}

/*
    @brief Hash data

    @param date                 search date (to rewrite old entry)  more important
    @param depth                depth                                      |
    @param mpc_level            MPC level                           less important
    @param lower                lower bound
    @param upper                upper bound
    @param moves                best moves
*/
class Hash_data {
    private:
        uint8_t mpc_level;
        uint8_t depth;
        uint8_t importance;
        int8_t lower;
        int8_t upper;
        uint8_t moves[N_TRANSPOSITION_MOVES];

    public:

        //Hash_data()
        //    : lower(-SCORE_MAX), upper(SCORE_MAX), moves({MOVE_UNDEFINED, MOVE_UNDEFINED}), level({0, 0}), importance(0) {}
        
        /*
            @brief Initialize a node
        */
        inline void init() {
            lower = -SCORE_MAX;
            upper = SCORE_MAX;
            moves[0] = MOVE_UNDEFINED;
            moves[1] = MOVE_UNDEFINED;
            mpc_level = 0;
            depth = 0;
            importance = 0;
        }

        /*
            @brief Register value (same level)

            @param alpha                alpha bound
            @param beta                 beta bound
            @param value                best value
            @param policy               best move
        */
        inline void reg_same_level(const int alpha, const int beta, const int value, const int policy) {
            if (value < beta && value < upper) {
                upper = (int8_t)value;
                if (alpha < value && value < lower) {
                    lower = value;
                }
            }
            if (alpha < value && lower < value) {
                lower = (int8_t)value;
                if (value < beta && upper < value) {
                    upper = value;
                }
            }
            if ((alpha < value || value == -SCORE_MAX) && moves[0] != policy && is_valid_policy(policy)) {
                moves[1] = moves[0];
                moves[0] = (uint8_t)policy;
            }
            importance = 1;
        }

        /*
            @brief Register value (level increase)

            update best moves, reset other data

            @param d                    depth
            @param ml                   MPC level
            @param dt                   date
            @param alpha                alpha bound
            @param beta                 beta bound
            @param value                best value
            @param policy               best move
        */
        inline void reg_new_level(const int d, const uint_fast8_t ml, const int alpha, const int beta, const int value, const int policy) {
            if (value < beta) {
                upper = (int8_t)value;
            } else {
                upper = SCORE_MAX;
            }
            if (alpha < value) {
                lower = (int8_t)value;
            } else {
                lower = -SCORE_MAX;
            }
            if ((alpha < value || value == -SCORE_MAX) && moves[0] != policy && is_valid_policy(policy)) {
                moves[1] = moves[0];
                moves[0] = policy;
            }
            depth = d;
            mpc_level = ml;
            importance = 1;
        }

        /*
            @brief Register value (new board)

            update best moves, reset other data

            @param d                    depth
            @param ml                   MPC level
            @param dt                   date
            @param alpha                alpha bound
            @param beta                 beta bound
            @param value                best value
            @param policy               best move
        */
        inline void reg_new_data(const int d, const uint_fast8_t ml, const int alpha, const int beta, const int value, const int policy) {
            if (value < beta) {
                upper = (int8_t)value;
            } else {
                upper = SCORE_MAX;
            }
            if (alpha < value) {
                lower = (int8_t)value;
            } else {
                lower = -SCORE_MAX;
            }
            if ((alpha < value || value == -SCORE_MAX) && moves[0] != policy) {
                moves[0] = policy;
            } else {
                moves[0] = MOVE_UNDEFINED;
            }
            moves[1] = MOVE_UNDEFINED;
            depth = d;
            mpc_level = ml;
            importance = 1;
        }

        /*
            @brief Get level of the element

            @return level
        */
        inline uint32_t get_level() {
            if (importance) {
                return get_level_common(depth, mpc_level);
            }
            return 0;
        }

        /*
            @brief Get level of the element

            @return level
        */
        inline uint32_t get_level_no_importance() {
            return get_level_common(depth, mpc_level);
        }

        inline int get_depth() {
            return depth;
        }

        inline int get_mpc_level() {
            return mpc_level;
        }

        inline int get_window_width() {
            return upper - lower;
        }

        /*
            @brief Get moves

            @param res_moves            array to store result
        */
        inline void get_moves(uint_fast8_t res_moves[]) {
            res_moves[0] = moves[0];
            res_moves[1] = moves[1];
        }

        /*
            @brief Get bounds

            @param l                    lower bound
            @param u                    upper bound
        */
        inline void get_bounds(int *l, int *u) {
            *l = lower;
            *u = upper;
        }

        /*
            @brief Reset date
        */
        inline void set_importance_zero() {
            importance = 0;
        }

        inline uint8_t get_importance() const {
            return importance;
        }
};

struct Hash_node {
    Board board;
    Hash_data data;
    Spinlock lock;

    //Hash_node() 
    //    : board(Board{0ULL, 0ULL}) {}

    void init() {
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
void init_transposition_table(Hash_node table[], size_t s, size_t e) {
#if USE_SIMD && USE_SIMD_TT_INIT
    // UNDER CONSTRUCTION
    Hash_node HASH_NODE_INIT;
    HASH_NODE_INIT.init();
    __m256i init_data = _mm256_load_si256(((__m256i*)&HASH_NODE_INIT));
    for(size_t i = s; i < e;) {
        std::cerr << sizeof(Hash_node) << " " << ((uintptr_t)(table + i) & 0x1f) << std::endl;
        if (sizeof(Hash_node) == 32 && (((uintptr_t)(table + i) & 0x1f) == 0) && i + 1 < e) {
            std::cerr << "a";
            _mm256_stream_si256(((__m256i*)(table + i)), init_data);
            _mm_sfence();
            i += 1; // 32 byte
        } else {
            table[i].init();
            ++i;
        }
    }
#else
    for(size_t i = s; i < e; ++i) {
        table[i].init();
    }
#endif
}

/*
    @brief Resetting date in parallel

    @param table                transposition table
    @param s                    start index
    @param e                    end index
*/
void set_importance_zero_transposition_table(Hash_node table[], size_t s, size_t e) {
    for(size_t i = s; i < e; ++i) {
        table[i].data.set_importance_zero();
    }
}
/*
#if TUNE_MOVE_ORDERING_MID || TUNE_MOVE_ORDERING_END
class Transposition_table{
    public:
        inline void update_date() {
        }
        inline uint8_t get_date() {
            return 0;
        }
        inline bool resize(int hash_level) {
            return true;
        }
        inline void init() {
        }
        inline void reset_date() {
        }
        inline void reset_date_new_thread(int thread_size) {
        }
        inline void reg(const Search *search, uint32_t hash, const int depth, int alpha, int beta, int value, int policy) {
        }
        inline void get(const Search *search, uint32_t hash, const int depth, int *lower, int *upper, uint_fast8_t moves[]) {
        }
        inline void get(const Search *search, uint32_t hash, const int depth, int *lower, int *upper) {
        }
        inline int get_best_move(const Board *board, uint32_t hash) {
            return MOVE_UNDEFINED;
        }
        inline bool get_if_perfect(const Board *board, uint32_t hash, int *val, int *best_move) {
            return false;
        }
};
#else
*/
/*
    @brief Best move transposition table structure

    @param table_stack          transposition table on stack
    @param table_heap           transposition table on heap
    @param table_size           total table size
*/
class Transposition_table {
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
        Transposition_table() 
#if USE_CHANGEABLE_HASH_LEVEL || !TT_USE_STACK
            : table_heap(nullptr), table_size(0), n_registered(0), n_registered_threshold(0) {}
#else
            : table_size(0), n_registered(0), n_registered_threshold(0) {}
#endif

#if USE_CHANGEABLE_HASH_LEVEL
        /*
            @brief Resize transposition table

            @param hash_level           hash level representing the size
            @return table initialized?
        */
        inline bool resize(int hash_level) {
            size_t n_table_size = hash_sizes[hash_level] + TRANSPOSITION_TABLE_N_LOOP - 1;
            table_size = 0;
            if (table_heap != nullptr) {
                free(table_heap);
                table_heap = nullptr;
            }
            #if TT_USE_STACK
                if (n_table_size > TRANSPOSITION_TABLE_STACK_SIZE) {
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
#else // USE_CHANGEABLE_HASH_LEVEL
        inline bool set_size() {
            table_size = TRANSPOSITION_TABLE_STACK_SIZE;
            n_registered_threshold = table_size * TT_REGISTER_THRESHOLD_RATE;
            init();
            return true;
        }
#endif // USE_CHANGEABLE_HASH_LEVEL

        /*
            @brief Initialize transposition table
        */
        inline void init() {
            int thread_size = thread_pool.size();
            if (thread_size == 0) {
#if TT_USE_STACK
                for (size_t i = 0; i < std::min(table_size, (size_t)TRANSPOSITION_TABLE_STACK_SIZE); ++i) {
                    table_stack[i].init();
                }
#if USE_CHANGEABLE_HASH_LEVEL
                if (table_size > TRANSPOSITION_TABLE_STACK_SIZE) {
                    for (size_t i = 0; i < table_size - (size_t)TRANSPOSITION_TABLE_STACK_SIZE; ++i)
                        table_heap[i].init();
                }
#endif // USE_CHANGEABLE_HASH_LEVEL
#else // TT_USE_STACK
                for (size_t i = 0; i < table_size; ++i)
                    table_heap[i].init();
#endif // TT_USE_STACK
            } else {
                size_t s, e;
                std::vector<std::future<void>> tasks;
#if TT_USE_STACK
                    size_t delta = (std::min(table_size, (size_t)TRANSPOSITION_TABLE_STACK_SIZE) + thread_size - 1) / thread_size;
                    s = 0;
                    for (int i = 0; i < thread_size; ++i) {
                        e = std::min(std::min(table_size, (size_t)TRANSPOSITION_TABLE_STACK_SIZE), s + delta);
                        bool pushed = false;
                        while (!pushed) {
                            tasks.emplace_back(thread_pool.push(&pushed, std::bind(&init_transposition_table, table_stack, s, e)));
                            if (!pushed)
                                tasks.pop_back();
                        }
                        s = e;
                    }
#if USE_CHANGEABLE_HASH_LEVEL
                    if (table_size > TRANSPOSITION_TABLE_STACK_SIZE) {
                        delta = (table_size - (size_t)TRANSPOSITION_TABLE_STACK_SIZE + thread_size - 1) / thread_size;
                        s = 0;
                        for (int i = 0; i < thread_size; ++i) {
                            e = std::min(table_size - (size_t)TRANSPOSITION_TABLE_STACK_SIZE, s + delta);
                            bool pushed = false;
                            while (!pushed) {
                                tasks.emplace_back(thread_pool.push(&pushed, std::bind(&init_transposition_table, table_heap, s, e)));
                                if (!pushed)
                                    tasks.pop_back();
                            }
                            s = e;
                        }
                    }
#endif // USE_CHANGEABLE_HASH_LEVEL
#else // TT_USE_STACK
                size_t delta = (table_size + thread_size - 1) / thread_size;
                s = 0;
                for (int i = 0; i < thread_size; ++i) {
                    e = std::min(table_size, s + delta);
                    bool pushed = false;
                    while (!pushed) {
                        tasks.emplace_back(thread_pool.push(&pushed, std::bind(&init_transposition_table, table_heap, s, e)));
                        if (!pushed)
                            tasks.pop_back();
                    }
                    s = e;
                }
#endif // TT_USE_STACK
                for (std::future<void> &task: tasks) {
                    task.get();
                }
            }
            n_registered.store(0);
        }

        /*
            @brief set all date to 0
        */
        inline void reset_importance() {
            std::lock_guard lock(mtx);
            reset_importance_proc();
        }

        /*
            @brief set all date to 0

            create new thread
        */
        inline void reset_importance_new_thread(int thread_size) {
            size_t s, e;
            std::vector<std::future<void>> tasks;
#if TT_USE_STACK
            size_t delta = (std::min(table_size, (size_t)TRANSPOSITION_TABLE_STACK_SIZE) + thread_size - 1) / thread_size;
            s = 0;
            for (int i = 0; i < thread_size; ++i) {
                e = std::min(std::min(table_size, (size_t)TRANSPOSITION_TABLE_STACK_SIZE), s + delta);
                tasks.emplace_back(std::async(std::launch::async, std::bind(&set_importance_zero_transposition_table, table_stack, s, e)));
                s = e;
            }
#if USE_CHANGEABLE_HASH_LEVEL
            if (table_size > TRANSPOSITION_TABLE_STACK_SIZE) {
                delta = (table_size - (size_t)TRANSPOSITION_TABLE_STACK_SIZE + thread_size - 1) / thread_size;
                s = 0;
                for (int i = 0; i < thread_size; ++i) {
                    e = std::min(table_size - (size_t)TRANSPOSITION_TABLE_STACK_SIZE, s + delta);
                    tasks.emplace_back(std::async(std::launch::async, std::bind(&set_importance_zero_transposition_table, table_heap, s, e)));
                    s = e;
                }
            }
#endif // USE_CHANGEABLE_HASH_LEVEL
#else // TT_USE_STACK
            size_t delta = (table_size + thread_size - 1) / thread_size;
            s = 0;
            for (int i = 0; i < thread_size; ++i) {
                e = std::min(table_size, s + delta);
                tasks.emplace_back(std::async(std::launch::async, std::bind(&set_importance_zero_transposition_table, table_heap, s, e)));
                s = e;
            }
#endif // TT_USE_STACK
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
        inline void reg(const Search *search, uint32_t hash, const int depth, int alpha, int beta, int value, int policy) {
            Hash_node *node = get_node(hash);
            const uint32_t level = get_level_common(depth, search->mpc_level);
            uint32_t node_level;
#if TT_REGISTER_MIN_LEVEL
            Hash_node *min_level_node = nullptr;
            uint32_t min_level = 0x4fffffff;
            bool registered = false;
#endif
            for (uint_fast8_t i = 0; i < TRANSPOSITION_TABLE_N_LOOP; ++i) {
                if (node->data.get_level() <= level) {
                    node->lock.lock();
                        node_level = node->data.get_level();
                        if (node_level <= level) {
                            if (node->board.player == search->board.player && node->board.opponent == search->board.opponent) {
                                if (node_level == level)
                                    node->data.reg_same_level(alpha, beta, value, policy);
                                else
                                    node->data.reg_new_level(depth, search->mpc_level, alpha, beta, value, policy);
                                node->lock.unlock();
#if TT_REGISTER_MIN_LEVEL
                                registered = true;
#endif
                                break;
                            } else{
#if TT_REGISTER_MIN_LEVEL
                                if (node_level < min_level) {
                                    min_level = node_level;
                                    min_level_node = node;
                                }
#else
                                if (node->data.get_importance() == 0) {
                                    n_registered.fetch_add(1);
                                }
                                node->board.player = search->board.player;
                                node->board.opponent = search->board.opponent;
                                node->data.reg_new_data(depth, search->mpc_level, alpha, beta, value, policy);
                                node->lock.unlock();
                                //if (node_level > 0) {
                                //    n_registered.fetch_add(1);
                                //}
                                break;
#endif
                            }
                        }
                    node->lock.unlock();
                }
                ++hash;
                node = get_node(hash);
            }
#if TT_REGISTER_MIN_LEVEL
            if (!registered && min_level_node != nullptr) {
                min_level_node->lock.lock();
                    min_level_node->board.player = search->board.player;
                    min_level_node->board.opponent = search->board.opponent;
                    min_level_node->data.reg_new_data(depth, search->mpc_level, alpha, beta, value, policy);
                    if (min_level_node->data.get_level() > 0) {
                        n_registered.fetch_add(1);
                    }
                min_level_node->lock.unlock();
            }
#endif
            if (n_registered >= n_registered_threshold && transposition_table_auto_reset_importance) {
                std::lock_guard lock(mtx);
                if (n_registered >= n_registered_threshold) {
                    reset_importance_proc();
                }
            }
        }


        inline void reg_overwrite(const Search *search, uint32_t hash, const int depth, int alpha, int beta, int value, int policy) {
            Hash_node *node = get_node(hash);
            //const uint32_t level = get_level_common(depth, search->mpc_level);
            uint32_t node_level;
#if TT_REGISTER_MIN_LEVEL
            Hash_node *min_level_node = nullptr;
            uint32_t min_level = 0x4fffffff;
            bool registered = false;
#endif
            for (uint_fast8_t i = 0; i < TRANSPOSITION_TABLE_N_LOOP; ++i) {
                node->lock.lock();
                    if (node->board.player == search->board.player && node->board.opponent == search->board.opponent) {
                        node->data.reg_new_level(depth, search->mpc_level, alpha, beta, value, policy);
                        node->lock.unlock();
#if TT_REGISTER_MIN_LEVEL
                        registered = true;
#endif
                        break;
                    }
                node->lock.unlock();
                ++hash;
                node = get_node(hash);
            }
            if (n_registered >= n_registered_threshold && transposition_table_auto_reset_importance) {
                std::lock_guard lock(mtx);
                if (n_registered >= n_registered_threshold) {
                    //std::cerr << "resetting transposition importance" << std::endl;
                    reset_importance_proc();
                }
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
        inline void get(const Search *search, const uint32_t hash, const int depth, int *lower, int *upper, uint_fast8_t moves[]) {
            Hash_node *node = get_node(hash);
            const uint32_t level = get_level_common(depth, search->mpc_level);
            for (uint_fast8_t i = 0; i < TRANSPOSITION_TABLE_N_LOOP; ++i) {
                if (node->board.player == search->board.player && node->board.opponent == search->board.opponent) {
                    node->lock.lock();
                        if (node->board.player == search->board.player && node->board.opponent == search->board.opponent) {
                            node->data.get_moves(moves);
                            if (node->data.get_level_no_importance() >= level) {
                                node->data.get_bounds(lower, upper);
                            }
                            node->lock.unlock();
                            return;
                        }
                    node->lock.unlock();
                }
                node = get_node(hash + i + 1);
            }
        }

        /*
            @brief get best move from transposition table

            @param search               Search information
            @param hash                 hash code
            @param depth                depth
            @param lower                lower bound to store
            @param upper                upper bound to store
        */
        inline bool get_bounds(const Search *search, uint32_t hash, int depth, int *lower, int *upper) {
            Hash_node *node = get_node(hash);
            const uint32_t level = get_level_common(depth, search->mpc_level);
            for (uint_fast8_t i = 0; i < TRANSPOSITION_TABLE_N_LOOP; ++i) {
                if (node->board.player == search->board.player && node->board.opponent == search->board.opponent) {
                    node->lock.lock();
                        if (node->board.player == search->board.player && node->board.opponent == search->board.opponent) {
                            if (node->data.get_level_no_importance() >= level) {
                                node->data.get_bounds(lower, upper);
                                node->lock.unlock();
                                return true;
                            }
                        }
                    node->lock.unlock();
                }
                ++hash;
                node = get_node(hash);
            }
            return false;
        }

        /*
            @brief get best move from transposition table

            @param search               Search information
            @param hash                 hash code
            @param depth                depth
            @param lower                lower bound to store
            @param upper                upper bound to store
        */
        inline bool get_bounds_any_level(const Search *search, uint32_t hash, int *lower, int *upper) {
            Hash_node *node = get_node(hash);
            for (uint_fast8_t i = 0; i < TRANSPOSITION_TABLE_N_LOOP; ++i) {
                if (node->board.player == search->board.player && node->board.opponent == search->board.opponent) {
                    node->lock.lock();
                        if (node->board.player == search->board.player && node->board.opponent == search->board.opponent) {
                            node->data.get_bounds(lower, upper);
                            node->lock.unlock();
                            return true;
                        }
                    node->lock.unlock();
                }
                ++hash;
                node = get_node(hash);
            }
            return false;
        }

        /*
            @brief get best move from transposition table

            @param board                Board
            @param hash                 hash code
            @param depth                depth
            @param lower                lower bound to store
            @param upper                upper bound to store
        */
        inline bool get_bounds_any_level(const Board *board, uint32_t hash, int *lower, int *upper) {
            Hash_node *node = get_node(hash);
            for (uint_fast8_t i = 0; i < TRANSPOSITION_TABLE_N_LOOP; ++i) {
                if (node->board.player == board->player && node->board.opponent == board->opponent) {
                    node->lock.lock();
                        if (node->board.player == board->player && node->board.opponent == board->opponent) {
                            node->data.get_bounds(lower, upper);
                            node->lock.unlock();
                            return true;
                        }
                    node->lock.unlock();
                }
                ++hash;
                node = get_node(hash);
            }
            return false;
        }

        /*
            @brief get best moves from transposition table with any level

            @param board                board
            @param hash                 hash code
            @return best move
        */
        inline bool get_moves_any_level(const Board *board, uint32_t hash, uint_fast8_t moves[]) {
            Hash_node *node = get_node(hash);
            for (uint_fast8_t i = 0; i < TRANSPOSITION_TABLE_N_LOOP; ++i) {
                if (node->board.player == board->player && node->board.opponent == board->opponent) {
                    node->lock.lock();
                        if (node->board.player == board->player && node->board.opponent == board->opponent) {
                            node->data.get_moves(moves);
                            node->lock.unlock();
                            return true;
                        }
                    node->lock.unlock();
                }
                ++hash;
                node = get_node(hash);
            }
            return false;
        }

        inline void get_info(Board board, int *lower, int *upper, uint_fast8_t moves[], int *depth, uint_fast8_t *mpc_level) {
            uint32_t hash = board.hash();
            Hash_node *node = get_node(hash);
            for (uint_fast8_t i = 0; i < TRANSPOSITION_TABLE_N_LOOP; ++i) {
                if (node->board.player == board.player && node->board.opponent == board.opponent) {
                    node->lock.lock();
                        if (node->board.player == board.player && node->board.opponent == board.opponent) {
                            node->data.get_bounds(lower, upper);
                            node->data.get_moves(moves);
                            *depth = node->data.get_depth();
                            *mpc_level = node->data.get_mpc_level();
                            node->lock.unlock();
                            return;
                        }
                    node->lock.unlock();
                }
                node = get_node(hash + i + 1);
            }
        }

        inline void del(const Board *board, uint32_t hash) {
            Hash_node *node = get_node(hash);
            uint32_t node_level;
            for (uint_fast8_t i = 0; i < TRANSPOSITION_TABLE_N_LOOP; ++i) {
                node->lock.lock();
                    if (node->board.player == board->player && node->board.opponent == board->opponent) {
                        node->init();
                    }
                node->lock.unlock();
                ++hash;
                node = get_node(hash);
            }
        }

        inline bool has_node(const Search *search, uint32_t hash, int depth) {
            Hash_node *node = get_node(hash);
            const uint32_t level = get_level_common(depth, search->mpc_level);
            for (uint_fast8_t i = 0; i < TRANSPOSITION_TABLE_N_LOOP; ++i) {
                if (node->board.player == search->board.player && node->board.opponent == search->board.opponent) {
                    node->lock.lock();
                        if (node->board.player == search->board.player && node->board.opponent == search->board.opponent) {
                            if (node->data.get_level_no_importance() >= level) {
                                node->lock.unlock();
                                return true;
                            }
                        }
                    node->lock.unlock();
                }
                ++hash;
                node = get_node(hash);
            }
            return false;
        }

        inline bool has_node_any_level(const Search *search, uint32_t hash) {
            Hash_node *node = get_node(hash);
            for (uint_fast8_t i = 0; i < TRANSPOSITION_TABLE_N_LOOP; ++i) {
                if (node->board.player == search->board.player && node->board.opponent == search->board.opponent) {
                    return true;
                }
                ++hash;
                node = get_node(hash);
            }
            return false;
        }

        inline int has_node_any_level_cutoff(const Search *search, uint32_t hash, int depth, int alpha, int beta) {
            Hash_node *node = get_node(hash);
            const uint32_t level = get_level_common(depth, search->mpc_level);
            int res = TRANSPOSITION_TABLE_NOT_HAS_NODE;
            int l, u;
            for (uint_fast8_t i = 0; i < TRANSPOSITION_TABLE_N_LOOP; ++i) {
                if (node->board.player == search->board.player && node->board.opponent == search->board.opponent) {
                    node->lock.lock();
                        if (node->board.player == search->board.player && node->board.opponent == search->board.opponent) {
                            res = TRANSPOSITION_TABLE_HAS_NODE;
                            if (node->data.get_level_no_importance() >= level) {
                                node->data.get_bounds(&l, &u);
                                if (u <= alpha) {
                                    res = u;
                                } else if (beta <= l) {
                                    res = l;
                                }
                            }
                        }
                    node->lock.unlock();
                    break;
                }
                ++hash;
                node = get_node(hash);
            }
            return res;

        }

        inline bool has_node_any_level_get_bounds(const Search *search, uint32_t hash, int depth, int* l, int* u) {
            Hash_node *node = get_node(hash);
            const uint32_t level = get_level_common(depth, search->mpc_level);
            for (uint_fast8_t i = 0; i < TRANSPOSITION_TABLE_N_LOOP; ++i) {
                if (node->board.player == search->board.player && node->board.opponent == search->board.opponent) {
                    node->lock.lock();
                        if (node->board.player == search->board.player && node->board.opponent == search->board.opponent) {
                            if (node->data.get_level_no_importance() >= level) {
                                node->data.get_bounds(l, u);
                            }
                            node->lock.unlock();
                            return true;
                        }
                    node->lock.unlock();
                }
                ++hash;
                node = get_node(hash);
            }
            return false;

        }

        inline void prefetch(uint32_t hash) {
            // idea from http://www.amy.hi-ho.ne.jp/okuhara/edaxopt.htm#hashprefetch
#if USE_SIMD // only SIMD version
            Hash_node *node = get_node(hash);
            _mm_prefetch((char const *)node, _MM_HINT_T0);
            //_mm_prefetch((char const *)(node + TRANSPOSITION_TABLE_N_LOOP - 1), _MM_HINT_T0);
#endif
        }

    private:
        inline Hash_node* get_node(const uint32_t hash) {
#if TT_USE_STACK
#if USE_CHANGEABLE_HASH_LEVEL
            if (hash < TRANSPOSITION_TABLE_STACK_SIZE) {
                return &table_stack[hash];
            }
            return &table_heap[hash - TRANSPOSITION_TABLE_STACK_SIZE];
#else // USE_CHANGEABLE_HASH_LEVEL
            return &table_stack[hash];
#endif // USE_CHANGEABLE_HASH_LEVEL
#else // TT_USE_STACK
            return &table_heap[hash];
#endif // TT_USE_STACK
        }

        inline void reset_importance_proc() {
            //std::cerr << "resetting transposition importance" << std::endl;
            //std::cerr << "importance reset n_registered " << n_registered << " threshold " << n_registered_threshold << " table_size " << table_size << std::endl;
#if TT_USE_STACK
            for (size_t i = 0; i < std::min(table_size, (size_t)TRANSPOSITION_TABLE_STACK_SIZE); ++i) {
                table_stack[i].data.set_importance_zero();
            }
#if USE_CHANGEABLE_HASH_LEVEL
            if (table_size > TRANSPOSITION_TABLE_STACK_SIZE) {
                for (size_t i = 0; i < table_size - (size_t)TRANSPOSITION_TABLE_STACK_SIZE; ++i)
                    table_heap[i].data.set_importance_zero();
            }
#endif // USE_CHANGEABLE_HASH_LEVEL
#else // TT_USE_STACK
            for (size_t i = 0; i < table_size; ++i) {
                table_heap[i].data.set_importance_zero();
            }
#endif // TT_USE_STACK
            n_registered.store(0);
        }
};
//#endif

Transposition_table transposition_table;

void transposition_table_init() {
    transposition_table.init();
}

#if USE_CHANGEABLE_HASH_LEVEL
/*
    @brief Resize hash and transposition tables

    @param hash_level_failed    hash level used when failed
    @param hash_level           new hash level
    @return hash resized?
*/
bool hash_resize(int hash_level_failed, int hash_level, bool show_log) {
    if (!transposition_table.resize(hash_level)) {
        std::cerr << "[ERROR] hash table resize failed. resize to level " << hash_level_failed << std::endl;
        transposition_table.resize(hash_level_failed);
        if (!hash_init(hash_level_failed)) {
            std::cerr << "[ERROR] can't get hash. you can ignore this error" << std::endl;
            hash_init_rand(hash_level_failed);
        }
        global_hash_level = hash_level_failed;
#if USE_CRC32C_HASH
        global_hash_bit_mask = (1U << global_hash_level) - 1;
#endif
        return false;
    }
    if (!hash_init(hash_level)) {
        std::cerr << "[ERROR] can't get hash. you can ignore this error" << std::endl;
        hash_init_rand(hash_level);
    }
    global_hash_level = hash_level;
#if USE_CRC32C_HASH
    global_hash_bit_mask = (1U << global_hash_level) - 1;
#endif
    if (show_log) {
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
bool hash_resize(int hash_level_failed, int hash_level, std::string binary_path, bool show_log) {
    if (!transposition_table.resize(hash_level)) {
        std::cerr << "[ERROR] hash table resize failed. resize to level " << hash_level_failed << std::endl;
        transposition_table.resize(hash_level_failed);
        if (!hash_init(hash_level_failed, binary_path)) {
            std::cerr << "[ERROR] can't get hash. you can ignore this error" << std::endl;
            hash_init_rand(hash_level_failed);
        }
        global_hash_level = hash_level_failed;
#if USE_CRC32C_HASH
        global_hash_bit_mask = (1U << global_hash_level) - 1;
#endif
        return false;
    }
    if (!hash_init(hash_level, binary_path)) {
        std::cerr << "[ERROR] can't get hash. you can ignore this error" << std::endl;
        hash_init_rand(hash_level);
    }
    global_hash_level = hash_level;
#if USE_CRC32C_HASH
    global_hash_bit_mask = (1U << global_hash_level) - 1;
#endif
    if (show_log) {
        double size_mb = (double)sizeof(Hash_node) / 1024 / 1024 * hash_sizes[hash_level];
        std::cerr << "hash resized to level " << hash_level << " elements " << hash_sizes[hash_level] << " size " << size_mb << " MB" << std::endl;
    }
    return true;
}
#else
bool hash_tt_init(std::string binary_path, bool show_log) {
    transposition_table.set_size();
    if (!hash_init(DEFAULT_HASH_LEVEL, binary_path)) {
        std::cerr << "[ERROR] can't get hash. you can ignore this error" << std::endl;
        hash_init_rand(DEFAULT_HASH_LEVEL);
    }
    global_hash_level = DEFAULT_HASH_LEVEL;
#if USE_CRC32C_HASH
    global_hash_bit_mask = (1U << DEFAULT_HASH_LEVEL) - 1;
#endif
    return true;
}

bool hash_tt_init(bool show_log) {
    transposition_table.set_size();
    if (!hash_init(DEFAULT_HASH_LEVEL)) {
        std::cerr << "[ERROR] can't get hash. you can ignore this error" << std::endl;
        hash_init_rand(DEFAULT_HASH_LEVEL);
    }
    global_hash_level = DEFAULT_HASH_LEVEL;
#if USE_CRC32C_HASH
    global_hash_bit_mask = (1U << DEFAULT_HASH_LEVEL) - 1;
#endif
    return true;
}
#endif

void delete_tt(Board *board, int depth) {
    transposition_table.del(board, board->hash());
    if (depth == 0) {
        return;
    }
    uint64_t legal = board->get_legal();
    if (legal == 0) {
        board->pass();
            if (board->get_legal()) {
                delete_tt(board, depth);
            }
        board->pass();
        return;
    }
    Flip flip;
    for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)) {
        calc_flip(&flip, board, cell);
        board->move_board(&flip);
            delete_tt(board, depth - 1);
        board->undo_board(&flip);
    }
}