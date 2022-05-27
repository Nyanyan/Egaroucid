#pragma once
#include "setting.hpp"
#include "common.hpp"
#include "board.hpp"
#if USE_MULTI_THREAD
    #include "thread_pool.hpp"
    #include <future>
#endif
#include <shared_mutex>

using namespace std;

#define TRANSPOSE_TABLE_SIZE 2097152
#define TRANSPOSE_TABLE_MASK 2097151

#define CACHE_SAVE_EMPTY 15

#define TRANSPOSE_TABLE_UNDEFINED -INF

class Node_child_transpose_table{
    private:
        uint64_t player;
        uint64_t opponent;
        int best_move;

    public:

        inline void init(){
            free(this);
        }

        inline bool register_value(const Board *board, const int policy){
            if (board->player == player && board->opponent == opponent){
                best_move = policy;
                return true;
            }
            return false;
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
            if (board->player == player && board->opponent == opponent)
                return best_move;
            return TRANSPOSE_TABLE_UNDEFINED;
        }

        inline int n_stones(){
            return pop_count_ull(player | opponent);
        }
};

class Node_shared_mutex_child_transpose_table{
    private:
        shared_mutex mtx;
    public:
        Node_child_transpose_table *node;
    
    public:

        inline bool register_value(const Board *board, const int policy){
            //lock_guard<shared_mutex> lock(mtx);
            return node->register_value(board, policy);
        }

        inline void register_value_with_board(const Board *board, const int policy){
            //lock_guard<shared_mutex> lock(mtx);
            node->register_value_with_board(board, policy);
        }

        inline void register_value_with_board(Node_child_transpose_table *from){
            //lock_guard<shared_mutex> lock(mtx);
            node->register_value_with_board(from);
        }

        inline int get(const Board *board){
            //lock_guard<shared_mutex> lock(mtx);
            return node->get(board);
            /*
            int res = TRANSPOSE_TABLE_UNDEFINED;
            {
                shared_lock<shared_mutex> lock(mtx);
                res = node->get(board);
            }
            return res;
            */
        }

        inline int n_stones() const{
            return node->n_stones();
        }

        inline bool is_null() const{
            return node == NULL;
        }

        inline void set(){
            lock_guard<shared_mutex> lock(mtx);
            Node_child_transpose_table* p = (Node_child_transpose_table*)malloc(sizeof(Node_child_transpose_table));
            if (node == NULL)
                node = p;
            else
                free(p);
        }

        inline void set_null(){
            node = NULL;
        }

        inline void init(){
            node->init();
            node = NULL;
        }
};

#if USE_MULTI_THREAD
    void init_child_transpose_table(Node_shared_mutex_child_transpose_table table[], int s, int e){
        for(int i = s; i < e; ++i){
            if (!table[i].is_null())
                table[i].init();
        }
    }

    void copy_child_transpose_table(Node_shared_mutex_child_transpose_table from[], Node_shared_mutex_child_transpose_table to[], int s, int e){
        for(int i = s; i < e; ++i){
            if (!from[i].is_null()){
                if (from[i].n_stones() < HW2 - CACHE_SAVE_EMPTY){
                    if (to[i].is_null())
                        to[i].set();
                    to[i].register_value_with_board(from[i].node);
                }
            } else{
                if (!to[i].is_null())
                    to[i].init();
            }
        }
    }
#endif

class Child_transpose_table{
    private:
        Node_shared_mutex_child_transpose_table table[TRANSPOSE_TABLE_SIZE];

    public:
        inline void first_init(){
            for(int i = 0; i < TRANSPOSE_TABLE_SIZE; ++i)
                table[i].set_null();
        }

        #if USE_MULTI_THREAD
            inline void init(){
                int thread_size = thread_pool.size();
                int delta = (TRANSPOSE_TABLE_SIZE + thread_size - 1) / thread_size;
                int s = 0, e;
                vector<future<void>> tasks;
                for (int i = 0; i < thread_size; ++i){
                    e = min(TRANSPOSE_TABLE_SIZE, s + delta);
                    tasks.emplace_back(thread_pool.push(bind(&init_child_transpose_table, table, s, e)));
                    s = e;
                }
                for (future<void> &task: tasks)
                    task.get();
            }
        #else
            inline void init(){
                for(int i = 0; i < TRANSPOSE_TABLE_SIZE; ++i){
                    if (!table[i].is_null()){
                        table[i].init();
                        table[i] = NULL;
                    }
                }
            }
        #endif

        inline void reg(const Board *board, const uint32_t hash, const int policy){
            if (table[hash].is_null())
                table[hash].set();
            table[hash].register_value_with_board(board, policy);
        }

        inline int get(const Board *board, const uint32_t hash){
            if (!table[hash].is_null())
                return table[hash].get(board);
            return TRANSPOSE_TABLE_UNDEFINED;
        }

        #if USE_MULTI_THREAD
            inline void copy(Child_transpose_table *to){
                int thread_size = thread_pool.size();
                int delta = (TRANSPOSE_TABLE_SIZE + thread_size - 1) / thread_size;
                int s = 0, e;
                vector<future<void>> tasks;
                for (int i = 0; i < thread_size; ++i){
                    e = min(TRANSPOSE_TABLE_SIZE, s + delta);
                    tasks.emplace_back(thread_pool.push(bind(&copy_child_transpose_table, table, to->table, s, e)));
                    s = e;
                }
                for (future<void> &task: tasks)
                    task.get();
            }
        #else
            inline void copy(Child_transpose_table *to){
                to->init();
                for(int i = 0; i < TRANSPOSE_TABLE_SIZE; ++i){
                    if (!table[i].is_null()){
                        to->table[i].set();
                        to->table[i]->register_value_with_board(table[i]);
                    } else{
                        to->table[i].set_null();
                    }
                }
            }
        #endif
};


class Node_parent_transpose_table{
    private:
        uint64_t player;
        uint64_t opponent;
        int lower;
        int upper;

    public:

        inline void init(){
            free(this);
        }

        inline void register_value_with_board(const Board *board, const int l, const int u){
            player = board->player;
            opponent = board->opponent;
            lower = l;
            upper = u;
        }

        inline bool register_value(const Board *board, const int l, const int u){
            if (board->player == player && board->opponent == opponent){
                lower = l;
                upper = u;
                return true;
            }
            return false;
        }

        inline void register_value_with_board(Node_parent_transpose_table *from){
            player = from->player;
            opponent = from->opponent;
            lower = from->lower;
            upper = from->upper;
        }

        inline void get(const Board *board, int *l, int *u){
            if (board->player == player && board->opponent == opponent){
                *l = lower;
                *u = upper;
            }
        }

        inline int n_stones() const{
            return pop_count_ull(player | opponent);
        }
};

class Node_shared_mutex_parent_transpose_table{
    private:
        shared_mutex mtx;
    public:
        Node_parent_transpose_table *node;
    
    public:
        inline void register_value_with_board(const Board *board, const int l, const int u){
            lock_guard<shared_mutex> lock(mtx);
            node->register_value_with_board(board, l, u);
        }

        inline bool register_value(const Board *board, const int l, const int u){
            lock_guard<shared_mutex> lock(mtx);
            return node->register_value(board, l, u);
        }

        inline void register_value_with_board(Node_parent_transpose_table *from){
            lock_guard<shared_mutex> lock(mtx);
            node->register_value_with_board(from);
        }

        inline void get(const Board *board, int *l, int *u){
            //lock_guard<shared_mutex> lock(mtx);
            {
                shared_lock<shared_mutex> lock(mtx);
                node->get(board, l, u);
            }
        }

        inline int n_stones() const{
            return node->n_stones();
        }

        inline bool is_null() const{
            return node == NULL;
        }

        inline void set(){
            lock_guard<shared_mutex> lock(mtx);
            node = (Node_parent_transpose_table*)malloc(sizeof(Node_parent_transpose_table));
        }

        inline void set_null(){
            node = NULL;
        }

        inline void init(){
            node->init();
            node = NULL;
        }
};

#if USE_MULTI_THREAD
    void init_parent_transpose_table(Node_shared_mutex_parent_transpose_table table[], int s, int e){
        for(int i = s; i < e; ++i){
            if (!table[i].is_null())
                table[i].init();
        }
    }

    void copy_parent_transpose_table(Node_shared_mutex_parent_transpose_table from[], Node_shared_mutex_parent_transpose_table to[], int s, int e){
        for(int i = s; i < e; ++i){
            if (!from[i].is_null()){
                if (from[i].n_stones() < HW2 - CACHE_SAVE_EMPTY){
                    if (to[i].is_null())
                        to[i].set();
                    to[i].register_value_with_board(from[i].node);
                }
            } else{
                if (!to[i].is_null())
                    to[i].init();
            }
        }
    }
#endif

class Parent_transpose_table{
    private:
        Node_shared_mutex_parent_transpose_table table[TRANSPOSE_TABLE_SIZE];

    public:
        inline void first_init(){
            for(int i = 0; i < TRANSPOSE_TABLE_SIZE; ++i)
                table[i].set_null();
        }

        #if USE_MULTI_THREAD
            inline void init(){
                int thread_size = thread_pool.size();
                int delta = (TRANSPOSE_TABLE_SIZE + thread_size - 1) / thread_size;
                int s = 0, e;
                vector<future<void>> tasks;
                for (int i = 0; i < thread_size; ++i){
                    e = min(TRANSPOSE_TABLE_SIZE, s + delta);
                    tasks.emplace_back(thread_pool.push(bind(&init_parent_transpose_table, table, s, e)));
                    s = e;
                }
                for (future<void> &task: tasks)
                    task.get();
            }
        #else
            inline void init(){
                for(int i = 0; i < TRANSPOSE_TABLE_SIZE; ++i){
                    if (!table[i].is_null())
                        table[i].init();
                }
            }
        #endif

        inline void reg(const Board *board, const uint32_t hash, const int l, const int u){
            if (table[hash].is_null())
                table[hash].set();
            table[hash].register_value_with_board(board, l, u);
        }

        inline void get(const Board *board, const uint32_t hash, int *l, int *u){
            *l = -INF;
            *u = INF;
            if (!table[hash].is_null())
                table[hash].get(board, l, u);
        }

        #if USE_MULTI_THREAD
            inline void copy(Parent_transpose_table *to){
                int thread_size = thread_pool.size();
                int delta = (TRANSPOSE_TABLE_SIZE + thread_size - 1) / thread_size;
                int s = 0, e;
                vector<future<void>> tasks;
                for (int i = 0; i < thread_size; ++i){
                    e = min(TRANSPOSE_TABLE_SIZE, s + delta);
                    tasks.emplace_back(thread_pool.push(bind(&copy_parent_transpose_table, table, to->table, s, e)));
                    s = e;
                }
                for (future<void> &task: tasks)
                    task.get();
            }
        #else
            inline void copy(Parent_transpose_table *to){
                for(int i = 0; i < TRANSPOSE_TABLE_SIZE; ++i){
                    if (!table[i].is_null()){
                        if (to->table[i].is_null())
                            to->table[i].set();
                        to->table[i].register_value_with_board(table[i]);
                    } else{
                        if (!to->table[i].is_null())
                            to->table[i].init();
                    }
                }
            }
        #endif
};

Parent_transpose_table parent_transpose_table;
Parent_transpose_table bak_parent_transpose_table;
Child_transpose_table child_transpose_table;
Child_transpose_table bak_child_transpose_table;
