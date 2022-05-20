#pragma once
#include "setting.hpp"
#include "common.hpp"
#include "board.hpp"
#if USE_MULTI_THREAD
    #include "thread_pool.hpp"
    #include <future>
#endif
#include <mutex>

using namespace std;

#define TRANSPOSE_TABLE_SIZE 33554432
#define TRANSPOSE_TABLE_MASK 33554431

#define TRANSPOSE_TABLE_UNDEFINED -INF

class Node_child_transpose_table{
    private:
        mutex mtx;
        uint64_t player;
        uint64_t opponent;
        int best_move;
        //atomic<Node_child_transpose_table*> p_n_node;

    public:

        inline void init(){
            free(this);
        }

        inline bool register_value(const Board *board, const int policy){
            lock_guard<mutex> lock(mtx);
            if (board->player == player && board->opponent == opponent){
                best_move = policy;
                return true;
            }
            return false;
        }

        inline void register_value_with_board(const Board *board, const int policy){
            lock_guard<mutex> lock(mtx);
            player = board->player;
            opponent = board->opponent;
            best_move = policy;
        }

        inline void register_value_with_board(Node_child_transpose_table *from){
            lock_guard<mutex> lock(mtx);
            player = from->player;
            opponent = from->opponent;
            best_move = from->best_move;
        }

        inline int get(const Board *board){
            lock_guard<mutex> lock(mtx);
            if (board->player == player && board->opponent == opponent)
                return best_move;
            return TRANSPOSE_TABLE_UNDEFINED;
        }
};

#if USE_MULTI_THREAD
    void init_child_transpose_table(int id, Node_child_transpose_table *table[], int s, int e){
        for(int i = s; i < e; ++i){
            if (table[i] != NULL){
                table[i]->init();
                table[i] = NULL;
            }
        }
    }

    void copy_child_transpose_table(int id, Node_child_transpose_table *from[], Node_child_transpose_table *to[], int s, int e){
        for(int i = s; i < e; ++i){
            if (from[i] != NULL){
                if (to[i] == NULL){
                    Node_child_transpose_table *p = (Node_child_transpose_table*)malloc(sizeof(Node_child_transpose_table));
                    to[i] = new(p) Node_child_transpose_table;
                }
                to[i]->register_value_with_board(from[i]);
            } else{
                if (to[i] != NULL)
                    to[i]->init();
                to[i] = NULL;
            }
        }
    }
#endif

class Child_transpose_table{
    private:
        Node_child_transpose_table *table[TRANSPOSE_TABLE_SIZE];

    public:
        inline void first_init(){
            for(int i = 0; i < TRANSPOSE_TABLE_SIZE; ++i)
                table[i] = NULL;
        }

        #if USE_MULTI_THREAD
            inline void init(){
                int thread_size = thread_pool.size();
                int delta = (TRANSPOSE_TABLE_SIZE + thread_size - 1) / thread_size;
                int s = 0, e;
                vector<future<void>> tasks;
                for (int i = 0; i < thread_size; ++i){
                    e = min(TRANSPOSE_TABLE_SIZE, s + delta);
                    tasks.emplace_back(thread_pool.push(init_child_transpose_table, table, s, e));
                    s = e;
                }
                for (future<void> &task: tasks)
                    task.get();
            }
        #else
            inline void init(){
                for(int i = 0; i < TRANSPOSE_TABLE_SIZE; ++i){
                    if (table[i] != NULL){
                        table[i]->init();
                        table[i] = NULL;
                    }
                }
            }
        #endif

        inline void reg(const Board *board, const uint32_t hash, const int policy){
            if (table[hash] == NULL){
                Node_child_transpose_table *p = (Node_child_transpose_table*)malloc(sizeof(Node_child_transpose_table));
                table[hash] = new(p) Node_child_transpose_table;
            }
            table[hash]->register_value_with_board(board, policy);
        }

        inline int get(const Board *board, const uint32_t hash) const{
            if (table[hash] != NULL)
                return table[hash]->get(board);
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
                    tasks.emplace_back(thread_pool.push(copy_child_transpose_table, table, to->table, s, e));
                    s = e;
                }
                for (future<void> &task: tasks)
                    task.get();
            }
        #else
            inline void copy(Child_transpose_table *to){
                to->init();
                for(int i = 0; i < TRANSPOSE_TABLE_SIZE; ++i){
                    if (table[i] != NULL){
                        to->table[i] = (Node_child_transpose_table*)malloc(sizeof(Node_child_transpose_table));
                        to->table[i]->register_value_with_board(table[i]);
                    } else{
                        to->table[i] = NULL;
                    }
                }
            }
        #endif
};


class Node_parent_transpose_table{
    private:
        mutex mtx;
        uint64_t player;
        uint64_t opponent;
        int lower;
        int upper;
        //atomic<Node_child_transpose_table*> p_n_node;

    public:

        inline void init(){
            free(this);
        }

        inline void register_value_with_board(const Board *board, const int l, const int u){
            lock_guard<mutex> lock(mtx);
            player = board->player;
            opponent = board->opponent;
            lower = l;
            upper = u;
        }

        inline bool register_value(const Board *board, const int l, const int u){
            lock_guard<mutex> lock(mtx);
            if (board->player == player && board->opponent == opponent){
                lower = l;
                upper = u;
                return true;
            }
            return false;
        }

        inline void register_value_with_board(Node_parent_transpose_table *from){
            lock_guard<mutex> lock(mtx);
            player = from->player;
            opponent = from->opponent;
            lower = from->lower;
            upper = from->upper;
        }

        inline void get(const Board *board, int *l, int *u){
            lock_guard<mutex> lock(mtx);
            if (board->player == player && board->opponent == opponent){
                *l = lower;
                *u = upper;
            }
        }
};

#if USE_MULTI_THREAD
    void init_parent_transpose_table(int id, Node_parent_transpose_table *table[], int s, int e){
        for(int i = s; i < e; ++i){
            if (table[i] != NULL){
                table[i]->init();
                table[i] = NULL;
            }
        }
    }

    void copy_parent_transpose_table(int id, Node_parent_transpose_table *from[], Node_parent_transpose_table *to[], int s, int e){
        for(int i = s; i < e; ++i){
            if (from[i] != NULL){
                if (to[i] == NULL){
                    Node_parent_transpose_table *p = (Node_parent_transpose_table*)malloc(sizeof(Node_parent_transpose_table));
                    to[i] = new(p) Node_parent_transpose_table;
                }
                to[i]->register_value_with_board(from[i]);
            } else{
                if (to[i] != NULL)
                    to[i]->init();
                to[i] = NULL;
            }
        }
    }
#endif

class Parent_transpose_table{
    private:
        Node_parent_transpose_table *table[TRANSPOSE_TABLE_SIZE];

    public:
        inline void first_init(){
            for(int i = 0; i < TRANSPOSE_TABLE_SIZE; ++i)
                table[i] = NULL;
        }

        #if USE_MULTI_THREAD
            inline void init(){
                int thread_size = thread_pool.size();
                int delta = (TRANSPOSE_TABLE_SIZE + thread_size - 1) / thread_size;
                int s = 0, e;
                vector<future<void>> tasks;
                for (int i = 0; i < thread_size; ++i){
                    e = min(TRANSPOSE_TABLE_SIZE, s + delta);
                    tasks.emplace_back(thread_pool.push(init_parent_transpose_table, table, s, e));
                    s = e;
                }
                for (future<void> &task: tasks)
                    task.get();
            }
        #else
            inline void init(){
                for(int i = 0; i < TRANSPOSE_TABLE_SIZE; ++i){
                    if (table[i] != NULL){
                        table[i]->init();
                        table[i] = NULL;
                    }
                }
            }
        #endif

        inline void reg(const Board *board, const uint32_t hash, const int l, const int u){
            if (table[hash] == NULL){
                Node_parent_transpose_table *p = (Node_parent_transpose_table*)malloc(sizeof(Node_parent_transpose_table));
                table[hash] = new(p) Node_parent_transpose_table;
            }
            table[hash]->register_value_with_board(board, l, u);
        }

        inline void get(const Board *board, const uint32_t hash, int *l, int *u) const{
            *l = -INF;
            *u = INF;
            if (table[hash] != NULL)
                table[hash]->get(board, l, u);
        }

        #if USE_MULTI_THREAD
            inline void copy(Parent_transpose_table *to){
                int thread_size = thread_pool.size();
                int delta = (TRANSPOSE_TABLE_SIZE + thread_size - 1) / thread_size;
                int s = 0, e;
                vector<future<void>> tasks;
                for (int i = 0; i < thread_size; ++i){
                    e = min(TRANSPOSE_TABLE_SIZE, s + delta);
                    tasks.emplace_back(thread_pool.push(copy_parent_transpose_table, table, to->table, s, e));
                    s = e;
                }
                for (future<void> &task: tasks)
                    task.get();
            }
        #else
            inline void copy(Parent_transpose_table *to){
                for(int i = 0; i < TRANSPOSE_TABLE_SIZE; ++i){
                    if (table[i] != NULL){
                        if (to->table[i] == NULL)
                            to->table[i] = (Node_parent_transpose_table*)malloc(sizeof(Node_parent_transpose_table));
                        to->table[i]->register_value_with_board(table[i]);
                    } else{
                        if (to->table[i] != NULL)
                            to->table[i]->init();
                        to->table[i] = NULL;
                    }
                }
            }
        #endif
};

Parent_transpose_table parent_transpose_table;
Parent_transpose_table bak_parent_transpose_table;
Child_transpose_table child_transpose_table;
Child_transpose_table bak_child_transpose_table;
