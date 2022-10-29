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
#include "search.hpp"
#include <future>
#include <mutex>

using namespace std;

#define TRANSPOSITION_TABLE_SIZE 4194304 // 8388608 // 16777216
#define TRANSPOSITION_TABLE_MASK 4194303 // 8388607 // 16777215

#define CACHE_SAVE_EMPTY 10

#define TRANSPOSITION_TABLE_UNDEFINED -INF

#define DEPTH_INIT_VALUE -1

inline double data_strength(const double t, const int d){
    return t + 4.0 * d;
}

inline bool data_importance(const int n_discs_old, const int n_discs_new, const int first_n_discs){
    if (n_discs_old == DEPTH_INIT_VALUE)
        return true;
    if (n_discs_old < first_n_discs)
        return true;
    return n_discs_new <= n_discs_old;
}

inline bool data_reusable(const int n_discs_old, const int n_discs_new){
    if (n_discs_old == DEPTH_INIT_VALUE)
        return true;
    return n_discs_old <= n_discs_new;
}

class Node_policy{
    private:
        uint64_t player;
        uint64_t opponent;
        int n_discs;
        int best_move;

    public:
        inline void init(){
            player = 0ULL;
            opponent = 0ULL;
            n_discs = DEPTH_INIT_VALUE;
            best_move = 0;
        }

        inline void reg(const Search *search, const int policy){
            player = search->board.player;
            opponent = search->board.opponent;
            n_discs = search->n_discs;
            best_move = policy;
        }

        inline void reg_important(const Search *search, const int f, const int policy){
            if (data_importance(n_discs, search->n_discs, f)){
                player = search->board.player;
                opponent = search->board.opponent;
                n_discs = search->n_discs;
                best_move = policy;
            }
        }

        inline void reg_reusable(const Search *search, const int policy){
            if (data_reusable(n_discs, search->n_discs)){
                player = search->board.player;
                opponent = search->board.opponent;
                n_discs = search->n_discs;
                best_move = policy;
            }
        }

        inline bool get(const Search *search, int *policy){
            if (search->board.player == player || search->board.opponent == opponent){
                *policy = best_move;
                return true;
            }
            return false;
        }
};

class Node_value{
    private:
        uint64_t player;
        uint64_t opponent;
        int n_discs;
        double mpct;
        int depth;
        int lower_bound;
        int upper_bound;

    public:
        inline void init(){
            player = 0ULL;
            opponent = 0ULL;
            n_discs = DEPTH_INIT_VALUE;
            mpct = 0.0;
            depth = 0;
            lower_bound = -INF;
            upper_bound = INF;
        }

        inline void reg(const Search *search, const int d, const int l, const int u){
            player = search->board.player;
            opponent = search->board.opponent;
            n_discs = search->n_discs;
            mpct = search->mpct;
            depth = d;
            lower_bound = l;
            upper_bound = u;
        }

        inline void reg_important(const Search *search, const int d, const int f, const int l, const int u){
            if (data_importance(n_discs, search->n_discs, f)){
                player = search->board.player;
                opponent = search->board.opponent;
                n_discs = search->n_discs;
                mpct = search->mpct;
                depth = d;
                lower_bound = l;
                upper_bound = u;
            }
        }

        inline void reg_reusable(const Search *search, const int d, const int l, const int u){
            if (data_reusable(n_discs, search->n_discs)){
                player = search->board.player;
                opponent = search->board.opponent;
                n_discs = search->n_discs;
                mpct = search->mpct;
                depth = d;
                lower_bound = l;
                upper_bound = u;
            }
        }

        inline bool get(const Search *search, const int d, int *l, int *u){
            if (search->board.player == player || search->board.opponent == opponent){
                if (data_strength(mpct, depth) >= data_strength(search->mpct, d)){
                    *l = lower_bound;
                    *u = upper_bound;
                    return true;
                }
            }
            return false;
        }
};

class Node_transposition_table{
    private:
        mutex mtx;
        Node_policy policy_new;
        Node_value value_new;
        Node_policy policy_important;
        Node_value value_important;
        Node_policy policy_reusable;
        Node_value value_reusable;

    public:
        inline void init(){
            policy_new.init();
            value_new.init();
            policy_important.init();
            value_important.init();
            policy_reusable.init();
            value_reusable.init();
        }

        inline void reg(const Search *search, const int depth, const int policy, const int lower_bound, const int upper_bound){
            lock_guard<mutex> lock(mtx);
            policy_new.reg(search, policy);
            value_new.reg(search, depth, lower_bound, upper_bound);
            const int first_n_discs = search->n_discs - (search->first_depth - depth);
            policy_important.reg_important(search, first_n_discs, policy);
            value_important.reg_important(search, depth, first_n_discs, lower_bound, upper_bound);
            policy_reusable.reg_reusable(search, policy);
            value_reusable.reg_reusable(search, depth, lower_bound, upper_bound);
        }

        inline void reg_policy(const Search *search, const int depth, const int policy){
            lock_guard<mutex> lock(mtx);
            policy_new.reg(search, policy);
            const int first_n_discs = search->n_discs - (search->first_depth - depth);
            policy_important.reg_important(search, first_n_discs, policy);
            policy_reusable.reg_reusable(search, policy);
        }

        inline void reg_value(const Search *search, const int depth, const int lower_bound, const int upper_bound){
            lock_guard<mutex> lock(mtx);
            value_new.reg(search, depth, lower_bound, upper_bound);
            const int first_n_discs = search->n_discs - (search->first_depth - depth);
            value_important.reg_important(search, depth, first_n_discs, lower_bound, upper_bound);
            value_reusable.reg_reusable(search, depth, lower_bound, upper_bound);
        }

        inline void get(const Search *search, const int depth, int *best_move, int *lower_bound, int *upper_bound){
            lock_guard<mutex> lock(mtx);
            *best_move = TRANSPOSITION_TABLE_UNDEFINED;
            *lower_bound = -INF;
            *upper_bound = INF;
            if (!policy_new.get(search, best_move)){
                if (!policy_reusable.get(search, best_move))
                    policy_important.get(search, best_move);
            }
            if (!value_new.get(search, depth, lower_bound, upper_bound)){
                if (value_reusable.get(search, depth, lower_bound, upper_bound))
                    value_important.get(search, depth, lower_bound, upper_bound);
            }
        }

        inline void get_policy(const Search *search, const int depth, int *best_move){
            lock_guard<mutex> lock(mtx);
            *best_move = TRANSPOSITION_TABLE_UNDEFINED;
            if (!policy_new.get(search, best_move)){
                if (!policy_reusable.get(search, best_move))
                    policy_important.get(search, best_move);
            }
        }

        inline void get_value(const Search *search, const int depth, int *lower_bound, int *upper_bound){
            lock_guard<mutex> lock(mtx);
            *lower_bound = -INF;
            *upper_bound = INF;
            if (!value_new.get(search, depth, lower_bound, upper_bound)){
                if (value_reusable.get(search, depth, lower_bound, upper_bound))
                    value_important.get(search, depth, lower_bound, upper_bound);
            }
        }
};


void init_transposition_table(Node_transposition_table table[], int s, int e){
    for(int i = s; i < e; ++i){
        table[i].init();
    }
}

class Transposition_table{
    private:
        Node_transposition_table table[TRANSPOSITION_TABLE_SIZE];

    public:
        inline void first_init(){
            init();
        }

        inline void init(){
            int thread_size = thread_pool.size();
            if (thread_size >= 2){
                int delta = (TRANSPOSITION_TABLE_SIZE + thread_size - 1) / thread_size;
                int s = 0, e;
                vector<future<void>> tasks;
                for (int i = 0; i < thread_size; ++i){
                    e = min(TRANSPOSITION_TABLE_SIZE, s + delta);
                    tasks.emplace_back(thread_pool.push(bind(&init_transposition_table, table, s, e)));
                    s = e;
                }
                for (future<void> &task: tasks)
                    task.get();
            } else{
                for (int i = 0; i < TRANSPOSITION_TABLE_SIZE; ++i)
                    table[i].init();
            }
        }

        inline void reg(const Search *search, const int depth, const uint32_t hash_code, const int policy, const int lower_bound, const int upper_bound){
            table[hash_code].reg(search, depth, policy, lower_bound, upper_bound);
        }

        inline void reg_policy(const Search *search, const int depth, const uint32_t hash_code, const int policy){
            table[hash_code].reg_policy(search, depth, policy);
        }

        inline void reg_value(const Search *search, const int depth, const uint32_t hash_code, const int lower_bound, const int upper_bound){
            table[hash_code].reg_value(search, depth, lower_bound, upper_bound);
        }

        inline void get(const Search *search, const int depth, const uint32_t hash_code, int *best_move, int *lower_bound, int *upper_bound){
            return table[hash_code].get(search, depth, best_move, lower_bound, upper_bound);
        }

        inline void get_policy(const Search *search, const int depth, const uint32_t hash_code, int *best_move){
            return table[hash_code].get_policy(search, depth, best_move);
        }

        inline void get_value(const Search *search, const int depth, const uint32_t hash_code, int *lower_bound, int *upper_bound){
            return table[hash_code].get_value(search, depth, lower_bound, upper_bound);
        }
};

Transposition_table transposition_table;

inline void register_tt(const Search *search, const int depth, const uint32_t hash_code, int first_alpha, int v, int best_move, int l, int u, int alpha, int beta, const bool *searching){
    if (search->n_discs <= HW2 - USE_TT_DEPTH_THRESHOLD && (*searching) && -HW2 <= v && v <= HW2 && global_searching){
        int lower_bound = l, upper_bound = u, policy = TRANSPOSITION_TABLE_UNDEFINED;
        bool policy_reg = true, value_reg = true;
        if (first_alpha < v && best_move != TRANSPOSITION_TABLE_UNDEFINED)
            policy = best_move;
        else
            policy_reg = false;
        if (first_alpha < v && v < beta){
            lower_bound = v;
            upper_bound = v;
        } else if (beta <= v && l < v)
            lower_bound = v;
        else if (v <= alpha && v < u)
            upper_bound = v;
        else
            value_reg = false;
        if (policy_reg && value_reg)
            transposition_table.reg(search, depth, hash_code, policy, lower_bound, upper_bound);
        else if (policy_reg)
            transposition_table.reg_policy(search, depth, hash_code, policy);
        else if (value_reg)
            transposition_table.reg_value(search, depth, hash_code, lower_bound, upper_bound);
    }
}

inline void register_tt_policy(const Search *search, const int depth, const uint32_t hash_code, int first_alpha, int v, int best_move, const bool *searching){
    if (search->n_discs <= HW2 - USE_TT_DEPTH_THRESHOLD && (*searching) && -HW2 <= v && v <= HW2 && global_searching){
        if (first_alpha < v && best_move != TRANSPOSITION_TABLE_UNDEFINED)
            transposition_table.reg_policy(search, depth, hash_code, best_move);
    }
}
