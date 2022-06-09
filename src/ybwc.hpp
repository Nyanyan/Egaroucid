#pragma once
#include <iostream>
#include "setting.hpp"
#include "common.hpp"
#include "transpose_table.hpp"
#include "search.hpp"
#if USE_CUDA
    #include "cuda_midsearch.hpp"
    #include "cuda_endsearch.hpp"
#else
    #include "midsearch.hpp"
    #include "endsearch.hpp"
#endif
#include "thread_pool.hpp"

//#define YBWC_SPLIT_DIV 7
#define YBWC_MID_SPLIT_MIN_DEPTH 6
//#define YBWC_MID_SPLIT_MAX_DEPTH 20
#define YBWC_END_SPLIT_MIN_DEPTH 10
//#define YBWC_MAX_SPLIT_COUNT 3
//#define YBWC_PC_OFFSET 3
#define YBWC_ORDERING_MAX_OFFSET 8
#define YBWC_OFFSET_DIV_DEPTH 32
#define YBWC_ORDERING_MAX_OFFSET_END 6
#define YBWC_OFFSET_DIV_DEPTH_END 40

inline int depth_to_offset(const int depth){
    return depth * YBWC_ORDERING_MAX_OFFSET / YBWC_OFFSET_DIV_DEPTH;
}

inline int depth_to_offset_end(const int depth){
    return depth * YBWC_ORDERING_MAX_OFFSET_END / YBWC_OFFSET_DIV_DEPTH_END;
}

int nega_alpha_ordering(Search *search, int alpha, int beta, int depth, bool skipped, uint64_t legal, bool is_end_search, const bool *searching);
int nega_alpha_end(Search *search, int alpha, int beta, bool skipped, uint64_t legal, const bool *searching);
int nega_scout(Search *search, int alpha, int beta, int depth, bool skipped, uint64_t legal, bool is_end_search, const bool *searching);

Parallel_task ybwc_do_task(uint64_t player, uint64_t opponent, uint_fast8_t n, uint_fast8_t p, uint_fast8_t parity, 
        bool use_mpc, double mpct, uint_fast8_t eval_feature_reversed, //vector<int> eval_features, 
        int alpha, int beta, int depth, uint64_t legal, bool is_end_search, const bool *searching, int policy){
    Search search;
    search.board.player = player;
    search.board.opponent = opponent;
    search.board.n = n;
    search.board.p = p;
    search.board.parity = parity;
    search.use_mpc = use_mpc;
    search.mpct = mpct;
    search.eval_feature_reversed = eval_feature_reversed;
    //for (int i = 0; i < N_SYMMETRY_PATTERNS; ++i)
    //    search.eval_features[i] = eval_features[i];
    search.n_nodes = 0ULL;
    calc_features(&search);
    int g = -nega_alpha_ordering(&search, alpha, beta, depth, false, legal, is_end_search, searching);
    Parallel_task task;
    if (*searching){
        task.value = g;
        task.n_nodes = search.n_nodes;
        task.cell = policy;
    } else{
        task.value = SCORE_UNDEFINED;
        task.n_nodes = search.n_nodes;
        task.cell = policy;
    }
    return task;
}
/*
pair<int, uint64_t> ybwc_do_task_end(Search search, int alpha, int beta, int depth, uint64_t legal, const bool *searching){
    //calc_features(&search);
    int g = -nega_alpha_end(&search, alpha, beta, false, legal, searching);
    if (*searching)
        return make_pair(g, search.n_nodes);
    return make_pair(SCORE_UNDEFINED, search.n_nodes);
}
*/
/*
inline bool ybwc_split(Search *search, const Flip *flip, int alpha, int beta, const int depth, uint64_t legal, bool is_end_search, const bool *searching, int policy, const int pv_idx, const int canput, const int split_count, vector<future<pair<int, uint64_t>>> &parallel_tasks, const int first_val){
    if (pv_idx > 0 && 
        depth >= YBWC_MID_SPLIT_MIN_DEPTH &&
        flip->value < first_val - depth_to_offset(depth)){
        if (thread_pool.n_idle()){
            Search copy_search;
            //eval_move(search, flip);
            search->board.move(flip);
                search->board.copy(&copy_search.board);
                for (int i = 0; i < N_SYMMETRY_PATTERNS; ++i)
                    copy_search.eval_features[i] = search->eval_features[i];
                copy_search.eval_feature_reversed = search->eval_feature_reversed;
            search->board.undo(flip);
            //eval_undo(search, flip);
            copy_search.use_mpc = search->use_mpc;
            copy_search.mpct = search->mpct;
            copy_search.n_nodes = 0;
            //copy_search.p = search->p;
            parallel_tasks.emplace_back(thread_pool.push(bind(&ybwc_do_task, copy_search, alpha, beta, depth, legal, is_end_search, searching, policy)));
            return true;
        }
    }
    return false;
}
*/
inline bool ybwc_split_without_move(const Search *search, const Flip *flip, int alpha, int beta, const int depth, uint64_t legal, bool is_end_search, const bool *searching, int policy, const int pv_idx, const int canput, const int split_count, vector<future<Parallel_task>> &parallel_tasks, const int first_val, const int last_val){
    if (pv_idx > 0 && 
        depth >= YBWC_MID_SPLIT_MIN_DEPTH/* &&
        first_val - flip->value > depth_to_offset(depth)*/){
        if (thread_pool.n_idle()){
            //vector<int> eval_features(N_SYMMETRY_PATTERNS);
            //for (int i = 0; i < N_SYMMETRY_PATTERNS; ++i)
            //    eval_features[i] = search->eval_features[i];
            parallel_tasks.emplace_back(thread_pool.push(bind(&ybwc_do_task, 
                search->board.player, search->board.opponent, search->board.n, search->board.p, search->board.parity, 
                search->use_mpc, search->mpct, search->eval_feature_reversed, //eval_features,
                alpha, beta, depth, legal, is_end_search, searching, policy)));
            return true;
        }
    }
    return false;
}
/*
inline bool ybwc_split_without_move_end(Search *search, const Flip *flip, int alpha, int beta, const int depth, uint64_t legal, const bool *searching, const int pv_idx, const int canput, const int split_count, vector<future<pair<int, uint64_t>>> &parallel_tasks, const int first_val){
    if (pv_idx > 0 && 
        depth >= YBWC_END_SPLIT_MIN_DEPTH &&
        flip->value < first_val - depth_to_offset_end(depth)){
        if (thread_pool.n_idle()){
            Search copy_search;
            search->board.copy(&copy_search.board);
            for (int i = 0; i < N_SYMMETRY_PATTERNS; ++i)
                copy_search.eval_features[i] = search->eval_features[i];
            copy_search.eval_feature_reversed = search->eval_feature_reversed;
            copy_search.use_mpc = search->use_mpc;
            copy_search.mpct = search->mpct;
            copy_search.n_nodes = 0;
            //copy_search.p = search->p;
            parallel_tasks.emplace_back(thread_pool.push(bind(&ybwc_do_task_end, copy_search, alpha, beta, depth, legal, searching)));
            return true;
        }
    }
    return false;
}

inline bool ybwc_split_without_move_negascout(Search *search, const Flip *flip, int alpha, int beta, const int depth, uint64_t legal, bool is_end_search, const bool *searching, int policy, const int pv_idx, const int canput, const int split_count, vector<future<pair<int, uint64_t>>> &parallel_tasks){
    if (pv_idx > 3 && 
        depth >= YBWC_MID_SPLIT_MIN_DEPTH){
        if (thread_pool.n_idle()){
            Search copy_search;
            search->board.copy(&copy_search.board);
            for (int i = 0; i < N_SYMMETRY_PATTERNS; ++i)
                copy_search.eval_features[i] = search->eval_features[i];
            copy_search.eval_feature_reversed = search->eval_feature_reversed;
            copy_search.use_mpc = search->use_mpc;
            copy_search.mpct = search->mpct;
            copy_search.n_nodes = 0;
            //copy_search.p = search->p;
            parallel_tasks.emplace_back(thread_pool.push(bind(&ybwc_do_task, copy_search, alpha, beta, depth, legal, is_end_search, searching, policy)));
            return true;
        }
    }
    return false;
}
*/
inline void ybwc_wait_all(Search *search, vector<future<Parallel_task>> &parallel_tasks, int *v, int *best_move){
    Parallel_task got_task;
    for (future<Parallel_task> &task: parallel_tasks){
        got_task = task.get();
        if (*v < got_task.value){
            *v = got_task.value;
            *best_move = got_task.cell;
        }
        search->n_nodes += got_task.n_nodes;
    }
}

inline void ybwc_wait_all(Search *search, vector<future<Parallel_task>> &parallel_tasks){
    Parallel_task got_task;
    for (future<Parallel_task> &task: parallel_tasks){
        got_task = task.get();
        search->n_nodes += got_task.n_nodes;
    }
}

/*
inline int ybwc_negascout_wait_all(Search *search, vector<future<pair<int, uint64_t>>> &parallel_tasks, vector<Flip> &flips, int before_alpha, int alpha, int beta, int depth, bool skipped, bool is_end_search, int *best_move){
    int v = alpha, g;
    pair<int, uint64_t> got_task;
    bool searching = true;
    for (int i = 0; i < (int)parallel_tasks.size(); ++i){
        got_task = parallel_tasks[i].get();
        g = got_task.first;
        search->n_nodes += got_task.second;
        if (before_alpha < g){
            v = max(v, g);
            search->board.move(&flips[i]);
                g = -nega_scout(search, -beta, -v, depth, skipped, flips[i].n_legal, is_end_search, &searching);
            search->board.undo(&flips[i]);
            if (v < g){
                v = g;
                *best_move = flips[i].pos;
            }
        }
    }
    return v;
}
*/