/*
    Egaroucid Project

    @date 2021-2022
    @author Takuto Yamana (a.k.a Nyanyan)
    @license GPL-3.0 license
*/

#pragma once
#include <iostream>
#include <fstream>
#include <algorithm>
#include "setting.hpp"
#include "common.hpp"
#include "board.hpp"
#include "thread_pool.hpp"
#include "evaluate.hpp"
#include "transpose_table.hpp"

using namespace std;

#ifndef N_SYMMETRY_PATTERNS
    #define N_SYMMETRY_PATTERNS 62
#endif
#define MID_FAST_DEPTH 1
#define END_FAST_DEPTH 7
#define MID_TO_END_DEPTH 13
#define USE_TT_DEPTH_THRESHOLD 10

#define SCORE_UNDEFINED -INF

constexpr int cell_weight[HW2] = {
    18,  4,  16, 12, 12, 16,  4, 18,
     4,  2,   6,  8,  8,  6,  2,  4,
    16,  6,  14, 10, 10, 14,  6, 16,
    12,  8,  10,  0,  0, 10,  8, 12,
    12,  8,  10,  0,  0, 10,  8, 12,
    16,  6,  14, 10, 10, 14,  6, 16,
     4,  2,   6,  8,  8,  6,  2,  4,
    18,  4,  16, 12, 12, 16,  4, 18
};


//from https://github.com/abulmo/edax-reversi/blob/1ae7c9fe5322ac01975f1b3196e788b0d25c1e10/src/search.c
//modified by Nyanyan

constexpr int nws_stability_threshold[61] = {
    99, 99, 99,  4,  6,  8, 10, 12,
    14, 16, 20, 22, 24, 26, 28, 30,
    32, 34, 36, 38, 40, 42, 44, 46,
    48, 48, 50, 50, 52, 52, 54, 54,
    56, 56, 58, 58, 60, 60, 62, 62,
    64, 64, 64, 64, 64, 64, 64, 64,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99
};
//end of modification

constexpr uint_fast8_t cell_div4[HW2] = {
    1, 1, 1, 1, 2, 2, 2, 2, 
    1, 1, 1, 1, 2, 2, 2, 2, 
    1, 1, 1, 1, 2, 2, 2, 2, 
    1, 1, 1, 1, 2, 2, 2, 2, 
    4, 4, 4, 4, 8, 8, 8, 8, 
    4, 4, 4, 4, 8, 8, 8, 8, 
    4, 4, 4, 4, 8, 8, 8, 8, 
    4, 4, 4, 4, 8, 8, 8, 8
};

struct Search_result{
    int_fast8_t policy;
    int value;
    int depth;
    uint64_t time;
    uint64_t nodes;
    uint64_t nps;
    bool is_end_search;
    int probability;
};

class Search{
    public:
        Board board;
        int_fast8_t n_discs;
        uint_fast8_t parity;
        bool use_mpc;
        double mpct;
        uint64_t n_nodes;
        int eval_features[N_SYMMETRY_PATTERNS];
        uint_fast8_t eval_feature_reversed;
        int first_depth;
        bool use_multi_thread;

    public:
        inline void init_board(Board *init_board){
            board = init_board->copy();
            n_discs = board.n_discs();
            uint64_t empty = ~(board.player | board.opponent);
            parity = 1 & pop_count_ull(empty & 0x000000000F0F0F0FULL);
            parity |= (1 & pop_count_ull(empty & 0x00000000F0F0F0F0ULL)) << 1;
            parity |= (1 & pop_count_ull(empty & 0x0F0F0F0F00000000ULL)) << 2;
            parity |= (1 & pop_count_ull(empty & 0xF0F0F0F000000000ULL)) << 3;
        }

        inline void init_board(){
            n_discs = board.n_discs();
            uint64_t empty = ~(board.player | board.opponent);
            parity = 1 & pop_count_ull(empty & 0x000000000F0F0F0FULL);
            parity |= (1 & pop_count_ull(empty & 0x00000000F0F0F0F0ULL)) << 1;
            parity |= (1 & pop_count_ull(empty & 0x0F0F0F0F00000000ULL)) << 2;
            parity |= (1 & pop_count_ull(empty & 0xF0F0F0F000000000ULL)) << 3;
        }

        inline void move(const Flip *flip) {
            board.move_board(flip);
            ++n_discs;
            parity ^= cell_div4[flip->pos];
        }

        inline void undo(const Flip *flip) {
            board.undo_board(flip);
            --n_discs;
            parity ^= cell_div4[flip->pos];
        }

        inline int phase(){
            return min(N_PHASES - 1, (n_discs - 4) / PHASE_N_STONES);
        }
};

inline void register_tt(Search *search, int depth, uint32_t hash_code, int v, int best_move, int l, int u, int first_alpha, int beta, const bool *searching){
    if (search->n_discs <= HW2 - USE_TT_DEPTH_THRESHOLD && (*searching) && -SCORE_MAX <= v && v <= SCORE_MAX && global_searching){
        if (first_alpha < v && best_move != TRANSPOSE_TABLE_UNDEFINED)
            child_transpose_table.reg(&search->board, hash_code, best_move);
        if (first_alpha < v && v < beta)
            parent_transpose_table.reg(&search->board, hash_code, v, v, search->mpct, depth);
        else if (beta <= v && l < v)
            parent_transpose_table.reg(&search->board, hash_code, v, u, search->mpct, depth);
        else if (v <= first_alpha && v < u)
            parent_transpose_table.reg(&search->board, hash_code, l, v, search->mpct, depth);
    }
}

inline void register_tt_mpct(Search *search, int depth, uint32_t hash_code, int v, int best_move, int l, int u, int first_alpha, int beta, const bool *searching, double mpct){
    if (search->n_discs <= HW2 - USE_TT_DEPTH_THRESHOLD && (*searching) && -SCORE_MAX <= v && v <= SCORE_MAX && global_searching){
        if (first_alpha < v && best_move != TRANSPOSE_TABLE_UNDEFINED)
            child_transpose_table.reg(&search->board, hash_code, best_move);
        if (first_alpha < v && v < beta)
            parent_transpose_table.reg(&search->board, hash_code, v, v, mpct, depth);
        else if (beta <= v && l < v)
            parent_transpose_table.reg(&search->board, hash_code, v, u, mpct, depth);
        else if (v <= first_alpha && v < u)
            parent_transpose_table.reg(&search->board, hash_code, l, v, mpct, depth);
    }
}

inline void register_tt_nws(Search *search, int depth, uint32_t hash_code, int alpha, int v, int l, int u, const bool *searching){
    if (search->n_discs <= HW2 - USE_TT_DEPTH_THRESHOLD && (*searching) && -SCORE_MAX <= v && v <= SCORE_MAX && global_searching){
        if (alpha < v)
            parent_transpose_table.reg(&search->board, hash_code, v, u, search->mpct, depth);
        else
            parent_transpose_table.reg(&search->board, hash_code, l, v, search->mpct, depth);
    }
}

inline void register_tt_nws(Search *search, int depth, uint32_t hash_code, int alpha, int v, int best_move, int l, int u, const bool *searching){
    if (search->n_discs <= HW2 - USE_TT_DEPTH_THRESHOLD && (*searching) && -SCORE_MAX <= v && v <= SCORE_MAX && global_searching){
        if (alpha < v && best_move != TRANSPOSE_TABLE_UNDEFINED)
            child_transpose_table.reg(&search->board, hash_code, best_move);
        if (alpha < v)
            parent_transpose_table.reg(&search->board, hash_code, v, u, search->mpct, depth);
        else
            parent_transpose_table.reg(&search->board, hash_code, l, v, search->mpct, depth);
    }
}

inline void register_tt_nws_mpct(Search *search, int depth, uint32_t hash_code, int alpha, int v, int best_move, int l, int u, const bool *searching, double mpct){
    if (search->n_discs <= HW2 - USE_TT_DEPTH_THRESHOLD && (*searching) && -SCORE_MAX <= v && v <= SCORE_MAX && global_searching){
        if (alpha < v && best_move != TRANSPOSE_TABLE_UNDEFINED)
            child_transpose_table.reg(&search->board, hash_code, best_move);
        if (alpha < v)
            parent_transpose_table.reg(&search->board, hash_code, v, u, mpct, depth);
        else
            parent_transpose_table.reg(&search->board, hash_code, l, v, mpct, depth);
    }
}