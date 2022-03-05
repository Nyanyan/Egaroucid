#pragma once
#include <iostream>
#include <vector>
#include "common.hpp"
#include "board.hpp"
#include "search.hpp"
#include "flip.hpp"
#include "transpose_table.hpp"
#include "midsearch.hpp"

#define N_MOVE_ORDERING_PATTERNS 10
#define MAX_MOVE_ORDERING_EVALUATE_IDX 65536
#define MOVE_ORDERING_PHASE_DIV 10
#define N_MOVE_ORDERING_PHASE 6

#define W_BEST_MOVE 900000000

#define W_CACHE_HIT 100000
//#define W_CACHE_HIGH 10000
#define W_WIPEOUT 1000000000

#define W_VALUE 6
#define W_CELL_WEIGHT 1
//#define W_EVALUATE 20
#define W_MOBILITY 14
#define W_SURROUND 8
#define W_PARITY 4
//#define W_STABILITY 20

#define MOVE_ORDERING_VALUE_OFFSET 4

#define W_END_MOBILITY 29
//#define W_END_SURROUND 10
#define W_END_PARITY 14
//#define W_END_EVALUATE 2

#define ALL_NODE_OFFSET 4

constexpr int move_ordering_depth[60] = {
    0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 
    2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 
    4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6
};

int nega_alpha_eval1(Search *search, int alpha, int beta, bool skipped);
int nega_alpha(Search *search, int alpha, int beta, int depth, bool skipped);
int nega_alpha_ordering_nomemo(Search *search, int alpha, int beta, int depth, bool skipped);
int nega_scout(Search *search, int alpha, int beta, int depth, bool skipped, bool is_end_search);

inline void move_evaluate(Search *search, Flip *flip, const int best_move, const int alpha, const int beta, const int depth){
    flip->value = 0;
    if (flip->pos == best_move)
        flip->value = W_BEST_MOVE;
    else{
        flip->value += cell_weight[flip->pos] * W_CELL_WEIGHT;
        if (search->board.parity & cell_div4[flip->pos])
            flip->value += W_PARITY;
        search->board.move(flip);
            if (search->board.player == 0)
                flip->value += W_WIPEOUT;
            else{
                unsigned long long empties = ~(search->board.player | search->board.opponent);
                flip->value += (calc_surround(search->board.player, empties) - calc_surround(search->board.opponent, empties)) * W_SURROUND;
                flip->value -= pop_count_ull(search->board.get_legal()) * W_MOBILITY;
                switch(depth){
                    case 0:
                        flip->value += (HW2 - mid_evaluate(&search->board)) * W_VALUE;
                        break;
                    case 1:
                        flip->value += (HW2 - nega_alpha_eval1(search, alpha, beta, false)) * W_VALUE;
                        break;
                    default:
                        bool use_mpc = search->use_mpc;
                        search->use_mpc = false;
                            flip->value += (HW2 - nega_alpha_ordering_nomemo(search, alpha, beta, depth, false)) * W_VALUE;
                        search->use_mpc = use_mpc;
                        break;
                }
            }
        search->board.undo(flip);
    }
}

bool cmp_move_ordering(Flip &a, Flip &b){
    return a.value > b.value;
}

inline void move_ordering(Search *search, vector<Flip> &move_list, int depth, int alpha, int beta, bool is_end_search){
    if (move_list.size() < 2)
        return;
    uint32_t hash_code = search->board.hash() & TRANSPOSE_TABLE_MASK;
    int best_move = child_transpose_table.get(&search->board, hash_code);
    int eval_alpha = -min(HW2, beta + MOVE_ORDERING_VALUE_OFFSET);
    int eval_beta = -max(-HW2, alpha - MOVE_ORDERING_VALUE_OFFSET);
    int eval_depth = depth / 8;
    for (Flip &mob: move_list)
        move_evaluate(search, &mob, best_move, eval_alpha, eval_beta, eval_depth);
    sort(move_list.begin(), move_list.end(), cmp_move_ordering);
}

inline void move_evaluate_fast_first(Search *search, Flip *flip, const int best_move){
    flip->value = 0;
    if (flip->pos == best_move)
        flip->value = W_BEST_MOVE;
    else{
        //flip->value += cell_weight[flip->pos] * W_END_CELL_WEIGHT;
        if (search->board.parity & cell_div4[flip->pos])
            flip->value += W_END_PARITY;
        search->board.move(flip);
            //flip->value += -mid_evaluate(&search->board) * W_END_EVALUATE;
            //flip->value += calc_surround(search->board.player, ~(search->board.player | search->board.opponent)) * W_END_SURROUND;
            flip->value -= pop_count_ull(search->board.get_legal()) * W_END_MOBILITY;
        search->board.undo(flip);
    }
}

inline void move_ordering_fast_first(Search *search, vector<Flip> &move_list){
    if (move_list.size() < 2)
        return;
    uint32_t hash_code = search->board.hash() & TRANSPOSE_TABLE_MASK;
    int best_move = child_transpose_table.get(&search->board, hash_code);
    for (Flip &mob: move_list)
        move_evaluate_fast_first(search, &mob, best_move);
    sort(move_list.begin(), move_list.end(), cmp_move_ordering);
}

inline void move_sorting(vector<Flip> &move_list){
    if (move_list.size() < 2)
        return;
    sort(move_list.begin(), move_list.end(), cmp_move_ordering);
}
