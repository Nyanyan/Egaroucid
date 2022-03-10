#pragma once
#include <iostream>
#include <vector>
#include "common.hpp"
#include "board.hpp"
#include "search.hpp"
#include "flip.hpp"
#include "transpose_table.hpp"
#include "midsearch.hpp"
#include "probcut.hpp"

#define N_MOVE_ORDERING_PATTERNS 10
#define MAX_MOVE_ORDERING_EVALUATE_IDX 65536
#define MOVE_ORDERING_PHASE_DIV 10
#define N_MOVE_ORDERING_PHASE 6

#define W_BEST_MOVE 900000000

#define W_WIPEOUT 1000000000

#define W_VALUE 8
#define W_VALUE_SHALLOW 4
#define W_CELL_WEIGHT 1
#define W_MOBILITY 16
//#define W_SURROUND 8
#define W_PARITY 4

#define MOVE_ORDERING_VALUE_OFFSET 14

#define W_END_MOBILITY 64
#define W_END_PARITY 14

int nega_alpha_eval1(Search *search, int alpha, int beta, bool skipped);
int nega_alpha(Search *search, int alpha, int beta, int depth, bool skipped);
int nega_alpha_ordering_nomemo(Search *search, int alpha, int beta, int depth, bool skipped, uint64_t legal);
int nega_scout(Search *search, int alpha, int beta, int depth, bool skipped, uint64_t legal, bool is_end_search);

inline void move_sort_top(vector<Flip> &move_list, int best_idx){
    if (best_idx != 0)
        swap(move_list[best_idx], move_list[0]);
}

bool cmp_move_ordering(Flip &a, Flip &b){
    return a.value > b.value;
}

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
                //uint64_t empties = ~(search->board.player | search->board.opponent);
                //flip->value += (calc_surround(search->board.player, empties) - calc_surround(search->board.opponent, empties)) * W_SURROUND;
                //if (search->board.n >= 44){
                flip->n_legal = search->board.get_legal();
                flip->value += -pop_count_ull(flip->n_legal) * W_MOBILITY;
                //}
                switch(depth){
                    case 0:
                        flip->value += -mid_evaluate(&search->board) * W_VALUE_SHALLOW;
                        break;
                    case 1:
                        flip->value += -nega_alpha_eval1(search, alpha, beta, false) * W_VALUE;
                        break;
                    default:
                        if (depth <= MID_FAST_DEPTH)
                            flip->value += -nega_alpha(search, alpha, beta, depth, false) * W_VALUE;
                        else{
                            bool use_mpc = search->use_mpc;
                            search->use_mpc = false;
                                flip->value += -nega_alpha_ordering_nomemo(search, alpha, beta, depth, false, flip->n_legal) * W_VALUE;
                            search->use_mpc = use_mpc;
                        }
                        break;
                }
            }
        search->board.undo(flip);
    }
}

inline void move_evaluate(Search *search, Flip *flip, const int alpha, const int beta, const int depth){
    flip->value = cell_weight[flip->pos] * W_CELL_WEIGHT;
    if (search->board.parity & cell_div4[flip->pos])
        flip->value += W_PARITY;
    search->board.move(flip);
        if (search->board.player == 0)
            flip->value += W_WIPEOUT;
        else{
            //uint64_t empties = ~(search->board.player | search->board.opponent);
            //flip->value += (calc_surround(search->board.player, empties) - calc_surround(search->board.opponent, empties)) * W_SURROUND;
            flip->n_legal = search->board.get_legal();
            flip->value += -pop_count_ull(flip->n_legal) * W_MOBILITY;
            switch(depth){
                case 0:
                    flip->value += ((HW2 - mid_evaluate(&search->board)) >> 1) * W_VALUE;
                    break;
                case 1:
                    flip->value += ((HW2 - nega_alpha_eval1(search, alpha, beta, false))) * W_VALUE;
                    break;
                default:
                    if (depth <= MID_FAST_DEPTH)
                        flip->value += (HW2 - nega_alpha(search, alpha, beta, depth, false)) * W_VALUE;
                    else{
                        bool use_mpc = search->use_mpc;
                        search->use_mpc = false;
                            flip->value += (HW2 - nega_alpha_ordering_nomemo(search, alpha, beta, depth, false, flip->n_legal)) * W_VALUE;
                        search->use_mpc = use_mpc;
                    }
                    break;
            }
        }
    search->board.undo(flip);
}

inline void move_ordering(Search *search, vector<Flip> &move_list, int depth, int alpha, int beta, bool is_end_search){
    if (move_list.size() < 2)
        return;
    int eval_alpha = -min(HW2, beta + MOVE_ORDERING_VALUE_OFFSET);
    int eval_beta = -max(-HW2, alpha - MOVE_ORDERING_VALUE_OFFSET);
    int eval_depth = depth / 8;
    for (Flip &flip: move_list)
        move_evaluate(search, &flip, eval_alpha, eval_beta, eval_depth);
    sort(move_list.begin(), move_list.end(), cmp_move_ordering);
}

inline void move_evaluate_fast_first(Search *search, Flip *flip, const int best_move){
    flip->value = 0;
    if (flip->pos == best_move)
        flip->value = W_BEST_MOVE;
    else{
        if (search->board.parity & cell_div4[flip->pos])
            flip->value += W_END_PARITY;
        search->board.move(flip);
            flip->n_legal = search->board.get_legal();
            flip->value -= pop_count_ull(flip->n_legal) * W_END_MOBILITY;
        search->board.undo(flip);
    }
}

inline void move_evaluate_fast_first(Search *search, Flip *flip){
    flip->value = 0;
    if (search->board.parity & cell_div4[flip->pos])
        flip->value += W_END_PARITY;
    search->board.move(flip);
        flip->n_legal = search->board.get_legal();
        flip->value -= pop_count_ull(flip->n_legal) * W_END_MOBILITY;
    search->board.undo(flip);
}

inline void move_ordering_fast_first(Search *search, vector<Flip> &move_list){
    if (move_list.size() < 2)
        return;
    for (Flip &flip: move_list)
        move_evaluate_fast_first(search, &flip);
    sort(move_list.begin(), move_list.end(), cmp_move_ordering);
}

