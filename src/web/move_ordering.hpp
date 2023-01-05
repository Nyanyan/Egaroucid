/*
    Egaroucid Project

    @date 2021-2023
    @author Takuto Yamana (a.k.a Nyanyan)
    @license GPL-3.0 license
*/

#pragma once
#include <iostream>
#include <vector>
#include "common.hpp"
#include "board.hpp"
#include "search.hpp"
#include "transpose_table.hpp"
#include "midsearch.hpp"
#include "stability.hpp"

#define W_WIPEOUT INF

#define W_PARITY1 2
#define W_PARITY2 4

#define W_VALUE_TT 6
#define W_VALUE_DEEP 12
#define W_VALUE 10
#define W_VALUE_SHALLOW 8
#define W_MOBILITY 8
#define W_PLAYER_POTENTIAL_MOBILITY 8
#define W_OPPONENT_POTENTIAL_MOBILITY 10
//#define W_OPENNESS 1

#define MOVE_ORDERING_VALUE_OFFSET 10
#define MAX_MOBILITY 30
#define MAX_OPENNESS 50

#define W_END_MOBILITY 16
//#define W_END_PARITY 2

#define MIDGAME_N_STONES 44
#define USE_OPPONENT_OPENNESS_DEPTH 16

//#define MOVE_ORDERING_TT_OFFSET 1
#define MOVE_ORDERING_TT_BONUS 4

#define MOVE_ORDERING_THRESHOLD 4

#define WORTH_SEARCHING_THRESHOLD 4


struct Flip_value{
    Flip flip;
    int value;
    uint64_t n_legal;
};

int nega_alpha_eval1(Search *search, int alpha, int beta, bool skipped, const bool *searching);
int nega_alpha(Search *search, int alpha, int beta, int depth, bool skipped, const bool *searching);
int nega_alpha_ordering_nomemo(Search *search, int alpha, int beta, int depth, bool skipped, uint64_t legal, const bool *searching);
int nega_alpha_ordering(Search *search, int alpha, int beta, int depth, bool skipped, uint64_t legal, bool is_end_search, const bool *searching);
int nega_scout(Search *search, int alpha, int beta, int depth, bool skipped, uint64_t legal, bool is_end_search, const bool *searching);

inline int calc_openness(const Board *board, const Flip *flip){
    uint64_t f = flip->flip;
    uint64_t around = 0ULL;
    for (uint_fast8_t cell = first_bit(&f); f; cell = next_bit(&f))
        around |= bit_around[cell];
    around &= ~flip->flip;
    return pop_count_ull(~(board->player | board->opponent | (1ULL << flip->pos)) & around);
}

inline int get_corner_mobility(uint64_t legal){
    //legal &= 0b10000001'00000000'00000000'00000000'00000000'00000000'00000000'10000001ULL;
    int res = (int)((legal & 0b10000001ULL) + ((legal >> 56) & 0b10000001ULL));
    return (res & 0b11) + (res >> 7);
}

inline int get_weighted_n_moves(uint64_t legal){
    return pop_count_ull(legal) * 2 + get_corner_mobility(legal);
}

inline int get_potential_mobility(uint64_t opponent, uint64_t empties){
    uint64_t hmask = opponent & 0x7E7E7E7E7E7E7E7EULL;
    uint64_t vmask = opponent & 0x00FFFFFFFFFFFF00ULL;
    uint64_t hvmask = opponent & 0x007E7E7E7E7E7E00ULL;
    uint64_t res = 
        (hmask << 1) | (hmask >> 1) | 
        (vmask << HW) | (vmask >> HW) | 
        (hvmask << HW_M1) | (hvmask >> HW_M1) | 
        (hvmask << HW_P1) | (hvmask >> HW_P1);
    return pop_count_ull(empties & res);
}

inline bool move_evaluate(Search *search, Flip_value *flip_value, const int alpha, const int beta, const int depth, const bool *searching, const int search_depth, const int search_alpha){
    if (flip_value->flip.flip == search->board.opponent){
        flip_value->value = W_WIPEOUT;
        return true;
    }
    flip_value->value = cell_weight[flip_value->flip.pos];
    //flip_value->value -= (calc_openness(&search->board, &flip_value->flip) >> 1) * W_OPENNESS;
    eval_move(search, &flip_value->flip);
    search->move(&flip_value->flip);
        flip_value->n_legal = search->board.get_legal();
        flip_value->value -= get_weighted_n_moves(flip_value->n_legal) * W_MOBILITY;
        uint64_t empties = ~(search->board.player | search->board.opponent);
        flip_value->value -= get_potential_mobility(search->board.opponent, empties) * W_OPPONENT_POTENTIAL_MOBILITY;
        flip_value->value += get_potential_mobility(search->board.player, empties) * W_PLAYER_POTENTIAL_MOBILITY;
        //int l, u;
        //parent_transpose_table.get(&search->board, search->board.hash() & TRANSPOSE_TABLE_MASK, &l, &u, 0.0, 0);
        //if (-INF < l && u < INF)
        //    flip_value->value += (-(l + u) / 2 + MOVE_ORDERING_TT_BONUS) * W_VALUE_DEEP;
        //else{
        switch (depth){
            case 0:
                flip_value->value += -mid_evaluate_diff(search) * W_VALUE_SHALLOW;
                break;
            case 1:
                flip_value->value += -nega_alpha_eval1(search, alpha, beta, false, searching) * W_VALUE;
                break;
            default:
                flip_value->value += -nega_alpha_ordering(search, alpha, beta, depth, false, flip_value->n_legal, false, searching) * (W_VALUE_DEEP + (depth - 1) * 2);
                break;
        }
        //}
    search->undo(&flip_value->flip);
    eval_undo(search, &flip_value->flip);
    return false;
}

inline bool move_evaluate_fast_first(Search *search, Flip_value *flip_value){
    if (flip_value->flip.flip == search->board.opponent){
        flip_value->value = W_WIPEOUT;
        return true;
    }
    flip_value->value = cell_weight[flip_value->flip.pos];
    //if (search->parity & cell_div4[flip_value->flip.pos])
    //    flip_value->value += W_END_PARITY;
    search->move(&flip_value->flip);
        flip_value->n_legal = search->board.get_legal();
        flip_value->value += -pop_count_ull(flip_value->n_legal) * W_END_MOBILITY;
    search->undo(&flip_value->flip);
    return false;
}

inline void swap_next_best_move(vector<Flip_value> &move_list, const int strt, const int siz){
    int top_idx = strt;
    int best_value = -INF;
    for (int i = strt; i < siz; ++i){
        if (best_value < move_list[i].value){
            best_value = move_list[i].value;
            top_idx = i;
        }
    }
    if (top_idx != strt)
        swap(move_list[strt], move_list[top_idx]);
}

inline void move_sort_top(vector<Flip_value> &move_list, int best_idx){
    if (best_idx != 0)
        swap(move_list[best_idx], move_list[0]);
}

bool cmp_move_ordering(Flip_value &a, Flip_value &b){
    return a.value > b.value;
}

inline void move_list_evaluate(Search *search, vector<Flip_value> &move_list, int depth, int alpha, int beta, bool is_end_search, const bool *searching){
    if (move_list.size() == 1){
        move_list[0].n_legal = LEGAL_UNDEFINED;
        return;
    }
    int eval_alpha = -min(SCORE_MAX, beta + MOVE_ORDERING_VALUE_OFFSET);
    int eval_beta = -max(-SCORE_MAX, alpha - MOVE_ORDERING_VALUE_OFFSET);
    int eval_depth = depth >> 3;
    if (depth >= 18)
        eval_depth += (depth - 16) >> 1;
    bool wipeout_found = false;
    for (Flip_value &flip_value: move_list){
        flip_value.n_legal = LEGAL_UNDEFINED;
        if (!wipeout_found)
            wipeout_found = move_evaluate(search, &flip_value, eval_alpha, eval_beta, eval_depth, searching, depth, alpha);
        else
            flip_value.value = -INF;
    }
}

inline void move_ordering(Search *search, vector<Flip_value> &move_list, int depth, int alpha, int beta, bool is_end_search, const bool *searching){
    move_list_evaluate(search, move_list, depth, alpha, beta, is_end_search, searching);
    sort(move_list.begin(), move_list.end(), cmp_move_ordering);
}

inline void move_list_evaluate_fast_first(Search *search, vector<Flip_value> &move_list){
    if (move_list.size() == 1){
        move_list[0].n_legal = LEGAL_UNDEFINED;
        return;
    }
    bool wipeout_found = false;
    for (Flip_value &flip_value: move_list){
        flip_value.n_legal = LEGAL_UNDEFINED;
        if (!wipeout_found)
            wipeout_found = move_evaluate_fast_first(search, &flip_value);
        else
            flip_value.value = -INF;
    }
}

inline void move_ordering_fast_first(Search *search, vector<Flip_value> &move_list){
    move_list_evaluate_fast_first(search, move_list);
    sort(move_list.begin(), move_list.end(), cmp_move_ordering);
}