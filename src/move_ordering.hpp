#pragma once
#include <iostream>
#include <vector>
#include "common.hpp"
#include "board.hpp"
#include "search.hpp"
#include "transpose_table.hpp"
#if USE_CUDA
    #include "cuda_midsearch.hpp"
#else
    #include "midsearch.hpp"
#endif
#include "probcut.hpp"
#include "flip_variation/flipping.hpp"
#include "flip_variation/position.hpp"
#include "flip_variation/stone.hpp"

#define W_WIPEOUT 1000000000

#define W_PARITY1 2
#define W_PARITY2 4

#define W_VALUE 10
#define W_VALUE_SHALLOW 6
#define W_CACHE_HIT 16
#define W_MOBILITY 12
#define W_PLAYER_POTENTIAL_MOBILITY 4
#define W_OPPONENT_POTENTIAL_MOBILITY 8
#define W_OPENNESS 1
#define W_OPPONENT_OPENNESS 1
#define W_FLIP_INSIDE 16
#define W_BOUND_FLIP -1
#define W_CREATE_WALL -4
#define W_BREAK_WALL -32
#define W_DOUBLE_FLIP -64

#define MOVE_ORDERING_VALUE_OFFSET 14
#define MAX_MOBILITY 30
#define MAX_OPENNESS 50

#define W_END_MOBILITY 32
#define W_END_STABILITY 4
#define W_END_ANTI_EVEN 16
#define W_END_PARITY 2

#define MIDGAME_N_STONES 44
#define USE_OPPONENT_OPENNESS_DEPTH 16

struct move_ordering_info{
    uint64_t stones;
    uint64_t outside;
    uint64_t opponent_bound_stones;
    uint64_t opponent_create_wall_stones;
    uint64_t opponent_break_wall_stones;
};

int nega_alpha_eval1(Search *search, int alpha, int beta, bool skipped, const bool *searching);
int nega_alpha(Search *search, int alpha, int beta, int depth, bool skipped, const bool *searching);
int nega_alpha_ordering_nomemo(Search *search, int alpha, int beta, int depth, bool skipped, uint64_t legal, const bool *searching);
int nega_scout(Search *search, int alpha, int beta, int depth, bool skipped, uint64_t legal, bool is_end_search, const bool *searching);

inline uint64_t or_dismiss_1(const uint64_t a, const uint64_t b, const uint64_t c, const uint64_t d){
    return (a | b | c) & (a | b | d) & (a | c | d) & (b | c | d);
}

inline uint64_t calc_legal_flip_inside(const uint64_t player, const uint64_t opponent){
    const uint64_t empty = ~(player | opponent);
    uint64_t masked = empty & 0x7E7E7E7E7E7E7E7EULL;
    const uint64_t shift1 = (masked << 1) | (masked >> 1);
    masked = empty & 0x00FFFFFFFFFFFF00ULL;
    const uint64_t shift8 = (masked << HW) | (masked >> HW);
    masked = empty & 0x007E7E7E7E7E7E00ULL;
    const uint64_t shift7 = (masked << HW_M1) | (masked >> HW_M1);
    const uint64_t shift9 = (masked << HW_P1) | (masked >> HW_P1);
    uint64_t outside_stones_mask = shift1 | shift8 | shift7 | shift9;
    outside_stones_mask &= or_dismiss_1(shift1, shift8, shift7, shift9);
    if ((opponent & ~outside_stones_mask) == 0ULL)
        return 0ULL;
    uint64_t legal = calc_legal(player, opponent & ~outside_stones_mask) & empty;
    if (legal == 0ULL)
        return 0ULL;
    return legal & ~calc_legal(player, opponent & outside_stones_mask);
}

inline bool is_flip_inside(const int pos, const uint64_t p_legal_flip_inside){
    return (p_legal_flip_inside >> pos) & 1;
}

inline int create_disturb_player_flip_inside(Board *board, const int n_p_legal_flip_inside){
    return n_p_legal_flip_inside - pop_count_ull(calc_legal_flip_inside(board->opponent, board->player));
}

inline int create_disturb_opponent_flip_inside(Board *board, const int n_o_legal_flip_inside){
    return n_o_legal_flip_inside - pop_count_ull(calc_legal_flip_inside(board->player, board->opponent));
}

inline int calc_openness(const Board *board, const Flip *flip){
    uint64_t hmask = flip->flip & 0x7E7E7E7E7E7E7E7EULL;
    uint64_t vmask = flip->flip & 0x00FFFFFFFFFFFF00ULL;
    uint64_t hvmask = flip->flip & 0x007E7E7E7E7E7E00ULL;
    uint64_t around = 
        (hmask << 1) | (hmask >> 1) | 
        (vmask << HW) | (vmask >> HW) | 
        (hvmask << HW_M1) | (hvmask >> HW_M1) | 
        (hvmask << HW_P1) | (hvmask >> HW_P1);
    return pop_count_ull(~(board->player | board->opponent | (1ULL << flip->pos)) & around);
}

inline int calc_opponent_openness(Search *search, uint64_t legal){
    if (legal == 0ULL)
        return 0;
    int res = INF;
    Flip flip;
    for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)){
        calc_flip(&flip, &search->board, cell);
        res = min(res, calc_openness(&search->board, &flip));
        if (res == 0)
            return 0;
    }
    return res;
}

inline void move_sort_top(vector<Flip> &move_list, int best_idx){
    if (best_idx != 0)
        swap(move_list[best_idx], move_list[0]);
}

bool cmp_move_ordering(Flip &a, Flip &b){
    return a.value > b.value;
}

inline int midgame_player_move_ordering_heuristic(Board *board, Flip *flip, move_ordering_info *minfo){
    int openness = calc_openness(board, flip);
    // 中割り
    if (openness == 0)
        return W_FLIP_INSIDE;
    // 境界の石返し
    if (flip->flip & minfo->opponent_bound_stones)
        return -openness * W_OPENNESS + W_BOUND_FLIP;
    // 壁化
    if (flip->flip & minfo->opponent_create_wall_stones)
        return -openness * W_OPENNESS + W_CREATE_WALL;
    // 壁破り
    if (flip->flip & minfo->opponent_break_wall_stones)
        return -openness * W_OPENNESS + W_BREAK_WALL;
    // 二重返し
    if (is_double_flipping(board, flip))
        return -openness * W_OPENNESS + W_DOUBLE_FLIP;
    return -openness * W_OPENNESS;
}

int mobility_search(Search *search, int alpha, int beta, const int depth){
    if (depth == 0)
        return pop_count_ull(search->board.get_legal());
    uint64_t legal = search->board.get_legal();
    int g = 0;
    if (legal == 0ULL){
        search->board.pass();
            g = -mobility_search(search, -beta, -alpha, depth);
        search->board.pass();
        return g;
    }
    Flip flip;
    for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)){
        calc_flip(&flip, &search->board, cell);
        search->board.move(&flip);
            g = -mobility_search(search, -beta, -alpha, depth - 1);
        search->board.undo(&flip);
        alpha = max(alpha, g);
        if (beta <= alpha)
            return alpha;
    }
    return alpha;
}

int openness_search(Search *search, int alpha, int beta, const int depth){
    uint64_t legal = search->board.get_legal();
    int g = 0;
    if (legal == 0ULL){
        search->board.pass();
            g = -mobility_search(search, -beta, -alpha, depth);
        search->board.pass();
        return g;
    }
    Flip flip;
    if (depth > 0){
        for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)){
            calc_flip(&flip, &search->board, cell);
            search->board.move(&flip);
                g = -openness_search(search, -beta, -alpha, depth - 1);
            search->board.undo(&flip);
            alpha = max(alpha, g);
            if (beta <= alpha)
                return alpha;
        }
    } else{
        for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)){
            calc_flip(&flip, &search->board, cell);
            g = calc_openness(&search->board, &flip);
            alpha = max(alpha, g);
            if (beta <= alpha)
                return alpha;
        }
    }
    return alpha;
}

inline int get_weighted_n_moves(uint64_t legal){
    return pop_count_ull(legal) + pop_count_ull(legal & 0b10000001'00000000'00000000'00000000'00000000'00000000'00000000'10000001ULL);
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

inline void move_evaluate(Search *search, Flip *flip, const int alpha, const int beta, const int depth, const int mobility_depth, const bool *searching, const int search_depth){
    if (flip->flip == search->board.opponent)
        flip->value = W_WIPEOUT;
    else{
        flip->value = cell_weight[flip->pos];
        if (search->board.n <= MIDGAME_N_STONES)
            flip->value += -calc_openness(&search->board, flip) * W_OPENNESS;
        if (depth < 0){
            search->board.move(flip);
                flip->n_legal = search->board.get_legal();
                flip->value += -get_weighted_n_moves(flip->n_legal) * W_MOBILITY;
                flip->value += -get_potential_mobility(search->board.opponent, ~(search->board.player | search->board.opponent)) * W_OPPONENT_POTENTIAL_MOBILITY;
                flip->value += get_potential_mobility(search->board.player, ~(search->board.player | search->board.opponent)) * W_PLAYER_POTENTIAL_MOBILITY;
            search->board.undo(flip);
        } else{
            eval_move(search, flip);
            search->board.move(flip);
                flip->n_legal = search->board.get_legal();
                flip->value += -get_weighted_n_moves(flip->n_legal) * W_MOBILITY;
                flip->value += -get_potential_mobility(search->board.opponent, ~(search->board.player | search->board.opponent)) * W_OPPONENT_POTENTIAL_MOBILITY;
                flip->value += get_potential_mobility(search->board.player, ~(search->board.player | search->board.opponent)) * W_PLAYER_POTENTIAL_MOBILITY;
                /*
                bool value_flag = false;
                if (search_depth >= USE_PARENT_TT_DEPTH_THRESHOLD){
                    int l, u;
                    parent_transpose_table.get(&search->board, search->board.hash() & TRANSPOSE_TABLE_MASK, &l, &u);
                    if (u <= SCORE_MAX){
                        value_flag = true;
                        flip->value += (HW2 - u) * W_VALUE + W_CACHE_HIT;
                    } else if (-SCORE_MAX <= l){
                        value_flag = true;
                        flip->value += (HW2 - l) * W_VALUE + W_CACHE_HIT;
                    }
                }
                if (!value_flag){
                */
                switch(depth){
                    case 0:
                        flip->value += (HW2 - mid_evaluate_diff(search, searching)) * W_VALUE_SHALLOW;
                        break;
                    case 1:
                        flip->value += (HW2 - nega_alpha_eval1(search, alpha, beta, false, searching)) * W_VALUE;
                        break;
                    default:
                        if (depth <= MID_FAST_DEPTH)
                            flip->value += (HW2 - nega_alpha(search, alpha, beta, depth, false, searching)) * W_VALUE;
                        else{
                            bool use_mpc = search->use_mpc;
                            search->use_mpc = false;
                                flip->value += (HW2 - nega_alpha_ordering_nomemo(search, alpha, beta, depth, false, flip->n_legal, searching)) * W_VALUE;
                            search->use_mpc = use_mpc;
                        }
                        break;
                }
                //}
            search->board.undo(flip);
            eval_undo(search, flip);
        }
    }
}

inline void move_evaluate_fast_first(Search *search, Flip *flip){
    if (flip->flip == search->board.opponent)
        flip->value = W_WIPEOUT;
    else{
        flip->value = cell_weight[flip->pos];
        if (search->board.parity & cell_div4[flip->pos])
            flip->value += W_END_PARITY;
        search->board.move(flip);
            //flip->value += calc_stability_edge_player(search->board.opponent, search->board.player) * W_STABILITY;
            calc_stability(&search->board, &flip->stab1, &flip->stab0);
            flip->value += flip->stab0 * W_END_STABILITY;
            flip->n_legal = search->board.get_legal();
            flip->value += -pop_count_ull(flip->n_legal) * W_END_MOBILITY;
        search->board.undo(flip);
    }
}

inline int is_anti_even(Search *search, uint_fast8_t cell){
    if (search->board.parity & cell_div4[cell]){
        int res = 1;
        res &= search->board.opponent >> (cell + 1);
        res &= search->board.opponent >> (cell - 1);
        res &= search->board.opponent >> (cell + HW);
        res &= search->board.opponent >> (cell - HW);
        res &= search->board.opponent >> (cell + HW_P1);
        res &= search->board.opponent >> (cell - HW_P1);
        res &= search->board.opponent >> (cell + HW_M1);
        res &= search->board.opponent >> (cell - HW_M1);
        return res;
    }
    return 0;
}

inline void move_evaluate_end(Search *search, Flip *flip){
    if (flip->flip == search->board.opponent)
        flip->value = W_WIPEOUT;
    else{
        if (search->board.parity & cell_div4[flip->pos]){
            if (search->board.n < 34)
                flip->value = W_PARITY1;
            else
                flip->value = W_PARITY2;
        }
        search->board.move(flip);
            /*
            uint64_t empties = ~(search->board.player | search->board.opponent);
            if (flip->pos < 63 && (1 & (empties >> (flip->pos + 1))))
                flip->value -= is_anti_even(search, flip->pos + 1) * W_END_ANTI_EVEN;
            if (flip->pos >= 1 && (1 & (empties >> (flip->pos - 1))))
                flip->value -= is_anti_even(search, flip->pos - 1) * W_END_ANTI_EVEN;
            if (flip->pos < 56 && (1 & (empties >> (flip->pos + HW))))
                flip->value -= is_anti_even(search, flip->pos + HW) * W_END_ANTI_EVEN;
            if (flip->pos >= 8 && (1 & (empties >> (flip->pos - HW))))
                flip->value -= is_anti_even(search, flip->pos - HW) * W_END_ANTI_EVEN;
            */
            int stab0, stab1;
            calc_stability(&search->board, &stab0, &stab1);
            flip->value += stab1 * W_END_STABILITY;
            //flip->value += calc_stability_edge_player(search->board.opponent, search->board.player) * W_END_STABILITY;
            flip->n_legal = search->board.get_legal();
            flip->value += -pop_count_ull(flip->n_legal) * W_MOBILITY;
        search->board.undo(flip);
    }
}

inline void move_ordering(Search *search, vector<Flip> &move_list, int depth, int alpha, int beta, bool is_end_search, const bool *searching){
    if (move_list.size() < 2)
        return;
    int eval_alpha = -min(SCORE_MAX, beta + MOVE_ORDERING_VALUE_OFFSET);
    int eval_beta = -max(-SCORE_MAX, alpha - MOVE_ORDERING_VALUE_OFFSET);
    int eval_depth = depth >> 3;
    if (depth >= 18)
        ++eval_depth;
    if (depth >= 20)
        ++eval_depth;
    if (depth >= 22)
        ++eval_depth;
    int mobility_depth = (depth >> 2) & 0b10;
    //eval_depth = max(0, eval_depth);
    /*
    move_ordering_info minfo;
    minfo.stones = search->board.player | search->board.opponent;
    minfo.outside = calc_outside_stones(&search->board);
    minfo.opponent_bound_stones = calc_opponent_bound_stones(&search->board, minfo.outside);
    minfo.opponent_create_wall_stones = calc_opponent_create_wall_stones(&search->board, minfo.outside, minfo.opponent_bound_stones);
    minfo.opponent_break_wall_stones = calc_opponent_break_wall_stones(minfo.outside & search->board.opponent, minfo.opponent_bound_stones);
    */
    for (Flip &flip: move_list)
        move_evaluate(search, &flip, eval_alpha, eval_beta, eval_depth, mobility_depth, searching, depth);
    sort(move_list.begin(), move_list.end(), cmp_move_ordering);
}
/*
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
*/
inline void move_ordering_fast_first(Search *search, vector<Flip> &move_list){
    if (move_list.size() < 2)
        return;
    for (Flip &flip: move_list)
        move_evaluate_fast_first(search, &flip);
    sort(move_list.begin(), move_list.end(), cmp_move_ordering);
    /*
    int best_score = -INF;
    int best_idx = 0, i = 0;
    for (Flip &flip: move_list){
        move_evaluate_fast_first(search, &flip);
        if (best_score < flip.value){
            best_score = flip.value;
            best_idx = i;
        }
        ++i;
    }
    move_sort_top(move_list, best_idx);
    */
}

inline void move_ordering_end(Search *search, vector<Flip> &move_list){
    if (move_list.size() < 2)
        return;
    for (Flip &flip: move_list)
        move_evaluate_end(search, &flip);
    //move_evaluate_fast_first(search, &flip);
    sort(move_list.begin(), move_list.end(), cmp_move_ordering);
}