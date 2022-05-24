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

#define W_WIPEOUT 1000000000

#define W_VALUE 8
#define W_VALUE_SHALLOW 4
#define W_MOBILITY 16
#define W_PARITY1 2
#define W_PARITY2 4
#define W_OPENNESS 2
#define W_OPPONENT_OPENNESS 1
#define W_PLAYER_FLIP_INSIDE 32
#define W_OPPONENT_FLIP_INSIDE 16

#define MOVE_ORDERING_VALUE_OFFSET 14

#define W_END_MOBILITY 32
#define W_END_STABILITY 4
#define W_END_ANTI_EVEN 16
#define W_END_PARITY 2

#define USE_OPENNESS_N_STONES 49
#define USE_OPPONENT_OPENNESS_DEPTH 16

struct Flip_inside_info{
    uint64_t player;
    uint64_t opponent;
    int n_player;
    int n_opponent;
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

inline void move_evaluate(Search *search, Flip *flip, const int alpha, const int beta, const int depth, const bool *searching, uint64_t stones, const int search_depth){
    if (flip->flip == search->board.opponent)
        flip->value = W_WIPEOUT;
    else{
        flip->value = cell_weight[flip->pos];
        if (search->board.parity & cell_div4[flip->pos]){
            if (search->board.n < 34)
                flip->value += W_PARITY1;
            else
                flip->value += W_PARITY2;
        }
        if (search->board.n <= USE_OPENNESS_N_STONES){
            int openness = calc_openness(&search->board, flip);
            if (openness == 0)
                flip->value += W_PLAYER_FLIP_INSIDE;
            else
                flip->value -= openness * W_OPENNESS;
        }
        if (depth < 0){
            search->board.move(flip);
                flip->n_legal = search->board.get_legal();
                flip->value += -pop_count_ull(flip->n_legal) * W_MOBILITY;
            search->board.undo(flip);
        } else{
            eval_move(search, flip);
            search->board.move(flip);
                flip->n_legal = search->board.get_legal();
                flip->value += -pop_count_ull(flip->n_legal) * W_MOBILITY;
                switch(depth){
                    case 0:
                        flip->value += (HW2 - value_to_score_int(mid_evaluate_diff(search, searching))) * W_VALUE_SHALLOW;
                        break;
                    case 1:
                        flip->value += (HW2 - value_to_score_int(nega_alpha_eval1(search, alpha, beta, false, searching))) * W_VALUE;
                        break;
                    default:
                        if (depth <= MID_FAST_DEPTH)
                            flip->value += (HW2 - value_to_score_int(nega_alpha(search, alpha, beta, depth, false, searching))) * W_VALUE;
                        else{
                            bool use_mpc = search->use_mpc;
                            search->use_mpc = false;
                                flip->value += (HW2 - value_to_score_int(nega_alpha_ordering_nomemo(search, alpha, beta, depth, false, flip->n_legal, searching))) * W_VALUE;
                            search->use_mpc = use_mpc;
                        }
                        break;
                }
                if (search->board.n <= USE_OPENNESS_N_STONES && search_depth >= USE_OPPONENT_OPENNESS_DEPTH && flip->n_legal){
                    int openness = calc_opponent_openness(search, flip->n_legal);
                    if (openness == 0)
                        flip->value -= W_OPPONENT_FLIP_INSIDE;
                    else
                        flip->value += openness * W_OPPONENT_OPENNESS;
                }
            search->board.undo(flip);
            eval_undo(search, flip);
        }
    }
}

/* old version
inline void move_evaluate(Search *search, Flip *flip, const int alpha, const int beta, const int depth, const bool *searching, uint64_t stones){
    if (flip->flip == search->board.opponent)
        flip->value = W_WIPEOUT;
    else{
        flip->value = cell_weight[flip->pos];
        if (search->board.parity & cell_div4[flip->pos]){
            if (search->board.n < 34)
                flip->value += W_PARITY1;
            else if (search->board.n < 44)
                flip->value += W_PARITY2;
            else
                flip->value += W_PARITY3;
        }
        if (depth < 0){
            search->board.move(flip);
                //flip->value += -calc_surround(search->board.opponent, ~(search->board.player | search->board.opponent)) * W_SURROUND;
                flip->value += calc_stability_edge_player(search->board.opponent, search->board.player) * W_STABILITY;
                flip->n_legal = search->board.get_legal();
                flip->value += -pop_count_ull(flip->n_legal) * W_MOBILITY;
            search->board.undo(flip);
        } else{
            eval_move(search, flip);
            search->board.move(flip);
                //flip->value += -calc_surround(search->board.opponent, ~(search->board.player | search->board.opponent)) * W_SURROUND;
                flip->value += calc_stability_edge_player(search->board.opponent, search->board.player) * W_STABILITY;
                flip->n_legal = search->board.get_legal();
                flip->value += -pop_count_ull(flip->n_legal) * W_MOBILITY;
                if (depth >= 0){
                    switch(depth){
                        case 0:
                            flip->value += (HW2 - value_to_score_int(mid_evaluate_diff(search, searching))) * W_VALUE_SHALLOW;
                            break;
                        case 1:
                            flip->value += (HW2 - value_to_score_int(nega_alpha_eval1(search, alpha, beta, false, searching))) * W_VALUE;
                            break;
                        default:
                            if (depth <= MID_FAST_DEPTH)
                                flip->value += (HW2 - value_to_score_int(nega_alpha(search, alpha, beta, depth, false, searching))) * W_VALUE;
                            else{
                                bool use_mpc = search->use_mpc;
                                search->use_mpc = false;
                                    flip->value += (HW2 - value_to_score_int(nega_alpha_ordering_nomemo(search, alpha, beta, depth, false, flip->n_legal, searching))) * W_VALUE;
                                search->use_mpc = use_mpc;
                            }
                            break;
                    }
                }
            search->board.undo(flip);
            eval_undo(search, flip);
        }
    }
}
*/

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
            if (1 & (empties >> (flip->pos + 1)))
                flip->value -= is_anti_even(search, flip->pos + 1) * W_END_ANTI_EVEN;
            if (1 & (empties >> (flip->pos - 1)))
                flip->value -= is_anti_even(search, flip->pos - 1) * W_END_ANTI_EVEN;
            if (1 & (empties >> (flip->pos + HW)))
                flip->value -= is_anti_even(search, flip->pos + HW) * W_END_ANTI_EVEN;
            if (1 & (empties >> (flip->pos - HW)))
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
    //eval_depth = max(0, eval_depth);
    const uint64_t stones = search->board.player | search->board.opponent;
    for (Flip &flip: move_list)
        move_evaluate(search, &flip, eval_alpha, eval_beta, eval_depth, searching, stones, depth);
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
    //move_evaluate_fast_first(search, &flip);
    sort(move_list.begin(), move_list.end(), cmp_move_ordering);
}

inline void move_ordering_end(Search *search, vector<Flip> &move_list){
    if (move_list.size() < 2)
        return;
    for (Flip &flip: move_list)
        move_evaluate_end(search, &flip);
    //move_evaluate_fast_first(search, &flip);
    sort(move_list.begin(), move_list.end(), cmp_move_ordering);
}