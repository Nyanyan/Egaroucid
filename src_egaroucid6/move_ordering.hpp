#pragma once
#include <iostream>
#include <vector>
#include "common.hpp"
#include "board.hpp"
#include "search.hpp"
#include "transpose_table.hpp"
#include "midsearch.hpp"
#include "stability.hpp"

#define W_WIPEOUT 1000000000

#define W_PARITY1 2
#define W_PARITY2 4

#define W_VALUE_DEEP 10
#define W_VALUE 8
#define W_VALUE_SHALLOW 6
#define W_CACHE_HIT 4
#define W_MOBILITY 12
#define W_PLAYER_POTENTIAL_MOBILITY 6
#define W_OPPONENT_POTENTIAL_MOBILITY 8
#define W_OPENNESS 1
#define W_OPPONENT_OPENNESS 2
#define W_FLIP_INSIDE 16
#define W_N_FLIP 2
#define W_N_FLIP_DIRECTION 4
#define W_BOUND_FLIP -1
#define W_CREATE_WALL -4
#define W_BREAK_WALL -32
#define W_DOUBLE_FLIP -64
#define W_GIVE_POTENTIAL_FLIP_INSIDE 32
#define W_GOOD_STICK 8

#define W_FIRST_MIDDLE_EDGE_FLIP_1_DIRECTION 16
#define W_SECOND_MIDDLE_EDGE_FLIP_HORIZONTAL 64
#define W_SECOND_MIDDLE_EDGE_NEXT_TO_PILLAR 16
#define W_FIRST_EDGE_FLIP_1_DIRECTION 8
#define W_SECOND_EDGE_FLIP_1_DIRECTION 16

#define MOVE_ORDERING_VALUE_OFFSET 14
#define MAX_MOBILITY 30
#define MAX_OPENNESS 50

#define W_END_MOBILITY 32
#define W_END_STABILITY 4
#define W_END_ANTI_EVEN 16
#define W_END_PARITY 2

#define MIDGAME_N_STONES 44
#define USE_OPPONENT_OPENNESS_DEPTH 16



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
int nega_scout(Search *search, int alpha, int beta, int depth, bool skipped, uint64_t legal, bool is_end_search, const bool *searching);

inline int calc_openness(const Board *board, const Flip *flip){
    uint64_t f = flip->flip;
    uint64_t around = 0ULL;
    for (uint_fast8_t cell = first_bit(&f); f; cell = next_bit(&f))
        around |= bit_around[cell];
    around &= ~flip->flip;
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

inline int get_corner_mobility(uint64_t legal){
    legal &= 0b10000001'00000000'00000000'00000000'00000000'00000000'00000000'10000001ULL;
    int res = (int)((legal & 0b10000001ULL) + (legal >> 56));
    return (res & 0b11) + (res >> 7);
}

inline int get_weighted_n_moves(uint64_t legal){
    return pop_count_ull(legal) + get_corner_mobility(legal);
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

inline bool is_good_stick(Search *search, Flip *flip){
    uint64_t place = 1ULL << flip->pos;
    if (place & 0x0081818181818100ULL){
        if ((search->board.player | search->board.opponent) & (place << 8 | place >> 8))
            return pop_count_ull(flip->flip & bit_around[flip->pos]) == 1;
    }
    if (place & 0x7E0000000000007EULL){
        if ((search->board.player | search->board.opponent) & (place << 1 | place >> 1))
            return pop_count_ull(flip->flip & bit_around[flip->pos]) == 1;
    }
    return false;
}

inline bool is_first_middle_edge(Search *search, Flip *flip){
    uint64_t place = 1ULL << flip->pos;
    uint64_t stones = search->board.player | search->board.opponent;
    if (place & 0x003C000000000000ULL)
        return (stones & 0x003C000000000000ULL) == 0ULL;
    if (place & 0x0000404040400000ULL)
        return (stones & 0x0000404040400000ULL) == 0ULL;
    if (place & 0x0000000000003C00ULL)
        return (stones & 0x0000000000003C00ULL) == 0ULL;
    if (place & 0x0000020202020000ULL)
        return (stones & 0x0000020202020000ULL) == 0ULL;
    return false;
}

inline bool is_flip_1_direction(Flip *flip){
    return pop_count_ull(flip->flip & bit_around[flip->pos]) == 1;
}

inline bool is_flip_horizontal(Flip *flip){
    uint64_t place = 1ULL << flip->pos;
    if (place & 0x003C000000000000ULL)
        return (flip->flip & 0x003C000000000000ULL) > 0ULL;
    if (place & 0x0000404040400000ULL)
        return (flip->flip & 0x0000404040400000ULL) > 0ULL;
    if (place & 0x0000000000003C00ULL)
        return (flip->flip & 0x0000000000003C00ULL) > 0ULL;
    if (place & 0x0000020202020000ULL)
        return (flip->flip & 0x0000020202020000ULL) > 0ULL;
    return false;
}

inline bool is_next_to_pillar(Search *search, Flip *flip){
    uint64_t place = 1ULL << flip->pos;
    uint64_t stones = search->board.player | search->board.opponent;
    if (place & 0x003C000000003C00ULL)
        return (stones & ((place << 1) | (place >> 1))) > 0ULL;
    if (place & 0x0000424242420000ULL)
        return (stones & ((place << HW) | (place >> HW))) > 0ULL;
    return false;
}

inline bool is_first_edge(Search *search, Flip *flip){
    uint64_t place = 1ULL << flip->pos;
    uint64_t stones = search->board.player | search->board.opponent;
    if (place & 0x3C00000000000000ULL)
        return (stones & 0x3C00000000000000ULL) == 0ULL;
    if (place & 0x0000808080800000ULL)
        return (stones & 0x0000808080800000ULL) == 0ULL;
    if (place & 0x000000000000003CULL)
        return (stones & 0x000000000000003CULL) == 0ULL;
    if (place & 0x0000010101010000ULL)
        return (stones & 0x0000010101010000ULL) == 0ULL;
    return false;
}

inline bool is_next_to_pillar_edge(Search *search, Flip *flip){
    uint64_t place = 1ULL << flip->pos;
    uint64_t stones = search->board.player | search->board.opponent;
    if (place & 0x3C0000000000003CULL)
        return (stones & ((place << 1) | (place >> 1))) > 0ULL;
    if (place & 0x0000818181810000ULL)
        return (stones & ((place << HW) | (place >> HW))) > 0ULL;
    return false;
}

inline bool is_flip_few_stones(Flip *flip, const int threshold){
    return pop_count_ull(flip->flip) <= threshold;
}


inline bool move_evaluate(Search *search, Flip_value *flip_value, const int alpha, const int beta, const int depth, const bool *searching, const int search_depth, const int search_alpha, bool *worth_searching){
    if (flip_value->flip.flip == search->board.opponent){
        flip_value->value = W_WIPEOUT;
        return true;
    }
    flip_value->value = cell_weight[flip_value->flip.pos];
    flip_value->value -= (calc_openness(&search->board, &flip_value->flip) >> 1) * W_OPENNESS;
    eval_move(search, &flip_value->flip);
    search->move(&flip_value->flip);
        flip_value->n_legal = search->board.get_legal();
        flip_value->value -= get_weighted_n_moves(flip_value->n_legal) * W_MOBILITY;
        flip_value->value -= get_potential_mobility(search->board.opponent, ~(search->board.player | search->board.opponent)) * W_OPPONENT_POTENTIAL_MOBILITY;
        flip_value->value += get_potential_mobility(search->board.player, ~(search->board.player | search->board.opponent)) * W_PLAYER_POTENTIAL_MOBILITY;
        int val;
        switch(depth){
            case 0:
                val = -mid_evaluate_diff(search);
                if (search_alpha - WORTH_SEARCHING_THRESHOLD <= val)
                    *worth_searching = true;
                flip_value->value += val * W_VALUE_SHALLOW;
                break;
            case 1:
                val = -nega_alpha_eval1(search, alpha, beta, false, searching);
                if (search_alpha - WORTH_SEARCHING_THRESHOLD <= val)
                    *worth_searching = true;
                flip_value->value += val * W_VALUE;
                break;
            default:
                if (parent_transpose_table.contain(&search->board, search->board.hash() & TRANSPOSE_TABLE_MASK))
                    flip_value->value += W_CACHE_HIT;
                val = -nega_alpha_ordering_nomemo(search, alpha, beta, depth, false, flip_value->n_legal, searching);
                if (search_alpha - WORTH_SEARCHING_THRESHOLD <= val)
                    *worth_searching = true;
                flip_value->value += val * (W_VALUE_DEEP + (depth - 1) * 2);
                break;
        }
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
    if (search->parity & cell_div4[flip_value->flip.pos])
        flip_value->value += W_END_PARITY;
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

inline bool move_list_evaluate(Search *search, vector<Flip_value> &move_list, int depth, int alpha, int beta, bool is_end_search, const bool *searching){
    if (move_list.size() < 2)
        return true;
    int eval_alpha = -min(SCORE_MAX, beta + MOVE_ORDERING_VALUE_OFFSET);
    int eval_beta = -max(-SCORE_MAX, alpha - MOVE_ORDERING_VALUE_OFFSET);
    int eval_depth = depth >> 3;
    if (depth >= 18 && is_end_search){
        ++eval_depth;
        if (depth >= 21){
            ++eval_depth;
            if (depth >= 23){
                ++eval_depth;
                if (depth >= 26){
                    ++eval_depth;
                }
            }
        }
    }
    bool wipeout_found = false;
    bool worth_searching = false;
    for (Flip_value &flip_value: move_list){
        if (!wipeout_found)
            wipeout_found = move_evaluate(search, &flip_value, eval_alpha, eval_beta, eval_depth, searching, depth, alpha, &worth_searching);
        else
            flip_value.value = -INF;
    }
    return worth_searching;
}

inline void move_ordering(Search *search, vector<Flip_value> &move_list, int depth, int alpha, int beta, bool is_end_search, const bool *searching){
    move_list_evaluate(search, move_list, depth, alpha, beta, is_end_search, searching);
    sort(move_list.begin(), move_list.end(), cmp_move_ordering);
}

inline void move_evaluate_fast_first(Search *search, vector<Flip_value> &move_list){
    if (move_list.size() < 2)
        return;
    bool wipeout_found = false;
    for (Flip_value &flip_value: move_list){
        if (!wipeout_found)
            wipeout_found = move_evaluate_fast_first(search, &flip_value);
        else
            flip_value.value = -INF;
    }
}

inline void move_ordering_fast_first(Search *search, vector<Flip_value> &move_list){
    move_evaluate_fast_first(search, move_list);
    sort(move_list.begin(), move_list.end(), cmp_move_ordering);
}
