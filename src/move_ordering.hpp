/*
    Egaroucid Project

    @date 2021-2022
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

#define W_VALUE_TT 6
#define W_VALUE_DEEP 12
#define W_VALUE 10
#define W_VALUE_SHALLOW 8
#define W_MOBILITY 8
#define W_PLAYER_POTENTIAL_MOBILITY 8
#define W_OPPONENT_POTENTIAL_MOBILITY 10
//#define W_OPENNESS 1

#define W_NWS_MOBILITY 14
#define W_NWS_N_FLIP 2
#define W_NWS_VALUE_DEEP 16
#define W_NWS_VALUE 14
#define W_NWS_VALUE_SHALLOW 12
#define W_NWS_N_NODES 2
#define W_NWS_PLAYER_POTENTIAL_MOBILITY 8
#define W_NWS_OPPONENT_POTENTIAL_MOBILITY 10

#define W_END_MOBILITY 16
#define W_END_PARITY 8

#define MOVE_ORDERING_VALUE_OFFSET_ALPHA 10
#define MOVE_ORDERING_VALUE_OFFSET_BETA 10
#define MOVE_ORDERING_NWS_VALUE_OFFSET_ALPHA 10
#define MOVE_ORDERING_NWS_VALUE_OFFSET_BETA 3

#define MOVE_ORDERING_MPCT 0.7

#define N_MOVE_ORDERING_WEIGHT 3
#define N_MOVE_ORDERING_DEPTH 12
#define N_MOVE_ORDERING_PHASES 6
#define N_MOVE_ORDERING_PHASE_DISCS 10

constexpr int move_ordering_weights[N_MOVE_ORDERING_PHASES][N_MOVE_ORDERING_DEPTH][N_MOVE_ORDERING_WEIGHT] = {
    {{-62, 58, -178}, {-10, 5, -40}, {-63, 30, -116}, {18, -26, -270}, {-28, 62, -145}, {-17, 73, -174}, {-76, 164, 1}, {-49, 45, -173}, {29, -12, -184}, {-24, 0, -115}, {18, 11, -225}, {-21, 0, -101}}, 
    {{-61, 22, -115}, {-52, 12, -140}, {-19, 10, -139}, {-52, 5, -143}, {-26, 25, -95}, {-3, 2, -49}, {-18, 17, -151}, {-19, 5, -159}, {-3, 4, -88}, {-9, 3, -119}, {-7, 1, -66}, {-14, 1, -164}},
    {{-24, 5, -60}, {-23, 0, -134}, {-34, 46, -121}, {7, 7, -114}, {9, 9, -86}, {-43, 0, -154}, {-12, -3, -85}, {-28, -21, -122}, {-10, 11, -126}, {-12, 3, -141}, {7, 14, -85}, {-13, 2, -130}}, 
    {{32, -13, -153}, {-6, -3, -107}, {21, 13, -177}, {-5, 3, -74}, {-15, 8, -95}, {-13, 2, -132}, {-6, 6, -94}, {-21, 23, -120}, {-16, 10, -96}, {-17, 21, -156}, {0, 7, -47}, {-13, 28, -131}},
    {{-26, 34, -86}, {-45, 9, -210}, {-24, 32, -59}, {-37, 2, -137}, {-32, 12, -170}, {0, 15, -29}, {-19, 63, -155}, {0, 33, -103}, {-45, 39, -137}, {-76, 61, -188}, {0, 38, -139}, {0, 45, -183}}, 
    {{-38, -19, -53}, {-69, -21, -139}, {-22, -11, -139}, {-44, -11, -101}, {-12, -2, -19}, {-58, -12, -142}, {-39, -9, -144}, {-67, -13, -130}, {-21, -4, -58}, {-128, -29, -140}, {-34, -2, -91}, {-53, -10, -136}}
};

struct Flip_value{
    Flip flip;
    int value;
    uint64_t n_legal;
    Flip_value(){
        n_legal = LEGAL_UNDEFINED;
    }
};

inline int get_move_ordering_phase(int n_discs){
    return (n_discs - 4) / N_MOVE_ORDERING_PHASE_DISCS;
}

int nega_alpha_eval1(Search *search, int alpha, int beta, bool skipped, const bool *searching);
#if MID_FAST_DEPTH > 1
    int nega_alpha(Search *search, int alpha, int beta, int depth, bool skipped, const bool *searching);
#endif
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
    int res = (int)((legal & 0b10000001ULL) + ((legal >> 56) & 0b10000001ULL));
    return (res & 0b11) + (res >> 7);
}

inline int get_weighted_n_moves(uint64_t legal){
    return pop_count_ull(legal) * 2 + get_corner_mobility(legal);
}

#if USE_SIMD
    inline int get_potential_mobility(uint64_t opponent, uint64_t empties){
        const u64_4 shift(1, HW, HW_M1, HW_P1);
        const u64_4 mask(0x7E7E7E7E7E7E7E7EULL, 0x00FFFFFFFFFFFF00ULL, 0x007E7E7E7E7E7E00ULL, 0x007E7E7E7E7E7E00ULL);
        u64_4 op(opponent);
        op = op & mask;
        return pop_count_ull(empties & all_or((op << shift) | (op >> shift)));
    }
#else
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
#endif
/*
inline bool move_evaluate(Search *search, Flip_value *flip_value, int alpha, int beta, int depth, const bool *searching){
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
        switch (depth){
            case 0:
                flip_value->value += -mid_evaluate_diff(search) * W_VALUE_SHALLOW;
                break;
            case 1:
                flip_value->value += -nega_alpha_eval1(search, alpha, beta, false, searching) * W_VALUE;
                break;
            default:
                #if MID_FAST_DEPTH > 1
                    if (depth <= MID_FAST_DEPTH)
                        flip_value->value += -nega_alpha(search, alpha, beta, depth, false, searching) * (W_VALUE_DEEP + (depth - 1) * 2);
                    else{
                        bool use_mpc = search->use_mpc;
                        double mpct = search->mpct;
                        search->use_mpc = true;
                        search->mpct = MOVE_ORDERING_MPCT;
                        flip_value->value += -nega_scout(search, alpha, beta, depth, false, flip_value->n_legal, false, searching) * (W_VALUE_DEEP + (depth - 1) * 2);
                        search->use_mpc = use_mpc;
                        search->mpct = mpct;
                    }
                #else
                    bool use_mpc = search->use_mpc;
                    double mpct = search->mpct;
                    search->use_mpc = true;
                    search->mpct = MOVE_ORDERING_MPCT;
                    flip_value->value += -nega_scout(search, alpha, beta, depth, false, flip_value->n_legal, false, searching) * (W_VALUE_DEEP + (depth - 1) * 2);
                    search->use_mpc = use_mpc;
                    search->mpct = mpct;
                #endif
                break;
        }
    search->undo(&flip_value->flip);
    eval_undo(search, &flip_value->flip);
    return false;
}
*/
inline bool move_evaluate(Search *search, Flip_value *flip_value, int alpha, int beta, int depth, const bool *searching, const int phase){
    if (flip_value->flip.flip == search->board.opponent){
        flip_value->value = W_WIPEOUT;
        return true;
    }
    eval_move(search, &flip_value->flip);
    search->move(&flip_value->flip);
        flip_value->n_legal = calc_legal(search->board.player, search->board.opponent);
        flip_value->value = get_weighted_n_moves(flip_value->n_legal) * move_ordering_weights[phase][depth][0];
        flip_value->value += get_weighted_n_moves(calc_legal(search->board.opponent, search->board.player)) * move_ordering_weights[phase][depth][1];
        switch (depth){
            case 0:
                flip_value->value += mid_evaluate_diff(search) * move_ordering_weights[phase][0][2];
                break;
            case 1:
                flip_value->value += nega_alpha_eval1(search, alpha, beta, false, searching) * move_ordering_weights[phase][1][2];
                break;
            default:
                #if MID_FAST_DEPTH > 1
                    if (depth <= MID_FAST_DEPTH)
                        flip_value->value += nega_alpha(search, alpha, beta, depth, false, searching) * move_ordering_weights[phase][depth][2];
                    else{
                        bool use_mpc = search->use_mpc;
                        double mpct = search->mpct;
                        search->use_mpc = true;
                        search->mpct = MOVE_ORDERING_MPCT;
                            flip_value->value += nega_scout(search, alpha, beta, depth, false, flip_value->n_legal, false, searching) * move_ordering_weights[phase][depth][2];
                        search->use_mpc = use_mpc;
                        search->mpct = mpct;
                    }
                #else
                    bool use_mpc = search->use_mpc;
                    double mpct = search->mpct;
                    search->use_mpc = true;
                    search->mpct = MOVE_ORDERING_MPCT;
                        flip_value->value += nega_scout(search, alpha, beta, depth, false, flip_value->n_legal, false, searching) * move_ordering_weights[phase][depth][2];
                    search->use_mpc = use_mpc;
                    search->mpct = mpct;
                #endif
                break;
        }
    search->undo(&flip_value->flip);
    eval_undo(search, &flip_value->flip);
    return false;
}

inline bool move_evaluate_end(Search *search, Flip_value *flip_value){
    if (flip_value->flip.flip == search->board.opponent){
        flip_value->value = W_WIPEOUT;
        return true;
    }
    flip_value->value = cell_weight[flip_value->flip.pos];
    if (search->parity & cell_div4[flip_value->flip.pos])
        flip_value->value += W_END_PARITY;
    search->move(&flip_value->flip);
        flip_value->n_legal = search->board.get_legal();
        flip_value->value -= pop_count_ull(flip_value->n_legal) * W_END_MOBILITY;
    search->undo(&flip_value->flip);
    return false;
}

inline bool move_evaluate_nws(Search *search, Flip_value *flip_value, int alpha, int beta, int depth, const bool *searching){
    if (flip_value->flip.flip == search->board.opponent){
        flip_value->value = W_WIPEOUT;
        return true;
    }
    flip_value->value = cell_weight[flip_value->flip.pos];
    flip_value->value -= pop_count_ull(flip_value->flip.flip) * W_NWS_N_FLIP;
    eval_move(search, &flip_value->flip);
    search->move(&flip_value->flip);
        uint64_t empties = ~(search->board.player | search->board.opponent);
        flip_value->value -= get_potential_mobility(search->board.opponent, empties) * W_NWS_OPPONENT_POTENTIAL_MOBILITY;
        flip_value->value += get_potential_mobility(search->board.player, empties) * W_NWS_PLAYER_POTENTIAL_MOBILITY;
        flip_value->n_legal = search->board.get_legal();
        //flip_value->value -= pop_count_ull(flip_value->n_legal) * W_NWS_MOBILITY;
        flip_value->value -= get_weighted_n_moves(flip_value->n_legal) * W_NWS_MOBILITY;
        //int64_t bef_n_nodes = search->n_nodes;
        switch (depth){
            case 0:
                flip_value->value += -mid_evaluate_diff(search) * W_NWS_VALUE_SHALLOW;
                break;
            default:
                flip_value->value += -nega_alpha_eval1(search, alpha, beta, false, searching) * W_NWS_VALUE;
                //flip_value->value -= (search->n_nodes - bef_n_nodes) * W_NWS_N_NODES;
                break;
            /*
            case 1:
                flip_value->value += -nega_alpha_eval1(search, alpha, beta, false, searching) * W_NWS_VALUE;
                //flip_value->value -= (search->n_nodes - bef_n_nodes) * W_NWS_N_NODES;
                break;
            default:
                #if MID_FAST_DEPTH > 1
                    if (depth <= MID_FAST_DEPTH)
                        flip_value->value += -nega_alpha(search, alpha, beta, depth, false, searching) * (W_NWS_VALUE_DEEP + (depth - 1) * 2);
                    else
                        flip_value->value += -nega_scout(search, alpha, beta, depth, false, flip_value->n_legal, false, searching) * (W_NWS_VALUE_DEEP + (depth - 1) * 2);
                #else
                    flip_value->value += -nega_scout(search, alpha, beta, depth, false, flip_value->n_legal, false, searching) * (W_NWS_VALUE_DEEP + (depth - 1) * 2);
                #endif
                //flip_value->value -= (search->n_nodes - bef_n_nodes) / W_NWS_N_NODES;
                break;
            */
        }
    search->undo(&flip_value->flip);
    eval_undo(search, &flip_value->flip);
    return false;
}

inline void swap_next_best_move(vector<Flip_value> &move_list, const int strt, const int siz){
    int top_idx = strt;
    int best_value = move_list[strt].value;
    for (int i = strt + 1; i < siz; ++i){
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
    if (move_list.size() == 1)
        return;
    int eval_alpha = -min(SCORE_MAX, beta + MOVE_ORDERING_VALUE_OFFSET_BETA);
    int eval_beta = -max(-SCORE_MAX, alpha - MOVE_ORDERING_VALUE_OFFSET_ALPHA);
    int phase = get_move_ordering_phase(search->n_discs);
    int eval_depth = depth >> 3;
    if (depth >= 18)
        eval_depth += (depth - 16) >> 1;
    eval_depth = min(eval_depth, N_MOVE_ORDERING_DEPTH - 1);
    bool wipeout_found = false;
    for (Flip_value &flip_value: move_list){
        if (!wipeout_found)
            wipeout_found = move_evaluate(search, &flip_value, eval_alpha, eval_beta, eval_depth, searching, phase);
        else
            flip_value.value = -INF;
    }
}

inline void move_ordering(Search *search, vector<Flip_value> &move_list, int depth, int alpha, int beta, bool is_end_search, const bool *searching){
    move_list_evaluate(search, move_list, depth, alpha, beta, is_end_search, searching);
    sort(move_list.begin(), move_list.end(), cmp_move_ordering);
}

inline void move_list_evaluate_end(Search *search, vector<Flip_value> &move_list){
    if (move_list.size() == 1)
        return;
    bool wipeout_found = false;
    for (Flip_value &flip_value: move_list){
        if (!wipeout_found)
            wipeout_found = move_evaluate_end(search, &flip_value);
        else
            flip_value.value = -INF;
    }
}

inline void move_ordering_end(Search *search, vector<Flip_value> &move_list){
    move_list_evaluate_end(search, move_list);
    sort(move_list.begin(), move_list.end(), cmp_move_ordering);
}

inline void move_list_evaluate_nws(Search *search, vector<Flip_value> &move_list, int depth, int alpha, bool is_end_search, const bool *searching){
    if (move_list.size() == 1)
        return;
    const int eval_alpha = -min(SCORE_MAX, alpha + MOVE_ORDERING_NWS_VALUE_OFFSET_BETA);
    const int eval_beta = -max(-SCORE_MAX, alpha - MOVE_ORDERING_NWS_VALUE_OFFSET_ALPHA);
    const int eval_depth = depth >> 4;
    bool wipeout_found = false;
    for (Flip_value &flip_value: move_list){
        if (!wipeout_found)
            wipeout_found = move_evaluate_nws(search, &flip_value, eval_alpha, eval_beta, eval_depth, searching);
        else
            flip_value.value = -INF;
    }
}
