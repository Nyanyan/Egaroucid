/*
    Egaroucid Project

    @file move_ordering.hpp
        Move ordering for each search algorithm
    @date 2021-2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once
#include <iostream>
#include <vector>
#if USE_SIMD
    #ifdef _MSC_VER
        #include <intrin.h>
    #else
        #include <x86intrin.h>
    #endif
#endif
#include "setting.hpp"
#include "common.hpp"
#include "board.hpp"
#include "search.hpp"
#include "midsearch.hpp"
#include "evaluate.hpp"
#include "stability.hpp"
#include "level.hpp"
#include "transposition_table.hpp"

/*
    @brief constant
*/
constexpr int W_WIPEOUT = 100000000;
constexpr int W_1ST_MOVE = 10000000;
constexpr int W_2ND_MOVE = 1000000;
constexpr int MO_OFFSET_L_PM = 38;

/*
    @brief constants for move ordering
*/
#if TUNE_MOVE_ORDERING
    constexpr int N_MOVE_ORDERING_PARAM = 20;
    int move_ordering_param_array[N_MOVE_ORDERING_PARAM] = {
        10, 6, 3, 35, 17, 485, 269, 94, 
        5, 3, 1, 17, 204, 7, 25, 
        40, 12, 
        18, 17, 300
    };

    int W_KILLER                    = move_ordering_param_array[0];
    int W_HISTORY_MOVE              = move_ordering_param_array[1];
    int W_COUNTER_MOVE              = move_ordering_param_array[2];
    int W_MOBILITY                  = move_ordering_param_array[3];
    int W_POTENTIAL_MOBILITY        = move_ordering_param_array[4];
    int W_TT_BONUS                  = move_ordering_param_array[5];
    int W_VALUE                     = move_ordering_param_array[6];
    int W_VALUE_DEEP_ADDITIONAL     = move_ordering_param_array[7];

    int W_NWS_KILLER                = move_ordering_param_array[8];
    int W_NWS_HISTORY_MOVE          = move_ordering_param_array[9];
    int W_NWS_COUNTER_MOVE          = move_ordering_param_array[10];
    int W_NWS_MOBILITY              = move_ordering_param_array[11];
    int W_NWS_TT_BONUS              = move_ordering_param_array[12];
    int W_NWS_VALUE                 = move_ordering_param_array[13];
    int W_NWS_VALUE_DEEP_ADDITIONAL = move_ordering_param_array[14];

    int W_END_NWS_MOBILITY          = move_ordering_param_array[15];
    int W_END_NWS_VALUE             = move_ordering_param_array[16];

    int W_END_NWS_SIMPLE_MOBILITY   = move_ordering_param_array[17];
    int W_END_NWS_SIMPLE_PARITY     = move_ordering_param_array[18];
    int W_END_NWS_SIMPLE_TT_BONUS   = move_ordering_param_array[19];

    int MOVE_ORDERING_PARAM_START = 0;
    int MOVE_ORDERING_PARAM_END = 10;
#else
    // midgame search
    constexpr int W_KILLER = 8;
    constexpr int W_HISTORY_MOVE = 6;
    constexpr int W_COUNTER_MOVE = 3;
    constexpr int W_MOBILITY = 35;
    constexpr int W_POTENTIAL_MOBILITY = 17;
    constexpr int W_TT_BONUS = 485;
    constexpr int W_VALUE = 269;
    constexpr int W_VALUE_DEEP_ADDITIONAL = 94;

    // midgame null window search
    constexpr int W_NWS_KILLER = 5;
    constexpr int W_NWS_HISTORY_MOVE = 3;
    constexpr int W_NWS_COUNTER_MOVE = 1;
    constexpr int W_NWS_MOBILITY = 17;
    constexpr int W_NWS_TT_BONUS = 204;
    constexpr int W_NWS_VALUE = 7;
    constexpr int W_NWS_VALUE_DEEP_ADDITIONAL = 25;

    // endgame null window search
    constexpr int W_END_NWS_MOBILITY = 40;
    constexpr int W_END_NWS_VALUE = 12;

    // endgame simple null window search
    constexpr int W_END_NWS_SIMPLE_MOBILITY = 18;
    constexpr int W_END_NWS_SIMPLE_PARITY = 17;
    constexpr int W_END_NWS_SIMPLE_TT_BONUS = 300;
#endif

constexpr int MOVE_ORDERING_VALUE_OFFSET_ALPHA = 12;
constexpr int MOVE_ORDERING_VALUE_OFFSET_BETA = 8;
constexpr int MOVE_ORDERING_NWS_VALUE_OFFSET_ALPHA = 16;
constexpr int MOVE_ORDERING_NWS_VALUE_OFFSET_BETA = 6;

constexpr int MOVE_ORDERING_MPC_LEVEL = MPC_74_LEVEL;

int nega_alpha_eval1(Search *search, int alpha, int beta, bool skipped);
int nega_scout(Search *search, int alpha, int beta, const int depth, const bool skipped, uint64_t legal, const bool is_end_search, bool *searching);
inline bool transposition_table_get_value(Search *search, uint32_t hash, int *l, int *u);
inline int mid_evaluate_diff(Search *search);
inline int mid_evaluate_move_ordering_end(Search *search);


#if USE_SIMD
__m256i eval_surround_mask;
__m256i eval_surround_shift1879;
inline void move_ordering_init() {
    eval_surround_mask = _mm256_set_epi64x(0x7E7E7E7E7E7E7E7EULL, 0x00FFFFFFFFFFFF00ULL, 0x007E7E7E7E7E7E00ULL, 0x007E7E7E7E7E7E00ULL);
    eval_surround_shift1879 = _mm256_set_epi64x(1, HW, HW_M1, HW_P1);
}
#else
inline void move_ordering_init() {
}
#endif

/*
    @brief Get number of corner mobility

    Optimized for corner mobility

    @param legal                legal moves as a bitboard
    @return number of legal moves on corners
*/
#if USE_BUILTIN_POPCOUNT
inline int get_corner_n_moves(uint64_t legal) {
    return pop_count_ull(legal & 0x8100000000000081ULL);
}
#else
inline int get_corner_n_moves(uint64_t legal) {
    int res = (int)((legal & 0b10000001ULL) + ((legal >> 56) & 0b10000001ULL));
    return (res & 0b11) + (res >> 7);
}
#endif

/*
    @brief Get a weighted mobility

    @param legal                legal moves as a bitboard
    @return weighted mobility
*/
inline int get_weighted_n_moves(uint64_t legal) {
    return pop_count_ull(legal) * 2 + get_corner_n_moves(legal);
}

#ifdef USE_SIMD
inline int get_n_moves_cornerX2(uint64_t legal) {
    return pop_count_ull(legal) + get_corner_n_moves(legal);
}
#else
inline int get_n_moves_cornerX2(uint64_t legal) {
    uint64_t b = legal;
    uint64_t c = b & 0x0100000000000001ull;
    b -= (b >> 1) & 0x1555555555555515ull;
    b = (b & 0x3333333333333333ull) + ((b >> 2) & 0x3333333333333333ull);
    b += c;
    b += (b >> 4);
    b &= 0x0f0f0f0f0f0f0f0full;
    b *= 0x0101010101010101ULL;
    return b >> 56;
}
#endif

/*
    @brief Get potential mobility

    Same idea as surround in evaluation function

    @param discs                a bitboard representing discs
    @param empties              a bitboard representing empty squares
    @return potential mobility
*/
#if USE_SIMD
inline int get_potential_mobility(uint64_t discs, uint64_t empties) {
    __m256i pl = _mm256_set1_epi64x(discs);
    pl = _mm256_and_si256(pl, eval_surround_mask);
    pl = _mm256_or_si256(_mm256_sllv_epi64(pl, eval_surround_shift1879), _mm256_srlv_epi64(pl, eval_surround_shift1879));
    __m128i res = _mm_or_si128(_mm256_castsi256_si128(pl), _mm256_extracti128_si256(pl, 1));
    res = _mm_or_si128(res, _mm_shuffle_epi32(res, 0x4e));
    return pop_count_ull(_mm_cvtsi128_si64(res) & empties);
}
#else
inline int get_potential_mobility(uint64_t discs, uint64_t empties) {
    uint64_t hmask = discs & 0x7E7E7E7E7E7E7E7EULL;
    uint64_t vmask = discs & 0x00FFFFFFFFFFFF00ULL;
    uint64_t hvmask = discs & 0x007E7E7E7E7E7E00ULL;
    uint64_t res = 
        (hmask << 1) | (hmask >> 1) | 
        (vmask << HW) | (vmask >> HW) | 
        (hvmask << HW_M1) | (hvmask >> HW_M1) | 
        (hvmask << HW_P1) | (hvmask >> HW_P1);
    return pop_count_ull(empties & res);
}
#endif

/*
    @brief Evaluate a move in midgame

    @param search               search information
    @param flip_value           flip with value
    @param alpha                alpha value to search
    @param beta                 beta value to search
    @param depth                depth to search
    @param searching            flag for terminating this search
    @return true if wipeout found else false
*/
inline void move_evaluate(Search *search, Flip_value *flip_value, int alpha, int beta, int depth, bool *searching) {
    flip_value->value = 0;
#if USE_KILLER_MOVE_MO
    flip_value->value += search->get_killer_bonus(flip_value->flip.pos) * W_KILLER;
    int prev_pos = search->get_prev_move();
    flip_value->value += search->get_history_bonus(prev_pos, flip_value->flip.pos) * W_HISTORY_MOVE;
    flip_value->value += search->get_counter_move_bonus(prev_pos, flip_value->flip.pos) * W_COUNTER_MOVE;
#endif
    search->move(&flip_value->flip);
        flip_value->n_legal = search->board.get_legal();
        flip_value->value += (MO_OFFSET_L_PM - get_weighted_n_moves(flip_value->n_legal)) * W_MOBILITY;
        flip_value->value += (MO_OFFSET_L_PM - get_potential_mobility(search->board.opponent, ~(search->board.player | search->board.opponent))) * W_POTENTIAL_MOBILITY;
        switch (depth) {
            case 0:
                flip_value->value += (SCORE_MAX - mid_evaluate_diff(search)) * W_VALUE;
                break;
            case 1:
                flip_value->value += (SCORE_MAX - nega_alpha_eval1(search, alpha, beta, false)) * (W_VALUE + W_VALUE_DEEP_ADDITIONAL);
                break;
            default:
                //if (transposition_table.has_node_any_level(search, search->board.hash())) {
                //    flip_value->value += W_TT_BONUS;
                //}
                uint_fast8_t mpc_level = search->mpc_level;
                search->mpc_level = MOVE_ORDERING_MPC_LEVEL;
                    flip_value->value += (SCORE_MAX - nega_scout(search, alpha, beta, depth, false, flip_value->n_legal, false, searching)) * (W_VALUE + depth * W_VALUE_DEEP_ADDITIONAL);
                search->mpc_level = mpc_level;
                break;
        }
    search->undo(&flip_value->flip);
}

/*
    @brief Evaluate a move in midgame for NWS

    @param search               search information
    @param flip_value           flip with value
    @param alpha                alpha value to search
    @param beta                 beta value to search
    @param depth                depth to search
    @param searching            flag for terminating this search
    @return true if wipeout found else false
*/
inline void move_evaluate_nws(Search *search, Flip_value *flip_value, int alpha, int beta, int depth, bool *searching) {
    flip_value->value = 0;
#if USE_KILLER_MOVE_MO && USE_KILLER_MOVE_NWS_MO
    flip_value->value += search->get_killer_bonus(flip_value->flip.pos) * W_NWS_KILLER;
    int prev_pos = search->get_prev_move();
    flip_value->value += search->get_history_bonus(prev_pos, flip_value->flip.pos) * W_NWS_HISTORY_MOVE;
    flip_value->value += search->get_counter_move_bonus(prev_pos, flip_value->flip.pos) * W_NWS_COUNTER_MOVE;
#endif
    search->move(&flip_value->flip);
        flip_value->n_legal = search->board.get_legal();
        flip_value->value += (MO_OFFSET_L_PM - get_weighted_n_moves(flip_value->n_legal)) * W_NWS_MOBILITY;
        switch (depth) {
            case 0:
                flip_value->value += (SCORE_MAX - mid_evaluate_diff(search)) * W_NWS_VALUE;
                break;
            case 1:
                flip_value->value += (SCORE_MAX - nega_alpha_eval1(search, alpha, beta, false)) * (W_NWS_VALUE + W_NWS_VALUE_DEEP_ADDITIONAL);
                break;
            default:
                //if (transposition_table.has_node_any_level(search, search->board.hash())) {
                //    flip_value->value += W_NWS_TT_BONUS;
                //}
                uint_fast8_t mpc_level = search->mpc_level;
                search->mpc_level = MOVE_ORDERING_MPC_LEVEL;
                    flip_value->value += (SCORE_MAX - nega_scout(search, alpha, beta, depth, false, flip_value->n_legal, false, searching)) * (W_NWS_VALUE + depth * W_NWS_VALUE_DEEP_ADDITIONAL);
                search->mpc_level = mpc_level;
                break;
        }
    search->undo(&flip_value->flip);
}

// /*
//     @brief Evaluate a move in endgame NWS

//     @param search               search information
//     @param flip_value           flip with value
//     @return true if wipeout found else false
// */
// inline void move_evaluate_end_nws(Search *search, Flip_value *flip_value) {
//     flip_value->value = 0;
//     // flip_value->value += search->get_killer_bonus(flip_value->flip.pos);
//     search->move_endsearch(&flip_value->flip);
//         flip_value->n_legal = search->board.get_legal();
//         flip_value->value += (MO_OFFSET_L_PM - get_n_moves_cornerX2(flip_value->n_legal)) * W_END_NWS_MOBILITY;
//         flip_value->value += (MO_OFFSET_L_PM - mid_evaluate_move_ordering_end(search)) * W_END_NWS_VALUE;
//     search->undo_endsearch(&flip_value->flip);
// }

// /*
//     @brief Evaluate a move in endgame NWS (simple)

//     @param search               search information
//     @param flip_value           flip with value
//     @return true if wipeout found else false
// */
// inline void move_evaluate_end_simple_nws(Search *search, Flip_value *flip_value) {
//     flip_value->value = 0;
//     // flip_value->value += search->get_killer_bonus(flip_value->flip.pos);
//     if (search->parity & cell_div4[flip_value->flip.pos]) {
//         flip_value->value += W_END_NWS_SIMPLE_PARITY;
//     }
//     search->move_noeval(&flip_value->flip);
//         flip_value->n_legal = search->board.get_legal();
//         flip_value->value += (MO_OFFSET_L_PM - get_n_moves_cornerX2(flip_value->n_legal)) * W_END_NWS_SIMPLE_MOBILITY;
//     search->undo_noeval(&flip_value->flip);
// }

/*
    @brief Set the best move to the first element

    @param move_list            list of moves
    @param strt                 the first index
    @param siz                  the size of move_list
*/
inline void swap_next_best_move(std::vector<Flip_value> &move_list, const int strt, const int siz) {
    if (strt == siz - 1) {
        return;
    }
    int top_idx = strt;
    int best_value = move_list[strt].value;
    for (int i = strt + 1; i < siz; ++i) {
        if (best_value < move_list[i].value) {
            best_value = move_list[i].value;
            top_idx = i;
        }
    }
    if (top_idx != strt) {
        std::swap(move_list[strt], move_list[top_idx]);
    }
}

/*
    @brief Set the best move to the first element

    @param move_list            list of moves
    @param strt                 the first index
    @param siz                  the size of move_list
*/
inline void swap_next_best_move(Flip_value move_list[], const int strt, const int siz) {
    if (strt == siz - 1) {
        return;
    }
    int top_idx = strt;
    int best_value = move_list[strt].value;
    for (int i = strt + 1; i < siz; ++i) {
        if (best_value < move_list[i].value) {
            best_value = move_list[i].value;
            top_idx = i;
        }
    }
    if (top_idx != strt) {
        std::swap(move_list[strt], move_list[top_idx]);
    }
}

/*
inline bool move_list_tt_check(Search *search, std::vector<Flip_value> &move_list, uint_fast8_t moves[], int depth, int alpha, int beta, int tt_bonus, int *best_move, int *best_score) {
    bool disable_move;
    *best_score = -SCORE_INF;
    for (Flip_value &flip_value: move_list) {
#if USE_MID_ETC
        if (flip_value.flip.flip) {
#endif
            flip_value.value = 0;
            disable_move = false;
            search->board.move_board(&flip_value.flip);
                //if (transposition_table.has_node_any_level(search, search->board.hash())) {
                //    flip_value.value += W_TT_BONUS;
                //}
                int tt_v = transposition_table.has_node_any_level_cutoff(search, search->board.hash(), depth - 1, -beta, -alpha);
                if (tt_v == TRANSPOSITION_TABLE_HAS_NODE) { // node found
                    flip_value.value += tt_bonus;
                } else if (tt_v != TRANSPOSITION_TABLE_NOT_HAS_NODE) { // cutoff
                    if (alpha < -tt_v) {
                        alpha = -tt_v;
                    }
                    disable_move = true;
                }
            search->board.undo_board(&flip_value.flip);
            if (disable_move) {
                flip_value.flip.flip = 0;
                flip_value.value = -INF;
                if (*best_score < -tt_v) {
                    *best_score = -tt_v;
                    *best_move = flip_value.flip.pos;
                    if (beta <= alpha) {
                        return true;
                    }
                }
            }
#if USE_MID_ETC
        } else {
            flip_value.value = -INF;
        }
#endif
    }
    return false;
}
*/

/*
    @brief Evaluate all legal moves for midgame

    @param search               search information
    @param move_list            list of moves
    @param moves                list of moves in transposition table
    @param depth                remaining depth
    @param alpha                alpha value
    @param beta                 beta value
    @param searching            flag for terminating this search
*/
inline bool move_list_evaluate(Search *search, std::vector<Flip_value> &move_list, uint_fast8_t moves[], int depth, int alpha, int beta, bool *searching) {
    if (move_list.size() == 1) {
        return false;
    }
    int eval_alpha = -std::min(SCORE_MAX, beta + MOVE_ORDERING_VALUE_OFFSET_BETA);
    int eval_beta = -std::max(-SCORE_MAX, alpha - MOVE_ORDERING_VALUE_OFFSET_ALPHA);
    int eval_depth = depth >> 3;
    if (depth >= 25 && search->mpc_level < MPC_100_LEVEL) {
        eval_depth = ((depth / 3) & 0b11111110) + (depth & 1); // depth / 3 + parity
    }
    for (Flip_value &flip_value: move_list) {
#if USE_MID_ETC
        if (flip_value.flip.flip) {
#endif
            if (flip_value.flip.pos == moves[0]) {
                flip_value.value = W_1ST_MOVE;
            } else if (flip_value.flip.pos == moves[1]) {
                flip_value.value = W_2ND_MOVE;
            } else {
                move_evaluate(search, &flip_value, eval_alpha, eval_beta, eval_depth, searching);
            }
#if USE_MID_ETC
        }
#endif
    }
    return false;
}

/*
    @brief Evaluate all legal moves for midgame

    @param search               search information
    @param move_list            list of moves
    @param moves                list of moves in transposition table
    @param depth                remaining depth
    @param alpha                alpha value
    @param beta                 beta value
    @param searching            flag for terminating this search
*/
inline bool move_list_evaluate(Search *search, Flip_value move_list[], int canput, uint_fast8_t moves[], int depth, int alpha, int beta, bool *searching) {
    if (canput == 1) {
        return false;
    }
    int eval_alpha = -std::min(SCORE_MAX, beta + MOVE_ORDERING_VALUE_OFFSET_BETA);
    int eval_beta = -std::max(-SCORE_MAX, alpha - MOVE_ORDERING_VALUE_OFFSET_ALPHA);
    int eval_depth = depth >> 3;
    if (depth >= 25 && search->mpc_level < MPC_100_LEVEL) {
        eval_depth = ((depth / 3) & 0b11111110) + (depth & 1); // depth / 3 + parity
    }
    for (int i = 0; i < canput; ++i) {
#if USE_MID_ETC
        if (move_list[i].flip.flip) {
#endif
            if (move_list[i].flip.pos == moves[0]) {
                move_list[i].value = W_1ST_MOVE;
            } else if (move_list[i].flip.pos == moves[1]) {
                move_list[i].value = W_2ND_MOVE;
            } else {
                move_evaluate(search, &move_list[i], eval_alpha, eval_beta, eval_depth, searching);
            }
#if USE_MID_ETC
        }
#endif
    }
    return false;
}

/*
    @brief Evaluate all legal moves for midgame NWS

    @param search               search information
    @param move_list            list of moves
    @param moves                list of moves in transposition table
    @param depth                remaining depth
    @param alpha                alpha value (beta = alpha + 1)
    @param searching            flag for terminating this search
*/
inline bool move_list_evaluate_nws(Search *search, std::vector<Flip_value> &move_list, uint_fast8_t moves[], int depth, int alpha, bool *searching) {
    if (move_list.size() <= 1) {
        return false;
    }
    const int eval_alpha = -std::min(SCORE_MAX, alpha + MOVE_ORDERING_NWS_VALUE_OFFSET_BETA);
    const int eval_beta = -std::max(-SCORE_MAX, alpha - MOVE_ORDERING_NWS_VALUE_OFFSET_ALPHA);
    int eval_depth = depth >> 4;
    for (Flip_value &flip_value: move_list) {
        if (flip_value.flip.flip) {
            if (flip_value.flip.pos == moves[0]) {
                flip_value.value = W_1ST_MOVE;
            } else if (flip_value.flip.pos == moves[1]) {
                flip_value.value = W_2ND_MOVE;
            } else{
                move_evaluate_nws(search, &flip_value, eval_alpha, eval_beta, eval_depth, searching);
            }
        }
    }
    return false;
}

/*
    @brief Evaluate all legal moves for midgame NWS

    @param search               search information
    @param move_list            list of moves
    @param moves                list of moves in transposition table
    @param depth                remaining depth
    @param alpha                alpha value (beta = alpha + 1)
    @param searching            flag for terminating this search
*/
inline bool move_list_evaluate_nws(Search *search, Flip_value move_list[], int canput, uint_fast8_t moves[], int depth, int alpha, bool *searching) {
    if (canput <= 1) {
        return false;
    }
    const int eval_alpha = -std::min(SCORE_MAX, alpha + MOVE_ORDERING_NWS_VALUE_OFFSET_BETA);
    const int eval_beta = -std::max(-SCORE_MAX, alpha - MOVE_ORDERING_NWS_VALUE_OFFSET_ALPHA);
    int eval_depth = depth >> 4;
    for (int i = 0; i < canput; ++i) {
        if (move_list[i].flip.flip) {
            if (move_list[i].flip.pos == moves[0]) {
                move_list[i].value = W_1ST_MOVE;
            } else if (move_list[i].flip.pos == moves[1]) {
                move_list[i].value = W_2ND_MOVE;
            } else{
                move_evaluate_nws(search, &move_list[i], eval_alpha, eval_beta, eval_depth, searching);
            }
        }
    }
    return false;
}

// /*
//     @brief Evaluate all legal moves for endgame NWS

//     @param search               search information
//     @param move_list            list of moves
// */
// inline void move_list_evaluate_end_nws(Search *search, std::vector<Flip_value> &move_list, uint_fast8_t moves[], bool *searching) {
//     if (move_list.size() <= 1) {
//         return;
//     }
//     for (Flip_value &flip_value: move_list) {
//         if (flip_value.flip.pos == moves[0]) {
//             flip_value.value = W_1ST_MOVE;
//         } else if (flip_value.flip.pos == moves[1]) {
//             flip_value.value = W_2ND_MOVE;
//         } else{
//             move_evaluate_end_nws(search, &flip_value);
//         }
//     }
// }

// /*
//     @brief Evaluate all legal moves for endgame NWS (simple)

//     @param search               search information
//     @param move_list            list of moves
// */
// inline void move_list_evaluate_end_simple_nws(Search *search, Flip_value move_list[], const int canput) {
//     if (canput <= 1) {
//         return;
//     }
//     for (int i = 0; i < canput; ++i) {
//         move_evaluate_end_simple_nws(search, &move_list[i]);
//     }
// }

inline void move_list_sort(std::vector<Flip_value> &move_list) {
    std::sort(move_list.begin(), move_list.end(), [](Flip_value &a, Flip_value &b) { return a.value > b.value; });
}

inline void move_list_sort(Flip_value move_list[], int canput) {
    std::sort(move_list, move_list + canput, [](Flip_value &a, Flip_value &b) { return a.value > b.value; });
}

/*
    @brief Parameter tuning for move ordering
*/
#if TUNE_MOVE_ORDERING
#include "ai.hpp"
inline Search_result tree_search_legal(Board board, int alpha, int beta, int depth, uint_fast8_t mpc_level, bool show_log, uint64_t use_legal, bool use_multi_thread, uint64_t time_limit, bool *searching);

uint64_t n_nodes_test(int level, std::vector<Board> testcase_arr) {
    uint64_t n_nodes = 0;
    for (Board &board: testcase_arr) {
        int depth;
        bool is_mid_search;
        uint_fast8_t mpc_level;
        get_level(level, board.n_discs() - 4, &is_mid_search, &depth, &mpc_level);
        transposition_table.init();
        bool searching = true;
        Search_result result = tree_search_legal(board, -SCORE_MAX, SCORE_MAX, depth, mpc_level, false, board.get_legal(), true, TIME_LIMIT_INF, &searching);
        n_nodes += result.nodes;
    }
    transposition_table.reset_importance();
    return n_nodes;
}

void tune_move_ordering(int level) {
    std::cout << "please input testcase file" << std::endl;
    std::string file;
    std::cin >> file;
    std::ifstream ifs(file);
    if (ifs.fail()) {
        std::cerr << "[ERROR] [FATAL] problem file " << file << " not found" << std::endl;
        return;
    }
    std::vector<Board> testcase_arr;
    std::string line;
    while (std::getline(ifs, line)) {
        Board b(line);
        testcase_arr.emplace_back(b);
    }
    std::cerr << testcase_arr.size() << " testcases loaded" << std::endl;
    int minute = 10;
    std::cout << "please input timelimit (minute)" << std::endl;
    std::cin >> minute;
    uint64_t tl = 60ULL * 1000ULL * minute; // 10 min
    uint64_t min_n_nodes = n_nodes_test(level, testcase_arr);
    double min_percentage = 100.0;
    uint64_t first_n_nodes = min_n_nodes;
    std::cerr << "min_n_nodes " << min_n_nodes << std::endl;
    int n_updated = 0;
    int n_try = 0;
    uint64_t strt = tim();
    while (tim() - strt < tl) {
        // update parameter randomly
        int idx = myrandrange(MOVE_ORDERING_PARAM_START, MOVE_ORDERING_PARAM_END + 1); // midgame search
        int delta = myrandrange(-5, 6);
        while (delta == 0) {
            delta = myrandrange(-5, 6);
        }
        if (move_ordering_param_array[idx] + delta < 0) {
            continue;
        }
        move_ordering_param_array[idx] += delta;
        uint64_t n_nodes = n_nodes_test(level, testcase_arr);
        double percentage = 100.0 * n_nodes / first_n_nodes;

        // simulated annealing
        constexpr double start_temp = 0.1; // percent
        constexpr double end_temp = 0.0001; // percent
        double temp = start_temp + (end_temp - start_temp) * (tim() - strt) / tl;
        double prob = exp((min_percentage - percentage) / temp);
        if (prob > myrandom()) {
            min_n_nodes = n_nodes;
            min_percentage = percentage;
            ++n_updated;
        } else {
            move_ordering_param_array[idx] -= delta;
        }
        ++n_try;

        std::cerr << "try " << n_try << " updated " << n_updated << " min_n_nodes " << min_n_nodes << " n_nodes " << n_nodes << " " << min_percentage << "% " << tim() - strt << " ms ";
        for (int i = 0; i < N_MOVE_ORDERING_PARAM; ++i) {
            std::cerr << ", " << move_ordering_param_array[i];
        }
        std::cerr << std::endl;
    }
    std::cout << "done " << min_percentage << "% ";
    for (int i = 0; i < N_MOVE_ORDERING_PARAM; ++i) {
        std::cout << ", " << move_ordering_param_array[i];
    }
    std::cout << std::endl;
}
#endif