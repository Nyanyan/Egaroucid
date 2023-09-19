/*
    Egaroucid Project

    @file move_ordering.hpp
        Move ordering for each search algorithm
    @date 2021-2023
    @author Takuto Yamana
    @license GPL-3.0 license
*/

#pragma once
#include <iostream>
#include <vector>
#include "common.hpp"
#include "board.hpp"
#include "search.hpp"
#include "midsearch.hpp"
#include "stability.hpp"
#include "level.hpp"
#if TUNE_MOVE_ORDERING_END
    #include "move_ordering_tune_end.hpp"
#endif

/*
    @brief if wipeout found, it must be searched first.
*/
// #define W_WIPEOUT INF
#define W_1ST_MOVE 10000000
#define W_2ND_MOVE 1000000

/*
    @brief constants for move ordering
*/
#define MOVE_ORDERING_VALUE_OFFSET_ALPHA 10
#define MOVE_ORDERING_VALUE_OFFSET_BETA 10
#define MOVE_ORDERING_NWS_VALUE_OFFSET_ALPHA 10
#define MOVE_ORDERING_NWS_VALUE_OFFSET_BETA 3
#define MOVE_ORDERING_MPC_LEVEL MPC_88_LEVEL

#define W_END_MOBILITY 32
#define W_END_PARITY 4
#define W_END_VALUE 4

#define N_CELL_TYPES 10
#define MAX_SAME_CELL_TYPE 8
#define MOVE_ORDERING_CELL_SCORE_MAX 256

#define W_CELL_WEIGHT 7
#define W_MOBILITY -14
//#define W_POTENTIAL_MOBILITY -28
#define W_VALUE -77
#define W_VALUE_DEEP_ADDITIONAL -218

#define W_NWS_MOBILITY -14
//#define W_NWS_POTENTIAL_MOBILITY -8
#define W_NWS_VALUE -16
#define W_NWS_VALUE_SHALLOW -14

/*
    @brief Flip structure with more information

    @param flip                 flip information
    @param value                the move ordering value
    @param n_legal              next legal moves as a bitboard for reusing
*/
struct Flip_value{
    Flip flip;
    int value;
    uint64_t n_legal;

    Flip_value(){
        n_legal = LEGAL_UNDEFINED;
    }

    bool operator<(const Flip_value &another) const{
        return value < another.value;
    }

    bool operator>(const Flip_value &another) const{
        return value > another.value;
    }
};

int nega_alpha_eval1(Search *search, int alpha, int beta, bool skipped, const bool *searching);
#if MID_FAST_DEPTH > 1
    int nega_alpha(Search *search, int alpha, int beta, int depth, bool skipped, const bool *searching);
#endif
int nega_scout(Search *search, int alpha, int beta, int depth, bool skipped, uint64_t legal, bool is_end_search, const bool *searching);

/*
    @brief Get number of corner mobility

    Optimized for corner mobility

    @param legal                legal moves as a bitboard
    @return number of legal moves on corners
*/
inline int get_corner_mobility(uint64_t legal){
    int res = (int)((legal & 0b10000001ULL) + ((legal >> 56) & 0b10000001ULL));
    return (res & 0b11) + (res >> 7);
}

/*
    @brief Get a weighted mobility

    @param legal                legal moves as a bitboard
    @return weighted mobility
*/
inline int get_weighted_n_moves(uint64_t legal){
    return pop_count_ull(legal) * 2 + get_corner_mobility(legal);
}

/*
    @brief Get potential mobility

    Same idea as surround in evaluation function

    @param opponent             a bitboard representing opponent
    @param empties              a bitboard representing empty squares
    @return potential mobility
*/
/*
#if USE_SIMD
    inline int get_potential_mobility(uint64_t opponent, uint64_t empties){
        const __m256i eval_surround_mask = _mm256_set_epi64x(0x7E7E7E7E7E7E7E7EULL, 0x00FFFFFFFFFFFF00ULL, 0x007E7E7E7E7E7E00ULL, 0x007E7E7E7E7E7E00ULL);
        const __m128i eval_surround_shift1879 = _mm_set_epi32(1, HW, HW_M1, HW_P1);
        __m256i pl = _mm256_set1_epi64x(opponent);
        pl = _mm256_and_si256(pl, eval_surround_mask);
        pl = _mm256_or_si256(_mm256_sll_epi64(pl, eval_surround_shift1879), _mm256_srl_epi64(pl, eval_surround_shift1879));
        __m128i res = _mm_or_si128(_mm256_castsi256_si128(pl), _mm256_extracti128_si256(pl, 1));
        res = _mm_or_si128(res, _mm_shuffle_epi32(res, 0x4e));
        return pop_count_ull(_mm_cvtsi128_si64(res));
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
*/

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
inline void move_evaluate(Search *search, Flip_value *flip_value, int alpha, int beta, int depth, const bool *searching){
    flip_value->value = cell_weight[flip_value->flip.pos] * W_CELL_WEIGHT;
    eval_move(search, &flip_value->flip);
    search->move(&flip_value->flip);
        flip_value->n_legal = search->board.get_legal();
        flip_value->value += get_weighted_n_moves(flip_value->n_legal) * W_MOBILITY;
        //uint64_t empties = ~(search->board.player | search->board.opponent);
        //flip_value->value += get_potential_mobility(search->board.player, empties) * W_POTENTIAL_MOBILITY;
        switch (depth){
            case 0:
                flip_value->value += mid_evaluate_diff(search) * W_VALUE;
                break;
            case 1:
                flip_value->value += nega_alpha_eval1(search, alpha, beta, false, searching) * (W_VALUE + W_VALUE_DEEP_ADDITIONAL);
                break;
            default:
                #if MID_FAST_DEPTH > 1
                    if (depth <= MID_FAST_DEPTH)
                        flip_value->value += nega_alpha(search, alpha, beta, depth, false, searching) * (W_VALUE + depth * W_VALUE_DEEP_ADDITIONAL);
                    else{
                        uint_fast8_t mpc_level = search->mpc_level;
                        search->mpc_level = MOVE_ORDERING_MPC_LEVEL;
                        flip_value->value += nega_scout(search, alpha, beta, depth, false, flip_value->n_legal, false, searching) * (W_VALUE + depth * W_VALUE_DEEP_ADDITIONAL);
                        search->mpc_level = mpc_level;
                    }
                #else
                    uint_fast8_t mpc_level = search->mpc_level;
                    search->mpc_level = MOVE_ORDERING_MPC_LEVEL;
                    flip_value->value += nega_scout(search, alpha, beta, depth, false, flip_value->n_legal, false, searching) * (W_VALUE + depth * W_VALUE_DEEP_ADDITIONAL);
                    search->mpc_level = mpc_level;
                #endif
                break;
        }
    search->undo(&flip_value->flip);
    eval_undo(search, &flip_value->flip);
}

/*
    @brief Evaluate a move in endgame

    @param search               search information
    @param flip_value           flip with value
    @return true if wipeout found else false
*/
inline void move_evaluate_end(Search *search, Flip_value *flip_value){
    flip_value->value = cell_weight[flip_value->flip.pos];
    if (search->parity & cell_div4[flip_value->flip.pos])
        flip_value->value += W_END_PARITY;
    search->move(&flip_value->flip);
        flip_value->n_legal = search->board.get_legal();
        flip_value->value -= pop_count_ull(flip_value->n_legal) * W_END_MOBILITY;
    search->undo(&flip_value->flip);
}

/*
    @brief Evaluate a move in endgame

    @param search               search information
    @param flip_value           flip with value
    @return true if wipeout found else false
*/
inline void move_evaluate_end_nws(Search *search, Flip_value *flip_value){
    flip_value->value = cell_weight[flip_value->flip.pos];
    if (search->parity & cell_div4[flip_value->flip.pos])
        flip_value->value += W_END_PARITY;
    search->move(&flip_value->flip);
        flip_value->n_legal = search->board.get_legal();
        flip_value->value -= pop_count_ull(flip_value->n_legal) * W_END_MOBILITY;
    search->undo(&flip_value->flip);
}

/*
    @brief Evaluate a move in endgame with eval

    @param search               search information
    @param flip_value           flip with value
    @return true if wipeout found else false
*/
inline void move_evaluate_end_nws_eval(Search *search, Flip_value *flip_value){
    flip_value->value = cell_weight[flip_value->flip.pos];
    if (search->parity & cell_div4[flip_value->flip.pos])
        flip_value->value += W_END_PARITY;
    eval_move(search, &flip_value->flip);
    search->move(&flip_value->flip);
        flip_value->n_legal = search->board.get_legal();
        flip_value->value -= pop_count_ull(flip_value->n_legal) * W_END_MOBILITY;
        flip_value->value -= mid_evaluate_diff(search) * W_END_VALUE;
    search->undo(&flip_value->flip);
    eval_undo(search, &flip_value->flip);
}

/*
    @brief Evaluate a move in endgame

    @param search               search information
    @param flip_value           flip with value
    @return true if wipeout found else false
*/
inline void move_evaluate_end_simple_nws(Search *search, Flip_value *flip_value){
    flip_value->value = cell_weight[flip_value->flip.pos];
    if (search->parity & cell_div4[flip_value->flip.pos])
        flip_value->value += W_END_PARITY;
    search->move(&flip_value->flip);
        flip_value->n_legal = search->board.get_legal();
        flip_value->value -= pop_count_ull(flip_value->n_legal) * W_END_MOBILITY;
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
inline void move_evaluate_nws(Search *search, Flip_value *flip_value, int alpha, int beta, int depth, const bool *searching){
    flip_value->value = cell_weight[flip_value->flip.pos];
    eval_move(search, &flip_value->flip);
    search->move(&flip_value->flip);
        flip_value->n_legal = search->board.get_legal();
        flip_value->value += get_weighted_n_moves(flip_value->n_legal) * W_NWS_MOBILITY;
        //uint64_t empties = ~(search->board.player | search->board.opponent);
        //flip_value->value += get_potential_mobility(search->board.player, empties) * W_NWS_POTENTIAL_MOBILITY;
        if (depth == 0)
            flip_value->value += mid_evaluate_diff(search) * W_NWS_VALUE_SHALLOW;
        else
            flip_value->value += nega_alpha_eval1(search, alpha, beta, false, searching) * W_NWS_VALUE;
    search->undo(&flip_value->flip);
    eval_undo(search, &flip_value->flip);
}

/*
    @brief Set the best move to the first element

    @param move_list            list of moves
    @param strt                 the first index
    @param siz                  the size of move_list
*/
inline void swap_next_best_move(std::vector<Flip_value> &move_list, const int strt, const int siz){
    if (strt == siz - 1)
        return;
    int top_idx = strt;
    int best_value = move_list[strt].value;
    for (int i = strt + 1; i < siz; ++i){
        if (best_value < move_list[i].value){
            best_value = move_list[i].value;
            top_idx = i;
        }
    }
    if (top_idx != strt)
        std::swap(move_list[strt], move_list[top_idx]);
}

/*
    @brief Set the best move to the first element

    @param move_list            list of moves
    @param strt                 the first index
    @param siz                  the size of move_list
*/
inline void swap_next_best_move(Flip_value move_list[], const int strt, const int siz){
    if (strt == siz - 1)
        return;
    int top_idx = strt;
    int best_value = move_list[strt].value;
    for (int i = strt + 1; i < siz; ++i){
        if (best_value < move_list[i].value){
            best_value = move_list[i].value;
            top_idx = i;
        }
    }
    if (top_idx != strt)
        std::swap(move_list[strt], move_list[top_idx]);
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
inline void move_list_evaluate(Search *search, std::vector<Flip_value> &move_list, uint_fast8_t moves[], int depth, int alpha, int beta, const bool *searching){
    if (move_list.size() == 1)
        return;
    int eval_alpha = -std::min(SCORE_MAX, beta + MOVE_ORDERING_VALUE_OFFSET_BETA);
    int eval_beta = -std::max(-SCORE_MAX, alpha - MOVE_ORDERING_VALUE_OFFSET_ALPHA);
    //int phase = get_move_ordering_phase(search->n_discs);
    int eval_depth = depth >> 3;
    if (depth >= 16)
        eval_depth += (depth - 14) >> 1;
    for (Flip_value &flip_value: move_list){
        #if USE_MID_ETC
            if (flip_value.flip.flip){
                if (flip_value.flip.pos == moves[0])
                    flip_value.value = W_1ST_MOVE;
                else if (flip_value.flip.pos == moves[1])
                    flip_value.value = W_2ND_MOVE;
                else
                    move_evaluate(search, &flip_value, eval_alpha, eval_beta, eval_depth, searching);
            } else
                flip_value.value = -INF;
        #else
            if (flip_value.flip.pos == moves[0])
                flip_value.value = W_1ST_MOVE;
            else if (flip_value.flip.pos == moves[1])
                flip_value.value = W_2ND_MOVE;
            else
                move_evaluate(search, &flip_value, eval_alpha, eval_beta, eval_depth, searching);
        #endif
    }
}

/*
    @brief Evaluate all legal moves for midgame

    @param search               search information
    @param move_list            list of moves
    @param depth                remaining depth
    @param alpha                alpha value
    @param beta                 beta value
    @param searching            flag for terminating this search
*/
inline void move_list_evaluate(Search *search, std::vector<Flip_value> &move_list, int depth, int alpha, int beta, const bool *searching){
    if (move_list.size() == 1)
        return;
    int eval_alpha = -std::min(SCORE_MAX, beta + MOVE_ORDERING_VALUE_OFFSET_BETA);
    int eval_beta = -std::max(-SCORE_MAX, alpha - MOVE_ORDERING_VALUE_OFFSET_ALPHA);
    //int phase = get_move_ordering_phase(search->n_discs);
    int eval_depth = depth >> 3;
    if (depth >= 16)
        eval_depth += (depth - 14) >> 1;
    for (Flip_value &flip_value: move_list){
        #if USE_MID_ETC
            if (flip_value.flip.flip)
                move_evaluate(search, &flip_value, eval_alpha, eval_beta, eval_depth, searching);
            else
                flip_value.value = -INF;
        #else
            move_evaluate(search, &flip_value, eval_alpha, eval_beta, eval_depth, searching);
        #endif
    }
}

#if USE_NEGA_ALPHA_END
    /*
        @brief Evaluate all legal moves for endgame

        @param search               search information
        @param move_list            list of moves
    */
    inline void move_list_evaluate_end(Search *search, std::vector<Flip_value> &move_list, const int canput){
        if (canput == 1)
            return;
        for (Flip_value &flip_value: move_list)
            move_evaluate_end(search, &flip_value);
    }
#endif

/*
    @brief Evaluate all legal moves for endgame

    @param search               search information
    @param move_list            list of moves
*/
inline void move_list_evaluate_end_simple_nws(Search *search, Flip_value move_list[], const int canput){
    if (canput == 1)
        return;
    for (int i = 0; i < canput; ++i)
        move_evaluate_end_simple_nws(search, &move_list[i]);
}

/*
    @brief Evaluate all legal moves for endgame

    @param search               search information
    @param move_list            list of moves
*/
inline void move_list_evaluate_end_nws(Search *search, std::vector<Flip_value> &move_list, const int canput, uint_fast8_t moves[], bool use_eval){
    if (canput == 1)
        return;
    if (use_eval){
        for (Flip_value &flip_value: move_list){
            if (flip_value.flip.pos == moves[0])
                flip_value.value = W_1ST_MOVE;
            else if (flip_value.flip.pos == moves[1])
                flip_value.value = W_2ND_MOVE;
            else
                move_evaluate_end_nws_eval(search, &flip_value);
        }
    } else{
        for (Flip_value &flip_value: move_list){
            if (flip_value.flip.pos == moves[0])
                flip_value.value = W_1ST_MOVE;
            else if (flip_value.flip.pos == moves[1])
                flip_value.value = W_2ND_MOVE;
            else
                move_evaluate_end_nws(search, &flip_value);
        }
    }
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
inline void move_list_evaluate_nws(Search *search, std::vector<Flip_value> &move_list, uint_fast8_t moves[], int depth, int alpha, const bool *searching){
    if (move_list.size() == 1)
        return;
    const int eval_alpha = -std::min(SCORE_MAX, alpha + MOVE_ORDERING_NWS_VALUE_OFFSET_BETA);
    const int eval_beta = -std::max(-SCORE_MAX, alpha - MOVE_ORDERING_NWS_VALUE_OFFSET_ALPHA);
    int eval_depth = depth >> 4;
    for (Flip_value &flip_value: move_list){
        #if USE_MID_ETC
            if (flip_value.flip.flip){
                if (flip_value.flip.pos == moves[0])
                    flip_value.value = W_1ST_MOVE;
                else if (flip_value.flip.pos == moves[1])
                    flip_value.value = W_2ND_MOVE;
                else
                    move_evaluate_nws(search, &flip_value, eval_alpha, eval_beta, eval_depth, searching);
            } else
                flip_value.value = -INF;
        #else
            if (flip_value.flip.pos == moves[0])
                flip_value.value = W_1ST_MOVE;
            else if (flip_value.flip.pos == moves[1])
                flip_value.value = W_2ND_MOVE;
            else
                move_evaluate_nws(search, &flip_value, eval_alpha, eval_beta, eval_depth, searching);
        #endif
    }
}

#if MID_FAST_NWS_DEPTH > 1
    /*
        @brief Evaluate all legal moves for midgame NWS

        @param search               search information
        @param move_list            list of moves
        @param moves                list of moves in transposition table
        @param depth                remaining depth
        @param alpha                alpha value (beta = alpha + 1)
        @param searching            flag for terminating this search
    */
    inline void move_list_evaluate_nws_fast(Search *search, std::vector<Flip_value> &move_list, uint_fast8_t moves[], int depth, int alpha, const bool *searching){
        if (move_list.size() == 1)
            return;
        const int eval_alpha = -std::min(SCORE_MAX, alpha + MOVE_ORDERING_NWS_VALUE_OFFSET_BETA);
        const int eval_beta = -std::max(-SCORE_MAX, alpha - MOVE_ORDERING_NWS_VALUE_OFFSET_ALPHA);
        int eval_depth = depth >> 4;
        for (Flip_value &flip_value: move_list){
            if (flip_value.flip.pos == moves[0])
                flip_value.value = W_1ST_MOVE;
            else if (flip_value.flip.pos == moves[1])
                flip_value.value = W_2ND_MOVE;
            else
                move_evaluate_nws(search, &flip_value, eval_alpha, eval_beta, eval_depth, searching);
        }
    }
#endif
