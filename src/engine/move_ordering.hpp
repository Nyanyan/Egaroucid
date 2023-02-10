/*
    Egaroucid Project

    @file move_ordering.hpp
        Move ordering for each search algorithm
    @date 2021-2023
    @author Takuto Yamana (a.k.a. Nyanyan)
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
#define W_WIPEOUT INF

/*
    @brief constants for move ordering
*/
#define MOVE_ORDERING_VALUE_OFFSET_ALPHA 10
#define MOVE_ORDERING_VALUE_OFFSET_BETA 10
#define MOVE_ORDERING_NWS_VALUE_OFFSET_ALPHA 10
#define MOVE_ORDERING_NWS_VALUE_OFFSET_BETA 3
#define MOVE_ORDERING_MPC_LEVEL MPC_95_LEVEL
#define W_END_MOBILITY 16
#define W_END_PARITY 8

// 5 -10 -20 -70 -25
// 4 -8 -16 -48 -157
// 7 -14 -28 -77 -218
#define W_CELL_WEIGHT 7
#define W_MOBILITY -14
#define W_POTENTIAL_MOBILITY -28
#define W_VALUE -77
#define W_VALUE_DEEP_ADDITIONAL -218

#define W_NWS_MOBILITY -14
#define W_NWS_POTENTIAL_MOBILITY -8
#define W_NWS_VALUE -16
#define W_NWS_VALUE_SHALLOW -14

#define N_MOVE_ORDERING_END_PATTERN_PARAMS 196830
#define N_MOVE_ORDERING_END_PATTERNS 4
#define N_MOVE_ORDERING_END_SYMMETRY_PATTERNS 16
#define N_MOVE_ORDERING_END_CELL_TYPES 10
#define N_MOVE_ORDERING_END_LEGAL 16
#define N_MOVE_ORDERING_END_POTENTIAL_MOBILITY 16

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

constexpr uint_fast8_t cell_types[HW2] = {
    0, 1, 2, 3, 3, 2, 1, 0, 
    1, 4, 5, 6, 6, 5, 4, 1, 
    2, 5, 7, 8, 8, 7, 5, 2, 
    3, 6, 8, 9, 9, 8, 6, 3, 
    3, 6, 8, 9, 9, 8, 6, 3, 
    2, 5, 7, 8, 8, 7, 5, 2, 
    1, 4, 5, 6, 6, 5, 4, 1, 
    0, 1, 2, 3, 3, 2, 1, 0
};

int16_t move_ordering_end_score_cell_types[N_MOVE_ORDERING_END_CELL_TYPES];
int16_t move_ordering_end_score_parity[2];
int16_t move_ordering_end_score_free_odd[2];
int16_t move_ordering_end_score_n_legal[N_MOVE_ORDERING_END_LEGAL];
int16_t move_ordering_end_score_potential_mobility[N_MOVE_ORDERING_END_POTENTIAL_MOBILITY];
int16_t move_ordering_end_pattern_arr[2][N_MOVE_ORDERING_END_PATTERN_PARAMS + 2];

int nega_alpha_eval1(Search *search, int alpha, int beta, bool skipped, const bool *searching);
#if MID_FAST_DEPTH > 1
    int nega_alpha(Search *search, int alpha, int beta, int depth, bool skipped, const bool *searching);
#endif
int nega_scout(Search *search, int alpha, int beta, int depth, bool skipped, uint64_t legal, bool is_end_search, const bool *searching);

void init_move_ordering_pattern_arr_rev(int pattern_idx, int siz, int strt){
    int ri;
    for (int i = 0; i < (int)pow3[siz]; ++i){
        ri = swap_player_idx(i, siz);
        move_ordering_end_pattern_arr[1][strt + ri] = move_ordering_end_pattern_arr[0][strt + i];
    }
}

inline bool get_move_ordering_end_eval(const char* file, bool show_log){
    if (show_log)
        std::cerr << "move ordering end evaluation file " << file << std::endl;
    FILE* fp;
    if (!file_open(&fp, file, "rb")){
        std::cerr << "[ERROR] [FATAL] can't open eval " << file << std::endl;
        return false;
    }
    constexpr int pattern_sizes[N_MOVE_ORDERING_END_PATTERNS] = {10, 10, 10, 9};
    constexpr int pattern_starts[N_MOVE_ORDERING_END_PATTERNS] = {1, 59050, 118099, 177148};
    if (fread(move_ordering_end_score_cell_types, 2, N_MOVE_ORDERING_END_CELL_TYPES, fp) < N_MOVE_ORDERING_END_CELL_TYPES){
        std::cerr << "[ERROR] [FATAL] evaluation file broken" << std::endl;
        fclose(fp);
        return false;
    }
    if (fread(move_ordering_end_score_parity, 2, 2, fp) < 2){
        std::cerr << "[ERROR] [FATAL] evaluation file broken" << std::endl;
        fclose(fp);
        return false;
    }
    if (fread(move_ordering_end_score_free_odd, 2, 2, fp) < 2){
        std::cerr << "[ERROR] [FATAL] evaluation file broken" << std::endl;
        fclose(fp);
        return false;
    }
    if (fread(move_ordering_end_score_n_legal, 2, N_MOVE_ORDERING_END_LEGAL, fp) < N_MOVE_ORDERING_END_LEGAL){
        std::cerr << "[ERROR] [FATAL] evaluation file broken" << std::endl;
        fclose(fp);
        return false;
    }
    if (fread(move_ordering_end_score_potential_mobility, 2, N_MOVE_ORDERING_END_POTENTIAL_MOBILITY, fp) < N_MOVE_ORDERING_END_POTENTIAL_MOBILITY){
        std::cerr << "[ERROR] [FATAL] evaluation file broken" << std::endl;
        fclose(fp);
        return false;
    }
    if (fread(move_ordering_end_pattern_arr[0] + 1, 2, N_MOVE_ORDERING_END_PATTERN_PARAMS, fp) < N_MOVE_ORDERING_END_PATTERN_PARAMS){
        std::cerr << "[ERROR] [FATAL] evaluation file broken" << std::endl;
        fclose(fp);
        return false;
    }
    int i, j, k, idx, cell, pattern_idx;
    for (i = 0; i < 2; ++i){
        move_ordering_end_pattern_arr[i][0] = 0;
        move_ordering_end_pattern_arr[i][N_MOVE_ORDERING_END_PATTERN_PARAMS + 1] = 0;
    }
    for (j = 0; j < N_MOVE_ORDERING_END_PATTERNS; ++j){
        for (k = 0; k < pow3[pattern_sizes[j]]; ++k){
            if (move_ordering_end_pattern_arr[0][pattern_starts[j] + k] < -SIMD_EVAL_OFFSET){
                move_ordering_end_pattern_arr[0][pattern_starts[j] + k] = -SIMD_EVAL_OFFSET;
                std::cerr << "[ERROR] evaluation value too low. you can ignore this error. feature " << j << " index " << k << " found " << move_ordering_end_pattern_arr[0][pattern_starts[j] + k] << std::endl;
            }
            if (move_ordering_end_pattern_arr[0][pattern_starts[j] + k] >= 0x7FFF - SIMD_EVAL_OFFSET){
                move_ordering_end_pattern_arr[0][pattern_starts[j] + k] = 0x7FFF - SIMD_EVAL_OFFSET - 1;
                std::cerr << "[ERROR] evaluation value too high. you can ignore this error. feature " << j << " index " << k << " found " << move_ordering_end_pattern_arr[0][pattern_starts[j] + k] << std::endl;
            }
            move_ordering_end_pattern_arr[0][pattern_starts[j] + k] += SIMD_EVAL_OFFSET;
        }
    }
    if (thread_pool.size() >= 2){
        std::future<void> tasks[N_MOVE_ORDERING_END_PATTERNS];
        int i = 0;
        for (pattern_idx = 0; pattern_idx < N_MOVE_ORDERING_END_PATTERNS; ++pattern_idx)
            tasks[i++] = thread_pool.push(std::bind(init_move_ordering_pattern_arr_rev, pattern_idx, pattern_sizes[pattern_idx], pattern_starts[pattern_idx]));
        for (std::future<void> &task: tasks)
            task.get();
    } else{
        for (pattern_idx = 0; pattern_idx < N_PATTERNS; ++pattern_idx)
            init_move_ordering_pattern_arr_rev(pattern_idx, pattern_sizes[pattern_idx], pattern_starts[pattern_idx]);
    }
    return true;
}

inline bool move_ordering_init(std::string end_file, bool show_log){
    return get_move_ordering_end_eval(end_file.c_str(), show_log);
}

inline int move_ordering_end_pattern_calc(Search *search){
    const int *pat_com = (int*)move_ordering_end_pattern_arr[search->eval_feature_reversed];
    __m256i res256 = _mm256_add_epi32(
        gather_eval(pat_com, calc_idx8_a(search->eval_features, 0)), 
        gather_eval(pat_com, calc_idx8_b(search->eval_features, 0))
    );
    __m128i res128 = _mm_add_epi32(_mm256_castsi256_si128(res256), _mm256_extracti128_si256(res256, 1));
    res128 = _mm_hadd_epi32(res128, res128);
    return _mm_cvtsi128_si32(res128) + _mm_extract_epi32(res128, 1) - SIMD_EVAL_OFFSET * N_MOVE_ORDERING_END_SYMMETRY_PATTERNS;
}



/*
    @brief Calculate openness

    Not used for now

    @param board                board
    @param flip                 flip information
    @return openness
*/
/*
inline int calc_openness(const Board *board, const Flip *flip){
    uint64_t f = flip->flip;
    uint64_t around = 0ULL;
    for (uint_fast8_t cell = first_bit(&f); f; cell = next_bit(&f))
        around |= bit_around[cell];
    around &= ~flip->flip;
    return pop_count_ull(~(board->player | board->opponent | (1ULL << flip->pos)) & around);
}
*/

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
    @brief Evaluate a move in midgame

    @param search               search information
    @param flip_value           flip with value
    @param alpha                alpha value to search
    @param beta                 beta value to search
    @param depth                depth to search
    @param searching            flag for terminating this search
    @return true if wipeout found else false
*/
inline bool move_evaluate(Search *search, Flip_value *flip_value, int alpha, int beta, int depth, const bool *searching){
    if (flip_value->flip.flip == search->board.opponent){
        flip_value->value = W_WIPEOUT;
        return true;
    }
    flip_value->value = cell_weight[flip_value->flip.pos] * W_CELL_WEIGHT;
    eval_move(search, &flip_value->flip);
    search->move(&flip_value->flip);
        flip_value->n_legal = search->board.get_legal();
        flip_value->value += get_weighted_n_moves(flip_value->n_legal) * W_MOBILITY;
        uint64_t empties = ~(search->board.player | search->board.opponent);
        flip_value->value += get_potential_mobility(search->board.player, empties) * W_POTENTIAL_MOBILITY;
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
    return false;
}

inline bool is_free_odd_empties(Search *search, uint_fast8_t pos){
    if (search->parity & cell_div4[pos]){
        uint64_t masked_empties = ~(search->board.player | search->board.opponent) & parity_table[cell_div4[pos]];
        return get_potential_mobility(search->board.player, masked_empties) > 0;
    }
    return false;
}

/*
    @brief Evaluate a move in endgame

    @param search               search information
    @param flip_value           flip with value
    @return true if wipeout found else false
*/
inline bool move_evaluate_end(Search *search, Flip_value *flip_value){
    if (flip_value->flip.flip == search->board.opponent){
        flip_value->value = W_WIPEOUT;
        return true;
    }
    
    flip_value->value = move_ordering_end_score_cell_types[cell_types[flip_value->flip.pos]];
    flip_value->value += move_ordering_end_score_parity[(search->parity & cell_div4[flip_value->flip.pos]) > 0];
    //flip_value->value += move_ordering_end_score_free_odd[is_free_odd_empties(search, flip_value->flip.pos)];
    eval_move(search, &flip_value->flip);
    search->move(&flip_value->flip);
        flip_value->n_legal = search->board.get_legal();
        flip_value->value += move_ordering_end_score_n_legal[pop_count_ull(flip_value->n_legal)];
        //flip_value->value += move_ordering_end_score_potential_mobility[get_potential_mobility(search->board.player, ~(search->board.player | search->board.opponent))];
        flip_value->value += move_ordering_end_pattern_calc(search);
    search->undo(&flip_value->flip);
    eval_undo(search, &flip_value->flip);
    /**/
    /*
    flip_value->value = cell_weight[flip_value->flip.pos];
    if (search->parity & cell_div4[flip_value->flip.pos])
        flip_value->value += W_END_PARITY;
    eval_move(search, &flip_value->flip);
    search->move(&flip_value->flip);
        flip_value->n_legal = search->board.get_legal();
        flip_value->value -= pop_count_ull(flip_value->n_legal) * W_END_MOBILITY;
        flip_value->value += move_ordering_end_pattern_calc(search) / 256;
        //flip_value->value -= mid_evaluate_diff(search) * 4;
    search->undo(&flip_value->flip);
    eval_undo(search, &flip_value->flip);
    /**/
    /*
    flip_value->value = cell_weight[flip_value->flip.pos];
    if (search->parity & cell_div4[flip_value->flip.pos])
        flip_value->value += W_END_PARITY;
    search->move(&flip_value->flip);
        flip_value->n_legal = search->board.get_legal();
        flip_value->value -= pop_count_ull(flip_value->n_legal) * W_END_MOBILITY;
    search->undo(&flip_value->flip);
    /**/
    return false;
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
inline bool move_evaluate_nws(Search *search, Flip_value *flip_value, int alpha, int beta, int depth, const bool *searching){
    if (flip_value->flip.flip == search->board.opponent){
        flip_value->value = W_WIPEOUT;
        return true;
    }
    flip_value->value = cell_weight[flip_value->flip.pos];
    //flip_value->value -= pop_count_ull(flip_value->flip.flip) * W_NWS_N_FLIP;
    eval_move(search, &flip_value->flip);
    search->move(&flip_value->flip);
        flip_value->n_legal = search->board.get_legal();
        flip_value->value += get_weighted_n_moves(flip_value->n_legal) * W_NWS_MOBILITY;
        uint64_t empties = ~(search->board.player | search->board.opponent);
        flip_value->value += get_potential_mobility(search->board.player, empties) * W_NWS_POTENTIAL_MOBILITY;
        //int64_t bef_n_nodes = search->n_nodes;
        if (depth == 0)
            flip_value->value += mid_evaluate_diff(search) * W_NWS_VALUE_SHALLOW;
        else
            flip_value->value += nega_alpha_eval1(search, alpha, beta, false, searching) * W_NWS_VALUE;
    search->undo(&flip_value->flip);
    eval_undo(search, &flip_value->flip);
    return false;
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
    bool wipeout_found = false;
    for (Flip_value &flip_value: move_list){
        if (!wipeout_found)
            wipeout_found = move_evaluate(search, &flip_value, eval_alpha, eval_beta, eval_depth, searching);
        else
            flip_value.value = -INF;
    }
}

/*
    @brief Evaluate all legal moves for endgame

    @param search               search information
    @param move_list            list of moves
*/
inline void move_list_evaluate_end(Search *search, std::vector<Flip_value> &move_list, const int canput){
    if (canput == 1)
        return;
    bool wipeout_found = false;
    for (Flip_value &flip_value: move_list){
        if (!wipeout_found)
            wipeout_found = move_evaluate_end(search, &flip_value);
        else
            flip_value.value = -INF;
    }
}

/*
    @brief Evaluate all legal moves for midgame NWS

    @param search               search information
    @param move_list            list of moves
    @param depth                remaining depth
    @param alpha                alpha value (beta = alpha + 1)
    @param searching            flag for terminating this search
*/
inline void move_list_evaluate_nws(Search *search, std::vector<Flip_value> &move_list, int depth, int alpha, const bool *searching){
    if (move_list.size() == 1)
        return;
    const int eval_alpha = -std::min(SCORE_MAX, alpha + MOVE_ORDERING_NWS_VALUE_OFFSET_BETA);
    const int eval_beta = -std::max(-SCORE_MAX, alpha - MOVE_ORDERING_NWS_VALUE_OFFSET_ALPHA);
    int eval_depth = depth >> 4;
    bool wipeout_found = false;
    for (Flip_value &flip_value: move_list){
        if (!wipeout_found)
            wipeout_found = move_evaluate_nws(search, &flip_value, eval_alpha, eval_beta, eval_depth, searching);
        else
            flip_value.value = -INF;
    }
}