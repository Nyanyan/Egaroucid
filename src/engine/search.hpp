/*
    Egaroucid Project

    @file search.hpp
        Search common structure
    @date 2021-2022
    @author Takuto Yamana (a.k.a. Nyanyan)
    @license GPL-3.0 license
*/

#pragma once
#include <iostream>
#include <fstream>
#include <algorithm>
#include "setting.hpp"
#include "common.hpp"
#include "board.hpp"
#include "thread_pool.hpp"
#include "evaluate.hpp"

/*
    @brief Evaluation constant
*/
#ifndef N_SYMMETRY_PATTERNS
    #define N_SYMMETRY_PATTERNS 62
#endif
#if USE_SIMD_EVALUATION
    #define N_SIMD_EVAL_FEATURES 8
#endif

/*
    @brief Search switch parameters
*/
#define MID_FAST_DEPTH 1
#define END_FAST_DEPTH 7
#define MID_TO_END_DEPTH 13
#define USE_TT_DEPTH_THRESHOLD 10

#define SCORE_UNDEFINED -SCORE_INF

/*
    @brief Search constant
*/
#ifndef SEARCH_BOOK
    #define SEARCH_BOOK -1
#endif


/*
    @brief Weights of each cell
*/
constexpr int cell_weight[HW2] = {
    18,  4,  16, 12, 12, 16,  4, 18,
     4,  2,   6,  8,  8,  6,  2,  4,
    16,  6,  14, 10, 10, 14,  6, 16,
    12,  8,  10,  0,  0, 10,  8, 12,
    12,  8,  10,  0,  0, 10,  8, 12,
    16,  6,  14, 10, 10, 14,  6, 16,
     4,  2,   6,  8,  8,  6,  2,  4,
    18,  4,  16, 12, 12, 16,  4, 18
};

/*
    @brief Stability cutoff threshold

    see nws_stability_threshold[n_discs] >= alpha ? 
*/
constexpr int nws_stability_threshold[HW2] = {
    -99, -99, -99, -99, -99, -99, -99, -99, 
    -99, -99, -99, -99, -99, -99, -99, -99, 
    -99, -99, -99, -99, -99, -99, -99, -99, 
    -99, -99, -99, -99, -99, -99, -64, -64, 
    -64, -64, -64, -64, -64, -64, -64, -62, 
    -54, -46, -38, -30, -26, -20, -18,  -8, 
     -2,   4,  10,  16,  22,  28,  34,  38, 
     42,  46,  50,  54,  58,  60,  62,  64
};

/*
    @brief board division

    used for parity calculation
*/
constexpr uint_fast8_t cell_div4[HW2] = {
    1, 1, 1, 1, 2, 2, 2, 2, 
    1, 1, 1, 1, 2, 2, 2, 2, 
    1, 1, 1, 1, 2, 2, 2, 2, 
    1, 1, 1, 1, 2, 2, 2, 2, 
    4, 4, 4, 4, 8, 8, 8, 8, 
    4, 4, 4, 4, 8, 8, 8, 8, 
    4, 4, 4, 4, 8, 8, 8, 8, 
    4, 4, 4, 4, 8, 8, 8, 8
};

/*
    @brief Search result structure

    Used for returning the result

    @param policy               selected move
    @param value                value
    @param depth                search depth
    @param time                 elapsed time
    @param nodes                number of nodes visited
    @param clog_time            elapsed time for clog search
    @param clog_nodes           number of nodes visited for clog search
    @param nps                  NPS (Nodes Per Second)
    @param is_end_search        search till the end?
    @param probability          MPC (Multi-ProbCut) probability in integer [%]
*/
struct Search_result{
    int_fast8_t policy;
    int value;
    int depth;
    uint64_t time;
    uint64_t nodes;
    uint64_t clog_time;
    uint64_t clog_nodes;
    uint64_t nps;
    bool is_end_search;
    int probability;

    bool operator<(const Search_result &another) const{
        if (depth == SEARCH_BOOK && another.depth != SEARCH_BOOK)
            return false;
        else if (depth != SEARCH_BOOK && another.depth == SEARCH_BOOK)
            return true;
        return value < another.value;
    }

    bool operator>(const Search_result &another) const{
        if (another.depth == SEARCH_BOOK && depth != SEARCH_BOOK)
            return false;
        else if (another.depth != SEARCH_BOOK && depth == SEARCH_BOOK)
            return true;
        return value > another.value;
    }
};

struct Analyze_result{
    int played_move;
    int played_score;
    int played_depth;
    int played_probability;
    int alt_move;
    int alt_score;
    int alt_depth;
    int alt_probability;
};

/*
    @brief Search structure

    Used in midgame / endgame search

    @param board                board to solve
    @param n_discs              number of discs on the board
    @param parity               parity of the board
    @param mpc_level            MPC (Multi-ProbCut) probability level
    @param n_nodes              number of visited nodes
    @param eval_features        features of pattern evaluation
    @param eval_feature_reversed    need to swap player in evaluation?
    @param use_multi_thread     use parallel search?
*/
class Search{
    public:
        Board board;
        int_fast8_t strt_n_discs;
        int_fast8_t n_discs;
        uint_fast8_t parity;
        uint_fast8_t mpc_level;
        uint64_t n_nodes;
        //Eval_features eval_features;
        #if USE_SIMD_EVALUATION
            __m256i eval_features[N_SIMD_EVAL_FEATURES];
        #else
            uint_fast16_t eval_features[N_SYMMETRY_PATTERNS];
        #endif
        bool eval_feature_reversed;
        bool use_multi_thread;
        #if USE_SEARCH_STATISTICS
            uint64_t n_nodes_discs[HW2];
        #endif
        uint8_t date;

    public:
        /*
            @brief Initialize with board

            @param init_board           a board to set
        */
        inline void init_board(Board *init_board){
            board = init_board->copy();
            n_discs = board.n_discs();
            strt_n_discs = n_discs;
            uint64_t empty = ~(board.player | board.opponent);
            parity = 1 & pop_count_ull(empty & 0x000000000F0F0F0FULL);
            parity |= (1 & pop_count_ull(empty & 0x00000000F0F0F0F0ULL)) << 1;
            parity |= (1 & pop_count_ull(empty & 0x0F0F0F0F00000000ULL)) << 2;
            parity |= (1 & pop_count_ull(empty & 0xF0F0F0F000000000ULL)) << 3;
        }

        /*
            @brief Initialize Search menber variables
        */
        inline void init_search(){
            n_discs = board.n_discs();
            strt_n_discs = n_discs;
            uint64_t empty = ~(board.player | board.opponent);
            parity = 1 & pop_count_ull(empty & 0x000000000F0F0F0FULL);
            parity |= (1 & pop_count_ull(empty & 0x00000000F0F0F0F0ULL)) << 1;
            parity |= (1 & pop_count_ull(empty & 0x0F0F0F0F00000000ULL)) << 2;
            parity |= (1 & pop_count_ull(empty & 0xF0F0F0F000000000ULL)) << 3;
        }

        /*
            @brief Move board and other variables except eval_features

            @param flip                 Flip information
        */
        inline void move(const Flip *flip) {
            board.move_board(flip);
            ++n_discs;
            parity ^= cell_div4[flip->pos];
        }

        /*
            @brief Undo board and other variables except eval_features

            @param flip                 Flip information
        */
        inline void undo(const Flip *flip) {
            board.undo_board(flip);
            --n_discs;
            parity ^= cell_div4[flip->pos];
        }

        /*
            @brief Get evaluation phase
        */
        inline int phase(){
            return std::min(N_PHASES - 1, (n_discs - 4) / PHASE_N_STONES);
        }
};

/*
    @brief Clog search structure

    @param board                board to solve
    @param n_nodes              number of visited nodes
*/
struct Clog_search{
    Board board;
    uint64_t n_nodes;
};

/*
    @brief Clog search result structure

    @param pos                  position to put
    @param val                  the exact score
*/
struct Clog_result{
    uint_fast8_t pos;
    int val;
};
