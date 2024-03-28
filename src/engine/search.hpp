/*
    Egaroucid Project

    @file search.hpp
        Search common structure
    @date 2021-2024
    @author Takuto Yamana
    @license GPL-3.0 license
    @notice I referred to codes written by others
*/

#pragma once
#include <iostream>
#include <fstream>
#include <algorithm>
#include "setting.hpp"
#include "common.hpp"
#include "board.hpp"
#include "thread_pool.hpp"
#include "evaluate_common.hpp"
#include "evaluate.hpp"
#include "flip.hpp"

/*
    @brief Search switch parameters
*/
#define END_FAST_DEPTH 6
#define END_SIMPLE_DEPTH 10
#define MID_TO_END_DEPTH 13

/*
    @brief Search hyperparameters
*/
#define MID_ETC_DEPTH 16



/*
    @brief Search constant
*/
#define SCORE_UNDEFINED -SCORE_INF
#define MOVE_UNDEFINED 127
#define MOVE_NOMOVE 65
#define MOVE_PASS 64
#ifndef SEARCH_BOOK
    #define SEARCH_BOOK -1
#endif

inline void calc_eval_features(Board *board, Eval_search *eval);
#if USE_SIMD
    inline void eval_move(Eval_search *eval, const Flip *flip, const Board *board);
    inline void eval_undo(Eval_search *eval);
    inline void eval_pass(Eval_search *eval, const Board *board);
    inline void eval_move_endsearch(Eval_search *eval, const Flip *flip, const Board *board);
    inline void eval_undo_endsearch(Eval_search *eval);
    inline void eval_pass_endsearch(Eval_search *eval, const Board *board);
#else
    inline void eval_move(Eval_search *eval, const Flip *flip);
    inline void eval_undo(Eval_search *eval, const Flip *flip);
    inline void eval_pass(Eval_search *eval);
    inline void eval_move_endsearch(Eval_search *eval, const Flip *flip);
    inline void eval_undo_endsearch(Eval_search *eval, const Flip *flip);
    inline void eval_pass_endsearch(Eval_search *eval);
#endif

/*
    @brief Stability cutoff threshold
    from https://github.com/abulmo/edax-reversi/blob/1ae7c9fe5322ac01975f1b3196e788b0d25c1e10/src/search.c#L108 and modified
*/
constexpr int stability_threshold_nws[HW2] = {
    99, 99, 99, 99, 99, 99, 99, 99, 
    99, 99, 99, 99, 99, 99, 99, 99, 
    99, 64, 64, 64, 64, 64, 64, 64, 
    64, 62, 62, 60, 60, 58, 58, 56, 
    56, 54, 54, 52, 52, 50, 50, 48, 
    48, 46, 44, 42, 40, 38, 36, 34, 
    32, 30, 28, 26, 24, 22, 20, 16, 
    14, 12, 10, 8, 6, 99, 99, 99
};

constexpr int stability_threshold[HW2] = {
    99, 99, 99, 99, 99, 99, 99, 99, 
    99, 99, 99, 99, 99, 99, 99, 99, 
    99, 62, 62, 60, 60, 58, 58, 56, 
    56, 54, 54, 52, 52, 50, 50, 48, 
    48, 46, 46, 44, 44, 42, 42, 40, 
    40, 38, 36, 34, 32, 30, 28, 26, 
    24, 22, 20, 18, 16, 14, 12, 8, 
    6, 4, 2, 0, -2, 99, 99, 99
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
    @brief a table for parity-based move ordering
*/
constexpr uint64_t parity_table[16] = {
    0x0000000000000000ULL, 0x000000000F0F0F0FULL, 0x00000000F0F0F0F0ULL, 0x00000000FFFFFFFFULL,
    0x0F0F0F0F00000000ULL, 0x0F0F0F0F0F0F0F0FULL, 0x0F0F0F0FF0F0F0F0ULL, 0x0F0F0F0FFFFFFFFFULL,
    0xF0F0F0F000000000ULL, 0xF0F0F0F00F0F0F0FULL, 0xF0F0F0F0F0F0F0F0ULL, 0xF0F0F0F0FFFFFFFFULL,
    0xFFFFFFFF00000000ULL, 0xFFFFFFFF0F0F0F0FULL, 0xFFFFFFFFF0F0F0F0ULL, 0xFFFFFFFFFFFFFFFFULL
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

    Search_result(){
        policy = HW2;
        value = SCORE_UNDEFINED;
        depth = -1;
        time = 0;
        nodes = 0;
        clog_time = 0;
        clog_nodes = 0;
        nps = 0;
        is_end_search = false;
        probability = 0;
    }

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

/*
    @brief Analyze result structure

    Used in `analyze` command

    @param played_move          played move
    @param played_score         score of played move
    @param played_depth         depth of search for played_score calculation
    @param played_probability   probability of search for played_score calculation
    @param alt_move             alternative best move
    @param alt_score            score of alternative move
    @param alt_depth            depth of search for alt_score calculation
    @param alt_probability      probability of search for alt_score calculation
*/
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
        Eval_search eval;
        bool use_multi_thread;
        bool need_to_see_tt_loop;
        #if USE_SEARCH_STATISTICS
            uint64_t n_nodes_discs[HW2];
        #endif

    public:

        inline void init(Board *init_board, uint_fast8_t init_mpc_level, bool init_use_multi_thread, bool init_need_to_see_tt_loop){
            board = init_board->copy();
            n_discs = board.n_discs();
            strt_n_discs = n_discs;
            uint64_t empty = ~(board.player | board.opponent);
            parity = 1 & pop_count_ull(empty & 0x000000000F0F0F0FULL);
            parity |= (1 & pop_count_ull(empty & 0x00000000F0F0F0F0ULL)) << 1;
            parity |= (1 & pop_count_ull(empty & 0x0F0F0F0F00000000ULL)) << 2;
            parity |= (1 & pop_count_ull(empty & 0xF0F0F0F000000000ULL)) << 3;
            calc_eval_features(&board, &eval);
            mpc_level = init_mpc_level;
            use_multi_thread = init_use_multi_thread;
            n_nodes = 0;
            need_to_see_tt_loop = init_need_to_see_tt_loop;
        }

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
            calc_eval_features(&board, &eval);
        }

        /*
            @brief Move board and other variables

            @param flip                 Flip information
        */
        inline void move(const Flip *flip) {
            #if USE_SIMD
                eval_move(&eval, flip, &board);
            #else
                eval_move(&eval, flip);
            #endif
            board.move_board(flip);
            ++n_discs;
            parity ^= cell_div4[flip->pos];
        }

        /*
            @brief Undo board and other variables

            @param flip                 Flip information
        */
        inline void undo(const Flip *flip) {
            #if USE_SIMD
                eval_undo(&eval);
            #else
                eval_undo(&eval, flip);
            #endif
            board.undo_board(flip);
            --n_discs;
            parity ^= cell_div4[flip->pos];
        }

        /*
            @brief pass board
        */
        inline void pass(){
            #if USE_SIMD
                eval_pass(&eval, &board);
            #else
                eval_pass(&eval);
            #endif
            board.pass();
        }

        /*
            @brief Move board and other variables

            @param flip                 Flip information
        */
        inline void move_endsearch(const Flip *flip) {
            #if USE_SIMD
                eval_move_endsearch(&eval, flip, &board);
            #else
                eval_move_endsearch(&eval, flip);
            #endif
            board.move_board(flip);
            ++n_discs;
            parity ^= cell_div4[flip->pos];
        }

        /*
            @brief Undo board and other variables

            @param flip                 Flip information
        */
        inline void undo_endsearch(const Flip *flip) {
            #if USE_SIMD
                eval_undo_endsearch(&eval);
            #else
                eval_undo_endsearch(&eval, flip);
            #endif
            board.undo_board(flip);
            --n_discs;
            parity ^= cell_div4[flip->pos];
        }

        /*
            @brief pass board
        */
        inline void pass_endsearch(){
            #if USE_SIMD
                eval_pass_endsearch(&eval, &board);
            #else
                eval_pass_endsearch(&eval);
            #endif
            board.pass();
        }

        /*
            @brief Move board and other variables except eval_features

            @param flip                 Flip information
        */
        inline void move_noeval(const Flip *flip) {
            board.move_board(flip);
            ++n_discs;
            parity ^= cell_div4[flip->pos];
        }

        /*
            @brief Undo board and other variables except eval_features

            @param flip                 Flip information
        */
        inline void undo_noeval(const Flip *flip) {
            board.undo_board(flip);
            --n_discs;
            parity ^= cell_div4[flip->pos];
        }

        /*
            @brief pass board except eval_features
        */
        inline void pass_noeval(){
            board.pass();
        }

        /*
            @brief Move board and other variables except eval_features

            @param flip                 Flip information
        */
        inline void move_lastN(const Flip *flip) {
            board.move_board(flip);
            #if !USE_SIMD
                parity ^= cell_div4[flip->pos];
            #endif
        }

        /*
            @brief Undo board and other variables except eval_features

            @param flip                 Flip information
        */
        inline void undo_lastN(const Flip *flip) {
            board.undo_board(flip);
            #if !USE_SIMD
                parity ^= cell_div4[flip->pos];
            #endif
        }

        /*
            @brief Get evaluation phase
        */
        inline int phase(){
            return (n_discs - 4) / PHASE_N_DISCS;
            //return std::min(N_PHASES - 1, (n_discs - 4) / PHASE_N_DISCS);
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
