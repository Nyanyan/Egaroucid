/*
    Egaroucid Project

    @file probcut.hpp
        MPC (Multi-ProbCut)
    @date 2021-2024
    @author Takuto Yamana
    @license GPL-3.0 license
*/

#pragma once
#include <iostream>
#include "board.hpp"
#include "evaluate.hpp"
#include "search.hpp"
#include "midsearch.hpp"
#include "util.hpp"

#define USE_MPC_DEPTH 2

#if USE_ALL_NODE_PREDICTION
    #define ALL_NODE_CHECK_MPCT 1.8
#endif

/*
    @brief constants for ProbCut error calculation
*/
#define probcut_a 0.20095710497524927
#define probcut_b -1.8797511943486194
#define probcut_c 0.1680064538820171
#define probcut_d -7.562467694781112
#define probcut_e -0.7390419266354427
#define probcut_f 12.726928314156629
#define probcut_g 3.3282889578984736

#define probcut_end_a -2.1545861246368383
#define probcut_end_b -5.227169693157927
#define probcut_end_c 0.6751858297229165
#define probcut_end_d 1.680597473398011
#define probcut_end_e 1.4883149166012901
#define probcut_end_f 7.097351603734277

#if USE_MPC_PRE_CALCULATION
    int mpc_error[N_SELECTIVITY_LEVEL][HW2 + 1][HW2 - 3][HW2 - 3];
    int mpc_error_end[N_SELECTIVITY_LEVEL][HW2 + 1][HW2 - 3];
#endif

constexpr int mpc_search_depth_arr[2][61] = {
    { // midgame
         0,  0,  0,  1,  2,  3,  2,  3,  4,  5, 
         4,  5,  6,  7,  6,  7,  8,  9,  8,  9, 
        10, 11, 10, 11, 12, 13, 12, 13, 14, 15, 
        14, 15, 16, 17, 16, 17, 18, 19, 18, 19, 
        20, 21, 20, 21, 22, 23, 22, 23, 24, 25, 
        24, 25, 26, 27, 26, 27, 28, 29, 28, 29, 
        30
    }, { // endgame
         0,  1,  0,  1,  0,  1,  0,  1,  2,  3, 
         2,  3,  2,  3,  2,  3,  4,  5,  4,  5, 
         4,  5,  4,  5,  6,  7,  6,  7,  6,  7, 
         6,  7,  8,  9,  8,  9,  8,  9,  8,  9, 
        10, 11, 10, 11, 10, 11, 10, 11, 12, 13, 
        12, 13, 12, 13, 12, 13, 14, 15, 14, 15, 
        14
    }
};

/*
    @brief ProbCut error calculation for midgame

    @param n_discs              number of discs on the board
    @param depth1               depth of shallow search
    @param depth2               depth of deep search
    @return expected error
*/
inline double probcut_sigma(int n_discs, int depth1, int depth2){
    double res = probcut_a * ((double)n_discs / 64.0) + probcut_b * ((double)depth1 / 60.0) + probcut_c * ((double)depth2 / 60.0);
    res = probcut_d * res * res * res + probcut_e * res * res + probcut_f * res + probcut_g;
    return res;
}

/*
    @brief ProbCut error calculation for midgame (when shallow depth = 0)

    @param n_discs              number of discs on the board
    @param depth2               depth of deep search
    @return expected error
*/
inline double probcut_sigma_depth0(int n_discs, int depth2){
    double res = probcut_a * ((double)n_discs / 64.0) + probcut_c * ((double)depth2 / 60.0);
    res = probcut_d * res * res * res + probcut_e * res * res + probcut_f * res + probcut_g;
    return res;
}

/*
    @brief ProbCut error calculation for endgame

    @param n_discs              number of discs on the board
    @param depth                depth of shallow search
    @return expected error
*/
inline double probcut_sigma_end(int n_discs, int depth){
    double res = probcut_end_a * ((double)n_discs / 64.0) + probcut_end_b * ((double)depth / 60.0);
    res = probcut_end_c * res * res * res + probcut_end_d * res * res + probcut_end_e * res + probcut_end_f;
    return res;
}

/*
    @brief ProbCut error calculation for endgame (when shallow depth = 0)

    @param n_discs              number of discs on the board
    @return expected error
*/
inline double probcut_sigma_end_depth0(int n_discs){
    double res = probcut_end_a * ((double)n_discs / 64.0);
    res = probcut_end_c * res * res * res + probcut_end_d * res * res + probcut_end_e * res + probcut_end_f;
    return res;
}

inline int nega_alpha_eval1_nws(Search *search, int alpha, bool skipped, const bool *searching);
#if USE_NEGA_ALPHA_ORDERING
    int nega_alpha_ordering(Search *search, int alpha, int beta, int depth, bool skipped, uint64_t legal, bool is_end_search, const bool *searching);
#endif
int nega_alpha_ordering_nws(Search *search, int alpha, int depth, bool skipped, uint64_t legal, bool is_end_search, const bool *searching);

/*
    @brief Multi-ProbCut for normal search

    @param search               search information
    @param alpha                alpha value
    @param beta                 beta value
    @param depth                depth of deep search
    @param legal                for use of previously calculated legal bitboard
    @param is_end_search        search till the end?
    @param v                    an integer to store result
    @param searching            flag for terminating this search
    @return cutoff occurred?
*/
inline bool mpc(Search* search, int alpha, int beta, int depth, uint64_t legal, bool is_end_search, int* v, const bool* searching){
    if (search->mpc_level == MPC_100_LEVEL)
        return false;
    int search_depth = mpc_search_depth_arr[is_end_search][depth];
    int error_search, error_0;
    #if USE_MPC_PRE_CALCULATION
        if (is_end_search){
            error_search = mpc_error_end[search->mpc_level][search->n_discs][search_depth];
            error_0 = mpc_error_end[search->mpc_level][search->n_discs][0];
        } else{
            error_search = mpc_error[search->mpc_level][search->n_discs][search_depth][depth];
            error_0 = mpc_error[search->mpc_level][search->n_discs][0][depth];
        }
    #else
        double mpct = SELECTIVITY_MPCT[search->mpc_level];
        if (is_end_search){
            error_search = ceil(mpct * probcut_sigma_end(search->n_discs, search_depth));
            error_0 = ceil(mpct * probcut_sigma_end(search->n_discs, 0));
        }else{
            error_search = ceil(mpct * probcut_sigma(search->n_discs, search_depth, depth));
            error_0 = ceil(mpct * probcut_sigma(search->n_discs, 0, depth));
        }
    #endif
    int d0value = mid_evaluate_diff(search);
    if (d0value >= beta + (error_search + error_0) / 2){
        int pc_beta = beta + error_search;
        if (pc_beta < SCORE_MAX){
            if (nega_alpha_ordering_nws(search, pc_beta - 1, search_depth, false, legal, false, searching) >= pc_beta){
                *v = beta;
                if (is_end_search)
                    *v += beta & 1;
                return true;
            }
        }
    }
    if (d0value <= alpha - (error_search + error_0) / 2){
        int pc_alpha = alpha - error_search;
        if (pc_alpha > -SCORE_MAX){
            if (nega_alpha_ordering_nws(search, pc_alpha, search_depth, false, legal, false, searching) <= pc_alpha){
                *v = alpha;
                if (is_end_search)
                    *v -= alpha & 1;
                return true;
            }
        }
    }
    return false;
}


#if USE_ALL_NODE_PREDICTION_NWS
    inline bool predict_all_node(Search* search, int alpha, int depth, uint64_t legal, bool is_end_search, const bool* searching){
        uint_fast8_t mpc_level = MPC_93_LEVEL;
        int search_depth = mpc_search_depth_arr[is_end_search][depth];
        int error_search, error_0;
        #if USE_MPC_PRE_CALCULATION
            if (is_end_search){
                error_search = mpc_error_end[mpc_level][search->n_discs][search_depth];
                error_0 = mpc_error_end[mpc_level][search->n_discs][0];
            } else{
                error_search = mpc_error[mpc_level][search->n_discs][search_depth][depth];
                error_0 = mpc_error[mpc_level][search->n_discs][0][depth];
            }
        #else
            double mpct = SELECTIVITY_MPCT[mpc_level];
            if (is_end_search){
                error_search = ceil(mpct * probcut_sigma_end(search->n_discs, search_depth));
                error_0 = ceil(mpct * probcut_sigma_end(search->n_discs, 0));
            }else{
                error_search = ceil(mpct * probcut_sigma(search->n_discs, search_depth, depth));
                error_0 = ceil(mpct * probcut_sigma(search->n_discs, 0, depth));
            }
        #endif
        int d0value = mid_evaluate_diff(search);
        if (d0value <= alpha - (error_search + error_0) / 2){
            int pc_alpha = alpha - error_search;
            if (pc_alpha > -SCORE_MAX){
                if (nega_alpha_ordering_nws(search, pc_alpha, search_depth, false, legal, false, searching) <= pc_alpha){
                    return true;
                }
            }
        }
        return false;
    }
#endif



#if USE_MPC_PRE_CALCULATION
    void mpc_init(){
        int mpc_level, n_discs, depth1, depth2;
        for (mpc_level = 0; mpc_level < N_SELECTIVITY_LEVEL; ++mpc_level){
            for (n_discs = 0; n_discs < HW2 + 1; ++n_discs){
                for (depth1 = 0; depth1 < HW2 - 3; ++depth1){
                    mpc_error_end[mpc_level][n_discs][depth1] = ceil(SELECTIVITY_MPCT[mpc_level] * probcut_sigma_end(n_discs, depth1));
                    for (depth2 = 0; depth2 < HW2 - 3; ++depth2)
                        mpc_error[mpc_level][n_discs][depth1][depth2] = ceil(SELECTIVITY_MPCT[mpc_level] * probcut_sigma(n_discs, depth1, depth2));
                }
            }
        }
    }
#endif

#if TUNE_PROBCUT_MID
    void get_data_probcut_mid(){
        std::ofstream ofs("probcut_mid.txt");
        Board board;
        Flip flip;
        Search_result short_ans, long_ans;
        for (int i = 0; i < 1000; ++i){
            for (int depth = 2; depth < 14; ++depth){
                for (int n_discs = 4; n_discs < HW2 - depth - 5; ++n_discs){
                    board.reset();
                    for (int j = 4; j < n_discs && board.check_pass(); ++j){ // random move
                        uint64_t legal = board.get_legal();
                        int random_idx = myrandrange(0, pop_count_ull(legal));
                        int t = 0;
                        for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)){
                            if (t == random_idx){
                                calc_flip(&flip, &board, cell);
                                break;
                            }
                            ++t;
                        }
                        board.move_board(&flip);
                    }
                    if (board.check_pass()){
                        int short_depth = myrandrange(1, depth - 1);
                        short_depth &= 0xfffffffe;
                        short_depth |= depth & 1;
                        //int short_depth = mpc_search_depth_arr[0][depth];
                        if (short_depth == 0){
                            short_ans.value = mid_evaluate(&board);
                        } else{
                            short_ans = tree_search(board, short_depth, MPC_100_LEVEL, false, true);
                        }
                        long_ans = tree_search(board, depth, MPC_100_LEVEL, false, true);
                        // n_discs short_depth long_depth error
                        std::cerr << i << " " << n_discs << " " << short_depth << " " << depth << " " << long_ans.value - short_ans.value << std::endl;
                        ofs << n_discs << " " << short_depth << " " << depth << " " << long_ans.value - short_ans.value << std::endl;
                    }
                }
            }
        }
    }
#endif

#if TUNE_PROBCUT_END
    void get_data_probcut_end(){
        std::ofstream ofs("probcut_end.txt");
        Board board;
        Flip flip;
        Search_result short_ans, long_ans;
        for (int i = 0; i < 1000; ++i){
            for (int depth = 6; depth < 24; ++depth){
                board.reset();
                for (int j = 0; j < HW2 - 4 - depth && board.check_pass(); ++j){ // random move
                    uint64_t legal = board.get_legal();
                    int random_idx = myrandrange(0, pop_count_ull(legal));
                    int t = 0;
                    for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)){
                        if (t == random_idx){
                            calc_flip(&flip, &board, cell);
                            break;
                        }
                        ++t;
                    }
                    board.move_board(&flip);
                }
                if (board.check_pass()){
                    int short_depth = myrandrange(1, std::min(15, depth - 1));
                    short_depth &= 0xfffffffe;
                    short_depth |= depth & 1;
                    //int short_depth = mpc_search_depth_arr[1][depth];
                    if (short_depth == 0){
                        short_ans.value = mid_evaluate(&board);
                    } else{
                        short_ans = tree_search(board, short_depth, MPC_100_LEVEL, false, true);
                    }
                    long_ans = tree_search(board, depth, MPC_100_LEVEL, false, true);
                    // n_discs short_depth error
                    std::cerr << i << " " << HW2 - depth << " " << short_depth << " " << long_ans.value - short_ans.value << std::endl;
                    ofs << HW2 - depth << " " << short_depth << " " << long_ans.value - short_ans.value << std::endl;
                }
            }
        }
    }
#endif