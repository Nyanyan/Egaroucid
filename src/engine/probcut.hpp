/*
    Egaroucid Project

    @file probcut.hpp
        MPC (Multi-ProbCut)
    @date 2021-2023
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
#define probcut_a 0.3751721838248846
#define probcut_b -1.8352941535241047
#define probcut_c 1.8368724462489843
#define probcut_d 0.8431454514755922
#define probcut_e -3.013459627561517
#define probcut_f 8.3298545018757
#define probcut_g 2.539773029643781

#define probcut_end_a 1.908798448361043
#define probcut_end_b 1.6468299594064413
#define probcut_end_c 2.0530449406091082
#define probcut_end_d -5.961374118848742
#define probcut_end_e -0.7753956186749736
#define probcut_end_f 10.95261264952042

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

int mpc_search_depth_mid_arr[61] = {
    
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
#if MID_FAST_DEPTH > 1
    int nega_alpha_nws(Search *search, int alpha, int depth, bool skipped, const bool *searching);
#endif
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
    if (is_end_search){
        alpha -= (alpha + SCORE_MAX) & 1;
        beta += (beta + SCORE_MAX) & 1;
    }
    int error_depth0, error_search;
    #if USE_MPC_PRE_CALCULATION
        if (is_end_search){
            error_depth0 = mpc_error_end[search->mpc_level][search->n_discs][0];
            error_search = mpc_error_end[search->mpc_level][search->n_discs][search_depth];
        }
        else {
            error_depth0 = mpc_error[search->mpc_level][search->n_discs][0][depth];
            error_search = mpc_error[search->mpc_level][search->n_discs][search_depth][depth];
        }
    #else
        double mpct = SELECTIVITY_MPCT[search->mpc_level];
        if (is_end_search){
            error_depth0 = ceil(mpct * probcut_sigma_end_depth0(search->n_discs));
            error_search = ceil(mpct * probcut_sigma_end(search->n_discs, search_depth));
        }
        else {
            error_depth0 = ceil(mpct * probcut_sigma_depth0(search->n_discs, depth));
            error_search = ceil(mpct * probcut_sigma(search->n_discs, search_depth, depth));
        }
    #endif
    if (alpha - error_depth0 < -SCORE_MAX && SCORE_MAX < beta + error_depth0)
        return false;
    const int depth0_value = mid_evaluate_diff(search);
    if (depth0_value >= beta + error_depth0){
        if (search_depth == 0){
            *v = beta;
            return true;
        }
        const int beta_mpc = beta + error_search;
        if (beta_mpc <= SCORE_MAX){
            bool res = false;
            if (search_depth == 1)
                res = nega_alpha_eval1_nws(search, beta_mpc - 1, false, searching) >= beta_mpc;
            else {
                #if MID_FAST_DEPTH > 1
                    if (search_depth <= MID_FAST_DEPTH)
                        res = nega_alpha_nws(search, beta_mpc - 1, search_depth, false, searching) >= beta_mpc;
                    else
                        res = nega_alpha_ordering_nws(search, beta_mpc - 1, search_depth, false, legal, false, searching) >= beta_mpc;
                #else
                    res = nega_alpha_ordering_nws(search, beta_mpc - 1, search_depth, false, legal, false, searching) >= beta_mpc;
                #endif
            }
            if (res){
                *v = beta;
                return true;
            }
        }
    } else if (depth0_value <= alpha - error_depth0){
        if (search_depth == 0){
            *v = alpha;
            return true;
            }
        const int alpha_mpc = alpha - error_search;
        if (alpha_mpc >= -SCORE_MAX){
            bool res = false;
            if (search_depth == 1)
                res = nega_alpha_eval1_nws(search, alpha_mpc, false, searching) <= alpha_mpc;
            else {
                #if MID_FAST_DEPTH > 1
                    if (search_depth <= MID_FAST_DEPTH)
                        res = nega_alpha_nws(search, alpha_mpc, search_depth, false, searching) <= alpha_mpc;
                    else
                        res = nega_alpha_ordering_nws(search, alpha_mpc, search_depth, false, legal, false, searching) <= alpha_mpc;
                #else
                    res = nega_alpha_ordering_nws(search, alpha_mpc, search_depth, false, legal, false, searching) <= alpha_mpc;
                #endif
            }
            if (res){
                *v = alpha;
                return true;
            }
        }
    }
    return false;
}

/*
    @brief Multi-ProbCut for NWS (Null Window Search)

    @param search               search information
    @param alpha                alpha value (beta = alpha + 1)
    @param depth                depth of deep search
    @param legal                for use of previously calculated legal bitboard
    @param is_end_search        search till the end?
    @param v                    an integer to store result
    @param searching            flag for terminating this search
    @return cutoff occurred?
*/
inline bool mpc_nws(Search *search, int alpha, int depth, uint64_t legal, bool is_end_search, int *v, const bool *searching){
    return mpc(search, alpha, alpha + 1, depth, legal, is_end_search, v, searching);
}

#if USE_ALL_NODE_PREDICTION
    /*
        @brief Predict ALL node (fail low node) with MPC algorithm

        @param search               search information
        @param alpha                alpha value
        @param depth                depth of deep search
        @param legal                for use of previously calculated legal bitboard
        @param is_end_search        search till the end?
        @param searching            flag for terminating this search
        @return this node seems to be ALL node?
    */
    inline bool predict_all_node(Search *search, int alpha, int depth, uint64_t legal, bool is_end_search, const bool *searching){
        alpha -= 2;
        uint_fast8_t mpc_level_memo = search->mpc_level;
        uint_fast8_t predict_mpc_level = std::max(0, search->mpc_level - 1);
        int search_depth;
        if (is_end_search)
            search_depth = ((depth >> 2) & 0xFE) ^ (depth & 1);
        else
            search_depth = ((depth >> 1) & 0xFE) ^ (depth & 1);
        if (is_end_search)
            alpha -= (alpha + SCORE_MAX) & 1;
        int error_depth0, error_search;
        #if USE_MPC_PRE_CALCULATION
            if (is_end_search){
                error_depth0 = mpc_error_end[predict_mpc_level][search->n_discs][0];
                error_search = mpc_error_end[predict_mpc_level][search->n_discs][search_depth];
            } else{
                error_depth0 = mpc_error[predict_mpc_level][search->n_discs][0][depth];
                error_search = mpc_error[predict_mpc_level][search->n_discs][search_depth][depth];
            }
        #else
            double mpct = SELECTIVITY_MPCT[predict_mpc_level];
            if (is_end_search){
                error_depth0 = ceil(mpct * probcut_sigma_end_depth0(search->n_discs));
                error_search = ceil(mpct * probcut_sigma_end(search->n_discs, search_depth));
            } else{
                error_depth0 = ceil(mpct * probcut_sigma_depth0(search->n_discs, depth));
                error_search = ceil(mpct * probcut_sigma(search->n_discs, search_depth, depth));
            }
        #endif
        if (alpha - error_depth0 < -SCORE_MAX)
            return false;
        const int alpha_mpc = alpha - error_search;
        if (search_depth && alpha_mpc < -SCORE_MAX)
            return false;
        const int depth0_value = mid_evaluate_diff(search);
        if (depth0_value > alpha - error_depth0)
            return false;
        if (search_depth == 0)
            return true;
        bool res = false;
        if (search_depth == 1)
            res = nega_alpha_eval1_nws(search, alpha_mpc, false, searching) <= alpha_mpc;
        else{
            search->mpc_level = predict_mpc_level;
            #if MID_FAST_DEPTH > 1
                if (search_depth <= MID_FAST_DEPTH)
                    res = nega_alpha_nws(search, alpha_mpc, search_depth, false, searching) <= alpha_mpc;
                else
                    res = nega_alpha_ordering_nws(search, alpha_mpc, search_depth, false, legal, false, searching) <= alpha_mpc;
            #else
                res = nega_alpha_ordering_nws(search, alpha_mpc, search_depth, false, legal, false, searching) <= alpha_mpc;
            #endif
            search->mpc_level = mpc_level_memo;
        }
        if (res)
            return true;
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

bool enhanced_mpc(Search *search, std::vector<Flip_value> &move_list, int depth, int alpha, int beta, bool is_end_search, const bool *searching, int *v){
    int val;
    bool mpc_cut;
    bool all_fail_low = true;
    for (Flip_value &flip_value: move_list){
        eval_move(search, &flip_value.flip);
        search->move(&flip_value.flip);
            mpc_cut = mpc(search, -beta, -alpha, depth - 1, LEGAL_UNDEFINED, is_end_search, &val, searching);
        search->undo(&flip_value.flip);
        eval_undo(search, &flip_value.flip);
        if (mpc_cut){ // mpc cutoff done
            flip_value.flip.flip = 0ULL;
            if (-val >= alpha) // not fail low at parent node
                all_fail_low = false;
            if (-val >= beta){ // fail high at parent node
                *v = -val;
                return true;
            }
        } else
            all_fail_low = false;
    }
    if (all_fail_low){
        *v = alpha;
        return true;
    }
    return false;
}