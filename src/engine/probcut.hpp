/*
    Egaroucid Project

    @file probcut.hpp
        MPC (Multi-ProbCut)
    @date 2021-2022
    @author Takuto Yamana (a.k.a. Nyanyan)
    @license GPL-3.0 license
*/

#pragma once
#include <iostream>
#include "board.hpp"
#include "evaluate.hpp"
#include "search.hpp"
#include "midsearch.hpp"
#include "util.hpp"

#define USE_MPC_N_DISCS 51

#if USE_ALL_NODE_PREDICTION
    #define ALL_NODE_CHECK_MPCT 1.8
#endif

/*
    @brief constants for ProbCut error calculation
*/
#define probcut_a -0.001265116404528472
#define probcut_b -1.758143972292579
#define probcut_c 1.7566279520842052
#define probcut_d -1.0938733019995888
#define probcut_e 1.1290649413162603
#define probcut_f 10.55327508419982
#define probcut_g 3.106465908394277

#define probcut_end_a 1.7618824674282139
#define probcut_end_b 1.6761979186286653
#define probcut_end_c -1.5368514045644108
#define probcut_end_d 5.649224423285564
#define probcut_end_e -12.020118030448877
#define probcut_end_f 14.261869478626625

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
inline bool mpc(Search *search, int alpha, int beta, int depth, uint64_t legal, bool is_end_search, int *v, const bool *searching){
    if (search->mpc_level == MPC_100_LEVEL)
        return false;
    int search_depth;
    if (is_end_search)
        search_depth = ((depth >> 2) & 0xFE) ^ (depth & 1);
    else
        search_depth = ((depth >> 1) & 0xFE) ^ (depth & 1);
    if (is_end_search){
        alpha -= (alpha + SCORE_MAX) & 1;
        beta += (beta + SCORE_MAX) & 1;
    }
    int error_depth0, error_search;
    double mpct = SELECTIVITY_MPCT[search->mpc_level];
    if (is_end_search){
        error_depth0 = ceil(mpct * probcut_sigma_end_depth0(search->n_discs));
        error_search = ceil(mpct * probcut_sigma_end(search->n_discs, search_depth));
    } else{
        error_depth0 = ceil(mpct * probcut_sigma_depth0(search->n_discs, depth));
        error_search = ceil(mpct * probcut_sigma(search->n_discs, search_depth, depth));
    }
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
            else{
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
            else{
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
        if (search->mpc_level == MPC_100_LEVEL)
            return false;
        int search_depth;
        if (is_end_search)
            search_depth = ((depth >> 2) & 0xFE) ^ (depth & 1);
        else
            search_depth = ((depth >> 1) & 0xFE) ^ (depth & 1);
        int error_depth0;
        double mpct = SELECTIVITY_MPCT[search->mpc_level];
        if (is_end_search)
            error_depth0 = ceil(mpct * probcut_sigma_end_depth0(search->n_discs));
        else
            error_depth0 = ceil(mpct * probcut_sigma_depth0(search->n_discs, depth));
        if (alpha - error_depth0 < -SCORE_MAX)
            return false;
        int depth0_value = mid_evaluate_diff(search);
        if (depth0_value <= alpha - error_depth0){
            if (search_depth == 0)
                return true;
            int error_search;
            if (is_end_search)
                error_search = ceil(mpct * probcut_sigma_end(search->n_discs, search_depth));
            else
                error_search = ceil(mpct * probcut_sigma(search->n_discs, search_depth, depth));
            const int alpha_mpc = alpha - error_search;
            if (alpha_mpc < -SCORE_MAX)
                return false;
            bool res = false;
            if (search_depth == 1)
                res = nega_alpha_eval1_nws(search, alpha_mpc, false, searching) <= alpha_mpc;
            else{
                #if MID_FAST_DEPTH > 1
                    if (search_depth <= MID_FAST_DEPTH)
                        res = nega_alpha_nws(search, alpha_mpc, search_depth, false, searching) <= alpha_mpc;
                    else
                        res = nega_alpha_ordering_nws(search, alpha_mpc, search_depth, false, legal, false, searching) <= alpha_mpc;
                #else
                    res = nega_alpha_ordering_nws(search, alpha_mpc, search_depth, false, legal, false, searching) <= alpha_mpc;
                #endif
            }
            return res;
        }
        return false;
    }
#endif