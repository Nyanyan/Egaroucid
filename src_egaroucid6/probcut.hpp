#pragma once
#include <iostream>
#include "board.hpp"
#include "evaluate.hpp"
#include "search.hpp"
#include "midsearch.hpp"
#include "util.hpp"

using namespace std;

#define PROBCUT_SHALLOW_IGNORE 5

#define probcut_a -0.004942680531225671
#define probcut_b -0.015275790948328411
#define probcut_c 0.022814812802627794
#define probcut_d 14.128678232108669
#define probcut_e 13.580288999705605
#define probcut_f -5.044978620977171
#define probcut_g 0.06855182392482333

#define probcut_end_a 0.27811777028350226
#define probcut_end_b 0.9393034706176613
#define probcut_end_c -0.0003466476344665104
#define probcut_end_d 0.026804233485840375
#define probcut_end_e -0.6919837072602527
#define probcut_end_f 13.88628573583576

inline double probcut_sigma(int n_stones, int depth1, int depth2){
    double w = n_stones;
    double x = depth1;
    double y = depth1 - depth2;
    double res = probcut_a * w + probcut_b * x + probcut_c * y;
    res = probcut_d * res * res * res + probcut_e * res * res + probcut_f * res + probcut_g;
    return res;
}

inline double probcut_sigma_depth0(int n_stones, int depth1){
    double w = n_stones;
    double x = depth1;
    double res = probcut_a * w + probcut_b * x + probcut_c * x;
    res = probcut_d * res * res * res + probcut_e * res * res + probcut_f * res + probcut_g;
    return res;
}

inline double probcut_sigma_end(int n_stones, int depth){
    double x = n_stones;
    double y = depth;
    double res = probcut_end_a * x + probcut_end_b * y;
    res = probcut_end_c * res * res * res + probcut_end_d * res * res + probcut_end_e * res + probcut_end_f;
    return res;
}

inline double probcut_sigma_end_depth0(int n_stones){
    double x = n_stones;
    double res = probcut_end_a * x;
    res = probcut_end_c * res * res * res + probcut_end_d * res * res + probcut_end_e * res + probcut_end_f;
    return res;
}

int nega_alpha_eval1(Search *search, int alpha, int beta, bool skipped, const bool *searching);
int nega_alpha(Search *search, int alpha, int beta, int depth, bool skipped, const bool *searching);
int nega_alpha_ordering_nomemo(Search *search, int alpha, int beta, int depth, bool skipped, uint64_t legal, const bool *searching);

inline bool mpc(Search *search, int alpha, int beta, int depth, uint64_t legal, bool is_end_search, int *v, const bool *searching){
    if (search->first_depth - depth < PROBCUT_SHALLOW_IGNORE)
        return false;
    bool res = false;
    int search_depth;
    if (is_end_search)
        search_depth = ((depth >> 4) & 0xFE) ^ (depth & 1);
    else
        search_depth = ((depth >> 3) & 0xFE) ^ (depth & 1);
    const int depth0_value = mid_evaluate_diff(search);
    int error_depth0, error_search;
    if (is_end_search){
        alpha -= alpha & 1;
        beta += beta & 1;
        error_depth0 = ceil(search->mpct * probcut_sigma_end_depth0(search->n_discs));
        error_search = ceil(search->mpct * probcut_sigma_end(search->n_discs, search_depth));
    } else{
        error_depth0 = ceil(search->mpct * probcut_sigma_depth0(search->n_discs, depth));
        error_search = ceil(search->mpct * probcut_sigma(search->n_discs, depth, search_depth));
    }
    if (depth0_value >= beta + error_depth0 && beta + error_search <= SCORE_MAX){
        switch(search_depth){
            case 0:
                res = depth0_value >= beta + error_search;
                break;
            case 1:
                res = nega_alpha_eval1(search, beta + error_search - 1, beta + error_search, false, searching) >= beta + error_search;
                break;
            default:
                if (search_depth <= MID_FAST_DEPTH)
                    res = nega_alpha(search, beta + error_search - 1, beta + error_search, search_depth, false, searching) >= beta + error_search;
                else{
                    //double mpct = search->mpct;
                    //search->mpct = 1.18;
                    search->use_mpc = false;
                        res = nega_alpha_ordering_nomemo(search, beta + error_search - 1, beta + error_search, search_depth, false, legal, searching) >= beta + error_search;
                    search->use_mpc = true;
                    //search->mpct = mpct;
                }
                break;
        }
        if (res){
            *v = beta;
            return true;
        }
    }
    if (depth0_value <= alpha - error_depth0 && alpha - error_search >= -SCORE_MAX){
        switch(search_depth){
            case 0:
                res = depth0_value <= alpha - error_search;
                break;
            case 1:
                res = nega_alpha_eval1(search, alpha - error_search, alpha - error_search + 1, false, searching) <= alpha - error_search;
                break;
            default:
                if (search_depth <= MID_FAST_DEPTH)
                    res = nega_alpha(search, alpha - error_search, alpha - error_search + 1, search_depth, false, searching) <= alpha - error_search;
                else{
                    //double mpct = search->mpct;
                    //search->mpct = 1.18;
                    search->use_mpc = false;
                        res = nega_alpha_ordering_nomemo(search, alpha - error_search, alpha - error_search + 1, search_depth, false, legal, searching) <= alpha - error_search;
                    search->use_mpc = true;
                    //search->mpct = mpct;
                }
                break;
        }
        if (res){
            *v = alpha;
            return true;
        }
    }
    return false;
}
