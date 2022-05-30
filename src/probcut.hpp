#pragma once
#include <iostream>
#include "board.hpp"
#include "evaluate.hpp"
#include "search.hpp"
#if USE_CUDA
    #include "cuda_midsearch.hpp"
#else
    #include "midsearch.hpp"
#endif
#include "util.hpp"

using namespace std;
/*
constexpr int mpcd[61] = {
    0, 1, 0, 1, 2, 3, 2, 3, 4, 5, 
    4, 5, 6, 7, 6, 7, 8, 9, 8, 9, 
    10, 11, 10, 11, 12, 13, 12, 13, 14, 15, 
    14, 15, 16, 17, 16, 17, 18, 19, 18, 19, 
    20, 21, 20, 21, 22, 23, 22, 23, 24, 25, 
    24, 25, 26, 27, 26, 27, 28, 29, 28, 29,
    30
};
*/
/*
constexpr int mpcd[61] = {
    0, 1, 0, 1, 2, 3, 2, 3, 4, 3, 
    4, 5, 4, 5, 6, 5, 6, 7, 6, 7, 
    8, 7, 8, 9, 8, 9, 10, 9, 10, 11, 
    10, 11, 12, 11, 12, 13, 12, 13, 14, 13, 
    14, 15, 14, 15, 16, 15, 16, 17, 16, 17, 
    18, 17, 18, 19, 18, 19, 20, 19, 20, 21,
    20
};
*/
/*
constexpr int mpcd[61] = {
    0, 1, 0, 1, 2, 3, 2, 3, 4, 3, 
    4, 3, 4, 5, 4, 5, 6, 5, 6, 5, 
    6, 7, 6, 7, 8, 7, 8, 7, 8, 9, 
    8, 9, 10, 9, 10, 9, 10, 11, 10, 11, 
    10, 11, 12, 11, 12, 11, 12, 13, 12, 13, 
    14, 13, 14, 13, 14, 15, 14, 15, 16, 15,
    16
};
*/
/*
constexpr int mpcd[61] = {
    0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 
    4, 3, 4, 3, 4, 5, 4, 5, 6, 5, 
    6, 5, 6, 7, 6, 7, 8, 7, 8, 7, 
    8, 9, 8, 9, 10, 9, 10, 9, 10, 11, 
    10, 11, 12, 11, 12, 11, 12, 13, 12, 13, 
    14, 13, 14, 13, 14, 15, 14, 15, 16, 15,
    16
};
*/
/*
constexpr int mpcd[61] = {
    0, 1, 0, 1, 0, 1, 2, 1, 2, 1, 
    2, 3, 2, 3, 2, 3, 4, 3, 4, 3, 
    4, 5, 4, 5, 4, 5, 6, 5, 6, 5, 
    6, 7, 6, 7, 6, 7, 8, 7, 8, 7, 
    8, 9, 8, 9, 8, 9, 10, 9, 10, 9, 
    10, 11, 10, 11, 10, 11, 12, 11, 12, 11,
    12
};
*/
/*
constexpr int mpcd[61] = {
    0, 1, 0, 1, 0, 1, 2, 1, 2, 1, 
    2, 1, 2, 3, 2, 3, 2, 3, 4, 3, 
    4, 3, 4, 3, 4, 5, 4, 5, 4, 5, 
    6, 5, 6, 5, 6, 5, 6, 7, 6, 7, 
    6, 7, 8, 7, 8, 7, 8, 7, 8, 9, 
    8, 9, 8, 9, 10, 9, 10, 9, 10, 9,
    10
};
*/
/*
constexpr int mpcd[61] = {
    0, 1, 0, 1, 0, 1, 0, 1, 2, 1, 
    2, 1, 2, 1, 2, 3, 2, 3, 2, 3, 
    2, 3, 4, 3, 4, 3, 4, 5, 4, 5, 
    4, 5, 4, 5, 6, 5, 6, 5, 6, 7, 
    6, 7, 6, 7, 6, 7, 8, 7, 8, 7, 
    8, 9, 8, 9, 8, 9, 8, 9, 10, 9,
    10
};
*/

#define probcut_a -0.0027183880227839127
#define probcut_b -0.010159330980892623
#define probcut_c 0.04069811753963199
#define probcut_d -3.126668257717306
#define probcut_e 8.513417624696324
#define probcut_f -9.55097169285451
#define probcut_g 8.03198419537373

#define probcut_end_a 0.15811777028350227
#define probcut_end_b 0.9393034706176613
#define probcut_end_c -0.0003466476344665104
#define probcut_end_d 0.026804233485840375
#define probcut_end_e -0.6919837072602527
#define probcut_end_f 9.38628573583576

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

/*
inline bool mpc_higher(Search *search, int beta, int depth, uint64_t legal, bool is_end_search){
    if ((!is_end_search && depth >= 17) || (is_end_search && depth >= 23))
        return false;
    bool res = false;
    int bound;
    if (search->board.n + depth < HW2)
        bound = beta + round(search->mpct * score_to_value(probcut_sigma(search->board.n, depth, mpcd[depth])));
    else
        bound = beta + round(search->mpct * score_to_value(probcut_sigma_end(search->board.n, mpcd[depth])));
    if (bound > SCORE_MAX)
        bound = SCORE_MAX; //return false;
    switch(mpcd[depth]){
        case 0:
            res = mid_evaluate(&search->board) >= bound;
            break;
        case 1:
            res = nega_alpha_eval1(search, bound - 1, bound, false) >= bound;
            break;
        default:
            if (mpcd[depth] <= MID_FAST_DEPTH)
                res = nega_alpha(search, bound - 1, bound, mpcd[depth], false) >= bound;
            else{
                //double mpct = search->mpct;
                //search->mpct = 1.18;
                //search->use_mpc = false;
                    res = nega_alpha_ordering_nomemo(search, bound - 1, bound, mpcd[depth], false, legal) >= bound;
                //search->use_mpc = true;
                //search->mpct = mpct;
            }
            break;
    }
    return res;
}

inline bool mpc_lower(Search *search, int alpha, int depth, uint64_t legal, bool is_end_search){
    if ((!is_end_search && depth >= 17) || (is_end_search && depth >= 23))
        return false;
    bool res = false;
    int bound;
    if (search->board.n + depth < HW2)
        bound = alpha - round(search->mpct * score_to_value(probcut_sigma(search->board.n, depth, mpcd[depth])));
    else
        bound = alpha - round(search->mpct * score_to_value(probcut_sigma_end(search->board.n, mpcd[depth])));
    if (bound < -SCORE_MAX)
        bound = -SCORE_MAX; //return false;
    switch(mpcd[depth]){
        case 0:
            res = mid_evaluate(&search->board) <= bound;
            break;
        case 1:
            res = nega_alpha_eval1(search, bound, bound + 1, false) <= bound;
            break;
        default:
            if (mpcd[depth] <= MID_FAST_DEPTH)
                res = nega_alpha(search, bound, bound + 1, mpcd[depth], false) <= bound;
            else{
                //double mpct = search->mpct;
                //search->mpct = 1.18;
                //search->use_mpc = false;
                    res = nega_alpha_ordering_nomemo(search, bound, bound + 1, mpcd[depth], false, legal) <= bound;
                //search->use_mpc = true;
                //search->mpct = mpct;
            }
            break;
    }
    return res;
}
*/

inline bool mpc(Search *search, int alpha, int beta, int depth, uint64_t legal, bool is_end_search, int *v, const bool *searching){
    if ((!is_end_search && depth >= 17) || (is_end_search && depth >= 23))
        return false;
    bool res = false;
    int search_depth;
    if (is_end_search)
        search_depth = ((depth >> 4) & 0xFE) ^ (depth & 1);
    else
        search_depth = ((depth >> 2) & 0xFE) ^ (depth & 1);
    const int depth0_value = mid_evaluate_diff(search, searching);
    int error_depth0, error_search;
    if (is_end_search){
        alpha = value_to_score_int(alpha);
        alpha -= alpha & 1;
        alpha = score_to_value(alpha);
        beta = value_to_score_int(beta);
        beta += beta & 1;
        beta = score_to_value(beta);
        error_depth0 = round(search->mpct * score_to_value(probcut_sigma_end_depth0(search->board.n)));
        error_search = round(search->mpct * score_to_value(probcut_sigma_end(search->board.n, search_depth)));
    } else{
        error_depth0 = round(search->mpct * score_to_value(probcut_sigma_depth0(search->board.n, depth)));
        error_search = round(search->mpct * score_to_value(probcut_sigma(search->board.n, depth, search_depth)));
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