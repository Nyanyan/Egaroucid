#pragma once
#include <iostream>
#include "board.hpp"
#include "evaluate.hpp"
#include "search.hpp"
#include "midsearch.hpp"
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

#define probcut_a -0.1237043372413518
#define probcut_b -0.10181021370042476
#define probcut_c 1.1166092914902872
#define probcut_d -0.0001047218699444686
#define probcut_e 0.01093547044745102
#define probcut_f -0.36362643370418024
#define probcut_g 4.071950100378323

#define probcut_end_a -0.007067691798918974
#define probcut_end_b -0.03625767020840132
#define probcut_end_c 7.057010691277178
#define probcut_end_d 21.002429687642383
#define probcut_end_e 21.02101092545754
#define probcut_end_f 10.510201246471908

inline double probcut_sigma(int n_stones, int depth1, int depth2){
    double w = n_stones;
    double x = depth1;
    double y = depth2;
    double res = probcut_a * w + probcut_b * x + probcut_c * y;
    res = probcut_d * res * res * res + probcut_e * res * res + probcut_f * res + probcut_g;
    return res;
}

inline double probcut_sigma_depth0(int n_stones, int depth1){
    double w = n_stones;
    double x = depth1;
    double res = probcut_a * w + probcut_b * x;
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

int nega_alpha_eval1(Search *search, int alpha, int beta, bool skipped);
int nega_alpha(Search *search, int alpha, int beta, int depth, bool skipped);
int nega_alpha_ordering_nomemo(Search *search, int alpha, int beta, int depth, bool skipped, uint64_t legal);

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

inline bool mpc(Search *search, int alpha, int beta, int depth, uint64_t legal, bool is_end_search, int *v){
    if ((!is_end_search && depth >= 17) || (is_end_search && depth >= 23))
        return false;
    bool res = false;
    int search_depth = depth >> 4;
    const int depth0_value = mid_evaluate(&search->board);
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
    if (depth0_value >= beta - error_depth0 && beta + error_search <= SCORE_MAX){
        switch(search_depth){
            case 0:
                res = depth0_value >= beta + error_search;
                break;
            case 1:
                res = nega_alpha_eval1(search, beta + error_search - 1, beta + error_search, false) >= beta + error_search;
                break;
            default:
                if (search_depth <= MID_FAST_DEPTH)
                    res = nega_alpha(search, beta + error_search - 1, beta + error_search, search_depth, false) >= beta + error_search;
                else{
                    //double mpct = search->mpct;
                    //search->mpct = 1.18;
                    //search->use_mpc = false;
                        res = nega_alpha_ordering_nomemo(search, beta + error_search - 1, beta + error_search, search_depth, false, legal) >= beta + error_search;
                    //search->use_mpc = true;
                    //search->mpct = mpct;
                }
                break;
        }
        if (res){
            *v = beta;
            return true;
        }
    }
    if (depth0_value <= alpha + error_depth0 && alpha - error_search >= -SCORE_MAX){
        switch(search_depth){
            case 0:
                res = depth0_value <= alpha - error_search;
                break;
            case 1:
                res = nega_alpha_eval1(search, alpha - error_search, alpha - error_search + 1, false) <= alpha - error_search;
                break;
            default:
                if (search_depth <= MID_FAST_DEPTH)
                    res = nega_alpha(search, alpha - error_search, alpha - error_search + 1, search_depth, false) <= alpha - error_search;
                else{
                    //double mpct = search->mpct;
                    //search->mpct = 1.18;
                    //search->use_mpc = false;
                        res = nega_alpha_ordering_nomemo(search, alpha - error_search, alpha - error_search + 1, search_depth, false, legal) <= alpha - error_search;
                    //search->use_mpc = true;
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