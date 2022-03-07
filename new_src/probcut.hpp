#pragma once
#include <iostream>
#include "board.hpp"
#include "evaluate.hpp"
#include "search.hpp"
#include "midsearch.hpp"

using namespace std;

constexpr int mpcd[61] = {
    0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 
    4, 3, 4, 3, 4, 5, 4, 5, 6, 5, 
    6, 5, 6, 7, 6, 7, 8, 7, 8, 7, 
    8, 9, 8, 9, 10, 9, 10, 9, 10, 11, 
    10, 11, 12, 11, 12, 11, 12, 13, 12, 13, 
    14, 13, 14, 13, 14, 15, 14, 15, 16, 15,
    16
};

inline double probcut_sigma(int n_stones){
    return 1.0 + 0.05 * n_stones;
}

int nega_alpha_eval1(Search *search, int alpha, int beta, bool skipped);
int nega_alpha(Search *search, int alpha, int beta, int depth, bool skipped);
int nega_alpha_ordering_nomemo(Search *search, int alpha, int beta, int depth, bool skipped, uint64_t legal);

inline bool mpc_higher(Search *search, int beta, int depth, uint64_t legal){
    int bound = beta + ceil(search->mpct * probcut_sigma(search->board.n));
    if (bound > HW2)
        bound = HW2; //return false;
    bool res;
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
                search->use_mpc = false;
                    res = nega_alpha_ordering_nomemo(search, bound - 1, bound, mpcd[depth], false, legal) >= bound;
                search->use_mpc = true;
                //search->mpct = mpct;
            }
            break;
    }
    return res;
}

inline bool mpc_lower(Search *search, int alpha, int depth, uint64_t legal){
    int bound = alpha - ceil(search->mpct * probcut_sigma(search->board.n));
    if (bound < -HW2)
        bound = -HW2; //return false;
    bool res;
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
                search->use_mpc = false;
                    res = nega_alpha_ordering_nomemo(search, bound, bound + 1, mpcd[depth], false, legal) <= bound;
                search->use_mpc = true;
                //search->mpct = mpct;
            }
            break;
    }
    return res;
}