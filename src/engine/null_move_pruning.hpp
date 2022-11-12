/*
    Egaroucid Project

    @file null_move_pruning.hpp
        ProbCut with Null Move Pruning
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

#define USE_NULL_MOVE_PRUNING_N_DISCS 51
#define USE_NULL_MOVE_PRUNING_DEPTH 9

/*
    @brief constants for Null Move Pruning
*/
constexpr double null_move_const_a[2] = {-0.8503110335696682, -0.8793653592885485};
constexpr double null_move_const_b[2] = {-5.7281549015782, -4.222035987304141};

#define null_move_pruning_a 1.7953883211601585
#define null_move_pruning_b 0.19563822250898116
#define null_move_pruning_c 14.492851079667709
#define null_move_pruning_d -37.67736919749262
#define null_move_pruning_e 31.32102383480849
#define null_move_pruning_f -1.2126603393649316

/*
    @brief Null Move Pruning value conversion

    @param v                    null move value
    @return expected value
*/
inline double nmp_convert(int v, uint_fast8_t parity){
    return null_move_const_a[parity] * v + null_move_const_b[parity];
}

/*
    @brief Null Move Pruning error calculation for midgame

    @param n_discs              number of discs on the board
    @param depth                depth of deep search
    @return expected error
*/
inline double nmp_sigma(int n_discs, int depth){
    double res = null_move_pruning_a * ((double)n_discs / 64.0) + null_move_pruning_b * ((double)depth / 60.0);
    res = null_move_pruning_c * res * res * res + null_move_pruning_d * res * res + null_move_pruning_e * res + null_move_pruning_f;
    return res;
}

/*
    @brief Null Move Pruning for normal search

    @param search               search information
    @param alpha                alpha value
    @param beta                 beta value
    @param depth                depth of deep search
    @param v                    an integer to store result
    @return cutoff occurred?
*/
inline bool nmp(Search *search, int alpha, int beta, int depth, int *v){
    double error = search->mpct * nmp_sigma(search->n_discs, depth);
    const double null_move_value = nmp_convert(mid_evaluate_diff_pass(search), search->n_discs & 1);
    if (null_move_value <= (double)alpha - error){
        *v = alpha;
        return true;
    }
    if (null_move_value >= (double)beta + error){
        *v = beta;
        return true;
    }
    return false;
}

/*
    @brief Null Move Pruning for NWS (Null Window Search)

    @param search               search information
    @param alpha                alpha value (beta = alpha + 1)
    @param depth                depth of deep search
    @param v                    an integer to store result
    @return cutoff occurred?
*/
inline bool nmp_nws(Search *search, int alpha, int depth, int *v){
    return nmp(search, alpha, alpha + 1, depth, v);
}
