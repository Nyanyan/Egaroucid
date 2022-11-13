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
constexpr double null_move_const_a[2] = {-61.3776920479145, -42.489688728004666};
constexpr double null_move_const_b[2] = {-6.889455582792511, 0.7090865608885384};
constexpr double null_move_const_c[2] = {-22.020567864648946, -29.95624285424537};
constexpr double null_move_const_d[2] = {-1.7492521920534927, -2.1414654948687826};

#define null_move_pruning_a 1.4906023693536687
#define null_move_pruning_b 0.01547301437682099
#define null_move_pruning_c 15.746679229773422
#define null_move_pruning_d -35.066778220644736
#define null_move_pruning_e 25.913544235113513
#define null_move_pruning_f 0.07597292971713729

/*
    @brief Null Move Pruning value conversion

    @param v                    null move value
    @return expected value
*/
inline double nmp_convert(int v, uint_fast8_t parity){
    double vv = (double)v / 64;
    double vv2 = vv * vv;
    return null_move_const_a[parity] * vv2 * vv + null_move_const_b[parity] * vv2 + null_move_const_c[parity] * vv + null_move_const_d[parity];
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
