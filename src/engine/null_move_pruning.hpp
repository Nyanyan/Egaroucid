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

#define USE_NULL_MOVE_PRUNING_N_DISCS 50

/*
    @brief constants for Null Move Pruning
*/
#define null_move_const_a 13.895352711158006
#define null_move_const_b -1.4594275635787501

#define null_move_pruning_a 2.9926628871203254
#define null_move_pruning_b -0.08800506474056283
#define null_move_pruning_c 1.370776862257715
#define null_move_pruning_d -5.274245516055893
#define null_move_pruning_e 7.126241341550173
#define null_move_pruning_f -0.9032409632420645

/*
    @brief Null Move Pruning value conversion

    @param v                    null move value
    @return expected value
*/
inline double nmp_convert(int v){
    return null_move_const_a * v + null_move_const_b;
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
    const double null_move_value = nmp_convert(mid_evaluate_diff_pass(search));
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
