/*
    Egaroucid Project

    @file stability_cutoff.hpp
        Cutoff using stable discs
    @date 2021-2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once
#include "stability.hpp"

/*
    @brief Stability cutoff

    If P (number of player's stable discs) and O (number of opponent's stable discs) are calculated, 
    then the final score should be 2 * P - 64 <= final_score <= 64 - 2 * O.
    Using this formula, we can narrow the search window.

    @param search               search information
    @param alpha                alpha value
    @param beta                 beta value
    @return SCORE_UNDEFINED if no cutoff found else the score
*/
inline int stability_cut(Search *search, int *alpha, int *beta) {
    if (*beta >= stability_threshold[search->n_discs]) {
        int n_beta = HW2 - 2 * pop_count_ull(calc_stability(search->board.opponent, search->board.player));
        if (n_beta <= *alpha) {
            return n_beta;
        } else if (n_beta < *beta) {
            *beta = n_beta;
        }
    }
    return SCORE_UNDEFINED;
}

// last4 (min stage)
inline int stability_cut_last4(Search *search, int *alpha, int beta) {
    if (*alpha <= -stability_threshold[60]) {
        int n_alpha = 2 * pop_count_ull(calc_stability(search->board.opponent, search->board.player)) - HW2;
        if (n_alpha >= beta) {
            return n_alpha;
        } else if (n_alpha > *alpha) {
            *alpha = n_alpha;
        }
    }
    return SCORE_UNDEFINED;
}

/*
    @brief Stability cutoff for NWS (Null Window Search)

    If P (number of player's stable discs) and O (number of opponent's stable discs) are calculated, 
    then the final score should be 2 * P - 64 <= final_score <= 64 - 2 * O.
    Using this formula, we can narrow the search window.

    @param search               search information
    @param alpha                alpha value (beta = alpha + 1)
    @return SCORE_UNDEFINED if no cutoff found else the score
*/
inline int stability_cut_nws(Search *search, int alpha) {
    if (alpha >= stability_threshold_nws[search->n_discs]) {
        int n_beta = HW2 - 2 * pop_count_ull(calc_stability(search->board.opponent, search->board.player));
        if (n_beta <= alpha) {
            return n_beta;
        }
    }
    return SCORE_UNDEFINED;
}

// last4 (min stage)
inline int stability_cut_last4_nws(Search *search, int alpha) {
    if (alpha < -stability_threshold_nws[60]) {
        int n_alpha = 2 * pop_count_ull(calc_stability(search->board.opponent, search->board.player)) - HW2;
        if (n_alpha > alpha) {
            return n_alpha;
        }
    }
    return SCORE_UNDEFINED;
}