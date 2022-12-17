/*
    Egaroucid Project

    @file endsearch_common.hpp
        Common things for endgame search
    @date 2021-2022
    @author Takuto Yamana (a.k.a. Nyanyan)
    @license GPL-3.0 license
*/

#pragma once
#include <iostream>
#include <vector>
#include <functional>
#include "setting.hpp"
#include "common.hpp"
#include "board.hpp"
#include "search.hpp"

/*
    @brief Get a final score with last 1 empty

    Special optimization from an idea of https://github.com/abulmo/edax-reversi/blob/1ae7c9fe5322ac01975f1b3196e788b0d25c1e10/src/endgame.c#L85

    @param search               search information
    @param alpha                alpha value
    @param p0                   last empty square
    @return the final score
*/
inline int last1(Search *search, int alpha, uint_fast8_t p0){
    ++search->n_nodes;
    #if USE_SEARCH_STATISTICS
        ++search->n_nodes_discs[search->n_discs];
    #endif
    int score = HW2 - 2 * search->board.count_opponent();
    int n_flip;
    n_flip = count_last_flip(search->board.player, p0);
    if (n_flip == 0){
        ++search->n_nodes;
        if (score <= 0){
            score -= 2;
            if (score >= alpha){
                n_flip = count_last_flip(search->board.opponent, p0);
                score -= 2 * n_flip;
            }
        } else{
            if (score >= alpha){
                n_flip = count_last_flip(search->board.opponent, p0);
                if (n_flip)
                    score -= 2 * n_flip + 2;
            }
        }
    } else
        score += 2 * n_flip;
    return score;
}