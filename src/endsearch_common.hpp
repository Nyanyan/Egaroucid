/*
    Egaroucid Project

    @date 2021-2022
    @author Takuto Yamana (a.k.a Nyanyan)
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

using namespace std;

inline int last1(Search *search, int alpha, uint_fast8_t p0){
    ++search->n_nodes;
    int score = HW2 - 2 * search->board.count_opponent();
    int n_flip;
    n_flip = count_last_flip(search->board.player, search->board.opponent, p0);
    if (n_flip == 0){
        ++search->n_nodes;
        if (score <= 0){
            score -= 2;
            if (score >= alpha){
                n_flip = count_last_flip(search->board.opponent, search->board.player, p0);
                score -= 2 * n_flip;
            }
        } else{
            if (score >= alpha){
                n_flip = count_last_flip(search->board.opponent, search->board.player, p0);
                if (n_flip)
                    score -= 2 * n_flip + 2;
            }
        }
    } else
        score += 2 * n_flip;
    return score;
}