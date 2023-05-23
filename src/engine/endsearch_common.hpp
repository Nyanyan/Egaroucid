/*
    Egaroucid Project

    @file endsearch_common.hpp
        Common things for endgame search
    @date 2021-2023
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

#if USE_SIMD
    __m128i parity_ordering_shuffle_mask[64];
#endif

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

void endsearch_init(){
    #if USE_SIMD
        constexpr uint32_t parity_ordering_shuffle_mask_32bit[64] = {
            0x3020100U, 0x3020100U, 0x3020100U, 0x3020100U, 0x3020100U, 0x3020100U, 0x3020100U, 0x3020100U, 
            0x3020100U, 0x3020100U, 0x1000302U, 0x3020100U, 0x3000201U, 0x3000201U, 0x3020100U, 0x2000301U,
            0x3020100U, 0x1000302U, 0x3020100U, 0x1000302U, 0x3020100U, 0x3020100U, 0x3020100U, 0x3020100U,
            0x3020100U, 0x3020100U, 0x1000302U, 0x3020100U, 0x3010200U, 0x2010300U, 0x3020100U, 0x3010200U,
            0x3020100U, 0x3000201U, 0x3020100U, 0x3010200U, 0x3020100U, 0x3000201U, 0x3020100U, 0x3010200U,
            0x3020100U, 0x3000201U, 0x3020100U, 0x2010300U, 0x3000201U, 0x3020100U, 0x2010300U, 0x3020100U,
            0x3020100U, 0x3020100U, 0x3020100U, 0x3020100U, 0x3020100U, 0x2010300U, 0x3020100U, 0x2000301U,
            0x3020100U, 0x2000301U, 0x3020100U, 0x3010200U, 0x3010200U, 0x3020100U, 0x2000301U, 0x3020100U
        };
        for (int i = 0; i < 64; ++i)
            parity_ordering_shuffle_mask[i] = _mm_cvtsi32_si128(parity_ordering_shuffle_mask_32bit[i]);
    #endif
}