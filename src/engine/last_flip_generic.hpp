/*
    Egaroucid Project

    @file last_flip_generic.hpp
        calculate number of flipped discs in the last move without AVX2
    @date 2021-2022
    @author Takuto Yamana (a.k.a. Nyanyan)
    @license GPL-3.0 license
*/

#pragma once
#include "setting.hpp"
#include "common.hpp"
#include "bit.hpp"



/*
    @brief calculate number of flipped discs in the last move

    @param player               a bitboard representing player
    @param place                a place to put
    @return number of flipping discs
*/
inline int count_last_flip(uint64_t player, const uint_fast8_t place){
    const int t = place >> 3;
    const int u = place & 7;
    return
        n_flip_pre_calc[join_h_line(player, t)][u] + 
        n_flip_pre_calc[join_v_line(player, u)][t] + 
        n_flip_pre_calc[join_d7_line(player, u + t)][std::min(t, 7 - u)] + 
        n_flip_pre_calc[join_d9_line(player, u + 7 - t)][std::min(t, u)];
    /*
    int t, u, p;
    int res = 0;

    t = place / HW;
    u = place % HW;
    p = (player >> (HW * t)) & 0b11111111;
    res += n_flip_pre_calc[p][u];

    p = join_v_line(player, u);
    res += n_flip_pre_calc[p][t];

    t = place / HW;
    u = place % HW + t;
    if (u >= 2 && u <= 12){
        p = join_d7_line(player, u) & d7_mask[place];
        res += n_flip_pre_calc[p][HW_M1 - t];
    }

    u -= t * 2;
    if (u >= -5 && u <= 5){
        p = join_d9_line(player, u) & d9_mask[place];
        res += n_flip_pre_calc[p][t];
    }
    return res;
    */
}