/*
    Egaroucid Project

    @file last_flip_avx2.hpp
        calculate number of flipped discs in the last move with AVX2
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
    uint64_t opponent = (~player) ^ (1ULL << place);
    int t, u, p, o;
    int res = 0;

    t = place / HW;
    u = place % HW;
    p = (player >> (HW * t)) & 0b11111111;
    o = (opponent >> (HW * t)) & 0b11111111;
    res += n_flip_pre_calc[p][o][u];

    p = join_v_line(player, u);
    o = join_v_line(opponent, u);
    res += n_flip_pre_calc[p][o][t];

    t = place / HW;
    u = place % HW + t;
    if (u >= 2 && u <= 12){
        p = join_d7_line(player, u) & d7_mask[place];
        o = join_d7_line(opponent, u) & d7_mask[place];
        res += pop_count_uchar(flip_pre_calc[p][o][HW_M1 - t] & d7_mask[place]);
    }

    u -= t * 2;
    if (u >= -5 && u <= 5){
        p = join_d9_line(player, u) & d9_mask[place];
        o = join_d9_line(opponent, u) & d9_mask[place];
        res += pop_count_uchar(flip_pre_calc[p][o][t] & d9_mask[place]);
    }
    return res;
}