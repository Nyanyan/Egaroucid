/*
    Egaroucid Project

    @file flip_generic.hpp
        Flip calculation without AVX2
    @date 2021-2023
    @author Takuto Yamana (a.k.a. Nyanyan)
    @license GPL-3.0 license
    @notice I referred to codes written by others
*/

#pragma once
#include "setting.hpp"
#include "common.hpp"
#include "bit.hpp"

uint8_t flip_pre_calc[N_8BIT][N_8BIT][HW];
uint64_t line_to_board_d7[N_8BIT][HW * 2];
uint64_t line_to_board_d9[N_8BIT][HW * 2];

inline uint64_t calc_flip(const uint64_t player, const uint64_t opponent, const int place){
    const int t = place >> 3;
    const int u = place & 7;
    uint64_t flip = split_h_line(flip_pre_calc[join_h_line(player, t)][join_h_line(opponent, t)][u], t);
    flip |= split_v_line(flip_pre_calc[join_v_line(player, u)][join_v_line(opponent, u)][t], u);
    flip |= split_d7_line(flip_pre_calc[join_d7_line(player, u + t)][join_d7_line(opponent, u + t)][std::min(t, 7 - u)], u + t);
    flip |= split_d9_line(flip_pre_calc[join_d9_line(player, u + 7 - t)][join_d9_line(opponent, u + 7 - t)][std::min(t, u)], u + 7 - t);
    return flip;
}

void flip_init(){
    int player, opponent, place;
    int wh, put, m1, m2, m3, m4, m5, m6;
    for (player = 0; player < N_8BIT; ++player){
        for (opponent = 0; opponent < N_8BIT; ++opponent){
            for (place = 0; place < HW; ++place){
                flip_pre_calc[player][opponent][place] = 0;
                if ((1 & (player >> place)) == 0 && (1 & (opponent >> place)) == 0 && (player & opponent) == 0){
                    put = 1 << place;
                    wh = opponent & 0b01111110;
                    m1 = put >> 1;
                    if( (m1 & wh) != 0 ) {
                        if( ((m2 = m1 >> 1) & wh) == 0  ) {
                            if( (m2 & player) != 0 )
                                flip_pre_calc[player][opponent][place] |= m1;
                        } else if( ((m3 = m2 >> 1) & wh) == 0 ) {
                            if( (m3 & player) != 0 )
                                flip_pre_calc[player][opponent][place] |= m1 | m2;
                        } else if( ((m4 = m3 >> 1) & wh) == 0 ) {
                            if( (m4 & player) != 0 )
                                flip_pre_calc[player][opponent][place] |= m1 | m2 | m3;
                        } else if( ((m5 = m4 >> 1) & wh) == 0 ) {
                            if( (m5 & player) != 0 )
                                flip_pre_calc[player][opponent][place] |= m1 | m2 | m3 | m4;
                        } else if( ((m6 = m5 >> 1) & wh) == 0 ) {
                            if( (m6 & player) != 0 )
                                flip_pre_calc[player][opponent][place] |= m1 | m2 | m3 | m4 | m5;
                        } else {
                            if( ((m6 >> 1) & player) != 0 )
                                flip_pre_calc[player][opponent][place] |= m1 | m2 | m3 | m4 | m5 | m6;
                        }
                    }
                    m1 = put << 1;
                    if( (m1 & wh) != 0 ) {
                        if( ((m2 = m1 << 1) & wh) == 0  ) {
                            if( (m2 & player) != 0 )
                                flip_pre_calc[player][opponent][place] |= m1;
                        } else if( ((m3 = m2 << 1) & wh) == 0 ) {
                            if( (m3 & player) != 0 )
                                flip_pre_calc[player][opponent][place] |= m1 | m2;
                        } else if( ((m4 = m3 << 1) & wh) == 0 ) {
                            if( (m4 & player) != 0 )
                                flip_pre_calc[player][opponent][place] |= m1 | m2 | m3;
                        } else if( ((m5 = m4 << 1) & wh) == 0 ) {
                            if( (m5 & player) != 0 )
                                flip_pre_calc[player][opponent][place] |= m1 | m2 | m3 | m4;
                        } else if( ((m6 = m5 << 1) & wh) == 0 ) {
                            if( (m6 & player) != 0 )
                                flip_pre_calc[player][opponent][place] |= m1 | m2 | m3 | m4 | m5;
                        } else {
                            if( ((m6 << 1) & player) != 0 )
                                flip_pre_calc[player][opponent][place] |= m1 | m2 | m3 | m4 | m5 | m6;
                        }
                    }
                }
            }
        }
    }
}