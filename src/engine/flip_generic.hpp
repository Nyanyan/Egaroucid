/*
    Egaroucid Project

    @file flip_generic.hpp
        Flip calculation without AVX2
    @date 2021-2022
    @author Takuto Yamana (a.k.a. Nyanyan)
    @license GPL-3.0 license
    @notice I referred to codes written by others
*/

#pragma once
#include "setting.hpp"
#include "common.hpp"
#include "bit.hpp"

uint8_t flip_pre_calc[N_8BIT][N_8BIT][HW];
uint8_t n_flip_pre_calc[N_8BIT][N_8BIT][HW];
uint64_t line_to_board_v[N_8BIT][HW];
uint64_t line_to_board_d7[N_8BIT][HW * 2];
uint64_t line_to_board_d9[N_8BIT][HW * 2];

/*
    @brief Flip class

    @param pos                  a cell to put disc
    @param flip                 a bitboard representing flipped discs
*/
class Flip{
    public:
        uint_fast8_t pos;
        uint64_t flip;
    
    public:
        inline void calc_flip(const uint64_t player, const uint64_t opponent, const int place){
            int t, u, p, o;
            flip = 0;
            pos = place;

            t = place / HW;
            u = place % HW;
            p = (player >> (HW * t)) & 0b11111111;
            o = (opponent >> (HW * t)) & 0b11111111;
            flip |= (uint64_t)flip_pre_calc[p][o][u] << (HW * t);

            p = join_v_line(player, u);
            o = join_v_line(opponent, u);
            flip |= line_to_board_v[flip_pre_calc[p][o][t]][u];

            u += t;
            if (u >= 2 && u <= 12){
                p = join_d7_line(player, u) & d7_mask[place];
                o = join_d7_line(opponent, u) & d7_mask[place];
                flip |= line_to_board_d7[flip_pre_calc[p][o][HW_M1 - t]][u];
            }

            u -= t * 2;
            if (u >= -5 && u <= 5){
                p = join_d9_line(player, u) & d9_mask[place];
                o = join_d9_line(opponent, u) & d9_mask[place];
                flip |= line_to_board_d9[flip_pre_calc[p][o][t]][u + HW];
            }
        }
};

void flip_init(){
    int player, opponent, place;
    int wh, put, m1, m2, m3, m4, m5, m6;
    int idx, t, i;
    for (player = 0; player < N_8BIT; ++player){
        for (opponent = 0; opponent < N_8BIT; ++opponent){
            for (place = 0; place < HW; ++place){
                flip_pre_calc[player][opponent][place] = 0;
                n_flip_pre_calc[player][opponent][place] = 0;
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
                    n_flip_pre_calc[player][opponent][place] = 0;
                    for (i = 1; i < HW_M1; ++i)
                        n_flip_pre_calc[player][opponent][place] += 1 & (flip_pre_calc[player][opponent][place] >> i);
                }
            }
        }
    }
    for (idx = 0; idx < N_8BIT; ++idx){
        for (t = 0; t < HW; ++t)
            line_to_board_v[idx][t] = split_v_line((uint8_t)idx, t);
        for (t = 0; t < HW * 2; ++t)
            line_to_board_d7[idx][t] = split_d7_line((uint8_t)idx, t);
        for (t = -HW; t < HW; ++t)
            line_to_board_d9[idx][t + HW] = split_d9_line((uint8_t)idx, t);
    }
}