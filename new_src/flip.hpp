#pragma once
#include <iostream>
#include "setting.hpp"
#include "common.hpp"
#include "bit.hpp"

using namespace std;

uint8_t flip_pre_calc[N_8BIT][N_8BIT][HW];
uint8_t n_flip_pre_calc[N_8BIT][N_8BIT][HW];

class Flip{
    public:
        int pos;
        unsigned long long flip;
        int value;
    
    public:
        inline void copy(Flip *mob) const{
            mob->pos = pos;
            mob->flip = flip;
            mob->value = value;
        }

        #if FLIP_CALC_MODE == 0
            inline void calc_flip(const unsigned long long player, const unsigned long long opponent, const int place){
                unsigned long long wh, put, m1, m2, m3, m4, m5, m6, rev;
                put = 1ULL << place;
                rev = 0;

                wh = opponent & 0b0111111001111110011111100111111001111110011111100111111001111110ULL;
                m1 = put >> 1;
                if( (m1 & wh) != 0 ) {
                    if( ((m2 = m1 >> 1) & wh) == 0  ) {
                        if( (m2 & player) != 0 )
                            rev |= m1;
                    } else if( ((m3 = m2 >> 1) & wh) == 0 ) {
                        if( (m3 & player) != 0 )
                            rev |= m1 | m2;
                    } else if( ((m4 = m3 >> 1) & wh) == 0 ) {
                        if( (m4 & player) != 0 )
                            rev |= m1 | m2 | m3;
                    } else if( ((m5 = m4 >> 1) & wh) == 0 ) {
                        if( (m5 & player) != 0 )
                            rev |= m1 | m2 | m3 | m4;
                    } else if( ((m6 = m5 >> 1) & wh) == 0 ) {
                        if( (m6 & player) != 0 )
                            rev |= m1 | m2 | m3 | m4 | m5;
                    } else {
                        if( ((m6 >> 1) & player) != 0 )
                            rev |= m1 | m2 | m3 | m4 | m5 | m6;
                    }
                }
                m1 = put << 1;
                if( (m1 & wh) != 0 ) {
                    if( ((m2 = m1 << 1) & wh) == 0  ) {
                        if( (m2 & player) != 0 )
                            rev |= m1;
                    } else if( ((m3 = m2 << 1) & wh) == 0 ) {
                        if( (m3 & player) != 0 )
                            rev |= m1 | m2;
                    } else if( ((m4 = m3 << 1) & wh) == 0 ) {
                        if( (m4 & player) != 0 )
                            rev |= m1 | m2 | m3;
                    } else if( ((m5 = m4 << 1) & wh) == 0 ) {
                        if( (m5 & player) != 0 )
                            rev |= m1 | m2 | m3 | m4;
                    } else if( ((m6 = m5 << 1) & wh) == 0 ) {
                        if( (m6 & player) != 0 )
                            rev |= m1 | m2 | m3 | m4 | m5;
                    } else {
                        if( ((m6 << 1) & player) != 0 )
                            rev |= m1 | m2 | m3 | m4 | m5 | m6;
                    }
                }

                wh = opponent & 0b0000000011111111111111111111111111111111111111111111111100000000ULL;
                m1 = put >> HW;
                if( (m1 & wh) != 0 ) {
                    if( ((m2 = m1 >> HW) & wh) == 0  ) {
                        if( (m2 & player) != 0 )
                            rev |= m1;
                    } else if( ((m3 = m2 >> HW) & wh) == 0 ) {
                        if( (m3 & player) != 0 )
                            rev |= m1 | m2;
                    } else if( ((m4 = m3 >> HW) & wh) == 0 ) {
                        if( (m4 & player) != 0 )
                            rev |= m1 | m2 | m3;
                    } else if( ((m5 = m4 >> HW) & wh) == 0 ) {
                        if( (m5 & player) != 0 )
                            rev |= m1 | m2 | m3 | m4;
                    } else if( ((m6 = m5 >> HW) & wh) == 0 ) {
                        if( (m6 & player) != 0 )
                            rev |= m1 | m2 | m3 | m4 | m5;
                    } else {
                        if( ((m6 >> HW) & player) != 0 )
                            rev |= m1 | m2 | m3 | m4 | m5 | m6;
                    }
                }
                m1 = put << HW;
                if( (m1 & wh) != 0 ) {
                    if( ((m2 = m1 << HW) & wh) == 0  ) {
                        if( (m2 & player) != 0 )
                            rev |= m1;
                    } else if( ((m3 = m2 << HW) & wh) == 0 ) {
                        if( (m3 & player) != 0 )
                            rev |= m1 | m2;
                    } else if( ((m4 = m3 << HW) & wh) == 0 ) {
                        if( (m4 & player) != 0 )
                            rev |= m1 | m2 | m3;
                    } else if( ((m5 = m4 << HW) & wh) == 0 ) {
                        if( (m5 & player) != 0 )
                            rev |= m1 | m2 | m3 | m4;
                    } else if( ((m6 = m5 << HW) & wh) == 0 ) {
                        if( (m6 & player) != 0 )
                            rev |= m1 | m2 | m3 | m4 | m5;
                    } else {
                        if( ((m6 << HW) & player) != 0 )
                            rev |= m1 | m2 | m3 | m4 | m5 | m6;
                    }
                }

                wh = opponent & 0b0000000001111110011111100111111001111110011111100111111000000000ULL;
                m1 = put >> (HW - 1);
                if( (m1 & wh) != 0 ) {
                    if( ((m2 = m1 >> (HW - 1)) & wh) == 0  ) {
                        if( (m2 & player) != 0 )
                            rev |= m1;
                    } else if( ((m3 = m2 >> (HW - 1)) & wh) == 0 ) {
                        if( (m3 & player) != 0 )
                            rev |= m1 | m2;
                    } else if( ((m4 = m3 >> (HW - 1)) & wh) == 0 ) {
                        if( (m4 & player) != 0 )
                            rev |= m1 | m2 | m3;
                    } else if( ((m5 = m4 >> (HW - 1)) & wh) == 0 ) {
                        if( (m5 & player) != 0 )
                            rev |= m1 | m2 | m3 | m4;
                    } else if( ((m6 = m5 >> (HW - 1)) & wh) == 0 ) {
                        if( (m6 & player) != 0 )
                            rev |= m1 | m2 | m3 | m4 | m5;
                    } else {
                        if( ((m6 >> (HW - 1)) & player) != 0 )
                            rev |= m1 | m2 | m3 | m4 | m5 | m6;
                    }
                }
                m1 = put << (HW - 1);
                if( (m1 & wh) != 0 ) {
                    if( ((m2 = m1 << (HW - 1)) & wh) == 0  ) {
                        if( (m2 & player) != 0 )
                            rev |= m1;
                    } else if( ((m3 = m2 << (HW - 1)) & wh) == 0 ) {
                        if( (m3 & player) != 0 )
                            rev |= m1 | m2;
                    } else if( ((m4 = m3 << (HW - 1)) & wh) == 0 ) {
                        if( (m4 & player) != 0 )
                            rev |= m1 | m2 | m3;
                    } else if( ((m5 = m4 << (HW - 1)) & wh) == 0 ) {
                        if( (m5 & player) != 0 )
                            rev |= m1 | m2 | m3 | m4;
                    } else if( ((m6 = m5 << (HW - 1)) & wh) == 0 ) {
                        if( (m6 & player) != 0 )
                            rev |= m1 | m2 | m3 | m4 | m5;
                    } else {
                        if( ((m6 << (HW - 1)) & player) != 0 )
                            rev |= m1 | m2 | m3 | m4 | m5 | m6;
                    }
                }

                m1 = put >> (HW + 1);
                if( (m1 & wh) != 0 ) {
                    if( ((m2 = m1 >> (HW + 1)) & wh) == 0  ) {
                        if( (m2 & player) != 0 )
                            rev |= m1;
                    } else if( ((m3 = m2 >> (HW + 1)) & wh) == 0 ) {
                        if( (m3 & player) != 0 )
                            rev |= m1 | m2;
                    } else if( ((m4 = m3 >> (HW + 1)) & wh) == 0 ) {
                        if( (m4 & player) != 0 )
                            rev |= m1 | m2 | m3;
                    } else if( ((m5 = m4 >> (HW + 1)) & wh) == 0 ) {
                        if( (m5 & player) != 0 )
                            rev |= m1 | m2 | m3 | m4;
                    } else if( ((m6 = m5 >> (HW + 1)) & wh) == 0 ) {
                        if( (m6 & player) != 0 )
                            rev |= m1 | m2 | m3 | m4 | m5;
                    } else {
                        if( ((m6 >> (HW + 1)) & player) != 0 )
                            rev |= m1 | m2 | m3 | m4 | m5 | m6;
                    }
                }
                m1 = put << (HW + 1);
                if( (m1 & wh) != 0 ) {
                    if( ((m2 = m1 << (HW + 1)) & wh) == 0  ) {
                        if( (m2 & player) != 0 )
                            rev |= m1;
                    } else if( ((m3 = m2 << (HW + 1)) & wh) == 0 ) {
                        if( (m3 & player) != 0 )
                            rev |= m1 | m2;
                    } else if( ((m4 = m3 << (HW + 1)) & wh) == 0 ) {
                        if( (m4 & player) != 0 )
                            rev |= m1 | m2 | m3;
                    } else if( ((m5 = m4 << (HW + 1)) & wh) == 0 ) {
                        if( (m5 & player) != 0 )
                            rev |= m1 | m2 | m3 | m4;
                    } else if( ((m6 = m5 << (HW + 1)) & wh) == 0 ) {
                        if( (m6 & player) != 0 )
                            rev |= m1 | m2 | m3 | m4 | m5;
                    } else {
                        if( ((m6 << (HW + 1)) & player) != 0 )
                            rev |= m1 | m2 | m3 | m4 | m5 | m6;
                    }
                }

                flip = rev;
                pos = place;
            }
            
        #elif FLIP_CALC_MODE == 1
            inline void calc_flip(const unsigned long long player, const unsigned long long opponent, const int place){
                int t, u, p, o;
                flip = 0;
                pos = place;

                t = place / HW;
                u = place % HW;
                p = (player >> (HW * t)) & 0b11111111;
                o = (opponent >> (HW * t)) & 0b11111111;
                flip |= (unsigned long long)flip_pre_calc[p][o][u] << (HW * t);

                p = join_v_line(player, u);
                o = join_v_line(opponent, u);
                flip |= split_v_lines[flip_pre_calc[p][o][t]] << u;

                u += t;
                if (u >= 2 && u <= 12){
                    p = join_d7_line(player, u) & d7_mask[place];
                    o = join_d7_line(opponent, u) & d7_mask[place];
                    flip |= split_d7_lines[flip_pre_calc[p][o][HW_M1 - t]][u - 2];
                }

                u -= t * 2;
                if (u >= -5 && u <= 5){
                    p = join_d9_line(player, u) & d9_mask[place];
                    o = join_d9_line(opponent, u) & d9_mask[place];
                    flip |= split_d9_lines[flip_pre_calc[p][o][t]][u + 5];
                }
            }
        #elif FLIP_CALC_MODE == 2
            inline void calc_flip(const unsigned long long player, const unsigned long long opponent, const int place){
                int t, u, p, o;
                flip = 0;
                pos = place;

                t = place / HW;
                u = place % HW;
                p = (player >> (HW * t)) & 0b11111111;
                o = (opponent >> (HW * t)) & 0b11111111;
                flip |= (unsigned long long)flip_pre_calc[p][o][u] << (HW * t);

                p = join_v_line(player, u);
                o = join_v_line(opponent, u);
                flip |= split_v_lines[flip_pre_calc[p][o][t]] << u;

                u += t;
                if (u >= 2 && u <= 12){
                    p = join_d7_lines[u - 2](player);
                    o = join_d7_lines[u - 2](opponent);
                    flip |= split_d7_lines[flip_pre_calc[p][o][HW_M1 - t]][u - 2];
                }

                u -= t * 2;
                if (u >= -5 && u <= 5){
                    p = join_d9_lines[u + 5](player);
                    o = join_d9_lines[u + 5](opponent);
                    flip |= split_d9_lines[flip_pre_calc[p][o][t]][u + 5];
                }
            }
        #endif
};

#if LAST_FLIP_CALC_MODE == 0

    inline int count_last_flip(unsigned long long player, unsigned long long opponent, const int place){
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

#elif LAST_FLIP_CALC_MODE == 1

    inline int count_last_flip(unsigned long long player, unsigned long long opponent, const int place){
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
            p = join_d7_lines[u - 2](player);
            o = join_d7_lines[u - 2](opponent);
            res += pop_count(flip_pre_calc[p][o][HW_M1 - t] & d7_mask[place]);
        }

        u -= t * 2;
        if (u >= -5 && u <= 5){
            p = join_d9_lines[u + 5](player);
            o = join_d9_lines[u + 5](opponent);
            res += pop_count(flip_pre_calc[p][o][t] & d9_mask[place]);
        }
        return res;
    }

#endif

void flip_init(){
    int player, opponent, place;
    int wh, put, m1, m2, m3, m4, m5, m6;
    int idx, i;
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
}