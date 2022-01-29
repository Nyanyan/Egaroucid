#pragma once
#include <iostream>
#include "common.hpp"

using namespace std;

#define n_8bit 256

int flip_pre_calc[n_8bit][n_8bit][hw];

constexpr int d7_mask[hw2] = {
    0b00000001, 0b00000011, 0b00000111, 0b00001111, 0b00011111, 0b00111111, 0b01111111, 0b11111111,
    0b00000011, 0b00000111, 0b00001111, 0b00011111, 0b00111111, 0b01111111, 0b11111111, 0b11111110,
    0b00000111, 0b00001111, 0b00011111, 0b00111111, 0b01111111, 0b11111111, 0b11111110, 0b11111100,
    0b00001111, 0b00011111, 0b00111111, 0b01111111, 0b11111111, 0b11111110, 0b11111100, 0b11111000,
    0b00011111, 0b00111111, 0b01111111, 0b11111111, 0b11111110, 0b11111100, 0b11111000, 0b11110000,
    0b00111111, 0b01111111, 0b11111111, 0b11111110, 0b11111100, 0b11111000, 0b11110000, 0b11100000,
    0b01111111, 0b11111111, 0b11111110, 0b11111100, 0b11111000, 0b11110000, 0b11100000, 0b11000000,
    0b11111111, 0b11111110, 0b11111100, 0b11111000, 0b11110000, 0b11100000, 0b11000000, 0b10000000
};
/*
constexpr int d9_mask[hw2] = {
    0b11111111, 0b11111110, 0b11111100, 0b11111000, 0b11110000, 0b11100000, 0b11000000, 0b10000000,
    0b01111111, 0b11111111, 0b11111110, 0b11111100, 0b11111000, 0b11110000, 0b11100000, 0b11000000,
    0b00111111, 0b01111111, 0b11111111, 0b11111110, 0b11111100, 0b11111000, 0b11110000, 0b11100000,
    0b00011111, 0b00111111, 0b01111111, 0b11111111, 0b11111110, 0b11111100, 0b11111000, 0b11110000,
    0b00001111, 0b00011111, 0b00111111, 0b01111111, 0b11111111, 0b11111110, 0b11111100, 0b11111000,
    0b00000111, 0b00001111, 0b00011111, 0b00111111, 0b01111111, 0b11111111, 0b11111110, 0b11111100,
    0b00000011, 0b00000111, 0b00001111, 0b00011111, 0b00111111, 0b01111111, 0b11111111, 0b11111110,
    0b00000001, 0b00000011, 0b00000111, 0b00001111, 0b00011111, 0b00111111, 0b01111111, 0b11111111
};

constexpr int d7_mask[hw2] = {
    0b10000000, 0b11000000, 0b11100000, 0b11110000, 0b11111000, 0b11111100, 0b11111110, 0b11111111,
    0b11000000, 0b11100000, 0b11110000, 0b11111000, 0b11111100, 0b11111110, 0b11111111, 0b01111111,
    0b11100000, 0b11110000, 0b11111000, 0b11111100, 0b11111110, 0b11111111, 0b01111111, 0b00111111,
    0b11110000, 0b11111000, 0b11111100, 0b11111110, 0b11111111, 0b01111111, 0b00111111, 0b00011111,
    0b11111000, 0b11111100, 0b11111110, 0b11111111, 0b01111111, 0b00111111, 0b00011111, 0b00001111,
    0b11111100, 0b11111110, 0b11111111, 0b01111111, 0b00111111, 0b00011111, 0b00001111, 0b00000111,
    0b11111110, 0b11111111, 0b01111111, 0b00111111, 0b00011111, 0b00001111, 0b00000111, 0b00000011,
    0b11111111, 0b01111111, 0b00111111, 0b00011111, 0b00001111, 0b00000111, 0b00000011, 0b00000001
};
*/
constexpr int d9_mask[hw2] = {
    0b11111111, 0b01111111, 0b00111111, 0b00011111, 0b00001111, 0b00000111, 0b00000011, 0b00000001,
    0b11111110, 0b11111111, 0b01111111, 0b00111111, 0b00011111, 0b00001111, 0b00000111, 0b00000011,
    0b11111100, 0b11111110, 0b11111111, 0b01111111, 0b00111111, 0b00011111, 0b00001111, 0b00000111,
    0b11111000, 0b11111100, 0b11111110, 0b11111111, 0b01111111, 0b00111111, 0b00011111, 0b00001111,
    0b11110000, 0b11111000, 0b11111100, 0b11111110, 0b11111111, 0b01111111, 0b00111111, 0b00011111,
    0b11100000, 0b11110000, 0b11111000, 0b11111100, 0b11111110, 0b11111111, 0b01111111, 0b00111111,
    0b11000000, 0b11100000, 0b11110000, 0b11111000, 0b11111100, 0b11111110, 0b11111111, 0b01111111,
    0b10000000, 0b11000000, 0b11100000, 0b11110000, 0b11111000, 0b11111100, 0b11111110, 0b11111111
};


class mobility{
    public:
        int pos;
        unsigned long long flip;
    
    public:
        inline void calc_flip_slow(const unsigned long long player, const unsigned long long opponent, const int place){
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
            m1 = put >> hw;
            if( (m1 & wh) != 0 ) {
                if( ((m2 = m1 >> hw) & wh) == 0  ) {
                    if( (m2 & player) != 0 )
                        rev |= m1;
                } else if( ((m3 = m2 >> hw) & wh) == 0 ) {
                    if( (m3 & player) != 0 )
                        rev |= m1 | m2;
                } else if( ((m4 = m3 >> hw) & wh) == 0 ) {
                    if( (m4 & player) != 0 )
                        rev |= m1 | m2 | m3;
                } else if( ((m5 = m4 >> hw) & wh) == 0 ) {
                    if( (m5 & player) != 0 )
                        rev |= m1 | m2 | m3 | m4;
                } else if( ((m6 = m5 >> hw) & wh) == 0 ) {
                    if( (m6 & player) != 0 )
                        rev |= m1 | m2 | m3 | m4 | m5;
                } else {
                    if( ((m6 >> hw) & player) != 0 )
                        rev |= m1 | m2 | m3 | m4 | m5 | m6;
                }
            }
            m1 = put << hw;
            if( (m1 & wh) != 0 ) {
                if( ((m2 = m1 << hw) & wh) == 0  ) {
                    if( (m2 & player) != 0 )
                        rev |= m1;
                } else if( ((m3 = m2 << hw) & wh) == 0 ) {
                    if( (m3 & player) != 0 )
                        rev |= m1 | m2;
                } else if( ((m4 = m3 << hw) & wh) == 0 ) {
                    if( (m4 & player) != 0 )
                        rev |= m1 | m2 | m3;
                } else if( ((m5 = m4 << hw) & wh) == 0 ) {
                    if( (m5 & player) != 0 )
                        rev |= m1 | m2 | m3 | m4;
                } else if( ((m6 = m5 << hw) & wh) == 0 ) {
                    if( (m6 & player) != 0 )
                        rev |= m1 | m2 | m3 | m4 | m5;
                } else {
                    if( ((m6 << hw) & player) != 0 )
                        rev |= m1 | m2 | m3 | m4 | m5 | m6;
                }
            }

            wh = opponent & 0b0000000001111110011111100111111001111110011111100111111000000000ULL;
            m1 = put >> (hw - 1);
            if( (m1 & wh) != 0 ) {
                if( ((m2 = m1 >> (hw - 1)) & wh) == 0  ) {
                    if( (m2 & player) != 0 )
                        rev |= m1;
                } else if( ((m3 = m2 >> (hw - 1)) & wh) == 0 ) {
                    if( (m3 & player) != 0 )
                        rev |= m1 | m2;
                } else if( ((m4 = m3 >> (hw - 1)) & wh) == 0 ) {
                    if( (m4 & player) != 0 )
                        rev |= m1 | m2 | m3;
                } else if( ((m5 = m4 >> (hw - 1)) & wh) == 0 ) {
                    if( (m5 & player) != 0 )
                        rev |= m1 | m2 | m3 | m4;
                } else if( ((m6 = m5 >> (hw - 1)) & wh) == 0 ) {
                    if( (m6 & player) != 0 )
                        rev |= m1 | m2 | m3 | m4 | m5;
                } else {
                    if( ((m6 >> (hw - 1)) & player) != 0 )
                        rev |= m1 | m2 | m3 | m4 | m5 | m6;
                }
            }
            m1 = put << (hw - 1);
            if( (m1 & wh) != 0 ) {
                if( ((m2 = m1 << (hw - 1)) & wh) == 0  ) {
                    if( (m2 & player) != 0 )
                        rev |= m1;
                } else if( ((m3 = m2 << (hw - 1)) & wh) == 0 ) {
                    if( (m3 & player) != 0 )
                        rev |= m1 | m2;
                } else if( ((m4 = m3 << (hw - 1)) & wh) == 0 ) {
                    if( (m4 & player) != 0 )
                        rev |= m1 | m2 | m3;
                } else if( ((m5 = m4 << (hw - 1)) & wh) == 0 ) {
                    if( (m5 & player) != 0 )
                        rev |= m1 | m2 | m3 | m4;
                } else if( ((m6 = m5 << (hw - 1)) & wh) == 0 ) {
                    if( (m6 & player) != 0 )
                        rev |= m1 | m2 | m3 | m4 | m5;
                } else {
                    if( ((m6 << (hw - 1)) & player) != 0 )
                        rev |= m1 | m2 | m3 | m4 | m5 | m6;
                }
            }

            m1 = put >> (hw + 1);
            if( (m1 & wh) != 0 ) {
                if( ((m2 = m1 >> (hw + 1)) & wh) == 0  ) {
                    if( (m2 & player) != 0 )
                        rev |= m1;
                } else if( ((m3 = m2 >> (hw + 1)) & wh) == 0 ) {
                    if( (m3 & player) != 0 )
                        rev |= m1 | m2;
                } else if( ((m4 = m3 >> (hw + 1)) & wh) == 0 ) {
                    if( (m4 & player) != 0 )
                        rev |= m1 | m2 | m3;
                } else if( ((m5 = m4 >> (hw + 1)) & wh) == 0 ) {
                    if( (m5 & player) != 0 )
                        rev |= m1 | m2 | m3 | m4;
                } else if( ((m6 = m5 >> (hw + 1)) & wh) == 0 ) {
                    if( (m6 & player) != 0 )
                        rev |= m1 | m2 | m3 | m4 | m5;
                } else {
                    if( ((m6 >> (hw + 1)) & player) != 0 )
                        rev |= m1 | m2 | m3 | m4 | m5 | m6;
                }
            }
            m1 = put << (hw + 1);
            if( (m1 & wh) != 0 ) {
                if( ((m2 = m1 << (hw + 1)) & wh) == 0  ) {
                    if( (m2 & player) != 0 )
                        rev |= m1;
                } else if( ((m3 = m2 << (hw + 1)) & wh) == 0 ) {
                    if( (m3 & player) != 0 )
                        rev |= m1 | m2;
                } else if( ((m4 = m3 << (hw + 1)) & wh) == 0 ) {
                    if( (m4 & player) != 0 )
                        rev |= m1 | m2 | m3;
                } else if( ((m5 = m4 << (hw + 1)) & wh) == 0 ) {
                    if( (m5 & player) != 0 )
                        rev |= m1 | m2 | m3 | m4;
                } else if( ((m6 = m5 << (hw + 1)) & wh) == 0 ) {
                    if( (m6 & player) != 0 )
                        rev |= m1 | m2 | m3 | m4 | m5;
                } else {
                    if( ((m6 << (hw + 1)) & player) != 0 )
                        rev |= m1 | m2 | m3 | m4 | m5 | m6;
                }
            }

            flip = rev;
            pos = place;
        }

        inline void calc_flip(const unsigned long long player, const unsigned long long opponent, const int place){
            unsigned long long h;
            int t, u, p, o;
            flip = 0;
            pos = place;

            t = place / hw;
            u = place % hw;
            p = (player >> (hw * t)) & 0b11111111;
            o = (opponent >> (hw * t)) & 0b11111111;
            h = flip_pre_calc[p][o][u];
            flip |= h << (hw * t);

            p = join_v_line(player, u);
            o = join_v_line(opponent, u);
            flip |= split_v_line(flip_pre_calc[p][o][t], u);

            t = place / hw;
            u = place % hw + t;
            p = join_d7_line(player, u) & d7_mask[place];
            o = join_d7_line(opponent, u) & d7_mask[place];
            flip |= split_d7_line(flip_pre_calc[p][o][t] & d7_mask[place], u);
            /*
            for (int i = hw_m1; i >= 0; --i){
                if (1 & ((d7_mask[place]) >> i))
                    cerr << '1';
                else
                    cerr << '.';
            }
            cerr << endl;
            cerr << t << " " << u << endl;
            for (int i = hw_m1; i >= 0; --i){
                if (1 & (p >> i))
                    cerr << '0';
                else if (1 & (o >> i))
                    cerr << '1';
                else
                    cerr << '.';
            }
            cerr << " " << place << " ";
            for (int i = hw_m1; i >= 0; --i){
                if (1 & ((flip_pre_calc[p][o][t]) >> i))
                    cerr << '1';
                else
                    cerr << '.';
            }
            cerr << endl;
            */
            //t = place / hw;
            u -= t * 2;
            p = join_d9_line(player, u) & d9_mask[place];
            o = join_d9_line(opponent, u) & d9_mask[place];
            flip |= split_d9_line(flip_pre_calc[p][o][t] & d9_mask[place], u);
            /*
            for (int i = hw_m1; i >= 0; --i){
                if (1 & ((d9_mask[place]) >> i))
                    cerr << '1';
                else
                    cerr << '.';
            }
            cerr << endl;
            cerr << t << " " << u << endl;
            for (int i = hw_m1; i >= 0; --i){
                if (1 & (p >> i))
                    cerr << '0';
                else if (1 & (o >> i))
                    cerr << '1';
                else
                    cerr << '.';
            }
            cerr << " " << place << " ";
            for (int i = hw_m1; i >= 0; --i){
                if (1 & ((flip_pre_calc[p][o][t]) >> i))
                    cerr << '1';
                else
                    cerr << '.';
            }
            cerr << endl;
            */
        }
};

void mobility_init(){
    int player, opponent, place;
    int wh, put, m1, m2, m3, m4, m5, m6;
    for (player = 0; player < n_8bit; ++player){
        for (opponent = 0; opponent < n_8bit; ++opponent){
            for (place = 0; place < hw; ++place){
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
                    /*
                    for (int i = 0; i < hw; ++i){
                        if (1 & (player >> i))
                            cerr << '0';
                        else if (1 & (opponent >> i))
                            cerr << '1';
                        else
                            cerr << '.';
                    }
                    cerr << " " << place << " ";
                    for (int i = 0; i < hw; ++i){
                        if (1 & (flip_pre_calc[player][opponent][place] >> i))
                            cerr << '1';
                        else
                            cerr << '.';
                    }
                    cerr << endl;
                    */
                }
            }
        }
    }
}