#pragma once
#include <iostream>
#include "setting.hpp"
#include "common.hpp"
#include "bit.hpp"

using namespace std;

//uint_fast8_t flip_pre_calc[N_8BIT][N_8BIT][HW];
constexpr uint_fast8_t n_flip_pre_calc[N_8BIT][HW] = {
    {0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 1, 2, 3, 4, 5, 6}, {0, 0, 0, 1, 2, 3, 4, 5}, {0, 0, 0, 1, 2, 3, 4, 5}, {1, 0, 0, 0, 1, 2, 3, 4}, {0, 0, 0, 0, 1, 2, 3, 4}, {0, 0, 0, 0, 1, 2, 3, 4}, {0, 0, 0, 0, 1, 2, 3, 4}, 
    {2, 1, 0, 0, 0, 1, 2, 3}, {0, 1, 1, 0, 0, 1, 2, 3}, {0, 0, 0, 0, 0, 1, 2, 3}, {0, 0, 0, 0, 0, 1, 2, 3}, {1, 0, 0, 0, 0, 1, 2, 3}, {0, 0, 0, 0, 0, 1, 2, 3}, {0, 0, 0, 0, 0, 1, 2, 3}, {0, 0, 0, 0, 0, 1, 2, 3},
    {3, 2, 1, 0, 0, 0, 1, 2}, {0, 2, 2, 2, 0, 0, 1, 2}, {0, 0, 1, 1, 0, 0, 1, 2}, {0, 0, 1, 1, 0, 0, 1, 2}, {1, 0, 0, 0, 0, 0, 1, 2}, {0, 0, 0, 0, 0, 0, 1, 2}, {0, 0, 0, 0, 0, 0, 1, 2}, {0, 0, 0, 0, 0, 0, 1, 2},
    {2, 1, 0, 0, 0, 0, 1, 2}, {0, 1, 1, 0, 0, 0, 1, 2}, {0, 0, 0, 0, 0, 0, 1, 2}, {0, 0, 0, 0, 0, 0, 1, 2}, {1, 0, 0, 0, 0, 0, 1, 2}, {0, 0, 0, 0, 0, 0, 1, 2}, {0, 0, 0, 0, 0, 0, 1, 2}, {0, 0, 0, 0, 0, 0, 1, 2},
    {4, 3, 2, 1, 0, 0, 0, 1}, {0, 3, 3, 3, 3, 0, 0, 1}, {0, 0, 2, 2, 2, 0, 0, 1}, {0, 0, 2, 2, 2, 0, 0, 1}, {1, 0, 0, 1, 1, 0, 0, 1}, {0, 0, 0, 1, 1, 0, 0, 1}, {0, 0, 0, 1, 1, 0, 0, 1}, {0, 0, 0, 1, 1, 0, 0, 1},
    {2, 1, 0, 0, 0, 0, 0, 1}, {0, 1, 1, 0, 0, 0, 0, 1}, {0, 0, 0, 0, 0, 0, 0, 1}, {0, 0, 0, 0, 0, 0, 0, 1}, {1, 0, 0, 0, 0, 0, 0, 1}, {0, 0, 0, 0, 0, 0, 0, 1}, {0, 0, 0, 0, 0, 0, 0, 1}, {0, 0, 0, 0, 0, 0, 0, 1},
    {3, 2, 1, 0, 0, 0, 0, 1}, {0, 2, 2, 2, 0, 0, 0, 1}, {0, 0, 1, 1, 0, 0, 0, 1}, {0, 0, 1, 1, 0, 0, 0, 1}, {1, 0, 0, 0, 0, 0, 0, 1}, {0, 0, 0, 0, 0, 0, 0, 1}, {0, 0, 0, 0, 0, 0, 0, 1}, {0, 0, 0, 0, 0, 0, 0, 1}, 
    {2, 1, 0, 0, 0, 0, 0, 1}, {0, 1, 1, 0, 0, 0, 0, 1}, {0, 0, 0, 0, 0, 0, 0, 1}, {0, 0, 0, 0, 0, 0, 0, 1}, {1, 0, 0, 0, 0, 0, 0, 1}, {0, 0, 0, 0, 0, 0, 0, 1}, {0, 0, 0, 0, 0, 0, 0, 1}, {0, 0, 0, 0, 0, 0, 0, 1},
    {5, 4, 3, 2, 1, 0, 0, 0}, {0, 4, 4, 4, 4, 4, 0, 0}, {0, 0, 3, 3, 3, 3, 0, 0}, {0, 0, 3, 3, 3, 3, 0, 0}, {1, 0, 0, 2, 2, 2, 0, 0}, {0, 0, 0, 2, 2, 2, 0, 0}, {0, 0, 0, 2, 2, 2, 0, 0}, {0, 0, 0, 2, 2, 2, 0, 0},
    {2, 1, 0, 0, 1, 1, 0, 0}, {0, 1, 1, 0, 1, 1, 0, 0}, {0, 0, 0, 0, 1, 1, 0, 0}, {0, 0, 0, 0, 1, 1, 0, 0}, {1, 0, 0, 0, 1, 1, 0, 0}, {0, 0, 0, 0, 1, 1, 0, 0}, {0, 0, 0, 0, 1, 1, 0, 0}, {0, 0, 0, 0, 1, 1, 0, 0},
    {3, 2, 1, 0, 0, 0, 0, 0}, {0, 2, 2, 2, 0, 0, 0, 0}, {0, 0, 1, 1, 0, 0, 0, 0}, {0, 0, 1, 1, 0, 0, 0, 0}, {1, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0}, 
    {2, 1, 0, 0, 0, 0, 0, 0}, {0, 1, 1, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0}, {1, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0},
    {4, 3, 2, 1, 0, 0, 0, 0}, {0, 3, 3, 3, 3, 0, 0, 0}, {0, 0, 2, 2, 2, 0, 0, 0}, {0, 0, 2, 2, 2, 0, 0, 0}, {1, 0, 0, 1, 1, 0, 0, 0}, {0, 0, 0, 1, 1, 0, 0, 0}, {0, 0, 0, 1, 1, 0, 0, 0}, {0, 0, 0, 1, 1, 0, 0, 0},
    {2, 1, 0, 0, 0, 0, 0, 0}, {0, 1, 1, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0}, {1, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0},
    {3, 2, 1, 0, 0, 0, 0, 0}, {0, 2, 2, 2, 0, 0, 0, 0}, {0, 0, 1, 1, 0, 0, 0, 0}, {0, 0, 1, 1, 0, 0, 0, 0}, {1, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0},
    {2, 1, 0, 0, 0, 0, 0, 0}, {0, 1, 1, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0}, {1, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0}, 
    {6, 5, 4, 3, 2, 1, 0, 0}, {0, 5, 5, 5, 5, 5, 5, 0}, {0, 0, 4, 4, 4, 4, 4, 0}, {0, 0, 4, 4, 4, 4, 4, 0}, {1, 0, 0, 3, 3, 3, 3, 0}, {0, 0, 0, 3, 3, 3, 3, 0}, {0, 0, 0, 3, 3, 3, 3, 0}, {0, 0, 0, 3, 3, 3, 3, 0},
    {2, 1, 0, 0, 2, 2, 2, 0}, {0, 1, 1, 0, 2, 2, 2, 0}, {0, 0, 0, 0, 2, 2, 2, 0}, {0, 0, 0, 0, 2, 2, 2, 0}, {1, 0, 0, 0, 2, 2, 2, 0}, {0, 0, 0, 0, 2, 2, 2, 0}, {0, 0, 0, 0, 2, 2, 2, 0}, {0, 0, 0, 0, 2, 2, 2, 0},
    {3, 2, 1, 0, 0, 1, 1, 0}, {0, 2, 2, 2, 0, 1, 1, 0}, {0, 0, 1, 1, 0, 1, 1, 0}, {0, 0, 1, 1, 0, 1, 1, 0}, {1, 0, 0, 0, 0, 1, 1, 0}, {0, 0, 0, 0, 0, 1, 1, 0}, {0, 0, 0, 0, 0, 1, 1, 0}, {0, 0, 0, 0, 0, 1, 1, 0},
    {2, 1, 0, 0, 0, 1, 1, 0}, {0, 1, 1, 0, 0, 1, 1, 0}, {0, 0, 0, 0, 0, 1, 1, 0}, {0, 0, 0, 0, 0, 1, 1, 0}, {1, 0, 0, 0, 0, 1, 1, 0}, {0, 0, 0, 0, 0, 1, 1, 0}, {0, 0, 0, 0, 0, 1, 1, 0}, {0, 0, 0, 0, 0, 1, 1, 0},
    {4, 3, 2, 1, 0, 0, 0, 0}, {0, 3, 3, 3, 3, 0, 0, 0}, {0, 0, 2, 2, 2, 0, 0, 0}, {0, 0, 2, 2, 2, 0, 0, 0}, {1, 0, 0, 1, 1, 0, 0, 0}, {0, 0, 0, 1, 1, 0, 0, 0}, {0, 0, 0, 1, 1, 0, 0, 0}, {0, 0, 0, 1, 1, 0, 0, 0}, 
    {2, 1, 0, 0, 0, 0, 0, 0}, {0, 1, 1, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0}, {1, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0},
    {3, 2, 1, 0, 0, 0, 0, 0}, {0, 2, 2, 2, 0, 0, 0, 0}, {0, 0, 1, 1, 0, 0, 0, 0}, {0, 0, 1, 1, 0, 0, 0, 0}, {1, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0},
    {2, 1, 0, 0, 0, 0, 0, 0}, {0, 1, 1, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0}, {1, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0},
    {5, 4, 3, 2, 1, 0, 0, 0}, {0, 4, 4, 4, 4, 4, 0, 0}, {0, 0, 3, 3, 3, 3, 0, 0}, {0, 0, 3, 3, 3, 3, 0, 0}, {1, 0, 0, 2, 2, 2, 0, 0}, {0, 0, 0, 2, 2, 2, 0, 0}, {0, 0, 0, 2, 2, 2, 0, 0}, {0, 0, 0, 2, 2, 2, 0, 0},
    {2, 1, 0, 0, 1, 1, 0, 0}, {0, 1, 1, 0, 1, 1, 0, 0}, {0, 0, 0, 0, 1, 1, 0, 0}, {0, 0, 0, 0, 1, 1, 0, 0}, {1, 0, 0, 0, 1, 1, 0, 0}, {0, 0, 0, 0, 1, 1, 0, 0}, {0, 0, 0, 0, 1, 1, 0, 0}, {0, 0, 0, 0, 1, 1, 0, 0}, 
    {3, 2, 1, 0, 0, 0, 0, 0}, {0, 2, 2, 2, 0, 0, 0, 0}, {0, 0, 1, 1, 0, 0, 0, 0}, {0, 0, 1, 1, 0, 0, 0, 0}, {1, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0},
    {2, 1, 0, 0, 0, 0, 0, 0}, {0, 1, 1, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0}, {1, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0},
    {4, 3, 2, 1, 0, 0, 0, 0}, {0, 3, 3, 3, 3, 0, 0, 0}, {0, 0, 2, 2, 2, 0, 0, 0}, {0, 0, 2, 2, 2, 0, 0, 0}, {1, 0, 0, 1, 1, 0, 0, 0}, {0, 0, 0, 1, 1, 0, 0, 0}, {0, 0, 0, 1, 1, 0, 0, 0}, {0, 0, 0, 1, 1, 0, 0, 0},
    {2, 1, 0, 0, 0, 0, 0, 0}, {0, 1, 1, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0}, {1, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0},
    {3, 2, 1, 0, 0, 0, 0, 0}, {0, 2, 2, 2, 0, 0, 0, 0}, {0, 0, 1, 1, 0, 0, 0, 0}, {0, 0, 1, 1, 0, 0, 0, 0}, {1, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0}, 
    {2, 1, 0, 0, 0, 0, 0, 0}, {0, 1, 1, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0}, {1, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0}
};

constexpr uint_fast8_t flip_place_d7[HW2] = {
    0, 0, 2, 3, 4, 5, 6, 7, 
    0, 1, 2, 3, 4, 5, 6, 6, 
    0, 1, 2, 3, 4, 5, 5, 5, 
    0, 1, 2, 3, 4, 4, 4, 4, 
    0, 1, 2, 3, 3, 3, 3, 3, 
    0, 1, 2, 2, 2, 2, 2, 2, 
    0, 1, 1, 1, 1, 1, 1, 0, 
    0, 0, 0, 0, 0, 0, 0, 0
};

constexpr uint_fast8_t flip_place_d9[HW2] = {
    0, 1, 2, 3, 4, 5, 6, 7, 
    0, 1, 2, 3, 4, 5, 6, 6, 
    0, 1, 2, 3, 4, 5, 5, 5, 
    0, 1, 2, 3, 4, 4, 4, 4, 
    0, 1, 2, 3, 3, 3, 3, 3, 
    0, 1, 2, 2, 2, 2, 2, 2, 
    0, 1, 1, 1, 1, 1, 1, 0, 
    0, 0, 0, 0, 0, 0, 0, 0
};

class Flip{
    public:
        uint_fast8_t pos;
        uint64_t flip;
        int32_t value;
        uint64_t n_legal;
        int stab0;
        int stab1;
    
    public:
        inline void copy(Flip *mob) const{
            mob->pos = pos;
            mob->flip = flip;
            mob->value = value;
        }

        #if FLIP_CALC_MODE == 0
            inline void calc_flip(const uint64_t player, const uint64_t opponent, const uint_fast8_t place){
                uint64_t wh, put, m1, m2, m3, m4, m5, m6, rev;
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
            inline void calc_flip(const uint64_t player, const uint64_t opponent, const uint_fast8_t place){
                int_fast8_t t, u;
                uint_fast8_t p, o;
                flip = 0;
                pos = place;

                t = place / HW;
                u = place % HW;
                p = join_h_line(player, t);
                o = join_h_line(opponent, t);
                //join_h_line_double(player, opponent, t, &p, &o);
                flip |= split_h_line((uint64_t)flip_pre_calc[p][o][u], t);

                p = join_v_line(player, u);
                o = join_v_line(opponent, u);
                //join_v_line_double(player, opponent, u, &p, &o);
                //flip |= split_v_line(flip_pre_calc[p][o][t], u);
                flip |= split_v_lines[flip_pre_calc[p][o][t]] << u;

                u += t;
                if (u >= 2 && u <= 12){
                    p = join_d7_line(player, u) & d7_mask[place];
                    o = join_d7_line(opponent, u) & d7_mask[place];
                    //flip |= split_d7_line(flip_pre_calc[p][o][t], u);
                    flip |= split_d7_lines[flip_pre_calc[p][o][t]] << u;
                }

                u -= t * 2;
                if (u >= -5 && u <= 5){
                    p = join_d9_line(player, u) & d9_mask[place];
                    o = join_d9_line(opponent, u) & d9_mask[place];
                    //flip |= split_d9_line(flip_pre_calc[p][o][t], u);
                    if (u > 0)
                        flip |= split_d9_lines[flip_pre_calc[p][o][t]] << u;
                    else
                        flip |= split_d9_lines[flip_pre_calc[p][o][t]] >> (-u);
                }
            }
        #elif FLIP_CALC_MODE == 2
            inline void calc_flip(const uint64_t player, const uint64_t opponent, const uint_fast8_t place){
                int_fast8_t t, u;
                uint8_t p, o;
                flip = 0;
                pos = place;

                t = place / HW;
                u = place % HW;
                p = (player >> (HW * t)) & 0b11111111;
                o = (opponent >> (HW * t)) & 0b11111111;
                flip |= (uint64_t)flip_pre_calc[p][o][u] << (HW * t);

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
        #elif FLIP_CALC_MODE == 3
            inline void calc_flip(const uint64_t player, const uint64_t opponent, const uint_fast8_t place){
                /*
                Original code: https://github.com/primenumber/issen/blob/72f450256878094ffe90b75f8674599e6869c238/src/move_generator.cpp
                modified by Nyanyan
                */
                pos = place;
                u64_4 p(player);
                u64_4 o(opponent);
                u64_4 flipped, om, outflank;
                u64_4 mask(0xFFFFFFFFFFFFFFFFULL, 0x7E7E7E7E7E7E7E7EULL, 0x7E7E7E7E7E7E7E7EULL, 0x7E7E7E7E7E7E7E7EULL);
                om = o & mask;
                mask = {0x0080808080808080ULL, 0x7F00000000000000ULL, 0x0102040810204000ULL, 0x0040201008040201ULL};
                mask = mask >> (HW2_M1 - place);
                outflank = upper_bit(andnot(om, mask)) & p;
                flipped = ((-outflank) << 1) & mask;
                mask = {0x0101010101010100ULL, 0x00000000000000FELL, 0x0002040810204080ULL, 0x8040201008040200ULL};
                mask = mask << place;
                outflank = mask & ((om | ~mask) + 1) & player;
                flipped = flipped | ((outflank - nonzero(outflank)) & mask);
                flip = all_or(flipped);
                /*
                end of modification
                */
            }
        #endif
};

#if LAST_FLIP_CALC_MODE == 0

    inline int_fast8_t count_last_flip(uint64_t player, uint64_t opponent, const uint_fast8_t place){
        int_fast8_t t, u;
        uint8_t p, o;
        int_fast8_t res = 0;

        t = place / HW;
        u = place % HW;
        p = join_h_line(player, t);
        o = join_h_line(opponent, t);
        res += n_flip_pre_calc[p][o][u];

        p = join_v_line(player, u);
        o = join_v_line(opponent, u);
        res += n_flip_pre_calc[p][o][t];

        t = place / HW;
        u = place % HW + t;
        if (u >= 2 && u <= 12){
            p = join_d7_line(player, u) & d7_mask[place];
            o = join_d7_line(opponent, u) & d7_mask[place];
            res += pop_count_uchar(flip_pre_calc[p][o][t] & d7_mask[place]);
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

    inline int_fast8_t count_last_flip(uint64_t player, uint64_t opponent, const uint_fast8_t place){
        int_fast8_t t, u;
        uint8_t p, o;
        int_fast8_t res = 0;

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
            res += pop_count_uchar(flip_pre_calc[p][o][HW_M1 - t] & d7_mask[place]);
        }

        u -= t * 2;
        if (u >= -5 && u <= 5){
            p = join_d9_lines[u + 5](player);
            o = join_d9_lines[u + 5](opponent);
            res += pop_count_uchar(flip_pre_calc[p][o][t] & d9_mask[place]);
        }
        return res;
    }

#elif LAST_FLIP_CALC_MODE == 2

    inline int_fast8_t count_last_flip(uint64_t player, uint64_t opponent, const uint_fast8_t place){
        /*
        Original code: https://github.com/primenumber/issen/blob/72f450256878094ffe90b75f8674599e6869c238/src/move_generator.cpp
        modified by Nyanyan
        */
        u64_4 p(player);
        u64_4 o(opponent);
        u64_4 flipped, om, outflank;
        u64_4 mask(0xFFFFFFFFFFFFFFFFULL, 0x7E7E7E7E7E7E7E7EULL, 0x7E7E7E7E7E7E7E7EULL, 0x7E7E7E7E7E7E7E7EULL);
        om = o & mask;
        mask = {0x0080808080808080ULL, 0x7F00000000000000ULL, 0x0102040810204000ULL, 0x0040201008040201ULL};
        mask = mask >> (HW2_M1 - place);
        outflank = upper_bit(andnot(om, mask)) & p;
        flipped = ((-outflank) << 1) & mask;
        mask = {0x0101010101010100ULL, 0x00000000000000FELL, 0x0002040810204080ULL, 0x8040201008040200ULL};
        mask = mask << place;
        outflank = mask & ((om | ~mask) + 1) & player;
        flipped = flipped | ((outflank - nonzero(outflank)) & mask);
        /*
        end of modification
        */
        return pop_count_ull(_mm256_extract_epi64(flipped.data, 3) | _mm256_extract_epi64(flipped.data, 2) | _mm256_extract_epi64(flipped.data, 1) | _mm256_extract_epi64(flipped.data, 0));
    }

#elif LAST_FLIP_CALC_MODE == 3

    inline int_fast8_t count_last_flip(uint64_t player, uint64_t opponent, const uint_fast8_t place){
        const uint_fast8_t t = place >> 3;
        const uint_fast8_t u = place & 7;
        return
            n_flip_pre_calc[join_h_line(player, t)][u] + 
            n_flip_pre_calc[join_v_line(player, u)][t] + 
            n_flip_pre_calc[join_d7_line(player, u + t) & d7_mask[place]][t] + 
            n_flip_pre_calc[join_d9_line(player, u - t) & d9_mask[place]][t];
    }

#elif LAST_FLIP_CALC_MODE == 4

    inline int_fast8_t count_last_flip(uint64_t player, uint64_t opponent, const uint_fast8_t place){
        const uint_fast8_t t = place >> 3;
        const uint_fast8_t u = place & 7;
        return
            n_flip_pre_calc[join_h_line(player, t)][u] + 
            n_flip_pre_calc[join_v_line(player, u)][t] + 
            n_flip_pre_calc[join_d7_line2(player, u + t)][flip_place_d7[place]] + 
            n_flip_pre_calc[join_d9_line2(player, u - t + HW)][flip_place_d9[place]];
    }

#endif

void flip_init(){
    /* DEBUGGING
    uint64_t p = 0x0123456789ABCDEF;
    for (int i = 2; i < 12; ++i){
        if (join_d7_line(p, i) != join_d7_lines[i](p))
            cerr << i << " d7 " << (int)join_d7_line(p, i) << " " << (int)join_d7_lines[i](p) << endl;
        if (join_d9_line(p, i - HW_M1) != join_d9_lines[i](p))
            cerr << i << " d9 " << (int)join_d9_line(p, i - HW_M1) << " " << (int)join_d9_lines[i](p) << endl;
    }
    */
    /*
    uint_fast16_t player, opponent, place;
    uint_fast8_t wh, put, m1, m2, m3, m4, m5, m6, i;
    for (player = 0; player < N_8BIT; ++player){
        for (place = 0; place < HW; ++place)
            n_flip_pre_calc[player][place] = 0;
    }
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
                    if (pop_count_uchar(player) + pop_count_uchar(opponent) == HW_M1 && pop_count_uchar(player | opponent) == HW_M1){
                        n_flip_pre_calc[player][place] = pop_count_uchar(flip_pre_calc[player][opponent][place]);
                    }
                }
            }
        }
    }
    */

    //for (player = 0; player < N_8BIT; ++player){
    //    cerr << "{";
    //    for (place = 0; place < HW_M1; ++place){
    //        cerr << (int)n_flip_pre_calc[player][place] << ", ";
    //    }
    //    cerr << (int)n_flip_pre_calc[player][HW_M1] << "}, ";
    //    if ((player & 0b111) == 0b111)
    //        cerr << endl;
    //}
}