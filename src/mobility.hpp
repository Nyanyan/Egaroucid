#pragma once
#include <iostream>
#include "common.hpp"

using namespace std;

class mobility{
    public:
        int pos;
        unsigned long long flip;
    
    public:
        inline void calc_flip(const unsigned long long player, const unsigned long long opponent, const int place){
            unsigned long long wh, put, m1, m2, m3, m4, m5, m6, rev;
            put = 1ULL << place;
            rev = 0;

            wh = opponent & 0b0111111001111110011111100111111001111110011111100111111001111110;
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
                cerr << endl;
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

            wh = opponent & 0b000000001111111111111111111111111111111111111111111111110000000;
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

            wh = opponent & 0b0000000001111110011111100111111001111110011111100111111000000000;
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
};
