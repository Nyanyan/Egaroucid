#pragma once
#include <iostream>
#include "setting.hpp"
#include "common.hpp"
#include "bit.hpp"

using namespace std;

class Flip{
    public:
        uint_fast8_t pos;
        uint64_t flip;
    
    public:
        //Original code: https://github.com/primenumber/issen/blob/72f450256878094ffe90b75f8674599e6869c238/src/move_generator.cpp
        //modified by Nyanyan
        inline void calc_flip(const uint64_t player, const uint64_t opponent, const uint_fast8_t place){
            pos = place;
            u64_4 p(player);
            u64_4 o(opponent);
            u64_4 flipped, om, outflank, mask;
            const u64_4 mask1(0xFFFFFFFFFFFFFFFFULL, 0x7E7E7E7E7E7E7E7EULL, 0x7E7E7E7E7E7E7E7EULL, 0x7E7E7E7E7E7E7E7EULL);
            const u64_4 mask2(0x0080808080808080ULL, 0x7F00000000000000ULL, 0x0102040810204000ULL, 0x0040201008040201ULL);
            const u64_4 mask3(0x0101010101010100ULL, 0x00000000000000FEULL, 0x0002040810204080ULL, 0x8040201008040200ULL);
            om = o & mask1;
            mask = mask2 >> (HW2_M1 - place);
            outflank = upper_bit(andnot(om, mask)) & p;
            flipped = ((-outflank) << 1) & mask;
            mask = mask3 << place;
            outflank = mask & ((om | ~mask) + 1) & p;
            flipped = flipped | ((outflank - nonzero(outflank)) & mask);
            flip = all_or(flipped);
        }
        //end of modification
};
