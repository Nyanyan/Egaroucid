/*
    Egaroucid Project

    @file flip_generic.hpp
        Flip calculation without AVX2
    @date 2021-2024
    @author Takuto Yamana
    @license GPL-3.0 license
    @notice I referred to codes written by others
*/

#pragma once
#include "setting.hpp"
#include "common.hpp"
#include "bit.hpp"

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
        inline uint64_t calc_flip(uint64_t player, uint64_t opponent, const int place){
            uint64_t line, outflank;
            uint64_t mopponent = opponent & 0x7e7e7e7e7e7e7e7eULL;
            uint64_t rplayer = rotate_180(player);
            uint64_t ropponent = rotate_180(opponent);
            uint64_t mropponent = ropponent & 0x7e7e7e7e7e7e7e7eULL;
            int rplace = HW2_M1 - place;
            pos = place;
            flip = 0;
            uint64_t rflip = 0;
            
            // downward
            line = 0x0101010101010100ULL << rplace;
            outflank = ((ropponent | ~line) + 1) & line & rplayer;
            rflip |= (outflank - (outflank != 0)) & line;
            // upward
            line = 0x0101010101010100ULL << place;
            outflank = ((opponent | ~line) + 1) & line & player;
            flip |= (outflank - (outflank != 0)) & line;
            
            // right
            line = 0x00000000000000feULL << rplace;
            outflank = ((mropponent | ~line) + 1) & line & rplayer;
            rflip |= (outflank - (outflank != 0)) & line;
            // left
            line = 0x00000000000000feULL << place;
            outflank = ((mopponent | ~line) + 1) & line & player;
            flip |= (outflank - (outflank != 0)) & line;

            // d7 downward
            line = 0x0002040810204080ULL << rplace;
            outflank = ((mropponent | ~line) + 1) & line & rplayer;
            rflip |= (outflank - (outflank != 0)) & line;
            // d7 upward
            line = 0x0002040810204080ULL << place;
            outflank = ((mopponent | ~line) + 1) & line & player;
            flip |= (outflank - (outflank != 0)) & line;

            // d9 downward
            line = 0x8040201008040200ULL << rplace;
            outflank = ((mropponent | ~line) + 1) & line & rplayer;
            rflip |= (outflank - (outflank != 0)) & line;
            // d9 upward
            line = 0x8040201008040200ULL << place;
            outflank = ((mopponent | ~line) + 1) & line & player;
            flip |= (outflank - (outflank != 0)) & line;
            
            flip |= rotate_180(rflip);
            return flip;
        }
};

void flip_init(){
}