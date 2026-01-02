/*
    Egaroucid Project

    @file mobility_generic.hpp
        Calculate legal moves without AVX2
    @date 2021-2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
    @notice I referred to codes written by others
*/

#pragma once
#include "bit.hpp"
#include "setting.hpp"


/*
    @brief mobility initialize
*/
void mobility_init() {
}

/*
    @brief Get a bitboard representing all legal moves

    @param P                    a bitboard representing player
    @param O                    a bitboard representing opponent
    @return all legal moves as a bitboard
*/
// original code from http://www.amy.hi-ho.ne.jp/okuhara/bitboard.htm
// modified by Nyanyan
inline uint64_t calc_legal(const uint64_t P, const uint64_t O) {
    uint64_t moves, mO;
    uint64_t flip1, flip7, flip9, flip8, pre1, pre7, pre9, pre8;
    mO = O & 0x7e7e7e7e7e7e7e7eULL;
    flip1 = mO & (P << 1);         flip7  = mO & (P << 7);        flip9  = mO & (P << 9);        flip8  = O & (P << 8);
    flip1 |= mO & (flip1 << 1);    flip7 |= mO & (flip7 << 7);    flip9 |= mO & (flip9 << 9);    flip8 |= O & (flip8 << 8);
    pre1 = mO & (mO << 1);         pre7 = mO & (mO << 7);         pre9 = mO & (mO << 9);         pre8 = O & (O << 8);
    flip1 |= pre1 & (flip1 << 2);  flip7 |= pre7 & (flip7 << 14); flip9 |= pre9 & (flip9 << 18); flip8 |= pre8 & (flip8 << 16);
    flip1 |= pre1 & (flip1 << 2);  flip7 |= pre7 & (flip7 << 14); flip9 |= pre9 & (flip9 << 18); flip8 |= pre8 & (flip8 << 16);
    moves = flip1 << 1;            moves |= flip7 << 7;           moves |= flip9 << 9;           moves |= flip8 << 8;
    flip1 = mO & (P >> 1);         flip7  = mO & (P >> 7);        flip9  = mO & (P >> 9);        flip8  = O & (P >> 8);
    flip1 |= mO & (flip1 >> 1);    flip7 |= mO & (flip7 >> 7);    flip9 |= mO & (flip9 >> 9);    flip8 |= O & (flip8 >> 8);
    pre1 >>= 1;                    pre7 >>= 7;                    pre9 >>= 9;                    pre8 >>= 8;
    flip1 |= pre1 & (flip1 >> 2);  flip7 |= pre7 & (flip7 >> 14); flip9 |= pre9 & (flip9 >> 18); flip8 |= pre8 & (flip8 >> 16);
    flip1 |= pre1 & (flip1 >> 2);  flip7 |= pre7 & (flip7 >> 14); flip9 |= pre9 & (flip9 >> 18); flip8 |= pre8 & (flip8 >> 16);
    moves |= flip1 >> 1;           moves |= flip7 >> 7;           moves |= flip9 >> 9;           moves |= flip8 >> 8;
    return moves & ~(P | O);
}
// end of modification