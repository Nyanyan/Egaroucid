/*
    Egaroucid Project

    @file mobility_simd.hpp
        Calculate legal moves with SIMD
    @date 2021-2023
    @author Takuto Yamana (a.k.a. Nyanyan)
    @license GPL-3.0 license
    @notice I referred to codes written by others
*/

#pragma once
#include "bit.hpp"
#include "setting.hpp"


/*
    @brief mobility initialize
*/
__m256i shift1897, mflipH;
void mobility_init(){
    shift1897 = _mm256_set_epi64x(7, 9, 8, 1);
    mflipH = _mm256_set_epi64x(0x7E7E7E7E7E7E7E7E, 0x7E7E7E7E7E7E7E7E, -1, 0x7E7E7E7E7E7E7E7E);
}

/*
    @brief Get a bitboard representing all legal moves

    @param P                    a bitboard representing player
    @param O                    a bitboard representing opponent
    @return all legal moves as a bitboard
*/
// original code from http://www.amy.hi-ho.ne.jp/okuhara/bitboard.htm
// modified by Nyanyan
inline uint64_t calc_legal(const uint64_t P, const uint64_t O){
    __m256i	PP, mOO, MM, flip_l, flip_r, pre_l, pre_r, shift2;
    __m128i	M;
    PP = _mm256_broadcastq_epi64(_mm_cvtsi64_si128(P));
    mOO = _mm256_and_si256(_mm256_broadcastq_epi64(_mm_cvtsi64_si128(O)), mflipH);
    flip_l = _mm256_and_si256(mOO, _mm256_sllv_epi64(PP, shift1897));
    flip_r = _mm256_and_si256(mOO, _mm256_srlv_epi64(PP, shift1897));
    flip_l = _mm256_or_si256(flip_l, _mm256_and_si256(mOO, _mm256_sllv_epi64(flip_l, shift1897)));
    flip_r = _mm256_or_si256(flip_r, _mm256_and_si256(mOO, _mm256_srlv_epi64(flip_r, shift1897)));
    pre_l = _mm256_and_si256(mOO, _mm256_sllv_epi64(mOO, shift1897));
    pre_r = _mm256_srlv_epi64(pre_l, shift1897);
    shift2 = _mm256_add_epi64(shift1897, shift1897);
    flip_l = _mm256_or_si256(flip_l, _mm256_and_si256(pre_l, _mm256_sllv_epi64(flip_l, shift2)));
    flip_r = _mm256_or_si256(flip_r, _mm256_and_si256(pre_r, _mm256_srlv_epi64(flip_r, shift2)));
    flip_l = _mm256_or_si256(flip_l, _mm256_and_si256(pre_l, _mm256_sllv_epi64(flip_l, shift2)));
    flip_r = _mm256_or_si256(flip_r, _mm256_and_si256(pre_r, _mm256_srlv_epi64(flip_r, shift2)));
    MM = _mm256_sllv_epi64(flip_l, shift1897);
    MM = _mm256_or_si256(MM, _mm256_srlv_epi64(flip_r, shift1897));
    M = _mm_or_si128(_mm256_castsi256_si128(MM), _mm256_extracti128_si256(MM, 1));
    M = _mm_or_si128(M, _mm_unpackhi_epi64(M, M));
    return _mm_cvtsi128_si64(M) & ~(P | O);
}
// end of modification