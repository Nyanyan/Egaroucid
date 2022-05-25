#pragma once
#include <iostream>
#include "./../common.hpp"
#include "./../board.hpp"
#include "around.hpp"
#include "position.hpp"


// 外側の石
/*
    空きマスに4近傍のいずれかが接する石
*/
inline uint64_t calc_outside_stones(const Board *board){
    uint64_t empties = ~(board->player | board->opponent);
    uint64_t hmask = empties & 0x7E7E7E7E7E7E7E7EULL;
    uint64_t vmask = empties & 0x00FFFFFFFFFFFF00ULL;
    uint64_t res = 
        (hmask << 1) | (hmask >> 1) | 
        (vmask << HW) | (vmask >> HW);
    return res & (board->player | board->opponent);
}


// 内側の石
/*
    外側でない石
*/
inline uint64_t calc_inside_stones(const Board *board){
    return (board->player | board->opponent) & ~calc_outside_stones(board);
}


// 壁の石
/*
    外側に接する同じ色の石の3つ以上の4近傍での並び
*/
inline uint64_t calc_wall_stones(uint64_t player_outside_stones){
    uint64_t n_res = player_outside_stones & (
        ((player_outside_stones & 0xFEFEFEFEFEFEFEFEULL) >> 1) | ((player_outside_stones & 0x7F7F7F7F7F7F7F7FULL) << 1) | 
        ((player_outside_stones & 0xFFFFFFFFFFFFFF00ULL) >> HW) | ((player_outside_stones & 0x00FFFFFFFFFFFFFFULL) << HW)
    );
    uint64_t res = 0ULL;
    while (n_res & ~res){
        res |= n_res;
        n_res |= player_outside_stones & (
            ((res & 0xFEFEFEFEFEFEFEFEULL) >> 1) | 
            ((res & 0x7F7F7F7F7F7F7F7FULL) << 1) | 
            ((res & 0xFFFFFFFFFFFFFF00ULL) >> HW) | 
            ((res & 0x00FFFFFFFFFFFFFFULL) << HW)
        );
    }
    return res;
}


// 端の石
/*
    外側の石で対面方向に両方接していない石
*/
inline uint64_t calc_end_stones(uint64_t outside, uint64_t empties){
    uint64_t res = outside & 
        (((empties & 0xFEFEFEFEFEFEFEFEULL) >> 1) | ((empties & 0x7F7F7F7F7F7F7F7FULL) << 1)) & 
        (((empties & 0xFFFFFFFFFFFFFF00ULL) >> HW) | ((empties & 0x00FFFFFFFFFFFFFFULL) << HW));
    return res;
}


// 表面の石
/*
    外側に接する石で端でない石
*/
inline uint64_t calc_face_stones(uint64_t outside, uint64_t empties){
    return outside & ~calc_end_stones(outside, empties);
}


// 境界の石
/*
    外側で2つ以上相手の石が連続する石のうち，白黒の境界の石
*/
inline uint64_t calc_opponent_bound_stones(const Board *board, uint64_t outside){
    uint64_t player_outside = outside & board->player;
    uint64_t opponent_outside = outside & board->opponent;
    opponent_outside &= 
        ((opponent_outside & 0xFEFEFEFEFEFEFEFEULL) >> 1) | ((opponent_outside & 0x7F7F7F7F7F7F7F7FULL) << 1) | 
        ((opponent_outside & 0xFFFFFFFFFFFFFF00ULL) >> HW) | ((opponent_outside & 0x00FFFFFFFFFFFFFFULL) << HW);
    uint64_t res = opponent_outside & (
        ((player_outside & 0xFEFEFEFEFEFEFEFEULL) >> 1) | ((player_outside & 0x7F7F7F7F7F7F7F7FULL) << 1) | 
        ((player_outside & 0xFFFFFFFFFFFFFF00ULL) >> HW) | ((player_outside & 0x00FFFFFFFFFFFFFFULL) << HW)
    );
    return res;
}


// 壁化の石
/*
    外側の白黒境界のうち，境界の石でない石
*/
inline uint64_t calc_opponent_create_wall_stones(const Board *board, uint64_t outside, uint64_t bound_stones){
    uint64_t player_outside = outside & board->player;
    uint64_t opponent_outside = outside & board->opponent;
    uint64_t res = opponent_outside & (
        ((player_outside & 0xFEFEFEFEFEFEFEFEULL) >> 1) | ((player_outside & 0x7F7F7F7F7F7F7F7FULL) << 1) | 
        ((player_outside & 0xFFFFFFFFFFFFFF00ULL) >> HW) | ((player_outside & 0x00FFFFFFFFFFFFFFULL) << HW)
    );
    return res & ~bound_stones;
}


// 壁破りの石
/*
    表面の石のうち，端でない石
*/
inline uint64_t calc_opponent_break_wall_stones(uint64_t opponent_outside_stones, uint64_t bound_stones){
    return calc_wall_stones(opponent_outside_stones) & ~bound_stones;
}


// 4近傍で連続しているか
inline bool is_joining4(uint64_t a, uint64_t b){
    uint64_t bb = 
        ((b & 0xFEFEFEFEFEFEFEFEULL) >> 1) | ((b & 0x7F7F7F7F7F7F7F7FULL) << 1) | 
        ((b & 0xFFFFFFFFFFFFFF00ULL) >> HW) | ((b & 0x00FFFFFFFFFFFFFFULL) << HW);
    return (a & bb) != 0ULL;
}