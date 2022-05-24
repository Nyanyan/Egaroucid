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
inline uint64_t calc_outside_stones(Board *board){
    uint64_t empties = ~(board->player | board->opponent);
    uint64_t hmask = empties & 0x7E7E7E7E7E7E7E7EULL;
    uint64_t vmask = empties & 0x00FFFFFFFFFFFF00ULL;
    uint64_t res = 
        (hmask << 1) | (hmask >> 1) | 
        (vmask << HW) | (vmask >> HW);
    return res & (board->player | board->opponent);
}


// 壁の石
/*
    外側に接する同じ色の石の3つ以上の4近傍での並び
*/
inline uint64_t calc_wall_stones(uint64_t player_outside_stones){
    uint64_t n_res = player_outside_stones & (
        ((player_outside_stones & 0xFEFEFEFEFEFEFEFEULL) >> 1) | 
        ((player_outside_stones & 0x7F7F7F7F7F7F7F7FULL) << 1) | 
        ((player_outside_stones & 0xFFFFFFFFFFFFFF00ULL) >> HW) | 
        ((player_outside_stones & 0x00FFFFFFFFFFFFFFULL) << HW)
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


// 表面の石
/*
    壁の真ん中の石
*/
inline uint64_t calc_opponent_face_stones(Board *board){
    uint64_t p_wall = calc_wall_stones(calc_outside_stones(board) & board->opponent);
    /*
    uint64_t hmask = p_wall & 0x7E7E7E7E7E7E7E7EULL;
    uint64_t vmask = p_wall & 0x00FFFFFFFFFFFF00ULL;
    uint64_t res = 
        ((hmask << 1) & p_out & (hmask >> 1)) | 
        ((vmask << HW) & p_out & (vmask >> HW)) | 
        ()
    return res;
    */
    return 0;
}



// 端の石
/*
    壁の端の石
*/
