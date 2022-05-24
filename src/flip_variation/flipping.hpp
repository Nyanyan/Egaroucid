#pragma once
#include <iostream>
#include "./../common.hpp"
#include "./../board.hpp"
#include "around.hpp"
#include "position.hpp"

// 二重返し
/*
    2方向に同時に返す
    悪手
*/

inline bool is_double_flipping(const Board *board, const Flip *flip){
    const uint64_t put = 1ULL << flip->pos;
    uint64_t empties = ~(board->player | board->opponent | (1ULL << flip->pos));
    uint_fast8_t flip_direction = 
        is_inside(flip, empties, put << 1) + is_inside(flip, empties, put >> 1) + 
        is_inside(flip, empties, put << HW) + is_inside(flip, empties, put >> HW) + 
        is_inside(flip, empties, put << HW_M1) + is_inside(flip, empties, put >> HW_M1) + 
        is_inside(flip, empties, put << HW_P1) + is_inside(flip, empties, put >> HW_P1);
    return flip_direction >= 2;
}

// 二石割り
/*
    縦に2石返し
    悪手
*/
inline bool is_two_stones_split(){

}