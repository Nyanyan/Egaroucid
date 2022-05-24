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


// 縦に返す
/*
    内側から外側の領域に返す
*/
inline bool is_flip_vertically(const Flip *flip){
    if (pos_is_box(flip))
        return false;
    if (pos_is_edge(flip) && !pos_is_edge(flip->flip))
        return true;
    if (pos_is_middle_edge(flip) && !pos_is_edge(flip->flip) && !pos_is_middle_edge(flip->flip))
        return true;
    return false;
}


// 横に返す
/*
    同じ領域内で返す
*/
inline bool is_flip_horizontally(const Flip *flip){
    if (pos_is_middle_edge(flip) && !pos_is_edge(flip->flip))
        return true;
    if (pos_is_box(flip) && !pos_is_middle_edge(flip->flip) && !pos_is_edge(flip->flip))
        return true;
    return false;
}


// 2石返し
/*
    2石以上の空きマスに接する石を返す
*/
inline bool is_flip_2_stones(const Board *board, const Flip *flip){
    if (pop_count_ull(flip->flip) < 2)
        return false;
    uint64_t f = flip->flip;
    uint64_t empties = ~(board->player | board->opponent | (1ULL << flip->pos));
    int res = 0;
    for (uint_fast8_t cell = first_bit(&f); f; cell = next_bit(&f))
        res += min(1, count_around_coord(empties, cell));
    return res >= 2;
}


// 二石割り
/*
    縦に2石返し
    悪手
*/
inline bool is_two_stones_split(const Board *board, const Flip *flip){
    if (count_around_coord(flip->flip, flip->pos) != 1)
        return false;
    if (count_around4_coord(flip->flip, flip->pos))
        return is_flip_vertically(flip) && is_flip_2_stones(board, flip);
    return false;
}


// 突き
/*
    表面の石を縦に返す
*/

inline bool is_thrust(const Board *board, const Flip *flip){
    
}
