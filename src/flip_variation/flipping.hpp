#pragma once
#include <iostream>
#include "./../common.hpp"
#include "./../board.hpp"
#include "around.hpp"
#include "position.hpp"
#include "stone.hpp"


// 中割り、潜在的中割り
inline int give_potential_flip_inside(const Board *board, const Flip *flip){
    uint64_t f = flip->flip;
    uint64_t empties = ~(board->player | board->opponent | (1ULL << flip->pos));
    uint_fast8_t openness;
    int res = 0;
    for (uint_fast8_t cell = first_bit(&f); f; cell = next_bit(&f)){
        openness = pop_count_ull(bit_around[cell] | empties);
        if (openness == 1)
            ++res;
    }
    return res;
}


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



// 斜めに返す
/*
    斜めに返す
*/
inline bool is_flip_diagonally(const Flip *flip){
    uint64_t pos = 1ULL << flip->pos;
    return 
        (flip->flip & (pos << HW_M1)) || (flip->flip & (pos >> HW_M1)) | 
        (flip->flip & (pos << HW_P1)) || (flip->flip & (pos >> HW_P1));
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
inline bool is_thrust(const Flip *flip, const uint64_t face_stones){
    return (flip->flip & face_stones) && is_flip_vertically(flip) && !is_flip_diagonally(flip);
}


// めくり
/*
    表面の石を斜めに返す
*/
inline bool is_turn_over(const Flip *flip, const uint64_t face_stones){
    return (flip->flip & face_stones) && is_flip_diagonally(flip);
}


// 切り
/*
    内側の石を斜めに返す
*/
inline bool is_cut(const Flip *flip, const uint64_t outside_stones){
    return ((flip->flip & outside_stones) == 0ULL) && is_flip_diagonally(flip);
}


// 被せ
/*
    端の石を表面側に斜めに返す
*/
inline bool is_cover(const Flip *flip, const uint64_t end_stones, const uint64_t face_stones){
    return (flip->flip & end_stones) && is_flip_diagonally(flip) && is_joining4(1ULL << flip->pos, face_stones);
}


// 引き
/*
    端の石を辺側に縦に返す
*/
inline bool is_pull(const Flip *flip, const uint64_t end_stones){
    return (flip->flip & end_stones) && is_flip_vertically(flip) && !is_flip_diagonally(flip);
}


// 刺し
/*
    端の石を端外側に斜めに返す
*/
inline bool is_stab(const Flip *flip, const uint64_t end_stones, const uint64_t face_stones){
    return (flip->flip & end_stones) && is_flip_diagonally(flip) && !is_joining4(1ULL << flip->pos, face_stones);
}


