#pragma once
#include <iostream>
#include "./../common.hpp"
#include "./../board.hpp"
#include "around.hpp"
#include "position.hpp"


// 外側の石
inline uint64_t calc_outside_stones(Board *board){
    uint64_t empties = ~(board->player | board->opponent);
    uint64_t hmask = empties & 0x7E7E7E7E7E7E7E7EULL;
    uint64_t vmask = empties & 0x00FFFFFFFFFFFF00ULL;
    uint64_t hvmask = empties & 0x007E7E7E7E7E7E00ULL;
    uint64_t res = 
        (hmask << 1) | (hmask >> 1) | 
        (vmask << HW) | (vmask >> HW) | 
        (hvmask << HW_M1) | (hvmask >> HW_M1) | 
        (hvmask << HW_P1) | (hvmask >> HW_P1);
    return res & (board->player | board->opponent);
}

// 端の石
/*
    同じ色の石の3つ以上の並びの端の石
*/
inline uint64_t calc_end_stones(Board *board){

}