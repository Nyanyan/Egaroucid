#pragma once
#include <iostream>
#include "./../common.hpp"
#include "./../board.hpp"

inline int count_around_pos(uint64_t stones, uint64_t pos){
    uint64_t hmask = pos & 0x7E7E7E7E7E7E7E7EULL;
    uint64_t vmask = pos & 0x00FFFFFFFFFFFF00ULL;
    uint64_t hvmask = pos & 0x007E7E7E7E7E7E00ULL;
    uint64_t res = stones & (
        (hmask << 1) | (hmask >> 1) | 
        (vmask << HW) | (vmask >> HW) | 
        (hvmask << HW_M1) | (hvmask >> HW_M1) | 
        (hvmask << HW_P1) | (hvmask >> HW_P1)
    );
    return pop_count_ull(res);
}

inline int count_around_coord(uint64_t stones, uint_fast8_t coord){
    return count_around_pos(stones, 1ULL << coord);
}

inline int count_around4_pos(uint64_t stones, uint64_t pos){
    uint64_t hmask = pos & 0x7E7E7E7E7E7E7E7EULL;
    uint64_t vmask = pos & 0x00FFFFFFFFFFFF00ULL;
    uint64_t res = stones & (
        (hmask << 1) | (hmask >> 1) | 
        (vmask << HW) | (vmask >> HW)
    );
    return pop_count_ull(res);
}

inline int count_around4_coord(uint64_t stones, uint_fast8_t coord){
    return count_around4_pos(stones, 1ULL << coord);
}

inline uint_fast8_t is_inside(Flip *flip, uint64_t empties, uint64_t put){
    if ((flip->flip & put) && count_around_pos(empties, put))
        return 1;
    return 0;
}