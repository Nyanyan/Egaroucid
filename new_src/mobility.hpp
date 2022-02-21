#pragma once
#include <iostream>
#include "bit.hpp"

inline uint8_t calc_mobility_line(uint8_t p, uint8_t o){
    uint8_t p1 = p << 1;
    return ~(p1 | o) & (p1 + o);
}

inline uint64_t calc_mobility_left(uint64_t p, uint64_t o){
    return (uint64_t)calc_mobility_line(p & 0xFF, o & 0xFF) | 
        ((uint64_t)calc_mobility_line((p >> 8) & 0xFF, (o >> 8) & 0xFF) << 8) | 
        ((uint64_t)calc_mobility_line((p >> 16) & 0xFF, (o >> 16) & 0xFF) << 16) | 
        ((uint64_t)calc_mobility_line((p >> 24) & 0xFF, (o >> 24) & 0xFF) << 24) | 
        ((uint64_t)calc_mobility_line((p >> 32) & 0xFF, (o >> 32) & 0xFF) << 32) | 
        ((uint64_t)calc_mobility_line((p >> 40) & 0xFF, (o >> 40) & 0xFF) << 40) | 
        ((uint64_t)calc_mobility_line((p >> 48) & 0xFF, (o >> 48) & 0xFF) << 48) | 
        ((uint64_t)calc_mobility_line((p >> 56) & 0xFF, (o >> 56) & 0xFF) << 56);
}

inline uint64_t calc_mobility(uint64_t p, uint64_t o){
    uint64_t res = calc_mobility_left(p, o) | 
        horizontal_mirror(calc_mobility_left(horizontal_mirror(p), horizontal_mirror(o))) | 
        black_line_mirror(calc_mobility_left(black_line_mirror(p), black_line_mirror(o))) | 
        white_line_mirror(calc_mobility_left(white_line_mirror(p), white_line_mirror(o)));
    constexpr uint8_t mask[N_DIAG_LINE] = {
        0b11100000, 0b11110000, 0b11111000, 0b11111100, 0b11111110, 0b11111111,
        0b01111111, 0b00111111, 0b00011111, 0b00001111, 0b00000111
    };
    int i;
    for (i = 0; i < N_DIAG_LINE; ++i){
        res |= split_d7_lines[calc_mobility_line(join_d7_lines[i](p), join_d7_lines[i](o)) & mask[i]][i];
        res |= split_d9_lines[calc_mobility_line(join_d9_lines[i](p), join_d9_lines[i](o)) & mask[10 - i]][i];
    }
    res = white_line_mirror(res);
    p = white_line_mirror(p);
    o = white_line_mirror(o);
    for (i = 0; i < N_DIAG_LINE; ++i){
        res |= split_d7_lines[calc_mobility_line(join_d7_lines[i](p), join_d7_lines[i](o)) & mask[i]][i];
        res |= split_d9_lines[calc_mobility_line(join_d9_lines[i](p), join_d9_lines[i](o)) & mask[10 - i]][i];
    }
    res = black_line_mirror(res);
    p = black_line_mirror(p);
    o = black_line_mirror(o);
    for (i = 0; i < N_DIAG_LINE; ++i){
        res |= split_d7_lines[calc_mobility_line(join_d7_lines[i](p), join_d7_lines[i](o)) & mask[i]][i];
        res |= split_d9_lines[calc_mobility_line(join_d9_lines[i](p), join_d9_lines[i](o)) & mask[10 - i]][i];
    }
    res &= ~(p | o);
    return rotate_180(res);
}