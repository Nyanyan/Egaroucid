#pragma once
#include <iostream>
#include "./../common.hpp"
#include "./../board.hpp"

// ボックス
inline bool pos_is_box(Flip *flip){
    return ((1ULL << flip->pos) & 0b00000000'00000000'00111100'00111100'00111100'00111100'00000000'00000000ULL) != 0ULL;
}

// ボックスコーナー
inline bool pos_is_box_corner(Flip *flip){
    return ((1ULL << flip->pos) & 0b00000000'00000000'00100100'00000000'00000000'00100100'00000000'00000000ULL) != 0ULL;
}

// 中辺
inline bool pos_is_middle_edge(Flip *flip){
    return ((1ULL << flip->pos) & 0b00000000'01111110'01000010'01000010'01000010'01000010'01111110'00000000ULL) != 0ULL;
}

// 中A
inline bool pos_is_a(Flip *flip){
    return ((1ULL << flip->pos) & 0b00000000'00100100'00000000'00000000'00000000'00000000'00100100'00000000ULL) != 0ULL;
}

// 中B
inline bool pos_is_b(Flip *flip){
    return ((1ULL << flip->pos) & 0b00000000'00011000'00000000'00000000'00000000'00000000'00011000'00000000ULL) != 0ULL;
}

// 辺
inline bool pos_is_edge(Flip *flip){
    return ((1ULL << flip->pos) & 0b11111111'10000001'10000001'10000001'10000001'10000001'10000001'11111111ULL) != 0ULL;
}

// X
inline bool pos_is_X(Flip *flip){
    return ((1ULL << flip->pos) & 0b00000000'01000010'00000000'00000000'00000000'00000000'01000010'00000000ULL) != 0ULL;
}

// 隅
inline bool pos_is_corner(Flip *flip){
    return ((1ULL << flip->pos) & 0b10000001'00000000'00000000'00000000'00000000'00000000'00000000'10000001ULL) != 0ULL;
}

// A
inline bool pos_is_A(Flip *flip){
    return ((1ULL << flip->pos) & 0b00100100'00000000'00000000'00000000'00000000'00000000'00000000'00100100ULL) != 0ULL;
}

// B
inline bool pos_is_B(Flip *flip){
    return ((1ULL << flip->pos) & 0b00011000'00000000'00000000'00000000'00000000'00000000'00000000'00011000ULL) != 0ULL;
}

// C
inline bool pos_is_C(Flip *flip){
    return ((1ULL << flip->pos) & 0b01000010'00000000'00000000'00000000'00000000'00000000'00000000'01000010ULL) != 0ULL;
}





// ボックス
inline bool pos_is_box(uint64_t stones){
    return (stones & 0b00000000'00000000'00111100'00111100'00111100'00111100'00000000'00000000ULL) != 0ULL;
}

// ボックスコーナー
inline bool pos_is_box_corner(uint64_t stones){
    return (stones & 0b00000000'00000000'00100100'00000000'00000000'00100100'00000000'00000000ULL) != 0ULL;
}

// 中辺
inline bool pos_is_middle_edge(uint64_t stones){
    return (stones & 0b00000000'01111110'01000010'01000010'01000010'01000010'01111110'00000000ULL) != 0ULL;
}

// 中A
inline bool pos_is_a(uint64_t stones){
    return (stones & 0b00000000'00100100'00000000'00000000'00000000'00000000'00100100'00000000ULL) != 0ULL;
}

// 中B
inline bool pos_is_b(uint64_t stones){
    return (stones & 0b00000000'00011000'00000000'00000000'00000000'00000000'00011000'00000000ULL) != 0ULL;
}

// 辺
inline bool pos_is_edge(uint64_t stones){
    return (stones & 0b11111111'10000001'10000001'10000001'10000001'10000001'10000001'11111111ULL) != 0ULL;
}

// X
inline bool pos_is_X(uint64_t stones){
    return (stones & 0b00000000'01000010'00000000'00000000'00000000'00000000'01000010'00000000ULL) != 0ULL;
}

// 隅
inline bool pos_is_corner(uint64_t stones){
    return (stones & 0b10000001'00000000'00000000'00000000'00000000'00000000'00000000'10000001ULL) != 0ULL;
}

// A
inline bool pos_is_A(uint64_t stones){
    return (stones & 0b00100100'00000000'00000000'00000000'00000000'00000000'00000000'00100100ULL) != 0ULL;
}

// B
inline bool pos_is_B(uint64_t stones){
    return (stones & 0b00011000'00000000'00000000'00000000'00000000'00000000'00000000'00011000ULL) != 0ULL;
}

// C
inline bool pos_is_C(uint64_t stones){
    return (stones & 0b01000010'00000000'00000000'00000000'00000000'00000000'00000000'01000010ULL) != 0ULL;
}