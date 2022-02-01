#pragma once
#include <iostream>
#include "setting.hpp"

inline int join_d7_line_0(const unsigned long long x){
    return 0;
}

inline int join_d7_line_1(const unsigned long long x){
    return 0;
}

inline int join_d7_line_2(const unsigned long long x){
    return ((x & 0b00000000'00000000'00000000'00000000'00000000'00000001'00000010'00000100ULL) * 
                0b00100000'00100000'00100000'00000000'00000000'00000000'00000000'00000000ULL) >> 56;
}

inline int join_d7_line_3(const unsigned long long x){
    return ((x & 0b00000000'00000000'00000000'00000000'00000001'00000010'00000100'00001000ULL) * 
                0b00010000'00010000'00010000'00010000'00000000'00000000'00000000'00000000ULL) >> 56;
}

inline int join_d7_line_4(const unsigned long long x){
    return ((x & 0b00000000'00000000'00000000'00000001'00000010'00000100'00001000'00010000ULL) * 
            0b00001000'00001000'00001000'00001000'00001000'00000000'00000000'00000000ULL) >> 56;
}

inline int join_d7_line_5(const unsigned long long x){
    return ((x & 0b00000000'00000000'00000001'00000010'00000100'00001000'00010000'00100000ULL) * 
            0b00000100'00000100'00000100'00000100'00000100'00000100'00000000'00000000ULL) >> 56;
}

inline int join_d7_line_6(const unsigned long long x){
    return ((x & 0b00000000'00000001'00000010'00000100'00001000'00010000'00100000'01000000ULL) * 
            0b00000010'00000010'00000010'00000010'00000010'00000010'00000010'00000000ULL) >> 56;
}

inline int join_d7_line_7(const unsigned long long x){
    return ((x & 0b00000001'00000010'00000100'00001000'00010000'00100000'01000000'10000000ULL) * 
            0b00000001'00000001'00000001'00000001'00000001'00000001'00000001'00000001ULL) >> 56;
}

inline int join_d7_line_8(const unsigned long long x){
    return ((x & 0b00000010'00000100'00001000'00010000'00100000'01000000'10000000'00000000ULL) * 
            0b00000000'00000001'00000001'00000001'00000001'00000001'00000001'00000001ULL) >> 57;
}

inline int join_d7_line_9(const unsigned long long x){
    return ((x & 0b00000100'00001000'00010000'00100000'01000000'10000000'00000000'00000000ULL) * 
            0b00000000'00000000'00000001'00000001'00000001'00000001'00000001'00000001ULL) >> 58;
}

inline int join_d7_line_10(const unsigned long long x){
    return ((x & 0b00001000'00010000'00100000'01000000'10000000'00000000'00000000'00000000ULL) * 
            0b00000000'00000000'00000000'00000001'00000001'00000001'00000001'00000001ULL) >> 59;
}

inline int join_d7_line_11(const unsigned long long x){
    return ((x & 0b00010000'00100000'01000000'10000000'00000000'00000000'00000000'00000000ULL) * 
            0b00000000'00000000'00000000'00000000'00000001'00000001'00000001'00000001ULL) >> 60;
}

inline int join_d7_line_12(const unsigned long long x){
    return ((x & 0b00100000'01000000'10000000'00000000'00000000'00000000'00000000'00000000ULL) * 
            0b00000000'00000000'00000000'00000000'00000000'00000001'00000001'00000001ULL) >> 61;
}

inline int join_d7_line_13(const unsigned long long x){
    return 0;
}

inline int join_d7_line_14(const unsigned long long x){
    return 0;
}

int (*join_d7_line_fast[])(const unsigned long long) = {
    join_d7_line_0, join_d7_line_1, join_d7_line_2, join_d7_line_3, 
    join_d7_line_4, join_d7_line_5, join_d7_line_6, join_d7_line_7,
    join_d7_line_8, join_d7_line_9, join_d7_line_10, join_d7_line_11,
    join_d7_line_12, join_d7_line_13, join_d7_line_14
};