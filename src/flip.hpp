#pragma once
#include <iostream>
#include "setting.hpp"

inline int join_v_line(unsigned long long x, int c){
    x = (x >> c) & 0b0000000100000001000000010000000100000001000000010000000100000001ULL;
    return (x * 0b0000000100000010000001000000100000010000001000000100000010000000ULL) >> 56;
}

inline unsigned long long split_v_line(unsigned char x, int c){
    unsigned long long res = 0;
    unsigned long long a = x & 0b00001111;
    unsigned long long b = x & 0b11110000;
    res = a | (b << 28);
    a = res & 0b0000000000000000000000000000001100000000000000000000000000000011ULL;
    b = res & 0b0000000000000000000000000000110000000000000000000000000000001100ULL;
    res = a | (b << 14);
    a = res & 0b0000000000000001000000000000000100000000000000010000000000000001ULL;
    b = res & 0b0000000000000010000000000000001000000000000000100000000000000010ULL;
    res = a | (b << 7);
    return res << c;
}

inline int join_d7_line(unsigned long long x, const int t){
    x = (x >> t) & 0b0000000000000010000001000000100000010000001000000100000010000001ULL;
    return (x * 0b1000000010000000100000001000000010000000100000001000000010000000ULL) >> 56;
}

inline unsigned long long split_d7_line(unsigned char x, int t){
    unsigned char c = x & 0b01010101;
    unsigned char d = x & 0b10101010;
    x = (c << 1) | (d >> 1);
    c = x & 0b00110011;
    d = x & 0b11001100;
    x = (c << 2) | (d >> 2);
    c = x & 0b00001111;
    d = x & 0b11110000;
    x = (c << 4) | (d >> 4);
    unsigned long long a = x & 0b00001111;
    unsigned long long b = x & 0b11110000;
    unsigned long long res = a | (b << 24);
    a = res & 0b0000000000000000000000000000000000110000000000000000000000000011ULL;
    b = res & 0b0000000000000000000000000000000011000000000000000000000000001100ULL;
    res = a | (b << 12);
    a = res & 0b0000000100000000000001000000000000010000000000000100000000000001ULL;
    b = res & 0b0000001000000000000010000000000000100000000000001000000000000010ULL;
    res = a | (b << 6);
    return res << t;
}

inline int join_d9_line(unsigned long long x, int t){
    if (t > 0)
        x >>= t;
    else if (t < 0)
        x <<= (-t);
    x &= 0b1000000001000000001000000001000000001000000001000000001000000001ULL;
    return (x * 0b0000000100000001000000010000000100000001000000010000000100000001ULL) >> 56;
}

inline unsigned long long split_d9_line(unsigned char x, int t){
    unsigned long long a = x & 0b00001111;
    unsigned long long b = x & 0b11110000;
    unsigned long long res = a | (b << 32);
    a = res & 0b0000000000000000000000000011000000000000000000000000000000000011ULL;
    b = res & 0b0000000000000000000000001100000000000000000000000000000000001100ULL;
    res = a | (b << 16);
    a = res & 0b0000000001000000000000000001000000000000000001000000000000000001ULL;
    b = res & 0b0000000010000000000000000010000000000000000010000000000000000010ULL;
    res = a | (b << 8);
    if (t > 0)
        return res << t;
    return res >> (-t);
}

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