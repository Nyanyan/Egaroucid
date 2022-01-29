#pragma once
#include <chrono>
#include <random>
#include "setting.hpp"

using namespace std;

#define inf 100000000
#define n_phases 15
#define phase_n_stones 4

#define n_line 6561
#define hw 8
#define hw_m1 7
#define hw_p1 9
#define hw2 64
#define hw22 128
#define hw2_m1 63
#define hw2_mhw 56
#define hw2_p1 65
#define black 0
#define white 1
#define vacant 2

inline long long tim(){
    return chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now().time_since_epoch()).count();
}

mt19937 raw_myrandom(tim());
bool global_searching = true;

inline double myrandom(){
    return (double)raw_myrandom() / mt19937::max();
}

inline int myrandrange(int s, int e){
    return s +(int)((e - s) * myrandom());
}

inline unsigned long long myrand_ull(){
    return (unsigned long long)(myrandom() * 18446744073709551615ULL);
}

/*
inline int pop_count_ull(unsigned long long x){
    unsigned long long a = x & 0b0101010101010101010101010101010101010101010101010101010101010101ULL;
    unsigned long long b = x & 0b1010101010101010101010101010101010101010101010101010101010101010ULL;
    x = a + (b >> 1);
    a = x & 0b0011001100110011001100110011001100110011001100110011001100110011ULL;
    b = x & 0b1100110011001100110011001100110011001100110011001100110011001100ULL;
    x = a + (b >> 2);
    a = x & 0b0000111100001111000011110000111100001111000011110000111100001111ULL;
    b = x & 0b1111000011110000111100001111000011110000111100001111000011110000ULL;
    x = a + (b >> 4);
    a = x & 0b0000000011111111000000001111111100000000111111110000000011111111ULL;
    b = x & 0b1111111100000000111111110000000011111111000000001111111100000000ULL;
    x = a + (b >> 8);
    a = x & 0b0000000000000000111111111111111100000000000000001111111111111111ULL;
    b = x & 0b1111111111111111000000000000000011111111111111110000000000000000ULL;
    x = a + (b >> 16);
    a = x & 0b0000000000000000000000000000000011111111111111111111111111111111ULL;
    b = x & 0b1111111111111111111111111111111100000000000000000000000000000000ULL;
    return (int)(a + (b >> 32));
}
*/

inline int pop_count_ull(unsigned long long x){
    x = x - ((x >> 1) & 0x5555555555555555ULL);
	x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL);
	x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0FULL;
	x = (x * 0x0101010101010101ULL) >> 56;
    return (int)x;
}

inline unsigned long long mirror_v(unsigned long long x){
    unsigned long long a = x & 0b0101010101010101010101010101010101010101010101010101010101010101ULL;
    unsigned long long b = x & 0b1010101010101010101010101010101010101010101010101010101010101010ULL;
    x = (a << 1) | (b >> 1);
    a = x & 0b0011001100110011001100110011001100110011001100110011001100110011ULL;
    b = x & 0b1100110011001100110011001100110011001100110011001100110011001100ULL;
    x = (a << 2) | (b >> 2);
    a = x & 0b0000111100001111000011110000111100001111000011110000111100001111ULL;
    b = x & 0b1111000011110000111100001111000011110000111100001111000011110000ULL;
    x = (a << 4) | (b >> 4);
    a = x & 0b0000000011111111000000001111111100000000111111110000000011111111ULL;
    b = x & 0b1111111100000000111111110000000011111111000000001111111100000000ULL;
    x = (a << 8) | (b >> 8);
    a = x & 0b0000000000000000111111111111111100000000000000001111111111111111ULL;
    b = x & 0b1111111111111111000000000000000011111111111111110000000000000000ULL;
    x = (a << 16) | (b >> 16);
    a = x & 0b0000000000000000000000000000000011111111111111111111111111111111ULL;
    b = x & 0b1111111111111111111111111111111100000000000000000000000000000000ULL;
    return (a << 32) | (b >> 32);
}

inline unsigned long long white_line(unsigned long long x){
    unsigned long long res = 0;
    int i, j;
    for (i = 0; i < hw; ++i){
        for (j = 0; j < hw; ++j){
            res |= (1 & (x >> (i * hw + j))) << (j * hw + i);
        }
    }
    return res;
}

inline unsigned long long black_line(unsigned long long x){
    unsigned long long res = 0;
    int i, j;
    for (i = 0; i < hw; ++i){
        for (j = 0; j < hw; ++j){
            res |= (1 & (x >> (i * hw + j))) << ((hw_m1 - j) * hw + hw_m1 - i);
        }
    }
    return res;
}

inline int join_v_line(unsigned long long x, int c){
    x = x & (0b0000000100000001000000010000000100000001000000010000000100000001ULL << c);
    unsigned long long a = x & 0b0000000011111111000000001111111100000000111111110000000011111111ULL;
    unsigned long long b = x & 0b1111111100000000111111110000000011111111000000001111111100000000ULL;
    x = a | (b >> 7);
    a = x & 0b0000000000000000111111111111111100000000000000001111111111111111ULL;
    b = x & 0b1111111111111111000000000000000011111111111111110000000000000000ULL;
    x = a | (b >> 14);
    a = x & 0b0000000000000000000000000000000011111111111111111111111111111111ULL;
    b = x & 0b1111111111111111111111111111111100000000000000000000000000000000ULL;
    x = a | (b >> 28);
    return (int)(x >> c);
    /*
    int res = 1 & (x >> c);
    res |= (1 & (x >> (hw + c))) << 1;
    res |= (1 & (x >> (2 * hw + c))) << 2;
    res |= (1 & (x >> (3 * hw + c))) << 3;
    res |= (1 & (x >> (4 * hw + c))) << 4;
    res |= (1 & (x >> (5 * hw + c))) << 5;
    res |= (1 & (x >> (6 * hw + c))) << 6;
    res |= (1 & (x >> (7 * hw + c))) << 7;
    return res;
    */
}

inline unsigned long long split_v_line(int x, int c){
    unsigned long long res = 0;
    unsigned long long a = x & 0b00001111;
    unsigned long long b = x & 0b11110000;
    res = a | (b << 28);
    a = res & 0b0011001100110011001100110011001100110011001100110011001100110011ULL;
    b = res & 0b1100110011001100110011001100110011001100110011001100110011001100ULL;
    res = a | (b << 14);
    a = res & 0b0101010101010101010101010101010101010101010101010101010101010101ULL;
    b = res & 0b1010101010101010101010101010101010101010101010101010101010101010ULL;
    res = a | (b << 7);
    return res << c;
    /*
    unsigned long long res = (1ULL & x) << c;
    res |= (1ULL & (x >> 1)) << (hw + c);
    res |= (1ULL & (x >> 2)) << (2 * hw + c);
    res |= (1ULL & (x >> 3)) << (3 * hw + c);
    res |= (1ULL & (x >> 4)) << (4 * hw + c);
    res |= (1ULL & (x >> 5)) << (5 * hw + c);
    res |= (1ULL & (x >> 6)) << (6 * hw + c);
    res |= (1ULL & (x >> 7)) << (7 * hw + c);
    return res;
    */
}

inline int join_d7_line(unsigned long long x, int t){
    /*
    x = x & (0b1000000100000010000001000000100000010000001000000100000010000001ULL << t);
    unsigned long long a = x & 0b0111111100000001111111000000011111110000000111111100000001111111ULL;
    unsigned long long b = x & 0b1000000011111110000000111111100000001111111000000011111110000000ULL;
    unsigned long long c;
    x = a | (b >> 6);
    cerr << x << endl;
    a = x & 0b1111111100000000000000111111111111110000000000000011111111111111ULL;
    b = x & 0b0000000011111111111111000000000000001111111111111100000000000000ULL;
    x = a | (b >> 12);
    cerr << x << endl;
    a = x & 0b0000000000000000000000000000000000001111111111111111111111111111ULL;
    b = x & 0b0000000011111111111111111111111111110000000000000000000000000000ULL;
    c = x & 0b1111111100000000000000000000000000000000000000000000000000000000ULL;
    x = a | (b >> 24) | (c >> 48);
    cerr << x << endl;
    return (int)(x >> t);
    */
    int res = 1 & (x >> t);
    res |= 0b10 & (x >> (6 + t));
    res |= 0b100 & (x >> (12 + t));
    res |= 0b1000 & (x >> (18 + t));
    res |= 0b10000 & (x >> (24 + t));
    res |= 0b100000 & (x >> (30 + t));
    res |= 0b1000000 & (x >> (36 + t));
    res |= 0b10000000 & (x >> (42 + t));
    return res;
    
}

inline unsigned long long split_d7_line(int x, int c){
    unsigned long long res = (1ULL & x) << c;
    res |= (0b10ULL & x) << (6 + c);
    res |= (0b100ULL & x) << (12 + c);
    res |= (0b1000ULL & x) << (18 + c);
    res |= (0b10000ULL & x) << (24 + c);
    res |= (0b100000ULL & x) << (30 + c);
    res |= (0b1000000ULL & x) << (36 + c);
    res |= (0b10000000ULL & x) << (42 + c);
    return res;
}

inline int join_d9_line(unsigned long long x, int t){
    int res = 0;
    if (t >= 0)
        res |= 1 & (x >> t);
    res |= 0b10 & (x >> (8 + t));
    res |= 0b100 & (x >> (16 + t));
    res |= 0b1000 & (x >> (24 + t));
    res |= 0b10000 & (x >> (32 + t));
    res |= 0b100000 & (x >> (40 + t));
    res |= 0b1000000 & (x >> (48 + t));
    res |= 0b10000000 & (x >> (56 + t));
    return res;
}

inline unsigned long long split_d9_line(int x, int c){
    unsigned long long res = 0;
    if (c >= 0)
        res |= (1ULL & x) << c;
    res |= (0b10ULL & x) << (8 + c);
    res |= (0b100ULL & x) << (16 + c);
    res |= (0b1000ULL & x) << (24 + c);
    res |= (0b10000ULL & x) << (32 + c);
    res |= (0b100000ULL & x) << (40 + c);
    res |= (0b1000000ULL & x) << (48 + c);
    res |= (0b10000000ULL & x) << (56 + c);
    return res;
}