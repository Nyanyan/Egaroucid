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

inline double myrandom(){
    return (double)raw_myrandom() / mt19937::max();
}

inline int myrandrange(int s, int e){
    return s +(int)((e - s) * myrandom());
}

inline int pop_count_ull(unsigned long long x){
    x = x - ((x >> 1) & 0x5555555555555555);
	x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333);
	x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0F;
	x = (x * 0x0101010101010101) >> 56;
    return (int)x;
}

bool global_searching = true;