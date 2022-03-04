#pragma once
#include <iostream>
#include <chrono>
#include <random>

using namespace std;

#define HW 8
#define HW_M1 7
#define HW_P1 9
#define HW2 64
#define HW2_M1 63
#define HW2_P1 65

#define N_8BIT 256
#define N_DIAG_LINE 11
#define N_DIAG_LINE_M1 10

#define BLACK 0
#define WHITE 1
#define VACANT 2

#define N_PHASES 30
#define PHASE_N_STONES 2

#define INF 100000000

inline uint64_t tim(){
    return chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now().time_since_epoch()).count();
}

mt19937 raw_myrandom(tim());

inline double myrandom(){
    return (double)raw_myrandom() / mt19937::max();
}

inline int32_t myrandrange(int32_t s, int32_t e){
    return s +(int)((e - s) * myrandom());
}

inline uint64_t myrand_ull(){
    return (uint64_t)raw_myrandom();
}