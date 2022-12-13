/*
    Egaroucid Project

    @file common.hpp
        Common things
    @date 2021-2022
    @author Takuto Yamana (a.k.a. Nyanyan)
    @license GPL-3.0 license
*/

#pragma once
#include <iostream>
#include <fstream>
#include <chrono>
#include <random>
#include <string>
#include "setting.hpp"

// board size definition
#define HW 8
#define HW_M1 7
#define HW_P1 9
#define HW2 64
#define HW2_M1 63
#define HW2_P1 65

// color definition
#define BLACK 0
#define WHITE 1
#define VACANT 2

// evaluation phase definition
#define N_PHASES 30
#define PHASE_N_STONES 2

// constant
#define N_8BIT 256
#define INF 100000000
#define SCORE_INF 127
#define SCORE_MAX 64

// undefined legal bitboard: set bit on d4, d5, e4, and e5
#define LEGAL_UNDEFINED 0x0000001818000000ULL

/*
    @brief timing function

    @return time in milliseconds
*/
inline uint64_t tim(){
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

std::mt19937 raw_myrandom(tim());

/*
    @brief random function

    @return random value from 0.0 to 1.0 (not including 1.0)
*/
inline double myrandom(){
    return (double)raw_myrandom() / std::mt19937::max();
}

/*
    @brief randrange function

    @param s                    minimum integer
    @param e                    maximum integer
    @return random integer from s to e - 1
*/
inline int32_t myrandrange(int32_t s, int32_t e){
    return s +(int)((e - s) * myrandom());
}

/*
    @brief random integer function

    @return random 32bit integer
*/
inline uint32_t myrand_uint(){
    return (uint32_t)raw_myrandom();
}

/*
    @brief random integer function with bit reversed

    @return random 32bit integer with reversed bits
*/
inline uint32_t myrand_uint_rev(){
    uint32_t x = raw_myrandom();
    x = ((x & 0x55555555U) << 1) | ((x & 0xAAAAAAAAU) >> 1);
    x = ((x & 0x33333333U) << 2) | ((x & 0xCCCCCCCCU) >> 2);
    x = ((x & 0x0F0F0F0FU) << 4) | ((x & 0xF0F0F0F0U) >> 4);
    x = ((x & 0x00FF00FFU) << 8) | ((x & 0xFF00FF00U) >> 8);
    return ((x & 0x0000FFFFU) << 16) | ((x & 0xFFFF0000U) >> 16);
}

/*
    @brief random integer function

    @return random 64bit integer
*/
inline uint64_t myrand_ull(){
    return ((uint64_t)raw_myrandom() << 32) | (uint64_t)raw_myrandom();
}

/*
    @brief open a file

    wrapper for cross pratform

    @param fp                   FILE
    @param file                 file name
    @param mode                 open mode
    @return file opened?
*/
inline bool file_open(FILE **fp, const char *file, const char *mode){
    #ifdef _WIN64
        return fopen_s(fp, file, mode) == 0;
    #else
        *fp = fopen(file, mode);
        return *fp != NULL;
    #endif
}

/*
    @brief caluculate NPS (Nodes Per Second)

    @param n_nodes              number of nodes
    @param elapsed              time
    @return NPS
*/
inline uint64_t calc_nps(uint64_t n_nodes, uint64_t elapsed){
    if (elapsed == 0ULL)
        elapsed = 1ULL;
    return n_nodes * 1000ULL / elapsed;
}

// set false to stop all search immediately
bool global_searching = true;
