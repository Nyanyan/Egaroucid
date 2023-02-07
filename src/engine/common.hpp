/*
    Egaroucid Project

    @file common.hpp
        Common things
    @date 2021-2023
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
    #elif _WIN32
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

/*
    @brief bits around the cell are set
    from https://eukaryote.hateblo.jp/entry/2020/04/26/031246
*/
constexpr uint64_t bit_around[HW2] = {
    0x0000000000000302ULL, 0x0000000000000604ULL, 0x0000000000000e0aULL, 0x0000000000001c14ULL, 0x0000000000003828ULL, 0x0000000000007050ULL, 0x0000000000006020ULL, 0x000000000000c040ULL,
    0x0000000000030200ULL, 0x0000000000060400ULL, 0x00000000000e0a00ULL, 0x00000000001c1400ULL, 0x0000000000382800ULL, 0x0000000000705000ULL, 0x0000000000602000ULL, 0x0000000000c04000ULL,
    0x0000000003020300ULL, 0x0000000006040600ULL, 0x000000000e0a0e00ULL, 0x000000001c141c00ULL, 0x0000000038283800ULL, 0x0000000070507000ULL, 0x0000000060206000ULL, 0x00000000c040c000ULL,
    0x0000000302030000ULL, 0x0000000604060000ULL, 0x0000000e0a0e0000ULL, 0x0000001c141c0000ULL, 0x0000003828380000ULL, 0x0000007050700000ULL, 0x0000006020600000ULL, 0x000000c040c00000ULL,
    0x0000030203000000ULL, 0x0000060406000000ULL, 0x00000e0a0e000000ULL, 0x00001c141c000000ULL, 0x0000382838000000ULL, 0x0000705070000000ULL, 0x0000602060000000ULL, 0x0000c040c0000000ULL,
    0x0003020300000000ULL, 0x0006040600000000ULL, 0x000e0a0e00000000ULL, 0x001c141c00000000ULL, 0x0038283800000000ULL, 0x0070507000000000ULL, 0x0060206000000000ULL, 0x00c040c000000000ULL,
    0x0002030000000000ULL, 0x0004060000000000ULL, 0x000a0e0000000000ULL, 0x00141c0000000000ULL, 0x0028380000000000ULL, 0x0050700000000000ULL, 0x0020600000000000ULL, 0x0040c00000000000ULL,
    0x0203000000000000ULL, 0x0406000000000000ULL, 0x0a0e000000000000ULL, 0x141c000000000000ULL, 0x2838000000000000ULL, 0x5070000000000000ULL, 0x2060000000000000ULL, 0x40c0000000000000ULL
};

constexpr uint64_t square_type_mask[9] = {
    0x8100000000000081ULL, // corner
    0x0000182424180000ULL, // box edge
    0x0000240000240000ULL, // box corner
    0x2400810000810024ULL, // edge a
    0x1800008181000018ULL, // edge b
    0x0018004242001800ULL, // midedge center
    0x0024420000422400ULL, // midedge corner
    0x4281000000008142ULL, // edge c
    0x0042000000004200ULL  // x
};