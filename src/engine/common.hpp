/*
    Egaroucid Project

    @file common.hpp
        Common things
    @date 2021-2023
    @author Takuto Yamana
    @license GPL-3.0 license
*/

#pragma once
#include <iostream>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <time.h>
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

/*
    @brief bits radiating the cell are set
*/
constexpr uint64_t bit_radiation[HW2] = {
    0x81412111090503FEULL, 0x02824222120A07FDULL, 0x0404844424150EFBULL, 0x08080888492A1CF7ULL, 0x10101011925438EFULL, 0x2020212224A870DFULL, 0x404142444850E0BFULL, 0x8182848890A0C07FULL, 
    0x412111090503FE03ULL, 0x824222120A07FD07ULL, 0x04844424150EFB0EULL, 0x080888492A1CF71CULL, 0x101011925438EF38ULL, 0x20212224A870DF70ULL, 0x4142444850E0BFE0ULL, 0x82848890A0C07FC0ULL, 
    0x2111090503FE0305ULL, 0x4222120A07FD070AULL, 0x844424150EFB0E15ULL, 0x0888492A1CF71C2AULL, 0x1011925438EF3854ULL, 0x212224A870DF70A8ULL, 0x42444850E0BFE050ULL, 0x848890A0C07FC0A0ULL,
    0x11090503FE030509ULL, 0x22120A07FD070A12ULL, 0x4424150EFB0E1524ULL, 0x88492A1CF71C2A49ULL, 0x11925438EF385492ULL, 0x2224A870DF70A824ULL, 0x444850E0BFE05048ULL, 0x8890A0C07FC0A090ULL,
    0x090503FE03050911ULL, 0x120A07FD070A1222ULL, 0x24150EFB0E152444ULL, 0x492A1CF71C2A4988ULL, 0x925438EF38549211ULL, 0x24A870DF70A82422ULL, 0x4850E0BFE0504844ULL, 0x90A0C07FC0A09088ULL,
    0x0503FE0305091121ULL, 0x0A07FD070A122242ULL, 0x150EFB0E15244484ULL, 0x2A1CF71C2A498808ULL, 0x5438EF3854921110ULL, 0xA870DF70A8242221ULL, 0x50E0BFE050484442ULL, 0xA0C07FC0A0908884ULL,
    0x03FE030509112141ULL, 0x07FD070A12224282ULL, 0x0EFB0E1524448404ULL, 0x1CF71C2A49880808ULL, 0x38EF385492111010ULL, 0x70DF70A824222120ULL, 0xE0BFE05048444241ULL, 0xC07FC0A090888482ULL,
    0xFE03050911214181ULL, 0xFD070A1222428202ULL, 0xFB0E152444840404ULL, 0xF71C2A4988080808ULL, 0xEF38549211101010ULL, 0xDF70A82422212020ULL, 0xBFE0504844424140ULL, 0x7FC0A09088848281ULL
};

#if _WIN64 || _WIN32
    int get_localtime(tm* a, time_t* b) {
        return localtime_s(a, b);
    }
#else
    int get_localtime(tm* a, time_t* b) {
        a = localtime(b);
        return 0;
    }
#endif

std::string calc_date() {
    time_t now;
    tm newtime;
    time(&now);
    get_localtime(&newtime, &now);
    std::stringstream sout;
    std::string year = std::to_string(newtime.tm_year + 1900);
    sout << std::setfill('0') << std::setw(2) << newtime.tm_mon + 1;
    std::string month = sout.str();
    sout.str("");
    sout.clear(std::stringstream::goodbit);
    sout << std::setfill('0') << std::setw(2) << newtime.tm_mday;
    std::string day = sout.str();
    sout.str("");
    sout.clear(std::stringstream::goodbit);
    sout << std::setfill('0') << std::setw(2) << newtime.tm_hour;
    std::string hour = sout.str();
    sout.str("");
    sout.clear(std::stringstream::goodbit);
    sout << std::setfill('0') << std::setw(2) << newtime.tm_min;
    std::string minute = sout.str();
    sout.str("");
    sout.clear(std::stringstream::goodbit);
    sout << std::setfill('0') << std::setw(2) << newtime.tm_sec;
    std::string second = sout.str();
    return year + "_" + month + "_" + day + "_" + hour + "_" + minute + "_" + second;
}

void calc_date(int *year, int *month, int *day, int *hour, int *minute, int *second) {
    time_t now;
    tm newtime;
    time(&now);
    get_localtime(&newtime, &now);
    *year = newtime.tm_year + 1900;
    *month = newtime.tm_mon + 1;
    *day = newtime.tm_mday;
    *hour = newtime.tm_hour;
    *minute = newtime.tm_min;
    *second = newtime.tm_sec;
}