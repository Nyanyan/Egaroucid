#pragma once
#include <chrono>
#include <random>
#ifdef _MSC_VER
    #include <intrin.h>
#else
    #include <x86intrin.h>
#endif
#include "setting.hpp"

using namespace std;

#define INF 100000000
#define N_PHASES 15
#define PHASE_N_STONES 4

#define HW 8
#define HW_M1 7
#define HW_P1 9
#define HW2 64
#define HW22 128
#define HW2_M1 63
#define HW2_MHW 56
#define HW2_P1 65
#define BLACK 0
#define WHITE 1
#define VACANT 2
#define N_8BIT 256

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
    return raw_myrandom();
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

inline int pop_count_uchar(unsigned char x){
    x = (x & 0b01010101) + ((x & 0b10101010) >> 1);
    x = (x & 0b00110011) + ((x & 0b11001100) >> 2);
    return (x & 0b00001111) + ((x & 11110000) >> 4);
}
/*
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
*/

#ifdef _MSC_VER
	#define	mirror_v(x)	_byteswap_uint64(x)
#else
	#define	mirror_v(x)	__builtin_bswap64(x)
#endif


inline unsigned long long white_line(unsigned long long x){
    unsigned long long a = (x ^ (x >> 7)) & 0x00AA00AA00AA00AAULL;
    x = x ^ a ^ (a << 7);
    a = (x ^ (x >> 14)) & 0x0000CCCC0000CCCCULL;
    x = x ^ a ^ (a << 14);
    a = (x ^ (x >> 28)) & 0x00000000F0F0F0F0ULL;
    return x = x ^ a ^ (a << 28);
}

inline unsigned long long black_line(unsigned long long x){
    unsigned long long a = (x ^ (x >> 9)) & 0x0055005500550055ULL;
    x = x ^ a ^ (a << 9);
    a = (x ^ (x >> 18)) & 0x0000333300003333ULL;
    x = x ^ a ^ (a << 18);
    a = (x ^ (x >> 36)) & 0x000000000F0F0F0FULL;
    return x = x ^ a ^ (a << 36);
}

inline uint64_t horizontal_mirror(uint64_t x){
    x = ((x >> 1) & 0x5555555555555555ULL) | ((x << 1) & 0xAAAAAAAAAAAAAAAAULL);
    x = ((x >> 2) & 0x3333333333333333ULL) | ((x << 2) & 0xCCCCCCCCCCCCCCCCULL);
    return ((x >> 4) & 0x0F0F0F0F0F0F0F0FULL) | ((x << 4) & 0xF0F0F0F0F0F0F0F0ULL);
}

inline uint64_t rotate_180(uint64_t x){
    x = ((x & 0x5555555555555555ULL) << 1) | ((x & 0xAAAAAAAAAAAAAAAAULL) >> 1);
    x = ((x & 0x3333333333333333ULL) << 2) | ((x & 0xCCCCCCCCCCCCCCCCULL) >> 2);
    x = ((x & 0x0F0F0F0F0F0F0F0FULL) << 4) | ((x & 0xF0F0F0F0F0F0F0F0ULL) >> 4);
    x = ((x & 0x00FF00FF00FF00FFULL) << 8) | ((x & 0xFF00FF00FF00FF00ULL) >> 8);
    x = ((x & 0x0000FFFF0000FFFFULL) << 16) | ((x & 0xFFFF0000FFFF0000ULL) >> 16);
    return ((x & 0x00000000FFFFFFFFULL) << 32) | ((x & 0xFFFFFFFF00000000ULL) >> 32);
}

inline int pop_digit(unsigned long long x, int place){
    return 1 & (x >> place);
}