#pragma once
#include <iostream>
#ifdef _MSC_VER
    #include <intrin.h>
#else
    #include <x86intrin.h>
#endif
#include "common.hpp"

using namespace std;

inline void bit_print_reverse(uint64_t x){
    for (uint32_t i = 0; i < HW2; ++i)
        cerr << (1 & (x >> i));
    cerr << endl;
}

inline void bit_print(uint64_t x){
    for (uint32_t i = 0; i < HW2; ++i)
        cerr << (1 & (x >> (HW2_M1 - i)));
    cerr << endl;
}

inline void bit_print_uchar(uint8_t x){
    for (uint32_t i = 0; i < HW; ++i)
        cerr << (1 & (x >> (HW_M1 - i)));
    cerr << endl;
}

inline void bit_print_board_reverse(uint64_t x){
    for (uint32_t i = 0; i < HW2; ++i){
        cerr << (1 & (x >> i));
        if (i % HW == HW_M1)
            cerr << endl;
    }
    cerr << endl;
}

inline void bit_print_board(uint64_t x){
    for (uint32_t i = 0; i < HW2; ++i){
        cerr << (1 & (x >> (HW2_M1 - i)));
        if (i % HW == HW_M1)
            cerr << endl;
    }
    cerr << endl;
}

void input_board(uint64_t *p, uint64_t *o){
    char elem;
    *p = 0ULL;
    *o = 0ULL;
    for (int i = 0; i < HW2; ++i){
        cin >> elem;
        if (elem == '0')
            *p |= 1ULL << (HW2_M1 - i);
        else if (elem == '1')
            *o |= 1ULL << (HW2_M1 - i);
    }
}

void print_board(uint64_t p, uint64_t o){
    for (int i = 0; i < HW2; ++i){
        if (1 & (p >> (HW2_M1 - i)))
            cerr << '0';
        else if (1 & (o >> (HW2_M1 - i)))
            cerr << '1';
        else
            cerr << '.';
        if (i % HW == HW_M1)
            cerr << endl;
    }
}

/*
Original code: https://github.com/primenumber/issen/blob/72f450256878094ffe90b75f8674599e6869c238/src/move_generator.cpp
modified by Nyanyan
*/
struct u64_4 {
    __m256i data;
    u64_4() = default;
    u64_4(uint64_t val)
        : data(_mm256_set1_epi64x(val)) {}
    u64_4(uint64_t w, uint64_t x, uint64_t y, uint64_t z)
        : data(_mm256_set_epi64x(w, x, y, z)) {}
    //u64_4(u64_2 x, u64_2 y)
    //    : data(_mm256_setr_epi64x(_mm_cvtsi128_si64(y.data), _mm_cvtsi128_si64(_mm_unpackhi_epi64(y.data, y.data)), _mm_cvtsi128_si64(x.data), _mm_cvtsi128_si64(_mm_unpackhi_epi64(x.data, x.data)))) {}
    u64_4(__m256i data) : data(data) {}
    operator __m256i() { return data; }
};

inline u64_4 operator>>(const u64_4 lhs, const size_t n) {
    return _mm256_srli_epi64(lhs.data, n);
}

inline u64_4 operator>>(const u64_4 lhs, const u64_4 n) {
    return _mm256_srlv_epi64(lhs.data, n.data);
}

inline u64_4 operator<<(const u64_4 lhs, const size_t n) {
    return _mm256_slli_epi64(lhs.data, n);
}

inline u64_4 operator<<(const u64_4 lhs, const u64_4 n) {
    return _mm256_sllv_epi64(lhs.data, n.data);
}

inline u64_4 operator&(const u64_4 lhs, const u64_4 rhs) {
    return _mm256_and_si256(lhs.data, rhs.data);
}

inline u64_4 operator|(const u64_4 lhs, const u64_4 rhs) {
    return _mm256_or_si256(lhs.data, rhs.data);
}

inline u64_4 operator^(const u64_4 lhs, const u64_4 rhs) {
    return _mm256_xor_si256(lhs.data, rhs.data);
}

inline u64_4 operator+(const u64_4 lhs, const u64_4 rhs) {
    return _mm256_add_epi64(lhs.data, rhs.data);
}

inline u64_4 operator+(const u64_4 lhs, const uint64_t rhs) {
    __m256i r64 = _mm256_set1_epi64x(rhs);
    return _mm256_add_epi64(lhs.data, r64);
}

inline u64_4 operator-(const u64_4 lhs, const u64_4 rhs) {
    return _mm256_sub_epi64(lhs.data, rhs.data);
}

inline u64_4 operator-(const u64_4 lhs) {
    return _mm256_sub_epi64(_mm256_setzero_si256(), lhs.data);
}

inline u64_4 andnot(const u64_4 lhs, const u64_4 rhs) {
    return _mm256_andnot_si256(lhs.data, rhs.data);
}

inline u64_4 operator~(const u64_4 lhs) {
    return _mm256_andnot_si256(lhs.data, _mm256_set1_epi8(0xFF));
}

inline u64_4 nonzero(const u64_4 lhs) {
    return _mm256_cmpeq_epi64(lhs.data, _mm256_setzero_si256()) + u64_4(1);
}

inline uint64_t all_or(const u64_4 lhs) {
    __m128i lhs_xz_yw = _mm_or_si128(_mm256_castsi256_si128(lhs.data), _mm256_extractf128_si256(lhs.data, 1));
    return _mm_extract_epi64(lhs_xz_yw, 0) | _mm_extract_epi64(lhs_xz_yw, 1);
}

inline uint64_t all_and(const u64_4 lhs) {
    __m128i lhs_xz_yw = _mm_and_si128(_mm256_castsi256_si128(lhs.data), _mm256_extractf128_si256(lhs.data, 1));
    return _mm_extract_epi64(lhs_xz_yw, 0) & _mm_extract_epi64(lhs_xz_yw, 1);
}

/*
end of modification
*/



#if USE_BUILTIN_POPCOUNT
    #define	pop_count_ull(x) (int)__popcnt64(x)
    #define pop_count_uint(x) (int)__popcnt(x)
    #define pop_count_uchar(x) (int)__popcnt(x)
#else

    inline int pop_count_ull(uint64_t x){
        x = x - ((x >> 1) & 0x5555555555555555ULL);
        x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL);
        x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0FULL;
        x = (x * 0x0101010101010101ULL) >> 56;
        return x;
    }

    inline int pop_count_uint(uint32_t x){
        x = (x & 0x55555555) + ((x & 0xAAAAAAAA) >> 1);
        x = (x & 0x33333333) + ((x & 0xCCCCCCCC) >> 2);
        return (x & 0x0F0F0F0F) + ((x & 0xF0F0F0F0) >> 4);
    }

    inline int pop_count_uchar(uint8_t x){
        x = (x & 0b01010101) + ((x & 0b10101010) >> 1);
        x = (x & 0b00110011) + ((x & 0b11001100) >> 2);
        return (x & 0b00001111) + ((x & 0b11110000) >> 4);
    }

#endif

inline uint32_t pop_digit(uint64_t x, int place){
    return (uint32_t)(1ULL & (x >> place));
}

inline uint64_t white_line_mirror(uint64_t x){
    uint64_t a = (x ^ (x >> 7)) & 0x00AA00AA00AA00AAULL;
    x = x ^ a ^ (a << 7);
    a = (x ^ (x >> 14)) & 0x0000CCCC0000CCCCULL;
    x = x ^ a ^ (a << 14);
    a = (x ^ (x >> 28)) & 0x00000000F0F0F0F0ULL;
    return x = x ^ a ^ (a << 28);
}

inline uint64_t black_line_mirror(uint64_t x){
    uint64_t a = (x ^ (x >> 9)) & 0x0055005500550055ULL;
    x = x ^ a ^ (a << 9);
    a = (x ^ (x >> 18)) & 0x0000333300003333ULL;
    x = x ^ a ^ (a << 18);
    a = (x ^ (x >> 36)) & 0x000000000F0F0F0FULL;
    return x = x ^ a ^ (a << 36);
}

inline u64_4 black_line_mirror(u64_4 x){
    u64_4 a = (x ^ (x >> 9)) & 0x0055005500550055ULL;
    x = x ^ a ^ (a << 9);
    a = (x ^ (x >> 18)) & 0x0000333300003333ULL;
    x = x ^ a ^ (a << 18);
    a = (x ^ (x >> 36)) & 0x000000000F0F0F0FULL;
    return x ^ a ^ (a << 36);
}

inline u64_4 black_line_mirror_3_4(u64_4 x){
    u64_4 mask(0ULL, 0ULL, 0x0055005500550055ULL, 0x0055005500550055ULL);
    u64_4 a = (x ^ (x >> 9)) & mask;
    x = x ^ a ^ (a << 9);
    mask = {0ULL, 0ULL, 0x0000333300003333ULL, 0x0000333300003333ULL};
    a = (x ^ (x >> 18)) & mask;
    x = x ^ a ^ (a << 18);
    mask = {0ULL, 0ULL, 0x000000000F0F0F0FULL, 0x000000000F0F0F0FULL};
    a = (x ^ (x >> 36)) & mask;
    return x ^ a ^ (a << 36);
}

/*
inline void black_line_mirror_double(uint64_t xin, uint64_t yin, uint64_t *x, uint64_t *y){
    __m128i	xy = _mm_set_epi64x(xin, yin);
    __m128i a = _mm_xor_si128(xy, _mm_srli_epi64(xy, 9));
    a = _mm_and_si128(a, _mm_set1_epi64x(0x0055005500550055ULL));
    xy = _mm_xor_si128(_mm_xor_si128(xy, a),  _mm_slli_epi64(a, 9));

    a = _mm_xor_si128(xy, _mm_srli_epi64(xy, 18));
    a = _mm_and_si128(a, _mm_set1_epi64x(0x0000333300003333ULL));
    xy = _mm_xor_si128(_mm_xor_si128(xy, a),  _mm_slli_epi64(a, 18));
    
    a = _mm_xor_si128(xy, _mm_srli_epi64(xy, 36));
    a = _mm_and_si128(a, _mm_set1_epi64x(0x000000000F0F0F0FULL));
    xy = _mm_xor_si128(_mm_xor_si128(xy, a),  _mm_slli_epi64(a, 36));

    *y = _mm_cvtsi128_si64(xy);
    *x = _mm_cvtsi128_si64(_mm_unpackhi_epi64(xy, xy));
}
*/
#if USE_FAST_VERTICAL_MIRROR
    #ifdef _MSC_VER
        #define	vertical_mirror(x)	_byteswap_uint64(x)
    #else
        #define	vertical_mirror(x)	__builtin_bswap64(x)
    #endif
#else
    inline uint64_t vertical_mirror(uint64_t x){
        x = ((x >> 8) & 0x00FF00FF00FF00FFULL) | ((x << 8) & 0xFF00FF00FF00FF00ULL);
        x = ((x >> 16) & 0x0000FFFF0000FFFFULL) | ((x << 16) & 0xFFFF0000FFFF0000ULL);
        return ((x >> 32) & 0x00000000FFFFFFFFULL) | ((x << 32) & 0xFFFFFFFF00000000ULL);
    }
#endif

inline uint64_t horizontal_mirror(uint64_t x){
    x = ((x >> 1) & 0x5555555555555555ULL) | ((x << 1) & 0xAAAAAAAAAAAAAAAAULL);
    x = ((x >> 2) & 0x3333333333333333ULL) | ((x << 2) & 0xCCCCCCCCCCCCCCCCULL);
    return ((x >> 4) & 0x0F0F0F0F0F0F0F0FULL) | ((x << 4) & 0xF0F0F0F0F0F0F0F0ULL);
}

inline u64_4 horizontal_mirror(u64_4 x){
    x = ((x >> 1) & 0x5555555555555555ULL) | ((x << 1) & 0xAAAAAAAAAAAAAAAAULL);
    x = ((x >> 2) & 0x3333333333333333ULL) | ((x << 2) & 0xCCCCCCCCCCCCCCCCULL);
    return ((x >> 4) & 0x0F0F0F0F0F0F0F0FULL) | ((x << 4) & 0xF0F0F0F0F0F0F0F0ULL);
}

inline u64_4 horizontal_mirror_1_3(u64_4 x){
    u64_4 mask(0ULL, 0x5555555555555555ULL, 0ULL, 0x5555555555555555ULL);
    u64_4 a = (x ^ (x >> 1)) & mask;
    x = x ^ a ^ (a << 1);
    mask = {0ULL, 0x3333333333333333ULL, 0ULL, 0x3333333333333333ULL};
    a = (x ^ (x >> 2)) & mask;
    x = x ^ a ^ (a << 2);
    mask = {0ULL, 0x0F0F0F0F0F0F0F0FULL, 0ULL, 0x0F0F0F0F0F0F0F0FULL};
    a = (x ^ (x >> 4)) & mask;
    return x ^ a ^ (a << 4);
}

/*
inline void horizontal_mirror_double(uint64_t *x, uint64_t *y){
    __m128i	xy = _mm_set_epi64x(*x, *y);
    xy = _mm_or_si128(
        _mm_and_si128(_mm_srli_epi64(xy, 1), _mm_set1_epi64x(0x5555555555555555ULL)),
        _mm_and_si128(_mm_slli_epi64(xy, 1), _mm_set1_epi64x(0xAAAAAAAAAAAAAAAAULL)));
    xy = _mm_or_si128(
        _mm_and_si128(_mm_srli_epi64(xy, 2), _mm_set1_epi64x(0x3333333333333333ULL)), 
        _mm_and_si128(_mm_slli_epi64(xy, 2), _mm_set1_epi64x(0xCCCCCCCCCCCCCCCCULL)));
    xy = _mm_or_si128(
        _mm_and_si128(_mm_srli_epi64(xy, 4), _mm_set1_epi64x(0x0F0F0F0F0F0F0F0FULL)), 
        _mm_and_si128(_mm_slli_epi64(xy, 4), _mm_set1_epi64x(0xF0F0F0F0F0F0F0F0ULL)));
    *y = _mm_cvtsi128_si64(xy);
    *x = _mm_cvtsi128_si64(_mm_unpackhi_epi64(xy, xy));
}

inline void horizontal_mirror_double(uint64_t xin, uint64_t yin, uint64_t *x, uint64_t *y){
    __m128i	xy = _mm_set_epi64x(xin, yin);
    xy = _mm_or_si128(
        _mm_and_si128(_mm_srli_epi64(xy, 1), _mm_set1_epi64x(0x5555555555555555ULL)),
        _mm_and_si128(_mm_slli_epi64(xy, 1), _mm_set1_epi64x(0xAAAAAAAAAAAAAAAAULL)));
    xy = _mm_or_si128(
        _mm_and_si128(_mm_srli_epi64(xy, 2), _mm_set1_epi64x(0x3333333333333333ULL)), 
        _mm_and_si128(_mm_slli_epi64(xy, 2), _mm_set1_epi64x(0xCCCCCCCCCCCCCCCCULL)));
    xy = _mm_or_si128(
        _mm_and_si128(_mm_srli_epi64(xy, 4), _mm_set1_epi64x(0x0F0F0F0F0F0F0F0FULL)), 
        _mm_and_si128(_mm_slli_epi64(xy, 4), _mm_set1_epi64x(0xF0F0F0F0F0F0F0F0ULL)));
    *y = _mm_cvtsi128_si64(xy);
    *x = _mm_cvtsi128_si64(_mm_unpackhi_epi64(xy, xy));
}

inline void horizontal_mirror_quad(uint64_t win, uint64_t xin, uint64_t yin, uint64_t zin, uint64_t *w, uint64_t *x, uint64_t *y, uint64_t *z){
    __m256i	xy = _mm256_set_epi64x(win, xin, yin, zin);
    xy = _mm256_or_si256(
        _mm256_and_si256(_mm256_srli_epi64(xy, 1), _mm256_set1_epi64x(0x5555555555555555ULL)), 
        _mm256_and_si256(_mm256_slli_epi64(xy, 1), _mm256_set1_epi64x(0xAAAAAAAAAAAAAAAAULL)));
    xy = _mm256_or_si256(
        _mm256_and_si256(_mm256_srli_epi64(xy, 2), _mm256_set1_epi64x(0x3333333333333333ULL)), 
        _mm256_and_si256(_mm256_slli_epi64(xy, 2), _mm256_set1_epi64x(0xCCCCCCCCCCCCCCCCULL)));
    xy = _mm256_or_si256(
        _mm256_and_si256(_mm256_srli_epi64(xy, 4), _mm256_set1_epi64x(0x0F0F0F0F0F0F0F0FULL)),
        _mm256_and_si256(_mm256_slli_epi64(xy, 4), _mm256_set1_epi64x(0xF0F0F0F0F0F0F0F0ULL)));
    *w = _mm256_extract_epi64(xy, 3);
    *x = _mm256_extract_epi64(xy, 2);
    *y = _mm256_extract_epi64(xy, 1);
    *z = _mm256_extract_epi64(xy, 0);
}
*/
// direction is couner clockwise
inline uint64_t rotate_90(uint64_t x){
    return vertical_mirror(white_line_mirror(x));
}

// direction is couner clockwise
inline uint64_t rotate_270(uint64_t x){
    return vertical_mirror(black_line_mirror(x));
}

inline uint8_t rotate_180_uchar(uint8_t x){
    x = ((x & 0x55U) << 1) | ((x & 0xAAU) >> 1);
    x = ((x & 0x33U) << 2) | ((x & 0xCCU) >> 2);
    return ((x & 0x0FU) << 4) | ((x & 0xF0U) >> 4);
}

inline uint64_t rotate_180(uint64_t x){
    x = ((x & 0x5555555555555555ULL) << 1) | ((x & 0xAAAAAAAAAAAAAAAAULL) >> 1);
    x = ((x & 0x3333333333333333ULL) << 2) | ((x & 0xCCCCCCCCCCCCCCCCULL) >> 2);
    x = ((x & 0x0F0F0F0F0F0F0F0FULL) << 4) | ((x & 0xF0F0F0F0F0F0F0F0ULL) >> 4);
    x = ((x & 0x00FF00FF00FF00FFULL) << 8) | ((x & 0xFF00FF00FF00FF00ULL) >> 8);
    x = ((x & 0x0000FFFF0000FFFFULL) << 16) | ((x & 0xFFFF0000FFFF0000ULL) >> 16);
    return ((x & 0x00000000FFFFFFFFULL) << 32) | ((x & 0xFFFFFFFF00000000ULL) >> 32);
}

// rotate 45 degrees counter clockwise
/*
      8  7  6  5  4  3  2  1
      9  8  7  6  5  4  3  2
     10  9  8  7  6  5  4  3
     11 10  9  8  7  6  5  4
     12 11 10  9  8  7  6  5
     13 12 11 10  9  8  7  6
     14 13 12 11 10  9  8  7
     15 14 13 12 11 10  9  8
    
    to

     14 14  6  6  6  6  6  6
     13 13 13  5  5  5  5  5
     12 12 12 12  4  4  4  4
     11 11 11 11 11  3  3  3
     10 10 10 10 10 10  2  2
      9  9  9  9  9  9  9  1
      8  8  8  8  8  8  8  8
     15  7  7  7  7  7  7  7
*/
inline uint64_t rotate_45(uint64_t x){
    uint64_t a = (x ^ (x >> 8)) & 0x0055005500550055ULL;
    x = x ^ a ^ (a << 8);
    a = (x ^ (x >> 16)) & 0x0000CC660000CC66ULL;
    x = x ^ a ^ (a << 16);
    a = (x ^ (x >> 32)) & 0x00000000C3E1F078ULL;
    return x ^ a ^ (a << 32);
}
/*
inline void rotate_45_double(uint64_t *x, uint64_t *y){
    __m128i xy = _mm_set_epi64x(*x, *y);
    __m128i a = _mm_and_si128(_mm_xor_si128(xy, _mm_srli_epi64(xy, 8)), _mm_set1_epi64x(0x0055005500550055ULL));
    xy = _mm_xor_si128(_mm_xor_si128(xy, a), _mm_slli_epi64(a, 8));
    a = _mm_and_si128(_mm_xor_si128(xy, _mm_srli_epi64(xy, 16)), _mm_set1_epi64x(0x0000CC660000CC66ULL));
    xy = _mm_xor_si128(_mm_xor_si128(xy, a), _mm_slli_epi64(a, 16));
    a = _mm_and_si128(_mm_xor_si128(xy, _mm_srli_epi64(xy, 32)), _mm_set1_epi64x(0x00000000C3E1F078ULL));
    xy = _mm_xor_si128(_mm_xor_si128(xy, a), _mm_slli_epi64(a, 32));
    *y = _mm_cvtsi128_si64(xy);
    *x = _mm_cvtsi128_si64(_mm_unpackhi_epi64(xy, xy));
}

inline void rotate_45_double(uint64_t x_in, uint64_t y_in, uint64_t *x, uint64_t *y){
    __m128i xy = _mm_set_epi64x(x_in, y_in);
    __m128i a = _mm_and_si128(_mm_xor_si128(xy, _mm_srli_epi64(xy, 8)), _mm_set1_epi64x(0x0055005500550055ULL));
    xy = _mm_xor_si128(_mm_xor_si128(xy, a), _mm_slli_epi64(a, 8));
    a = _mm_and_si128(_mm_xor_si128(xy, _mm_srli_epi64(xy, 16)), _mm_set1_epi64x(0x0000CC660000CC66ULL));
    xy = _mm_xor_si128(_mm_xor_si128(xy, a), _mm_slli_epi64(a, 16));
    a = _mm_and_si128(_mm_xor_si128(xy, _mm_srli_epi64(xy, 32)), _mm_set1_epi64x(0x00000000C3E1F078ULL));
    xy = _mm_xor_si128(_mm_xor_si128(xy, a), _mm_slli_epi64(a, 32));
    *y = _mm_cvtsi128_si64(xy);
    *x = _mm_cvtsi128_si64(_mm_unpackhi_epi64(xy, xy));
}
*/
// unrotate 45 degrees counter clockwise
inline uint64_t unrotate_45(uint64_t x){
    uint64_t a = (x ^ (x >> 32)) & 0x00000000C3E1F078ULL;
    x = x ^ a ^ (a << 32);
    a = (x ^ (x >> 16)) & 0x0000CC660000CC66ULL;
    x = x ^ a ^ (a << 16);
    a = (x ^ (x >> 8)) & 0x0055005500550055ULL;
    return x ^ a ^ (a << 8);
}

inline uint64_t rotate_225(uint64_t x){
    return rotate_45(rotate_180(x));
}

inline uint64_t unrotate_225(uint64_t x){
    return rotate_180(unrotate_45(x));
}

inline uint64_t rotate_135(uint64_t x){
    uint64_t a = (x ^ (x >> 8)) & 0x00AA00AA00AA00AAULL;
    x = x ^ a ^ (a << 8);
    a = (x ^ (x >> 16)) & 0x0000336600003366ULL;
    x = x ^ a ^ (a << 16);
    a = (x ^ (x >> 32)) & 0x00000000C3870F1EULL;
    return x ^ a ^ (a << 32);
}
/*
inline void rotate_135_double(uint64_t *x, uint64_t *y){
    __m128i xy = _mm_set_epi64x(*x, *y);
    __m128i a = _mm_and_si128(_mm_xor_si128(xy, _mm_srli_epi64(xy, 8)), _mm_set1_epi64x(0x00AA00AA00AA00AAULL));
    xy = _mm_xor_si128(_mm_xor_si128(xy, a), _mm_slli_epi64(a, 8));
    a = _mm_and_si128(_mm_xor_si128(xy, _mm_srli_epi64(xy, 16)), _mm_set1_epi64x(0x0000336600003366ULL));
    xy = _mm_xor_si128(_mm_xor_si128(xy, a), _mm_slli_epi64(a, 16));
    a = _mm_and_si128(_mm_xor_si128(xy, _mm_srli_epi64(xy, 32)), _mm_set1_epi64x(0x00000000C3870F1EULL));
    xy = _mm_xor_si128(_mm_xor_si128(xy, a), _mm_slli_epi64(a, 32));
    *y = _mm_cvtsi128_si64(xy);
    *x = _mm_cvtsi128_si64(_mm_unpackhi_epi64(xy, xy));
}

inline void rotate_135_double(uint64_t x_in, uint64_t y_in, uint64_t *x, uint64_t *y){
    __m128i xy = _mm_set_epi64x(x_in, y_in);
    __m128i a = _mm_and_si128(_mm_xor_si128(xy, _mm_srli_epi64(xy, 8)), _mm_set1_epi64x(0x00AA00AA00AA00AAULL));
    xy = _mm_xor_si128(_mm_xor_si128(xy, a), _mm_slli_epi64(a, 8));
    a = _mm_and_si128(_mm_xor_si128(xy, _mm_srli_epi64(xy, 16)), _mm_set1_epi64x(0x0000336600003366ULL));
    xy = _mm_xor_si128(_mm_xor_si128(xy, a), _mm_slli_epi64(a, 16));
    a = _mm_and_si128(_mm_xor_si128(xy, _mm_srli_epi64(xy, 32)), _mm_set1_epi64x(0x00000000C3870F1EULL));
    xy = _mm_xor_si128(_mm_xor_si128(xy, a), _mm_slli_epi64(a, 32));
    *y = _mm_cvtsi128_si64(xy);
    *x = _mm_cvtsi128_si64(_mm_unpackhi_epi64(xy, xy));
}
*/
inline uint64_t unrotate_135(uint64_t x){
    uint64_t a = (x ^ (x >> 32)) & 0x00000000C3870F1EULL;
    x = x ^ a ^ (a << 32);
    a = (x ^ (x >> 16)) & 0x0000336600003366ULL;
    x = x ^ a ^ (a << 16);
    a = (x ^ (x >> 8)) & 0x00AA00AA00AA00AAULL;
    return x ^ a ^ (a << 8);
}
/*
inline uint64_t unrotate_45_135(uint64_t x, uint64_t y){
    __m128i xy = _mm_set_epi64x(x, y);
    __m128i a = _mm_and_si128(_mm_xor_si128(xy, _mm_srli_epi64(xy, 32)), _mm_set_epi64x(0x00000000C3E1F078ULL, 0x00000000C3870F1EULL));
    xy = _mm_xor_si128(_mm_xor_si128(xy, a), _mm_slli_epi64(a, 32));
    a = _mm_and_si128(_mm_xor_si128(xy, _mm_srli_epi64(xy, 16)), _mm_set_epi64x(0x0000CC660000CC66ULL, 0x0000336600003366ULL));
    xy = _mm_xor_si128(_mm_xor_si128(xy, a), _mm_slli_epi64(a, 16));
    a = _mm_and_si128(_mm_xor_si128(xy, _mm_srli_epi64(xy, 8)), _mm_set_epi64x(0x0055005500550055ULL, 0x00AA00AA00AA00AAULL));
    xy = _mm_xor_si128(_mm_xor_si128(xy, a), _mm_slli_epi64(a, 8));
    return _mm_cvtsi128_si64(xy) | _mm_cvtsi128_si64(_mm_unpackhi_epi64(xy, xy));
}

inline void rotate_45_double_135_double(uint64_t x_in, uint64_t y_in, uint64_t *w, uint64_t *x, uint64_t *y, uint64_t *z){
    __m256i xy = _mm256_set_epi64x(x_in, y_in, x_in, y_in);
    __m256i a = _mm256_and_si256(_mm256_xor_si256(xy, _mm256_srli_epi64(xy, 8)), _mm256_set_epi64x(0x0055005500550055ULL, 0x0055005500550055ULL, 0x00AA00AA00AA00AAULL, 0x00AA00AA00AA00AAULL));
    xy = _mm256_xor_si256(_mm256_xor_si256(xy, a), _mm256_slli_epi64(a, 8));
    a = _mm256_and_si256(_mm256_xor_si256(xy, _mm256_srli_epi64(xy, 16)), _mm256_set_epi64x(0x0000CC660000CC66ULL, 0x0000CC660000CC66ULL, 0x0000336600003366ULL, 0x0000336600003366ULL));
    xy = _mm256_xor_si256(_mm256_xor_si256(xy, a), _mm256_slli_epi64(a, 16));
    a = _mm256_and_si256(_mm256_xor_si256(xy, _mm256_srli_epi64(xy, 32)), _mm256_set_epi64x(0x00000000C3E1F078ULL, 0x00000000C3E1F078ULL, 0x00000000C3870F1EULL, 0x00000000C3870F1EULL));
    xy = _mm256_xor_si256(_mm256_xor_si256(xy, a), _mm256_slli_epi64(a, 32));
    *w = _mm256_extract_epi64(xy, 3);
    *x = _mm256_extract_epi64(xy, 2);
    *y = _mm256_extract_epi64(xy, 1);
    *z = _mm256_extract_epi64(xy, 0);
}
*/

inline u64_4 rotate_45_45_135_135(u64_4 x){
    u64_4 mask(0x0055005500550055ULL, 0x0055005500550055ULL, 0x00AA00AA00AA00AAULL, 0x00AA00AA00AA00AAULL);
    u64_4 a = (x ^ (x >> 8)) & mask;
    x = x ^ a ^ (a << 8);
    mask = {0x0000CC660000CC66ULL, 0x0000CC660000CC66ULL, 0x0000336600003366ULL, 0x0000336600003366ULL};
    a = (x ^ (x >> 16)) & mask;
    x = x ^ a ^ (a << 16);
    mask = {0x00000000C3E1F078ULL, 0x00000000C3E1F078ULL, 0x00000000C3870F1EULL, 0x00000000C3870F1EULL};
    a = (x ^ (x >> 32)) & mask;
    return x ^ a ^ (a << 32);
}

inline u64_4 unrotate_45_45_135_135(u64_4 x){
    u64_4 mask(0x00000000C3E1F078ULL, 0x00000000C3E1F078ULL, 0x00000000C3870F1EULL, 0x00000000C3870F1EULL);
    u64_4 a = (x ^ (x >> 32)) & mask;
    x = x ^ a ^ (a << 32);
    mask = {0x0000CC660000CC66ULL, 0x0000CC660000CC66ULL, 0x0000336600003366ULL, 0x0000336600003366ULL};
    a = (x ^ (x >> 16)) & mask;
    x = x ^ a ^ (a << 16);
    mask = {0x0055005500550055ULL, 0x0055005500550055ULL, 0x00AA00AA00AA00AAULL, 0x00AA00AA00AA00AAULL};
    a = (x ^ (x >> 8)) & mask;
    return x ^ a ^ (a << 8);
}



inline uint64_t rotate_315(uint64_t x){
    return rotate_135(rotate_180(x));
}

inline uint64_t unrotate_315(uint64_t x){
    return rotate_180(unrotate_135(x));
}

#if USE_MINUS_NTZ
    inline uint_fast8_t ntz(uint64_t *x){
        return pop_count_ull((*x & (-(*x))) - 1);
    }
#else
    inline uint_fast8_t ntz(uint64_t *x){
        return pop_count_ull((*x & (~(*x) + 1)) - 1);
    }
#endif

inline u64_4 pop_count_ull_quad(u64_4 x){
    u64_4 mask1(0x5555555555555555ULL);
    u64_4 mask2(0xAAAAAAAAAAAAAAAAULL);
    x = (x & mask1) + ((x & mask2) >> 1);
    mask1 = {0x3333333333333333ULL, 0x3333333333333333ULL, 0x3333333333333333ULL, 0x3333333333333333ULL};
    mask2 = {0xCCCCCCCCCCCCCCCCULL, 0xCCCCCCCCCCCCCCCCULL, 0xCCCCCCCCCCCCCCCCULL, 0xCCCCCCCCCCCCCCCCULL};
    x = (x & mask1) + ((x & mask2) >> 2);
    mask1 = {0x0F0F0F0F0F0F0F0FULL, 0x0F0F0F0F0F0F0F0FULL, 0x0F0F0F0F0F0F0F0FULL, 0x0F0F0F0F0F0F0F0FULL};
    mask2 = {0xF0F0F0F0F0F0F0F0ULL, 0xF0F0F0F0F0F0F0F0ULL, 0xF0F0F0F0F0F0F0F0ULL, 0xF0F0F0F0F0F0F0F0ULL};
    x = (x & mask1) + ((x & mask2) >> 4);
    mask1 = {0x00FF00FF00FF00FFULL, 0x00FF00FF00FF00FFULL, 0x00FF00FF00FF00FFULL, 0x00FF00FF00FF00FFULL};
    mask2 = {0xFF00FF00FF00FF00ULL, 0xFF00FF00FF00FF00ULL, 0xFF00FF00FF00FF00ULL, 0xFF00FF00FF00FF00ULL};
    x = (x & mask1) + ((x & mask2) >> 8);
    mask1 = {0x0000FFFF0000FFFFULL, 0x0000FFFF0000FFFFULL, 0x0000FFFF0000FFFFULL, 0x0000FFFF0000FFFFULL};
    mask2 = {0xFFFF0000FFFF0000ULL, 0xFFFF0000FFFF0000ULL, 0xFFFF0000FFFF0000ULL, 0xFFFF0000FFFF0000ULL};
    x = (x & mask1) + ((x & mask2) >> 16);
    mask1 = {0x00000000FFFFFFFFULL, 0x00000000FFFFFFFFULL, 0x00000000FFFFFFFFULL, 0x00000000FFFFFFFFULL};
    mask2 = {0xFFFFFFFF00000000ULL, 0xFFFFFFFF00000000ULL, 0xFFFFFFFF00000000ULL, 0xFFFFFFFF00000000ULL};
    x = (x & mask1) + ((x & mask2) >> 32);
    return x;
}

/*
Original code: https://github.com/primenumber/issen/blob/72f450256878094ffe90b75f8674599e6869c238/src/move_generator.cpp
modified by Nyanyan
*/
inline u64_4 nlz_quad(u64_4 x){
    x = x | (x >> 1);
    x = x | (x >> 2);
    x = x | (x >> 4);
    x = x | (x >> 8);
    x = x | (x >> 16);
    return pop_count_ull_quad(~x);
}

__m256i flip_vertical_shuffle_table;

inline void upper_bit_init(){
    flip_vertical_shuffle_table = _mm256_set_epi8(
        24, 25, 26, 27, 28, 29, 30, 31,
        16, 17, 18, 19, 20, 21, 22, 23,
        8, 9, 10, 11, 12, 13, 14, 15,
        0, 1, 2, 3, 4, 5, 6, 7
    );
}

inline u64_4 upper_bit(u64_4 p) {
    p = p | (p >> 1);
    p = p | (p >> 2);
    p = p | (p >> 4);
    p = andnot(p >> 1, p);
    p.data = _mm256_shuffle_epi8(p.data, flip_vertical_shuffle_table);
    p = p & -p;
    return _mm256_shuffle_epi8(p.data, flip_vertical_shuffle_table);
}
/*
end of modification
*/


inline uint_fast8_t first_bit(uint64_t *x){
    return ntz(x);
}

inline uint_fast8_t next_bit(uint64_t *x){
    *x &= *x - 1;
    return ntz(x);
}

inline uint_fast8_t next_odd_bit(uint64_t *x, uint_fast8_t parity){
    *x &= *x - 1;
    uint_fast8_t res = ntz(x);
    while (*x && (parity & cell_div4[res]) == 0)
        res = next_odd_bit(x, parity);
    return res;
}

inline uint_fast8_t first_odd_bit(uint64_t *x, uint_fast8_t parity){
    uint_fast8_t res = ntz(x);
    while (*x && (parity & cell_div4[res]) == 0)
        res = next_odd_bit(x, parity);
    return res;
}

inline uint_fast8_t next_even_bit(uint64_t *x, uint_fast8_t parity){
    *x &= *x - 1;
    uint_fast8_t res = ntz(x);
    while (*x && (parity & cell_div4[res]))
        res = next_even_bit(x, parity);
    return res;
}

inline uint_fast8_t first_even_bit(uint64_t *x, uint_fast8_t parity){
    uint_fast8_t res = ntz(x);
    while (*x && (parity & cell_div4[res]))
        res = next_even_bit(x, parity);
    return res;
}

constexpr uint64_t bit_around[HW2] = {
    0x0000000000000302ULL, 0x0000000000000705ULL, 0x0000000000000E0AULL, 0x0000000000001C14ULL, 0x0000000000003828ULL, 0x0000000000007050ULL, 0x000000000000E0A0ULL, 0x000000000000C040ULL, 
    0x0000000000030203ULL, 0x0000000000070507ULL, 0x00000000000E0A0EULL, 0x00000000001C141CULL, 0x0000000000382838ULL, 0x0000000000705070ULL, 0x0000000000E0A0E0ULL, 0x0000000000C040C0ULL, 
    0x0000000003020300ULL, 0x0000000007050700ULL, 0x000000000E0A0E00ULL, 0x000000001C141C00ULL, 0x0000000038283800ULL, 0x0000000070507000ULL, 0x00000000E0A0E000ULL, 0x00000000C040C000ULL,
    0x0000000302030000ULL, 0x0000000705070000ULL, 0x0000000E0A0E0000ULL, 0x0000001C141C0000ULL, 0x0000003828380000ULL, 0x0000007050700000ULL, 0x000000E0A0E00000ULL, 0x000000C040C00000ULL,
    0x0000030203000000ULL, 0x0000070507000000ULL, 0x00000E0A0E000000ULL, 0x00001C141C000000ULL, 0x0000382838000000ULL, 0x0000705070000000ULL, 0x0000E0A0E0000000ULL, 0x0000C040C0000000ULL,
    0x0003020300000000ULL, 0x0007050700000000ULL, 0x000E0A0E00000000ULL, 0x001C141C00000000ULL, 0x0038283800000000ULL, 0x0070507000000000ULL, 0x00E0A0E000000000ULL, 0x00C040C000000000ULL,
    0x0302030000000000ULL, 0x0705070000000000ULL, 0x0E0A0E0000000000ULL, 0x1C141C0000000000ULL, 0x3828380000000000ULL, 0x7050700000000000ULL, 0xE0A0E00000000000ULL, 0xC040C00000000000ULL,
    0x0203000000000000ULL, 0x0507000000000000ULL, 0x0A0E000000000000ULL, 0x141C000000000000ULL, 0x2838000000000000ULL, 0x5070000000000000ULL, 0xA0E0000000000000ULL, 0x40C0000000000000ULL
};

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

constexpr uint64_t bit_radiation_dismiss_around[HW2] = {
    0x81412111090500FCULL, 0x02824222120A00F8ULL, 0x04048444241500F1ULL, 0x08080888492A00E3ULL, 0x10101011925400C7ULL, 0x2020212224A8008FULL, 0x404142444850001FULL, 0x8182848890A0003FULL, 
    0x412111090500FC00ULL, 0x824222120A00F800ULL, 0x048444241500F100ULL, 0x080888492A00E300ULL, 0x101011925400C700ULL, 0x20212224A8008F00ULL, 0x4142444850001F00ULL, 0x82848890A0003F00ULL, 
    0x2111090500FC0005ULL, 0x4222120A00F8000AULL, 0x8444241500F10015ULL, 0x0888492A00E3002AULL, 0x1011925400C70054ULL, 0x212224A8008F00A8ULL, 0x42444850001F0050ULL, 0x848890A0003F00A0ULL,
    0x11090500FC000509ULL, 0x22120A00F8000A12ULL, 0x44241500F1001524ULL, 0x88492A00E3002A49ULL, 0x11925400C7005492ULL, 0x2224A8008F00A824ULL, 0x444850001F005048ULL, 0x8890A0003F00A090ULL,
    0x090500FC00050911ULL, 0x120A00F8000A1222ULL, 0x241500F100152444ULL, 0x492A00E3002A4988ULL, 0x925400C700549211ULL, 0x24A8008F00A82422ULL, 0x4850001F00504844ULL, 0x90A0003F00A09088ULL,
    0x0500FC0005091121ULL, 0x0A00F8000A122242ULL, 0x1500F10015244484ULL, 0x2A00E3002A498808ULL, 0x5400C70054921110ULL, 0xA8008F00A8242221ULL, 0x50001F0050484442ULL, 0xA0003F00A0908884ULL,
    0x00FC000509112141ULL, 0x00F8000A12224282ULL, 0x00F1001524448404ULL, 0x00E3002A49880808ULL, 0x00C7005492111010ULL, 0x008F00A824222120ULL, 0x001F005048444241ULL, 0x003F00A090888482ULL,
    0xFC00050911214181ULL, 0xF8000A1222428202ULL, 0xF100152444840404ULL, 0xE3002A4988080808ULL, 0xC700549211101010ULL, 0x8F00A82422212020ULL, 0x1F00504844424140ULL, 0x3F00A09088848281ULL
};

/*
constexpr uint8_t d7_mask[HW2] = {
    0b10000000, 0b11000000, 0b11100000, 0b11110000, 0b11111000, 0b11111100, 0b11111110, 0b11111111,
    0b11000000, 0b11100000, 0b11110000, 0b11111000, 0b11111100, 0b11111110, 0b11111111, 0b01111111,
    0b11100000, 0b11110000, 0b11111000, 0b11111100, 0b11111110, 0b11111111, 0b01111111, 0b00111111,
    0b11110000, 0b11111000, 0b11111100, 0b11111110, 0b11111111, 0b01111111, 0b00111111, 0b00011111,
    0b11111000, 0b11111100, 0b11111110, 0b11111111, 0b01111111, 0b00111111, 0b00011111, 0b00001111,
    0b11111100, 0b11111110, 0b11111111, 0b01111111, 0b00111111, 0b00011111, 0b00001111, 0b00000111,
    0b11111110, 0b11111111, 0b01111111, 0b00111111, 0b00011111, 0b00001111, 0b00000111, 0b00000011,
    0b11111111, 0b01111111, 0b00111111, 0b00011111, 0b00001111, 0b00000111, 0b00000011, 0b00000001
};
*/
constexpr uint8_t d7_mask[HW2] = {
    0b00000001, 0b00000011, 0b00000111, 0b00001111, 0b00011111, 0b00111111, 0b01111111, 0b11111111, 
    0b00000011, 0b00000111, 0b00001111, 0b00011111, 0b00111111, 0b01111111, 0b11111111, 0b11111110, 
    0b00000111, 0b00001111, 0b00011111, 0b00111111, 0b01111111, 0b11111111, 0b11111110, 0b11111100, 
    0b00001111, 0b00011111, 0b00111111, 0b01111111, 0b11111111, 0b11111110, 0b11111100, 0b11111000, 
    0b00011111, 0b00111111, 0b01111111, 0b11111111, 0b11111110, 0b11111100, 0b11111000, 0b11110000, 
    0b00111111, 0b01111111, 0b11111111, 0b11111110, 0b11111100, 0b11111000, 0b11110000, 0b11100000, 
    0b01111111, 0b11111111, 0b11111110, 0b11111100, 0b11111000, 0b11110000, 0b11100000, 0b11000000, 
    0b11111111, 0b11111110, 0b11111100, 0b11111000, 0b11110000, 0b11100000, 0b11000000, 0b10000000
};

constexpr uint8_t d9_mask[HW2] = {
    0b11111111, 0b01111111, 0b00111111, 0b00011111, 0b00001111, 0b00000111, 0b00000011, 0b00000001,
    0b11111110, 0b11111111, 0b01111111, 0b00111111, 0b00011111, 0b00001111, 0b00000111, 0b00000011,
    0b11111100, 0b11111110, 0b11111111, 0b01111111, 0b00111111, 0b00011111, 0b00001111, 0b00000111,
    0b11111000, 0b11111100, 0b11111110, 0b11111111, 0b01111111, 0b00111111, 0b00011111, 0b00001111,
    0b11110000, 0b11111000, 0b11111100, 0b11111110, 0b11111111, 0b01111111, 0b00111111, 0b00011111,
    0b11100000, 0b11110000, 0b11111000, 0b11111100, 0b11111110, 0b11111111, 0b01111111, 0b00111111,
    0b11000000, 0b11100000, 0b11110000, 0b11111000, 0b11111100, 0b11111110, 0b11111111, 0b01111111,
    0b10000000, 0b11000000, 0b11100000, 0b11110000, 0b11111000, 0b11111100, 0b11111110, 0b11111111
};

inline uint_fast8_t join_h_line(uint64_t x, int t){
    return (x >> (HW * t)) & 0b11111111U;
}

inline void join_h_line_double(uint64_t player, uint64_t opponent, int_fast8_t t, uint_fast8_t *p, uint_fast8_t *o){
    __m128i po = _mm_set_epi64x(player, opponent);
    po = _mm_srli_epi64(po, HW * t);
    po = _mm_and_si128(po, _mm_set1_epi64x(0b11111111ULL));
    *o = _mm_cvtsi128_si64(po);
    *p = _mm_cvtsi128_si64(_mm_unpackhi_epi64(po, po));
}

inline uint64_t split_h_line(uint_fast8_t x, int_fast8_t t){
    return (uint64_t)x << (HW * t);
}
/*
inline uint_fast8_t join_v_line(uint64_t x, int t){
    return _pext_u64(x >> t, 0x0101010101010101ULL);
}
*/

inline uint8_t join_v_line(uint64_t x, int_fast8_t t){
    x = (x >> t) & 0b0000000100000001000000010000000100000001000000010000000100000001ULL;
    return (x * 0b0000000100000010000001000000100000010000001000000100000010000000ULL) >> 56;
}

inline void join_v_line_double(uint64_t player, uint64_t opponent, int_fast8_t t, uint_fast8_t *p, uint_fast8_t *o){
    __m128i po = _mm_set_epi64x(player, opponent);
    po = _mm_srli_epi64(po, t);
    po = _mm_and_si128(po, _mm_set1_epi64x(0x0101010101010101ULL));
    po = _mm_mullo_epi64(po, _mm_set1_epi64x(0x0102040810204080ULL));
    po = _mm_srli_epi64(po, 56);
    *o = _mm_cvtsi128_si64(po);
    *p = _mm_cvtsi128_si64(_mm_unpackhi_epi64(po, po));
}

inline uint64_t split_v_line(uint_fast8_t x, int_fast8_t t){
    return _pdep_u64((uint64_t)x, 0x0101010101010101ULL) << t;
}

inline uint_fast8_t join_d7_line(uint64_t x, int_fast8_t t){
    return _pext_u64(x >> t, 0x0002040810204081ULL);
}

inline uint64_t split_d7_line(uint8_t x, int_fast8_t t){
    return _pdep_u64((uint64_t)x, 0x0002040810204081ULL) << t;
}

inline uint_fast8_t join_d9_line(uint64_t x, int_fast8_t t){
    return _pext_u64(t > 0 ? x >> t : x << (-t), 0x8040201008040201ULL);
}

inline uint64_t split_d9_line(uint8_t x, int_fast8_t t){
    uint64_t res = _pdep_u64((uint64_t)x, 0x8040201008040201ULL);
    return t > 0 ? res << t : res >> (-t);
}

constexpr uint64_t join_d7_line_mask[15] = {
    0ULL, 0ULL, 0x0000000000010204ULL, 0x0000000001020408ULL, 
    0x0000000102040810ULL, 0x0000010204081020ULL, 0x0001020408102040ULL, 0x0102040810204080ULL, 
    0x0204081020408000ULL, 0x0408102040800000ULL, 0x0810204080000000ULL, 0x1020408000000000ULL, 
    0x2040800000000000ULL, 0ULL, 0ULL
};

inline uint_fast8_t join_d7_line2(const uint64_t x, const uint_fast8_t t){
    return _pext_u64(x, join_d7_line_mask[t]);
}

constexpr uint64_t join_d9_line_mask[15] = {
    0ULL, 0ULL, 0x0402010000000000ULL, 0x0804020100000000ULL, 
    0x1008040201000000ULL, 0x2010080402010000ULL, 0x4020100804020100ULL, 0x8040201008040201ULL, 
    0x0080402010080402ULL, 0x0000804020100804ULL, 0x0000008040201008ULL, 0x0000000080402010ULL, 
    0x0000000000804020ULL, 0ULL, 0ULL
};

inline uint_fast8_t join_d9_line2(const uint64_t x, const uint_fast8_t t){
    return _pext_u64(x, join_d9_line_mask[t]);
}

uint64_t split_v_lines[N_8BIT];
uint64_t split_d7_lines[N_8BIT];
uint64_t split_d9_lines[N_8BIT];

void bit_init(){
    uint32_t i;
    for (i = 0; i < N_8BIT; ++i){
        split_v_lines[i] = split_v_line(i, 0);
        split_d7_lines[i] = split_d7_line(i, 0);
        split_d9_lines[i] = split_d9_line(i, 0);
    }
    upper_bit_init();
}