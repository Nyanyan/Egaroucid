/*
    Egaroucid Project

    @file nnue_eval.hpp
        Pattern NNUE evaluation (weights from src/tools/nnue/export_nnue.py)
    @date 2026
    @author Takuto Yamana
    @license GPL-3.0-or-later

    The integer model is defined in src/tools/nnue/export_nnue.py and must stay identical to it:
        acc   = sum of the 64 instance rows (int16, 1.0 = 512) + bias
        a     = clamp(acc, 0, 512);   x = min((a[k] * a[k + W/2]) >> 11, 127)
        zacc  = l1 . x + bias (1.0 = 8192);   z = clamp(zacc >> 6, 0, 127)
        oacc  = out . z + bias (1 disc = 8192)
        lin   = sum of the 64 linear rows + disc-count term (1 disc = 256)
        total = lin * 32 + oacc (direct), raw_pattern * 256 + oacc (residual), lin * 32 (linear)
        score = round half away from zero (total / 8192), clamped to [-64, 64]
    Instance rows: the board is mapped by the instance's inverse symmetry and the base pattern mask
    is extracted with pext, digits 1 = side to move, 2 = opponent, 0 = empty, weight 3^k.
    Square input (a file with SQUARE_ROWS = 129 rows, residual mode): the first layer sums the rows of the
    occupied squares instead, row k for a disc of the side to move on square k (bit 63 - k), row 64 + k for
    an opponent disc, and row 128 once per empty square (zero in models trained after 2026-10-20; kept exact
    through a table by the number of empties).
*/

#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#if USE_SIMD
#include <immintrin.h>
#endif
#include "nnue_patterns.hpp"

#if !USE_SIMD
#error "EGAROUCID_NNUE needs the AVX2 (SIMD) build"
#endif

#ifndef EGAROUCID_NNUE_SCOPE
// 1: midgame search leaves only (move ordering keeps the pattern evaluation), 2: every midgame evaluation
#define EGAROUCID_NNUE_SCOPE 1
#endif

namespace nnue {

constexpr int N_BUCKETS = 15;
constexpr int SQUARE_ROWS = 129;
constexpr int MAX_W = 256;
constexpr int H_FIXED = 32;
enum Mode { MODE_DIRECT = 0, MODE_RESIDUAL = 1, MODE_LINEAR = 2 };

struct Weights {
    bool loaded = false;
    int mode = MODE_DIRECT;
    int W = 0;
    int H = 0;
    int16_t *ft = nullptr;            // [rows][W] (int16 first layer, file version 1)
    int8_t *ft8 = nullptr;            // [rows][W] (int8 first layer, file version 2, residual only)
    int ft_shift = 0;                 // int8 rows are in 1 / (512 >> ft_shift)
    alignas(32) int16_t ft_bias[MAX_W];
    int16_t *l1_w = nullptr;          // [buckets][H][W/2], widened from int8
    int32_t l1_b[N_BUCKETS][H_FIXED];
    int32_t out_w[N_BUCKETS][H_FIXED];  // widened from int16
    int32_t out_b[N_BUCKETS];
    int16_t *lin = nullptr;           // [buckets][rows]
    int16_t num[N_BUCKETS][65];
    // packed rows read by the search: [W first-layer lanes][16 linear lanes, one per bucket] (linear
    // lanes only when the model has L), so one row read serves both parts
    int16_t *packed = nullptr;
    int stride = 0;
    bool has_lin = false;
    bool input_squares = false;  // square input (SQUARE_ROWS rows) instead of the pattern rows
    // square input, by the number of empties e: int16 rows: saturate(bias + e * row 128); int8 rows: e * row 128
    std::vector<int16_t> sq_empty;
};
inline Weights weights;
inline Weights uncertainty_weights;  // optional second network (per-position uncertainty for MPC)
inline uint16_t bits_to_ternary[512];

inline void *aligned_alloc64(size_t bytes) {
    return _mm_malloc((bytes + 63) / 64 * 64, 64);
}

template <typename T>
inline bool read_array(std::ifstream &ifs, T *dst, size_t n) {
    ifs.read(reinterpret_cast<char*>(dst), n * sizeof(T));
    return (bool)ifs;
}

#if USE_SIMD
inline void init_incremental();
#endif

inline bool init_weights(Weights &w, const std::string &file, bool show_log) {
    for (int b = 0; b < 512; ++b) {
        int t = 0, p = 1;
        for (int k = 0; k < 9; ++k, p *= 3) if ((b >> k) & 1) t += p;
        bits_to_ternary[b] = (uint16_t)t;
    }
    std::ifstream ifs(file, std::ios::binary);
    if (!ifs) {
        std::cerr << "[ERROR] [FATAL] NNUE weight file not found: " << file << std::endl;
        return false;
    }
    char magic[8];
    int32_t head[8];
    ifs.read(magic, 8);
    ifs.read(reinterpret_cast<char*>(head), sizeof(head));
    if (!ifs || std::memcmp(magic, "EGNNUE01", 8) != 0 || (head[0] != 1 && head[0] != 2)) {
        std::cerr << "[ERROR] [FATAL] not an NNUE weight file: " << file << std::endl;
        return false;
    }
    // a reload starts from an empty object (a version-1 file must not keep version-2 int8 rows)
    for (void **ptr : {(void**)&w.ft, (void**)&w.ft8, (void**)&w.l1_w, (void**)&w.lin, (void**)&w.packed}) {
        if (*ptr) {
            _mm_free(*ptr);
            *ptr = nullptr;
        }
    }
    w.loaded = false;
    w.mode = head[1];
    w.W = head[4];
    w.H = head[5];
    const bool int8_ft = head[0] == 2;
    w.ft_shift = int8_ft ? head[7] : 0;
    const bool shape_ok = w.mode == MODE_LINEAR
        ? (w.W == 0 && w.H == 0 && !int8_ft)
        : ((w.mode == MODE_DIRECT || w.mode == MODE_RESIDUAL) && w.W > 0 && w.W % 32 == 0 && w.W <= MAX_W && w.H == H_FIXED &&
           (!int8_ft || (w.mode == MODE_RESIDUAL && w.ft_shift >= 0 && w.ft_shift <= 3)));
    const bool squares = head[2] == SQUARE_ROWS && w.mode == MODE_RESIDUAL;
    if ((head[2] != NNUE_N_ROWS && !squares) || head[3] != NNUE_N_INSTANCES || head[6] != N_BUCKETS || !shape_ok) {
        std::cerr << "[ERROR] [FATAL] NNUE weight file does not match this build: " << file << std::endl;
        return false;
    }
    if (squares && !(w.W == 32 || w.W == 64 || w.W == 128 || w.W == 256)) {
        std::cerr << "[ERROR] [FATAL] NNUE square input needs a width of 32, 64, 128 or 256: " << file << std::endl;
        return false;
    }
    w.input_squares = squares;
    const size_t rows = head[2];
    const int W = w.W, H = w.H, half = W / 2;
    if (int8_ft) {
        w.ft8 = (int8_t*)aligned_alloc64(rows * W + 64);
    } else {
        w.ft = (int16_t*)aligned_alloc64(rows * W * sizeof(int16_t) + 64);
    }
    w.l1_w = (int16_t*)aligned_alloc64((size_t)N_BUCKETS * H_FIXED * (half > 0 ? half : 1) * sizeof(int16_t));
    w.lin = (int16_t*)aligned_alloc64((size_t)N_BUCKETS * rows * sizeof(int16_t));
    std::memset(w.l1_b, 0, sizeof(w.l1_b));
    std::memset(w.out_w, 0, sizeof(w.out_w));
    std::memset(w.out_b, 0, sizeof(w.out_b));
    bool ok = (int8_ft ? read_array(ifs, w.ft8, rows * W) : read_array(ifs, w.ft, rows * W)) && read_array(ifs, w.ft_bias, W);
    std::vector<int8_t> l1(N_BUCKETS * H * half);
    std::vector<int32_t> l1b(N_BUCKETS * H);
    std::vector<int16_t> ow(N_BUCKETS * H);
    ok = ok && read_array(ifs, l1.data(), l1.size()) && read_array(ifs, l1b.data(), l1b.size()) &&
         read_array(ifs, ow.data(), ow.size()) && read_array(ifs, w.out_b, N_BUCKETS) &&
         read_array(ifs, w.lin, (size_t)N_BUCKETS * rows) && read_array(ifs, &w.num[0][0], N_BUCKETS * 65);
    if (!ok) {
        std::cerr << "[ERROR] [FATAL] NNUE weight file is truncated: " << file << std::endl;
        return false;
    }
    for (int b = 0; b < N_BUCKETS; ++b) {
        for (int h = 0; h < H; ++h) {
            for (int k = 0; k < half; ++k) w.l1_w[((size_t)b * H_FIXED + h) * half + k] = l1[((size_t)b * H + h) * half + k];
            w.l1_b[b][h] = l1b[b * H + h];
            w.out_w[b][h] = ow[b * H + h];
        }
    }
    w.has_lin = w.mode != MODE_RESIDUAL;
    w.stride = W + (w.has_lin ? 16 : 0);
    if (!int8_ft) {
        w.packed = (int16_t*)aligned_alloc64(rows * w.stride * sizeof(int16_t) + 64);
        for (size_t r = 0; r < rows; ++r) {
            int16_t *dst = w.packed + r * w.stride;
            std::memcpy(dst, w.ft + r * W, W * sizeof(int16_t));
            if (w.has_lin) {
                for (int b = 0; b < 16; ++b) dst[W + b] = b < N_BUCKETS ? w.lin[(size_t)b * rows + r] : 0;
            }
        }
    }
    if (squares) {
        w.sq_empty.assign((size_t)65 * W, 0);
        for (int e = 0; e <= 64; ++e) {
            for (int j = 0; j < W; ++j) {
                const int32_t r = int8_ft ? w.ft8[(size_t)128 * W + j] : w.ft[(size_t)128 * W + j];
                w.sq_empty[(size_t)e * W + j] = int8_ft ? (int16_t)(e * r) : (int16_t)std::clamp(w.ft_bias[j] + e * r, -32768, 32767);
            }
        }
    }
#if USE_SIMD
    init_incremental();
#endif
    w.loaded = true;
    if (show_log) {
        std::cerr << "NNUE loaded " << file << " mode " << w.mode << " width " << W << (int8_ft ? " int8" : "") << (squares ? " square input" : "") << " scope " << EGAROUCID_NNUE_SCOPE << std::endl;
    }
    return true;
}

inline bool init(const std::string &file, bool show_log) {
    return init_weights(weights, file, show_log);
}

#ifndef EGAROUCID_NNUE_FILENAME
#define EGAROUCID_NNUE_FILENAME "nnue.egnn"
#endif

// path: $EGAROUCID_NNUE_FILE if set, otherwise EGAROUCID_NNUE_FILENAME next to the pattern evaluation file
inline std::string default_path(const std::string &eval_file, const char *filename = EGAROUCID_NNUE_FILENAME, const char *env_name = "EGAROUCID_NNUE_FILE") {
    if (const char *env = std::getenv(env_name)) {
        if (env[0]) return env;
    }
    const size_t sep = eval_file.find_last_of("/\\");
    return (sep == std::string::npos ? std::string() : eval_file.substr(0, sep + 1)) + filename;
}

#ifndef EGAROUCID_NNUE_UNCERTAINTY_FILENAME
#define EGAROUCID_NNUE_UNCERTAINTY_FILENAME "nnue_uncertainty_w32.egnn"
#endif

inline uint64_t mirror_cols(uint64_t x) {  // (x, y) -> (7 - x, y): reverse the bits of each byte
    x = ((x >> 1) & 0x5555555555555555ULL) | ((x & 0x5555555555555555ULL) << 1);
    x = ((x >> 2) & 0x3333333333333333ULL) | ((x & 0x3333333333333333ULL) << 2);
    return ((x >> 4) & 0x0F0F0F0F0F0F0F0FULL) | ((x & 0x0F0F0F0F0F0F0F0FULL) << 4);
}

inline uint64_t transpose_bits(uint64_t x) {  // bit (8r + c) -> (8c + r)
    uint64_t t;
    t = 0x0F0F0F0F00000000ULL & (x ^ (x << 28)); x ^= t ^ (t >> 28);
    t = 0x3333000033330000ULL & (x ^ (x << 14)); x ^= t ^ (t >> 14);
    t = 0x5500550055005500ULL & (x ^ (x << 7)); x ^= t ^ (t >> 7);
    return x;
}

inline uint64_t antitranspose_bits(uint64_t x) {  // bit (8r + c) -> (8 (7 - c) + 7 - r)
    uint64_t t;
    t = x ^ (x << 36); x ^= 0xF0F0F0F00F0F0F0FULL & (t ^ (x >> 36));
    t = 0xCCCC0000CCCC0000ULL & (x ^ (x << 18)); x ^= t ^ (t >> 18);
    t = 0xAA00AA00AA00AA00ULL & (x ^ (x << 9)); x ^= t ^ (t >> 9);
    return x;
}

// t[s] = board mapped by symmetry s of src/tools/nnue/nnue_patterns.py
inline void symmetries(uint64_t b, uint64_t t[8]) {
    const uint64_t m = mirror_cols(b), tr = transpose_bits(b);
    t[0] = b;
    t[1] = m;
    t[2] = __builtin_bswap64(b);
    t[3] = __builtin_bswap64(m);
    t[4] = tr;
    t[5] = antitranspose_bits(b);
    t[6] = mirror_cols(tr);
    t[7] = __builtin_bswap64(tr);
}

inline uint64_t pext64(uint64_t x, uint64_t mask) {
#if defined(__BMI2__)
    return _pext_u64(x, mask);
#else
    uint64_t res = 0;
    for (uint64_t bb = 1; mask; bb += bb) {
        if (x & mask & -mask) res |= bb;
        mask &= mask - 1;
    }
    return res;
#endif
}

inline void compute_indices(uint64_t player, uint64_t opponent, int idx[NNUE_N_INSTANCES]) {
    uint64_t tp[8], to[8];
    symmetries(player, tp);
    symmetries(opponent, to);
    for (int i = 0; i < NNUE_N_INSTANCES; ++i) {
        const Nnue_instance &in = nnue_instances[i];
        const Nnue_pattern &pt = nnue_patterns[in.pattern];
        idx[i] = pt.offset + bits_to_ternary[pext64(tp[in.inv_sym], pt.mask)] + 2 * bits_to_ternary[pext64(to[in.inv_sym], pt.mask)];
    }
}

inline int bucket_of(int empties) {
    const int b = (60 - empties) / 4;
    return b < 0 ? 0 : (b >= N_BUCKETS ? N_BUCKETS - 1 : b);
}

inline int64_t forward_scalar(const int idx[NNUE_N_INSTANCES], int bucket, const Weights &w = weights) {
    const int W = w.W, half = W / 2;
    int32_t acc[MAX_W];
    if (w.ft8) {
        // int8 rows: sum (fits int16), clamp to +-4095, scale to 1/512 and add the bias (saturating)
        for (int j = 0; j < W; ++j) acc[j] = 0;
        for (int i = 0; i < NNUE_N_INSTANCES; ++i) {
            const int8_t *row = w.ft8 + (size_t)idx[i] * W;
            for (int j = 0; j < W; ++j) acc[j] += row[j];
        }
        for (int j = 0; j < W; ++j) acc[j] = std::clamp(std::clamp(acc[j], -4095, 4095) * (1 << w.ft_shift) + w.ft_bias[j], -32768, 32767);
    } else {
        for (int j = 0; j < W; ++j) acc[j] = w.ft_bias[j];
        for (int i = 0; i < NNUE_N_INSTANCES; ++i) {
            const int16_t *row = w.ft + (size_t)idx[i] * W;
            for (int j = 0; j < W; ++j) acc[j] = std::clamp(acc[j] + row[j], -32768, 32767);  // saturate like the vector path
        }
    }
    int32_t x[MAX_W / 2];
    for (int k = 0; k < half; ++k) {
        const int32_t a = acc[k] < 0 ? 0 : (acc[k] > 512 ? 512 : acc[k]);
        const int32_t b = acc[k + half] < 0 ? 0 : (acc[k + half] > 512 ? 512 : acc[k + half]);
        const int32_t v = (a * b) >> 11;
        x[k] = v > 127 ? 127 : v;
    }
    int64_t oacc = w.out_b[bucket];
    for (int h = 0; h < H_FIXED; ++h) {
        const int16_t *row = w.l1_w + ((size_t)bucket * H_FIXED + h) * half;
        int32_t s = w.l1_b[bucket][h];
        for (int k = 0; k < half; ++k) s += x[k] * row[k];
        int32_t z = s >> 6;
        z = z < 0 ? 0 : (z > 127 ? 127 : z);
        oacc += (int64_t)z * w.out_w[bucket][h];
    }
    return oacc;
}

#if USE_SIMD
inline __m256i hsum8_epi32(__m256i s0, __m256i s1, __m256i s2, __m256i s3, __m256i s4, __m256i s5, __m256i s6, __m256i s7) {
    __m256i t0 = _mm256_hadd_epi32(s0, s1), t1 = _mm256_hadd_epi32(s2, s3);
    __m256i t2 = _mm256_hadd_epi32(s4, s5), t3 = _mm256_hadd_epi32(s6, s7);
    t0 = _mm256_hadd_epi32(t0, t1);
    t2 = _mm256_hadd_epi32(t2, t3);
    return _mm256_add_epi32(_mm256_permute2x128_si256(t0, t2, 0x20), _mm256_permute2x128_si256(t0, t2, 0x31));
}

// first layer: acc = ft_bias + sum of the instance rows (saturating), and the linear head's lanes
template <int W, bool LIN>
inline void accumulate_packed(const int idx[NNUE_N_INSTANCES], __m256i *acc, __m256i *lin_lo, __m256i *lin_hi, const Weights &w = weights) {
    constexpr int NV = W / 16;     // int16 vectors in the accumulator
    constexpr int STRIDE = W + (LIN ? 16 : 0);
    for (int v = 0; v < NV; ++v) acc[v] = _mm256_load_si256((const __m256i*)&w.ft_bias[v * 16]);
    __m256i llo = _mm256_setzero_si256(), lhi = _mm256_setzero_si256();
    for (int i = 0; i < NNUE_N_INSTANCES; ++i) {
        const __m256i *row = (const __m256i*)(w.packed + (size_t)idx[i] * STRIDE);
        for (int v = 0; v < NV; ++v) acc[v] = _mm256_adds_epi16(acc[v], _mm256_load_si256(row + v));  // saturating
        if constexpr (LIN) {
            const __m256i l = _mm256_load_si256(row + NV);
            llo = _mm256_add_epi32(llo, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(l)));
            lhi = _mm256_add_epi32(lhi, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(l, 1)));
        }
    }
    *lin_lo = llo;
    *lin_hi = lhi;
}

// first-layer features kept for the uncertainty head (dual network): clamped accumulator a (1.0 = 512),
// pairwise products x (1.0 = 128) and the bucket's clamped hidden layer z (1.0 = 128)
struct Trunk {
    alignas(32) int16_t a[MAX_W];
    alignas(32) int16_t x[MAX_W / 2];
    alignas(32) int32_t z[H_FIXED];
};

// pairwise product and the bucket's hidden and output layers: oacc (t: also keep the features)
template <int W>
inline int64_t head_packed(const __m256i *acc, int bucket, const Weights &w = weights, Trunk *t = nullptr) {
    constexpr int HALF = W / 2;
    constexpr int NX = HALF / 16;  // int16 vectors in x
    const __m256i zero = _mm256_setzero_si256(), c512 = _mm256_set1_epi16(512), c127 = _mm256_set1_epi16(127);
    __m256i x[NX];
    for (int v = 0; v < NX; ++v) {
        const __m256i a = _mm256_min_epi16(_mm256_max_epi16(acc[v], zero), c512);
        const __m256i b = _mm256_min_epi16(_mm256_max_epi16(acc[v + NX], zero), c512);
        x[v] = _mm256_min_epi16(_mm256_mulhi_epi16(_mm256_slli_epi16(a, 5), b), c127);  // (a * b) >> 11
        if (t) {
            _mm256_store_si256((__m256i*)&t->a[v * 16], a);
            _mm256_store_si256((__m256i*)&t->a[HALF + v * 16], b);
            _mm256_store_si256((__m256i*)&t->x[v * 16], x[v]);
        }
    }
    const int16_t *l1 = w.l1_w + (size_t)bucket * H_FIXED * HALF;
    __m256i o = _mm256_setzero_si256();
    for (int g = 0; g < H_FIXED / 8; ++g) {  // 8 outputs at a time keeps the partial sums in registers
        __m256i s[8];
        for (int h = 0; h < 8; ++h) {
            const int16_t *row = l1 + (8 * g + h) * HALF;
            __m256i t = _mm256_madd_epi16(x[0], _mm256_loadu_si256((const __m256i*)row));
            for (int v = 1; v < NX; ++v) t = _mm256_add_epi32(t, _mm256_madd_epi16(x[v], _mm256_loadu_si256((const __m256i*)(row + v * 16))));
            s[h] = t;
        }
        __m256i zs = hsum8_epi32(s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]);
        zs = _mm256_add_epi32(zs, _mm256_loadu_si256((const __m256i*)&w.l1_b[bucket][8 * g]));
        zs = _mm256_min_epi32(_mm256_max_epi32(_mm256_srai_epi32(zs, 6), _mm256_setzero_si256()), _mm256_set1_epi32(127));
        if (t) _mm256_store_si256((__m256i*)&t->z[8 * g], zs);
        o = _mm256_add_epi32(o, _mm256_mullo_epi32(zs, _mm256_loadu_si256((const __m256i*)&w.out_w[bucket][8 * g])));
    }
    __m128i o128 = _mm_add_epi32(_mm256_castsi256_si128(o), _mm256_extracti128_si256(o, 1));
    o128 = _mm_hadd_epi32(o128, o128);
    o128 = _mm_hadd_epi32(o128, o128);
    return (int64_t)_mm_cvtsi128_si32(o128) + w.out_b[bucket];
}

// first-layer row prefetch: 0 off, -1 all rows before summing, k > 0 k rows ahead inside the loop (12: about 1-1.5%
// faster at one thread on the 2026-10-21 benchmarks, identical values)
#ifndef EGAROUCID_NNUE_PREFETCH
#define EGAROUCID_NNUE_PREFETCH 12
#endif

// first layer from int8 rows: one cache line per row for W = 64
template <int W>
inline void accumulate_packed8(const int idx[NNUE_N_INSTANCES], __m256i *acc, const Weights &w = weights) {
    constexpr int NV = W / 16;
    __m256i a8[NV];
    for (int v = 0; v < NV; ++v) a8[v] = _mm256_setzero_si256();
#if EGAROUCID_NNUE_PREFETCH < 0
    for (int i = 0; i < NNUE_N_INSTANCES; ++i) _mm_prefetch((const char*)(w.ft8 + (size_t)idx[i] * W), _MM_HINT_T0);
#endif
    for (int i = 0; i < NNUE_N_INSTANCES; ++i) {
#if EGAROUCID_NNUE_PREFETCH > 0
        if (i + EGAROUCID_NNUE_PREFETCH < NNUE_N_INSTANCES) _mm_prefetch((const char*)(w.ft8 + (size_t)idx[i + EGAROUCID_NNUE_PREFETCH] * W), _MM_HINT_T0);
#endif
        const __m128i *row = (const __m128i*)(w.ft8 + (size_t)idx[i] * W);
        for (int v = 0; v < NV; ++v) a8[v] = _mm256_add_epi16(a8[v], _mm256_cvtepi8_epi16(_mm_load_si128(row + v)));
    }
    const __m256i lo = _mm256_set1_epi16(-4095), hi = _mm256_set1_epi16(4095);
    const __m128i shift = _mm_cvtsi32_si128(w.ft_shift);
    for (int v = 0; v < NV; ++v) {
        const __m256i c = _mm256_min_epi16(_mm256_max_epi16(a8[v], lo), hi);
        acc[v] = _mm256_adds_epi16(_mm256_sll_epi16(c, shift), _mm256_load_si256((const __m256i*)&w.ft_bias[v * 16]));
    }
}

template <int W>
inline int64_t forward_packed8(const int idx[NNUE_N_INSTANCES], int bucket, const Weights &w = weights, Trunk *t = nullptr) {
    __m256i acc[W / 16];
    accumulate_packed8<W>(idx, acc, w);
    return head_packed<W>(acc, bucket, w, t);
}

// square input: the rows of the occupied squares (player: row k, opponent: row 64 + k for bit 63 - k)
template <int W>
inline void accumulate_squares(uint64_t player, uint64_t opponent, __m256i *acc, const Weights &w = weights) {
    constexpr int NV = W / 16;
    const int16_t *empty_row = w.sq_empty.data() + (size_t)(64 - __builtin_popcountll(player | opponent)) * W;
    if (w.ft8) {
        __m256i a8[NV];
        for (int v = 0; v < NV; ++v) a8[v] = _mm256_loadu_si256((const __m256i*)(empty_row + v * 16));
        for (int side = 0; side < 2; ++side) {
            const int8_t *base = w.ft8 + (side ? (size_t)64 * W : 0);
            for (uint64_t b = side ? opponent : player; b; b &= b - 1) {
                const __m128i *row = (const __m128i*)(base + (size_t)(63 - __builtin_ctzll(b)) * W);
                for (int v = 0; v < NV; ++v) a8[v] = _mm256_add_epi16(a8[v], _mm256_cvtepi8_epi16(_mm_load_si128(row + v)));
            }
        }
        const __m256i lo = _mm256_set1_epi16(-4095), hi = _mm256_set1_epi16(4095);
        const __m128i shift = _mm_cvtsi32_si128(w.ft_shift);
        for (int v = 0; v < NV; ++v) {
            const __m256i c = _mm256_min_epi16(_mm256_max_epi16(a8[v], lo), hi);
            acc[v] = _mm256_adds_epi16(_mm256_sll_epi16(c, shift), _mm256_load_si256((const __m256i*)&w.ft_bias[v * 16]));
        }
        return;
    }
    for (int v = 0; v < NV; ++v) acc[v] = _mm256_loadu_si256((const __m256i*)(empty_row + v * 16));
    for (int side = 0; side < 2; ++side) {
        const int16_t *base = w.packed + (side ? (size_t)64 * w.stride : 0);
        for (uint64_t b = side ? opponent : player; b; b &= b - 1) {
            const __m256i *row = (const __m256i*)(base + (size_t)(63 - __builtin_ctzll(b)) * w.stride);
            for (int v = 0; v < NV; ++v) acc[v] = _mm256_adds_epi16(acc[v], _mm256_load_si256(row + v));  // saturating
        }
    }
}

template <int W>
inline int64_t forward_squares(uint64_t player, uint64_t opponent, int bucket, const Weights &w = weights, Trunk *t = nullptr) {
    __m256i acc[W / 16];
    accumulate_squares<W>(player, opponent, acc, w);
    return head_packed<W>(acc, bucket, w, t);
}

// residual model with an int16 first layer, keeping the features
template <int W>
inline int64_t forward_packed_trunk(const int idx[NNUE_N_INSTANCES], int bucket, Trunk *t, const Weights &w = weights) {
    __m256i acc[W / 16];
    __m256i lin_lo, lin_hi;
    accumulate_packed<W, false>(idx, acc, &lin_lo, &lin_hi, w);
    return head_packed<W>(acc, bucket, w, t);
}

template <int W, bool LIN>
inline int64_t forward_packed(const int idx[NNUE_N_INSTANCES], int bucket, uint64_t player, const Weights &w = weights) {
    constexpr int NV = W / 16;
    __m256i acc[NV > 0 ? NV : 1];
    __m256i lin_lo, lin_hi;
    accumulate_packed<W, LIN>(idx, acc, &lin_lo, &lin_hi, w);
    int64_t total = 0;
    if constexpr (W > 0) total += head_packed<W>(acc, bucket, w);
    if constexpr (LIN) {
        alignas(32) int32_t lanes[16];
        _mm256_store_si256((__m256i*)&lanes[0], lin_lo);
        _mm256_store_si256((__m256i*)&lanes[8], lin_hi);
        total += (int64_t)(lanes[bucket] + w.num[bucket][__builtin_popcountll(player)]) * 32;
    }
    return total;
}
#endif

#if USE_SIMD
// Incremental instance indices, kept beside the search's pattern features (one 16-bit lane per
// instance, pattern-local index without the row offset).
// The lanes use fixed colors: digit 1 is a disc of the side to move at the position where the
// features were built (the "root side"), digit 2 a disc of the other side, so a move only changes
// the placed square and the flipped discs:
//     root side moves:  lanes += W(placed) - W(flipped)      (0 -> 1, 2 -> 1)
//     other side moves: lanes += 2 W(placed) + W(flipped)    (0 -> 2, 1 -> 2)
// and a pass changes nothing. W(bits) is the per-lane sum of 3^k over the instance cells in bits.
// The occupancy lanes occ = W(occupied) give the side-to-move view when the other side is to move:
// swapping digits 1 and 2 maps x to 3 occ - x.
union Lanes {
    __m256i v[4];
    uint16_t u[64];
};
inline Lanes square_table[65];  // W(1 << square); entry 64 is zero (tzcnt of 0)
inline int instance_offset[NNUE_N_INSTANCES];

inline void init_incremental() {
    static_assert(NNUE_N_INSTANCES == 64, "one 16-bit lane per instance");
    std::memset(square_table, 0, sizeof(square_table));
    for (int i = 0; i < NNUE_N_INSTANCES; ++i) {
        const Nnue_instance &in = nnue_instances[i];
        const uint64_t mask = nnue_patterns[in.pattern].mask;
        instance_offset[i] = nnue_patterns[in.pattern].offset;
        for (int b = 0; b < 64; ++b) {
            uint64_t t[8];
            symmetries(1ULL << b, t);
            const int bb = __builtin_ctzll(t[in.inv_sym]);
            if ((mask >> bb) & 1) {
                const int k = __builtin_popcountll(mask & ((1ULL << bb) - 1));
                int p3 = 1;
                for (int j = 0; j < k; ++j) p3 *= 3;
                square_table[b].u[i] = (uint16_t)p3;
            }
        }
    }
}

inline void lanes_from_board(uint64_t player, uint64_t opponent, Lanes *dst) {
    int idx[NNUE_N_INSTANCES];
    compute_indices(player, opponent, idx);
    for (int i = 0; i < NNUE_N_INSTANCES; ++i) dst->u[i] = (uint16_t)(idx[i] - instance_offset[i]);
}

// lanes and occupancy of a freshly built position (the side to move becomes the root side)
inline void lanes_init(uint64_t player, uint64_t opponent, Lanes *lanes, Lanes *occ) {
    lanes_from_board(player, opponent, lanes);
    lanes_from_board(player | opponent, 0, occ);
}

// other_side: the mover is not the root side
inline void lanes_move(const Lanes *src, const Lanes *src_occ, Lanes *dst, Lanes *dst_occ, uint64_t flipped, int pos, bool other_side) {
    const Lanes &put = square_table[pos];
    __m256i fl0 = _mm256_setzero_si256(), fl1 = fl0, fl2 = fl0, fl3 = fl0;
    // the first 6 flipped discs without branches (a variable trip count mispredicts), the rest in a loop
    uint64_t f = flipped;
    for (int k = 0; k < 6; ++k) {
        const Lanes &w = square_table[_tzcnt_u64(f)];
        fl0 = _mm256_add_epi16(fl0, w.v[0]);
        fl1 = _mm256_add_epi16(fl1, w.v[1]);
        fl2 = _mm256_add_epi16(fl2, w.v[2]);
        fl3 = _mm256_add_epi16(fl3, w.v[3]);
        f = _blsr_u64(f);
    }
    for (; f; f &= f - 1) {
        const Lanes &w = square_table[__builtin_ctzll(f)];
        fl0 = _mm256_add_epi16(fl0, w.v[0]);
        fl1 = _mm256_add_epi16(fl1, w.v[1]);
        fl2 = _mm256_add_epi16(fl2, w.v[2]);
        fl3 = _mm256_add_epi16(fl3, w.v[3]);
    }
    if (other_side) {
        dst->v[0] = _mm256_add_epi16(_mm256_add_epi16(src->v[0], _mm256_slli_epi16(put.v[0], 1)), fl0);
        dst->v[1] = _mm256_add_epi16(_mm256_add_epi16(src->v[1], _mm256_slli_epi16(put.v[1], 1)), fl1);
        dst->v[2] = _mm256_add_epi16(_mm256_add_epi16(src->v[2], _mm256_slli_epi16(put.v[2], 1)), fl2);
        dst->v[3] = _mm256_add_epi16(_mm256_add_epi16(src->v[3], _mm256_slli_epi16(put.v[3], 1)), fl3);
    } else {
        dst->v[0] = _mm256_sub_epi16(_mm256_add_epi16(src->v[0], put.v[0]), fl0);
        dst->v[1] = _mm256_sub_epi16(_mm256_add_epi16(src->v[1], put.v[1]), fl1);
        dst->v[2] = _mm256_sub_epi16(_mm256_add_epi16(src->v[2], put.v[2]), fl2);
        dst->v[3] = _mm256_sub_epi16(_mm256_add_epi16(src->v[3], put.v[3]), fl3);
    }
    dst_occ->v[0] = _mm256_add_epi16(src_occ->v[0], put.v[0]);
    dst_occ->v[1] = _mm256_add_epi16(src_occ->v[1], put.v[1]);
    dst_occ->v[2] = _mm256_add_epi16(src_occ->v[2], put.v[2]);
    dst_occ->v[3] = _mm256_add_epi16(src_occ->v[3], put.v[3]);
}

// side-to-move view of the lanes
inline void lanes_view(const Lanes *lanes, const Lanes *occ, bool other_side, Lanes *dst) {
    if (other_side) {
        for (int v = 0; v < 4; ++v) {
            const __m256i o = occ->v[v];
            dst->v[v] = _mm256_sub_epi16(_mm256_add_epi16(_mm256_add_epi16(o, o), o), lanes->v[v]);
        }
    } else {
        for (int v = 0; v < 4; ++v) dst->v[v] = lanes->v[v];
    }
}
#endif

// network part only (oacc), scalar reference
inline int64_t forward(const int idx[NNUE_N_INSTANCES], int bucket) {
    return forward_scalar(idx, bucket);
}

// total in 1/8192 discs; pattern_raw: production pattern evaluation in 1/32 discs (residual mode only)
inline int64_t evaluate_total_scalar(const int idx[NNUE_N_INSTANCES], uint64_t player, int empties, int pattern_raw, const Weights &w = weights) {
    const int bucket = bucket_of(empties);
    int64_t total = 0;
    if (w.mode != MODE_LINEAR) total += forward_scalar(idx, bucket, w);
    if (w.mode == MODE_RESIDUAL) {
        total += (int64_t)pattern_raw * 256;
    } else {
        const int16_t *lin = w.lin + (size_t)bucket * NNUE_N_ROWS;
        int32_t l = w.num[bucket][__builtin_popcountll(player)];
        for (int i = 0; i < NNUE_N_INSTANCES; ++i) l += lin[idx[i]];
        total += (int64_t)l * 32;
    }
    return total;
}

inline int64_t evaluate_total_idx(const int idx[NNUE_N_INSTANCES], uint64_t player, int empties, int pattern_raw, const Weights &w = weights) {
#if USE_SIMD
    const int bucket = bucket_of(empties);
    int64_t total;
    if (w.ft8) {
        switch (w.W) {
            case 32: total = forward_packed8<32>(idx, bucket, w); break;
            case 64: total = forward_packed8<64>(idx, bucket, w); break;
            case 128: total = forward_packed8<128>(idx, bucket, w); break;
            case 256: total = forward_packed8<256>(idx, bucket, w); break;
            default: return evaluate_total_scalar(idx, player, empties, pattern_raw, w);
        }
        return total + (int64_t)pattern_raw * 256;
    }
    switch (w.W * 2 + (w.has_lin ? 1 : 0)) {
        case 1: total = forward_packed<0, true>(idx, bucket, player, w); break;
        case 64: total = forward_packed<32, false>(idx, bucket, player, w); break;
        case 65: total = forward_packed<32, true>(idx, bucket, player, w); break;
        case 128: total = forward_packed<64, false>(idx, bucket, player, w); break;
        case 129: total = forward_packed<64, true>(idx, bucket, player, w); break;
        case 256: total = forward_packed<128, false>(idx, bucket, player, w); break;
        case 257: total = forward_packed<128, true>(idx, bucket, player, w); break;
        case 512: total = forward_packed<256, false>(idx, bucket, player, w); break;
        case 513: total = forward_packed<256, true>(idx, bucket, player, w); break;
        default: return evaluate_total_scalar(idx, player, empties, pattern_raw, w);
    }
    if (w.mode == MODE_RESIDUAL) total += (int64_t)pattern_raw * 256;
    return total;
#else
    return evaluate_total_scalar(idx, player, empties, pattern_raw, w);
#endif
}

#if USE_SIMD
// square-input model: total in 1/8192 discs (residual), pattern_raw as in evaluate_total_scalar
inline int64_t evaluate_total_squares(uint64_t player, uint64_t opponent, int empties, int pattern_raw, const Weights &w = weights) {
    const int bucket = bucket_of(empties);
    int64_t total;
    switch (w.W) {
        case 32: total = forward_squares<32>(player, opponent, bucket, w); break;
        case 64: total = forward_squares<64>(player, opponent, bucket, w); break;
        case 128: total = forward_squares<128>(player, opponent, bucket, w); break;
        default: total = forward_squares<256>(player, opponent, bucket, w); break;
    }
    return total + (int64_t)pattern_raw * 256;
}
#endif

inline int64_t evaluate_total(uint64_t player, uint64_t opponent, int empties, int pattern_raw) {
#if USE_SIMD
    if (weights.input_squares) return evaluate_total_squares(player, opponent, empties, pattern_raw);
#endif
    int idx[NNUE_N_INSTANCES];
    compute_indices(player, opponent, idx);
    return evaluate_total_idx(idx, player, empties, pattern_raw);
}

#if USE_SIMD
inline int64_t evaluate_total_lanes(const Lanes *l, uint64_t player, int empties, int pattern_raw, const Weights &w = weights) {
    alignas(32) int idx[NNUE_N_INSTANCES];
    for (int v = 0; v < 8; ++v) {
        const __m128i lo = _mm_loadu_si128((const __m128i*)&l->u[8 * v]);
        const __m256i w32 = _mm256_add_epi32(_mm256_cvtepu16_epi32(lo), _mm256_loadu_si256((const __m256i*)&instance_offset[8 * v]));
        _mm256_store_si256((__m256i*)&idx[8 * v], w32);
    }
    return evaluate_total_idx(idx, player, empties, pattern_raw, w);
}
#endif

#if USE_SIMD
/*
    Uncertainty head on the main network's first layer (dual network, src/tools/nnue/train_uncertainty_head.py):
    inputs f = x / 128 (inputs 0), (x / 128, z / 128) (1), (a / 512, x / 128, z / 128) (2) or z / 128 (3);
    h = clamp(l1_w[bucket] f + l1_b[bucket], 0, 1);  u = 16 (out_w[bucket] . h + out_b[bucket]),
    or with hidden 0 (a linear output; inputs 3: the second output of a dual network trained with
    train_nnue.py --u-weight) u = 16 (out_w[bucket] . f + out_b[bucket]),
    an estimate of log sigma^2 of (final result - evaluation). It runs only at MPC nodes. The hidden layer
    uses int16 multiply-adds on the integer features: the loader folds the feature scales (1/512, 1/128)
    into the weights and quantizes them to int16 with one scale per head (sums stay below 2^31).
    Version 2 adds P private lanes read only by the head (8, 16 or 32): their own int8 table over the pattern rows,
    p = clamp(sum of the 64 rows + bias, 0, 64) / 64, appended to the inputs (n_in counts them).
    File (little endian, src/tools/nnue/export_uncertainty_head.py): "EGUHEAD1", int32 [version 1 or 2, inputs,
    n_in, hidden, main W, main H, buckets, P (version 2)], uint64 FNV-1a 64 of the main weight file,
    float l1_w[buckets][hidden][n_in], l1_b[buckets][hidden], out_w[buckets][hidden] (hidden 0: no l1, out_w[buckets][n_in]),
    out_b[buckets],
    then (version 2, P > 0) int8 private[rows][P], int16 private_bias[P].
*/
constexpr int UHEAD_MAX_PRIVATE = 32;
struct Uhead {
    bool loaded = false;
    int inputs = 0, n_in = 0, hidden = 0;
    int n_private = 0;        // private lanes (the last n_private inputs)
    int n_pad = 0;            // n_in rounded up to 16 (zero weights beyond n_in)
    std::vector<float> l1_w, l1_b, out_w, out_b;
    int16_t *l1_q = nullptr;  // [buckets][hidden][n_pad], weight / feature scale * q_scale
    float q_inv = 0.0f;       // 1 / q_scale
    int8_t *private_rows = nullptr;  // [rows][n_private]
    int16_t private_bias[UHEAD_MAX_PRIVATE];
};
inline Uhead uhead;

inline uint64_t fnv1a64_file(const std::string &file, bool *ok) {
    std::ifstream ifs(file, std::ios::binary);
    *ok = (bool)ifs;
    uint64_t h = 1469598103934665603ULL;
    std::vector<char> buf(1 << 20);
    while (ifs) {
        ifs.read(buf.data(), buf.size());
        const std::streamsize n = ifs.gcount();
        for (std::streamsize i = 0; i < n; ++i) {
            h = (h ^ (uint8_t)buf[i]) * 1099511628211ULL;
        }
    }
    return h;
}

// main_file: the weight file of the loaded main network (a head belongs to one main weight file);
// expected_hash: FNV-1a 64 of the head file that the compiled MPC coefficients were fitted with (0: any)
inline bool init_uhead(const std::string &file, const std::string &main_file, uint64_t expected_hash, bool show_log) {
    uhead.loaded = false;
    std::ifstream ifs(file, std::ios::binary);
    if (!ifs) {
        std::cerr << "[ERROR] NNUE uncertainty head not found: " << file << std::endl;
        return false;
    }
    bool file_ok = false;
    if (expected_hash != 0 && (fnv1a64_file(file, &file_ok) != expected_hash || !file_ok)) {
        std::cerr << "[ERROR] NNUE uncertainty head is not the one the MPC coefficients were fitted with: " << file << std::endl;
        return false;
    }
    char magic[8];
    int32_t head[8];
    uint64_t main_hash = 0;
    ifs.read(magic, 8);
    ifs.read(reinterpret_cast<char*>(head), sizeof(head));
    ifs.read(reinterpret_cast<char*>(&main_hash), sizeof(main_hash));
    const int inputs = head[1], n_in = head[2], hidden = head[3];
    const int n_private = head[0] >= 2 ? head[7] : 0;
    const int W = weights.W, H = weights.H;
    const int expected_n_in = (inputs == 0 ? W / 2 : (inputs == 1 ? W / 2 + H : (inputs == 2 ? W + W / 2 + H : H))) + n_private;
    if (!ifs || std::memcmp(magic, "EGUHEAD1", 8) != 0 || head[0] < 1 || head[0] > 2 || inputs < 0 || inputs > 3 || hidden < 0 || hidden > 256 || hidden % 8 != 0 ||
        !(n_private == 0 || n_private == 8 || n_private == 16 || n_private == 32) || n_in > MAX_W + MAX_W / 2 + H_FIXED + UHEAD_MAX_PRIVATE || head[4] != W || head[5] != H || head[6] != N_BUCKETS || n_in != expected_n_in ||
        weights.mode != MODE_RESIDUAL || !weights.loaded) {
        std::cerr << "[ERROR] NNUE uncertainty head does not match the main network: " << file << std::endl;
        return false;
    }
    if (hidden == 0 && (inputs != 3 || n_private != 0)) {
        std::cerr << "[ERROR] NNUE uncertainty head: a linear output needs inputs z and no private lanes: " << file << std::endl;
        return false;
    }
    bool hash_ok = false;
    if (fnv1a64_file(main_file, &hash_ok) != main_hash || !hash_ok) {
        std::cerr << "[ERROR] NNUE uncertainty head belongs to another main weight file: " << file << std::endl;
        return false;
    }
    uhead.inputs = inputs;
    uhead.n_in = n_in;
    uhead.hidden = hidden;
    uhead.n_private = n_private;
    uhead.n_pad = (n_in + 15) / 16 * 16;
    uhead.l1_w.resize((size_t)N_BUCKETS * hidden * n_in);
    uhead.l1_b.resize((size_t)N_BUCKETS * hidden);
    uhead.out_w.resize((size_t)N_BUCKETS * (hidden ? hidden : n_in));
    uhead.out_b.resize(N_BUCKETS);
    if (!(read_array(ifs, uhead.l1_w.data(), uhead.l1_w.size()) && read_array(ifs, uhead.l1_b.data(), uhead.l1_b.size()) &&
          read_array(ifs, uhead.out_w.data(), uhead.out_w.size()) && read_array(ifs, uhead.out_b.data(), uhead.out_b.size()))) {
        std::cerr << "[ERROR] NNUE uncertainty head is truncated: " << file << std::endl;
        return false;
    }
    if (uhead.private_rows) {
        _mm_free(uhead.private_rows);
        uhead.private_rows = nullptr;
    }
    if (n_private) {
        uhead.private_rows = (int8_t*)aligned_alloc64((size_t)NNUE_N_ROWS * n_private + 64);
        if (!(read_array(ifs, uhead.private_rows, (size_t)NNUE_N_ROWS * n_private) && read_array(ifs, uhead.private_bias, n_private))) {
            std::cerr << "[ERROR] NNUE uncertainty head is truncated: " << file << std::endl;
            return false;
        }
    }
    if (hidden == 0) {
        // linear output: fold the feature scale (z: 1.0 = 128) into the weights
        for (float &x : uhead.out_w) x *= 1.0f / 128.0f;
        uhead.loaded = true;
        if (show_log) {
            std::cerr << "NNUE uncertainty output loaded " << file << " (linear on the last hidden layer)" << std::endl;
        }
        return true;
    }
    // integer features: a (1.0 = 512), x and z (1.0 = 128), private lanes (1.0 = 64)
    const int n_a = inputs == 2 ? W : 0;
    auto feature_scale = [&](int k) { return k < n_a ? 512.0 : (k < n_in - n_private ? 128.0 : 64.0); };
    double max_w = 0.0;
    for (size_t r = 0; r < (size_t)N_BUCKETS * hidden; ++r) {
        for (int k = 0; k < n_in; ++k) max_w = std::max(max_w, std::fabs(uhead.l1_w[r * n_in + k] / feature_scale(k)));
    }
    // largest |weight| q_max: |sum| <= n_in * 512 * q_max <= 2e9 < 2^31
    const double q_max = std::min(32000.0, 2.0e9 / (uhead.n_pad * 512.0));
    const double q_scale = max_w > 0.0 ? q_max / max_w : 1.0;
    if (uhead.l1_q) _mm_free(uhead.l1_q);
    const int n_pad = uhead.n_pad;
    uhead.l1_q = (int16_t*)aligned_alloc64((size_t)N_BUCKETS * hidden * n_pad * sizeof(int16_t));
    for (size_t r = 0; r < (size_t)N_BUCKETS * hidden; ++r) {
        for (int k = 0; k < n_pad; ++k) {
            uhead.l1_q[r * n_pad + k] = k < n_in ? (int16_t)std::lround(uhead.l1_w[r * n_in + k] / feature_scale(k) * q_scale) : 0;
        }
    }
    uhead.q_inv = (float)(1.0 / q_scale);
    uhead.loaded = true;
    if (show_log) {
        std::cerr << "NNUE uncertainty head loaded " << file << " inputs " << inputs << " hidden " << hidden << std::endl;
    }
    return true;
}

// idx: the position's 64 row indices (for the private lanes)
inline double uhead_u(const Trunk &t, const int idx[NNUE_N_INSTANCES], int bucket, int W) {
    if (uhead.hidden == 0) {
        // the second output of a dual network: linear on the last hidden layer z
        const float *w2 = uhead.out_w.data() + (size_t)bucket * uhead.n_in;
        float u = uhead.out_b[bucket];
        for (int j = 0; j < H_FIXED; ++j) u += w2[j] * t.z[j];
        return 16.0 * u;
    }
    // integer features in the head's input order (a, x, z, private lanes), int16, zero up to n_pad
    alignas(32) int16_t f[MAX_W + MAX_W / 2 + H_FIXED + UHEAD_MAX_PRIVATE + 16];
    int k = 0;
    if (uhead.inputs == 2) {
        std::memcpy(f, t.a, W * sizeof(int16_t));
        k = W;
    }
    std::memcpy(f + k, t.x, W / 2 * sizeof(int16_t));
    k += W / 2;
    if (uhead.inputs >= 1) {
        for (int j = 0; j < H_FIXED; ++j) f[k + j] = (int16_t)t.z[j];
        k += H_FIXED;
    }
    if (const int P = uhead.n_private) {
        // private lanes: int8 rows summed in int16 (64 * 127 fits), then bias and clamp to [0, 64]
        __m256i p0 = _mm256_setzero_si256(), p1 = _mm256_setzero_si256();
        for (int i = 0; i < NNUE_N_INSTANCES; ++i) {
            const int8_t *row = uhead.private_rows + (size_t)idx[i] * P;
            if (P == 8) {
                p0 = _mm256_add_epi16(p0, _mm256_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)row)));
            } else {
                p0 = _mm256_add_epi16(p0, _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)row)));
                if (P == 32) p1 = _mm256_add_epi16(p1, _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(row + 16))));
            }
        }
        alignas(32) int16_t ps[32];
        _mm256_store_si256((__m256i*)ps, p0);
        _mm256_store_si256((__m256i*)(ps + 16), p1);
        for (int j = 0; j < P; ++j) f[k + j] = (int16_t)std::clamp(ps[j] + uhead.private_bias[j], 0, 64);
        k += P;
    }
    for (; k < uhead.n_pad; ++k) f[k] = 0;
    const int n_in = uhead.n_pad, hidden = uhead.hidden;
    const int16_t *w1 = uhead.l1_q + (size_t)bucket * hidden * n_in;
    const float *b1 = uhead.l1_b.data() + (size_t)bucket * hidden;
    const float *w2 = uhead.out_w.data() + (size_t)bucket * hidden;
    const __m256 q_inv = _mm256_set1_ps(uhead.q_inv), zero = _mm256_setzero_ps(), one = _mm256_set1_ps(1.0f);
    __m256 o = _mm256_setzero_ps();
    for (int g = 0; g < hidden; g += 8) {  // 8 hidden units at a time keep the partial sums in registers
        __m256i s[8];
        for (int h = 0; h < 8; ++h) {
            const int16_t *row = w1 + (size_t)(g + h) * n_in;
            __m256i t = _mm256_madd_epi16(_mm256_load_si256((const __m256i*)row), _mm256_load_si256((const __m256i*)f));
            for (int j = 16; j < n_in; j += 16) t = _mm256_add_epi32(t, _mm256_madd_epi16(_mm256_load_si256((const __m256i*)(row + j)), _mm256_load_si256((const __m256i*)(f + j))));
            s[h] = t;
        }
        const __m256 z = _mm256_fmadd_ps(_mm256_cvtepi32_ps(hsum8_epi32(s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7])), q_inv, _mm256_loadu_ps(b1 + g));
        o = _mm256_fmadd_ps(_mm256_min_ps(_mm256_max_ps(z, zero), one), _mm256_loadu_ps(w2 + g), o);
    }
    __m128 r = _mm_add_ps(_mm256_castps256_ps128(o), _mm256_extractf128_ps(o, 1));
    r = _mm_add_ps(r, _mm_movehl_ps(r, r));
    r = _mm_add_ss(r, _mm_shuffle_ps(r, r, 1));
    return 16.0 * (_mm_cvtss_f32(r) + uhead.out_b[bucket]);
}

// total (as evaluate_total_idx, residual model) and the uncertainty head's u from one first-layer pass;
// false (and no u) for a network shape without the shared pass
inline bool evaluate_total_and_u_idx(const int idx[NNUE_N_INSTANCES], uint64_t player, int empties, int pattern_raw, int64_t *total_out, double *u, const Weights &w = weights) {
    const int bucket = bucket_of(empties);
    Trunk t;
    int64_t total;
    switch (w.W * 2 + (w.ft8 ? 1 : 0)) {
        case 65: total = forward_packed8<32>(idx, bucket, w, &t); break;
        case 129: total = forward_packed8<64>(idx, bucket, w, &t); break;
        case 257: total = forward_packed8<128>(idx, bucket, w, &t); break;
        case 513: total = forward_packed8<256>(idx, bucket, w, &t); break;
        case 64: total = forward_packed_trunk<32>(idx, bucket, &t, w); break;
        case 128: total = forward_packed_trunk<64>(idx, bucket, &t, w); break;
        case 256: total = forward_packed_trunk<128>(idx, bucket, &t, w); break;
        case 512: total = forward_packed_trunk<256>(idx, bucket, &t, w); break;
        default:
            *total_out = evaluate_total_idx(idx, player, empties, pattern_raw, w);
            return false;
    }
    *u = uhead_u(t, idx, bucket, w.W);
    *total_out = total + (int64_t)pattern_raw * 256;
    return true;
}

inline bool evaluate_total_and_u_lanes(const Lanes *l, uint64_t player, int empties, int pattern_raw, int64_t *total_out, double *u, const Weights &w = weights) {
    alignas(32) int idx[NNUE_N_INSTANCES];
    for (int v = 0; v < 8; ++v) {
        const __m128i lo = _mm_loadu_si128((const __m128i*)&l->u[8 * v]);
        const __m256i w32 = _mm256_add_epi32(_mm256_cvtepu16_epi32(lo), _mm256_loadu_si256((const __m256i*)&instance_offset[8 * v]));
        _mm256_store_si256((__m256i*)&idx[8 * v], w32);
    }
    return evaluate_total_and_u_idx(idx, player, empties, pattern_raw, total_out, u, w);
}
#endif

inline int score_of_total(int64_t total) {
    const int64_t s = total >= 0 ? (total + 4096) / 8192 : -((-total + 4096) / 8192);
    return (int)(s < -64 ? -64 : (s > 64 ? 64 : s));
}

inline int evaluate(uint64_t player, uint64_t opponent, int empties, int pattern_raw) {
    return score_of_total(evaluate_total(player, opponent, empties, pattern_raw));
}

}  // namespace nnue
