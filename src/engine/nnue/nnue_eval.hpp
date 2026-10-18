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
*/

#pragma once
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

#ifndef EGAROUCID_NNUE_SCOPE
// 1: midgame search leaves only (move ordering keeps the pattern evaluation), 2: every midgame evaluation
#define EGAROUCID_NNUE_SCOPE 1
#endif

namespace nnue {

constexpr int N_BUCKETS = 15;
constexpr int MAX_W = 256;
constexpr int H_FIXED = 32;
enum Mode { MODE_DIRECT = 0, MODE_RESIDUAL = 1, MODE_LINEAR = 2 };

struct Weights {
    bool loaded = false;
    int mode = MODE_DIRECT;
    int W = 0;
    int H = 0;
    int16_t *ft = nullptr;            // [rows][W]
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
};
inline Weights weights;
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

inline bool init(const std::string &file, bool show_log) {
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
    if (!ifs || std::memcmp(magic, "EGNNUE01", 8) != 0 || head[0] != 1) {
        std::cerr << "[ERROR] [FATAL] not an NNUE weight file: " << file << std::endl;
        return false;
    }
    Weights &w = weights;
    w.mode = head[1];
    w.W = head[4];
    w.H = head[5];
    if (head[2] != NNUE_N_ROWS || head[3] != NNUE_N_INSTANCES || head[6] != N_BUCKETS ||
        w.W % 32 != 0 || w.W > MAX_W || (w.mode != MODE_LINEAR && w.H != H_FIXED)) {
        std::cerr << "[ERROR] [FATAL] NNUE weight file does not match this build: " << file << std::endl;
        return false;
    }
    const size_t rows = NNUE_N_ROWS;
    const int W = w.W, H = w.H, half = W / 2;
    w.ft = (int16_t*)aligned_alloc64(rows * W * sizeof(int16_t) + 64);
    w.l1_w = (int16_t*)aligned_alloc64((size_t)N_BUCKETS * H_FIXED * (half > 0 ? half : 1) * sizeof(int16_t));
    w.lin = (int16_t*)aligned_alloc64((size_t)N_BUCKETS * rows * sizeof(int16_t));
    std::memset(w.l1_b, 0, sizeof(w.l1_b));
    std::memset(w.out_w, 0, sizeof(w.out_w));
    std::memset(w.out_b, 0, sizeof(w.out_b));
    bool ok = read_array(ifs, w.ft, rows * W) && read_array(ifs, w.ft_bias, W);
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
    w.packed = (int16_t*)aligned_alloc64(rows * w.stride * sizeof(int16_t) + 64);
    for (size_t r = 0; r < rows; ++r) {
        int16_t *dst = w.packed + r * w.stride;
        std::memcpy(dst, w.ft + r * W, W * sizeof(int16_t));
        if (w.has_lin) {
            for (int b = 0; b < 16; ++b) dst[W + b] = b < N_BUCKETS ? w.lin[(size_t)b * rows + r] : 0;
        }
    }
#if USE_SIMD
    init_incremental();
#endif
    w.loaded = true;
    if (show_log) {
        std::cerr << "NNUE loaded " << file << " mode " << w.mode << " width " << W << " scope " << EGAROUCID_NNUE_SCOPE << std::endl;
    }
    return true;
}

#ifndef EGAROUCID_NNUE_FILENAME
#define EGAROUCID_NNUE_FILENAME "nnue.egnn"
#endif

// path: $EGAROUCID_NNUE_FILE if set, otherwise EGAROUCID_NNUE_FILENAME next to the pattern evaluation file
inline std::string default_path(const std::string &eval_file) {
    if (const char *env = std::getenv("EGAROUCID_NNUE_FILE")) {
        if (env[0]) return env;
    }
    const size_t sep = eval_file.find_last_of("/\\");
    return (sep == std::string::npos ? std::string() : eval_file.substr(0, sep + 1)) + EGAROUCID_NNUE_FILENAME;
}

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

inline int64_t forward_scalar(const int idx[NNUE_N_INSTANCES], int bucket) {
    const Weights &w = weights;
    const int W = w.W, half = W / 2;
    int32_t acc[MAX_W];
    for (int j = 0; j < W; ++j) acc[j] = w.ft_bias[j];
    for (int i = 0; i < NNUE_N_INSTANCES; ++i) {
        const int16_t *row = w.ft + (size_t)idx[i] * W;
        for (int j = 0; j < W; ++j) acc[j] = (int16_t)(acc[j] + row[j]);  // int16 wrap like the vector path
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

template <int W, bool LIN>
inline int64_t forward_packed(const int idx[NNUE_N_INSTANCES], int bucket, uint64_t player) {
    const Weights &w = weights;
    constexpr int NV = W / 16;     // int16 vectors in the accumulator
    constexpr int HALF = W / 2;
    constexpr int NX = HALF / 16;  // int16 vectors in x
    constexpr int STRIDE = W + (LIN ? 16 : 0);
    __m256i acc[NV > 0 ? NV : 1];
    __m256i lin_lo = _mm256_setzero_si256(), lin_hi = _mm256_setzero_si256();
    for (int v = 0; v < NV; ++v) acc[v] = _mm256_load_si256((const __m256i*)&w.ft_bias[v * 16]);
    for (int i = 0; i < NNUE_N_INSTANCES; ++i) {
        const __m256i *row = (const __m256i*)(w.packed + (size_t)idx[i] * STRIDE);
        for (int v = 0; v < NV; ++v) acc[v] = _mm256_add_epi16(acc[v], _mm256_load_si256(row + v));
        if constexpr (LIN) {
            const __m256i l = _mm256_load_si256(row + NV);
            lin_lo = _mm256_add_epi32(lin_lo, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(l)));
            lin_hi = _mm256_add_epi32(lin_hi, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(l, 1)));
        }
    }
    int64_t total = 0;
    if constexpr (W > 0) {
        const __m256i zero = _mm256_setzero_si256(), c512 = _mm256_set1_epi16(512), c127 = _mm256_set1_epi16(127);
        __m256i x[NX > 0 ? NX : 1];
        for (int v = 0; v < NX; ++v) {
            const __m256i a = _mm256_min_epi16(_mm256_max_epi16(acc[v], zero), c512);
            const __m256i b = _mm256_min_epi16(_mm256_max_epi16(acc[v + NX], zero), c512);
            x[v] = _mm256_min_epi16(_mm256_mulhi_epi16(_mm256_slli_epi16(a, 5), b), c127);  // (a * b) >> 11
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
            o = _mm256_add_epi32(o, _mm256_mullo_epi32(zs, _mm256_loadu_si256((const __m256i*)&w.out_w[bucket][8 * g])));
        }
        __m128i o128 = _mm_add_epi32(_mm256_castsi256_si128(o), _mm256_extracti128_si256(o, 1));
        o128 = _mm_hadd_epi32(o128, o128);
        o128 = _mm_hadd_epi32(o128, o128);
        total += (int64_t)_mm_cvtsi128_si32(o128) + w.out_b[bucket];
    }
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
// instance, pattern-local index without the row offset). With digits 0 empty, 1 side to move,
// 2 opponent, a move by the side to move becomes, from the next side to move's view:
//     lanes += T(mover's discs) - T(opponent's unflipped discs) + 2 * T(placed square)
// (flipped discs stay digit 2), and a pass adds T(side to move) - T(opponent).
// T(bits) is the per-lane sum of 3^k over the instance cells in bits, read 8 squares at a time.
union Lanes {
    __m256i v[4];
    uint16_t u[64];
};
inline Lanes byte_table[8][256];
inline Lanes put_table[64];
inline int instance_offset[NNUE_N_INSTANCES];

inline void init_incremental() {
    static_assert(NNUE_N_INSTANCES == 64, "one 16-bit lane per instance");
    std::memset(byte_table, 0, sizeof(byte_table));
    std::memset(put_table, 0, sizeof(put_table));
    static uint16_t weight[64][NNUE_N_INSTANCES];  // [original bit][instance] -> 3^k or 0
    std::memset(weight, 0, sizeof(weight));
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
                weight[b][i] = (uint16_t)p3;
            }
        }
    }
    for (int g = 0; g < 8; ++g) {
        for (int byte = 0; byte < 256; ++byte) {
            for (int j = 0; j < 8; ++j) {
                if ((byte >> j) & 1) {
                    for (int i = 0; i < NNUE_N_INSTANCES; ++i) byte_table[g][byte].u[i] += weight[8 * g + j][i];
                }
            }
        }
    }
    for (int b = 0; b < 64; ++b) {
        for (int i = 0; i < NNUE_N_INSTANCES; ++i) put_table[b].u[i] = (uint16_t)(2 * weight[b][i]);
    }
}

inline void lanes_from_board(uint64_t player, uint64_t opponent, Lanes *dst) {
    int idx[NNUE_N_INSTANCES];
    compute_indices(player, opponent, idx);
    for (int i = 0; i < NNUE_N_INSTANCES; ++i) dst->u[i] = (uint16_t)(idx[i] - instance_offset[i]);
}

inline void lanes_move(const Lanes *src, Lanes *dst, uint64_t mover, uint64_t opponent_unflipped, int pos) {
    __m256i f0 = _mm256_add_epi16(src->v[0], put_table[pos].v[0]);
    __m256i f1 = _mm256_add_epi16(src->v[1], put_table[pos].v[1]);
    __m256i f2 = _mm256_add_epi16(src->v[2], put_table[pos].v[2]);
    __m256i f3 = _mm256_add_epi16(src->v[3], put_table[pos].v[3]);
    for (int g = 0; g < 8; ++g) {
        const Lanes &a = byte_table[g][(mover >> (8 * g)) & 0xFF];
        const Lanes &b = byte_table[g][(opponent_unflipped >> (8 * g)) & 0xFF];
        f0 = _mm256_sub_epi16(_mm256_add_epi16(f0, a.v[0]), b.v[0]);
        f1 = _mm256_sub_epi16(_mm256_add_epi16(f1, a.v[1]), b.v[1]);
        f2 = _mm256_sub_epi16(_mm256_add_epi16(f2, a.v[2]), b.v[2]);
        f3 = _mm256_sub_epi16(_mm256_add_epi16(f3, a.v[3]), b.v[3]);
    }
    dst->v[0] = f0; dst->v[1] = f1; dst->v[2] = f2; dst->v[3] = f3;
}

inline void lanes_pass(Lanes *l, uint64_t player, uint64_t opponent) {
    for (int g = 0; g < 8; ++g) {
        const Lanes &a = byte_table[g][(player >> (8 * g)) & 0xFF];
        const Lanes &b = byte_table[g][(opponent >> (8 * g)) & 0xFF];
        for (int v = 0; v < 4; ++v) l->v[v] = _mm256_sub_epi16(_mm256_add_epi16(l->v[v], a.v[v]), b.v[v]);
    }
}
#endif

// network part only (oacc), scalar reference
inline int64_t forward(const int idx[NNUE_N_INSTANCES], int bucket) {
    return forward_scalar(idx, bucket);
}

// total in 1/8192 discs; pattern_raw: production pattern evaluation in 1/32 discs (residual mode only)
inline int64_t evaluate_total_scalar(const int idx[NNUE_N_INSTANCES], uint64_t player, int empties, int pattern_raw) {
    const Weights &w = weights;
    const int bucket = bucket_of(empties);
    int64_t total = 0;
    if (w.mode != MODE_LINEAR) total += forward_scalar(idx, bucket);
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

inline int64_t evaluate_total_idx(const int idx[NNUE_N_INSTANCES], uint64_t player, int empties, int pattern_raw) {
#if USE_SIMD
    const Weights &w = weights;
    const int bucket = bucket_of(empties);
    int64_t total;
    switch (w.W * 2 + (w.has_lin ? 1 : 0)) {
        case 1: total = forward_packed<0, true>(idx, bucket, player); break;
        case 64: total = forward_packed<32, false>(idx, bucket, player); break;
        case 65: total = forward_packed<32, true>(idx, bucket, player); break;
        case 128: total = forward_packed<64, false>(idx, bucket, player); break;
        case 129: total = forward_packed<64, true>(idx, bucket, player); break;
        case 256: total = forward_packed<128, false>(idx, bucket, player); break;
        case 257: total = forward_packed<128, true>(idx, bucket, player); break;
        case 512: total = forward_packed<256, false>(idx, bucket, player); break;
        case 513: total = forward_packed<256, true>(idx, bucket, player); break;
        default: return evaluate_total_scalar(idx, player, empties, pattern_raw);
    }
    if (w.mode == MODE_RESIDUAL) total += (int64_t)pattern_raw * 256;
    return total;
#else
    return evaluate_total_scalar(idx, player, empties, pattern_raw);
#endif
}

inline int64_t evaluate_total(uint64_t player, uint64_t opponent, int empties, int pattern_raw) {
    int idx[NNUE_N_INSTANCES];
    compute_indices(player, opponent, idx);
    return evaluate_total_idx(idx, player, empties, pattern_raw);
}

#if USE_SIMD
inline int64_t evaluate_total_lanes(const Lanes *l, uint64_t player, int empties, int pattern_raw) {
    alignas(32) int idx[NNUE_N_INSTANCES];
    for (int v = 0; v < 8; ++v) {
        const __m128i lo = _mm_loadu_si128((const __m128i*)&l->u[8 * v]);
        const __m256i w32 = _mm256_add_epi32(_mm256_cvtepu16_epi32(lo), _mm256_loadu_si256((const __m256i*)&instance_offset[8 * v]));
        _mm256_store_si256((__m256i*)&idx[8 * v], w32);
    }
    return evaluate_total_idx(idx, player, empties, pattern_raw);
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
