/*
    Egaroucid Project

    @file current_fm_tri_type_stream12_score_compare.cpp
        Direct consistency check for tri-type FM scalar scoring and streaming dim12 SIMD scoring
    @date 2026
    @author Takuto Yamana, Codex
    @license GPL-3.0-or-later
*/

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "../../../engine/common.hpp"
#include "../../../engine/evaluate_common.hpp"
#include "../../../engine/evaluate_experiment_current_fm_tri_type_stream12_common.hpp"

constexpr int STREAM12_COMPARE_N_PATTERN_FEATURES = 64;
constexpr int STREAM12_COMPARE_N_ACTIVE = STREAM12_COMPARE_N_PATTERN_FEATURES + 1;

struct FmCompareCase {
    int phase;
    int ids[STREAM12_COMPARE_N_ACTIVE];
};

inline int pattern_span(const int pattern_idx) {
    if (pattern_idx + 1 < N_PATTERNS) {
        return current_fm_pattern_offsets[pattern_idx + 1] - current_fm_pattern_offsets[pattern_idx];
    }
    return CURRENT_FM_N_PATTERN_PARAMS_RAW - current_fm_pattern_offsets[pattern_idx];
}

std::vector<FmCompareCase> make_cases(const uint64_t n_cases, const uint32_t seed) {
    std::mt19937 rng(seed);
    std::vector<FmCompareCase> cases(n_cases);
    for (FmCompareCase &c : cases) {
        c.phase = (int)(rng() % N_PHASES);
        for (int i = 0; i < STREAM12_COMPARE_N_PATTERN_FEATURES; ++i) {
            const int pattern_idx = i / 4;
            const int local = (int)(rng() % (uint32_t)pattern_span(pattern_idx));
            c.ids[i] = current_fm_pattern_offsets[pattern_idx] + local;
        }
        c.ids[STREAM12_COMPARE_N_PATTERN_FEATURES] =
            CURRENT_FM_N_PATTERN_PARAMS_RAW + (int)(rng() % MAX_STONE_NUM);
    }
    return cases;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "usage: current_fm_tri_type_stream12_score_compare <cross.egev4@same.egev4@count.egev4@cross_weight@same_weight@count_weight> [cases] [seed]" << std::endl;
        return 1;
    }
#if !USE_SIMD
    std::cerr << "this checker must be built with SIMD enabled" << std::endl;
    return 1;
#else
    const std::string eval_spec = argv[1];
    const uint64_t n_cases = argc >= 3 ? std::strtoull(argv[2], nullptr, 10) : 1000000ULL;
    const uint32_t seed = argc >= 4 ? (uint32_t)std::strtoul(argv[3], nullptr, 10) : 20260709U;

    if (!current_fm_load_egev4(eval_spec.c_str(), true)) {
        return 1;
    }
    if (current_fm_cross_file.dim != 12 || current_fm_same_file.dim != 12 || current_fm_count_file.dim != 12) {
        std::cerr << "stream12 checker expects all banks to be dim12, found dims="
                  << current_fm_cross_file.dim << ','
                  << current_fm_same_file.dim << ','
                  << current_fm_count_file.dim << std::endl;
        return 1;
    }

    const std::vector<FmCompareCase> cases = make_cases(n_cases, seed);
    uint64_t mismatches = 0;
    int max_abs_diff = 0;
    int64_t scalar_checksum = 0;
    int64_t stream12_checked_checksum = 0;
    int64_t stream12_unchecked_checksum = 0;
    int64_t stream12_idx8_checksum = 0;

    for (uint64_t i = 0; i < n_cases; ++i) {
        const FmCompareCase &c = cases[(size_t)i];
        __m256i idx8_groups[8];
        for (int group = 0; group < 8; ++group) {
            alignas(32) int values[8];
            for (int lane = 0; lane < 8; ++lane) {
                values[lane] = c.ids[group * 8 + lane] + 1;
            }
            idx8_groups[group] = _mm256_load_si256((const __m256i*)values);
        }
        const int scalar_score = current_fm_score_from_ids_quant(c.phase, c.ids, STREAM12_COMPARE_N_ACTIVE);
        const int stream12_checked_score = current_fm_score_from_ids_stream12(c.phase, c.ids, STREAM12_COMPARE_N_ACTIVE);
        const int stream12_unchecked_score = current_fm_score_from_ids_stream12_unchecked(c.phase, c.ids);
        const int stream12_idx8_score = current_fm_score_from_idx8_stream12_unchecked(
            c.phase, idx8_groups, c.ids[STREAM12_COMPARE_N_PATTERN_FEATURES]
        );
        scalar_checksum += scalar_score;
        stream12_checked_checksum += stream12_checked_score;
        stream12_unchecked_checksum += stream12_unchecked_score;
        stream12_idx8_checksum += stream12_idx8_score;

        const int diff = std::max(
            std::max(
                std::abs(scalar_score - stream12_checked_score),
                std::abs(scalar_score - stream12_unchecked_score)
            ),
            std::abs(scalar_score - stream12_idx8_score)
        );
        max_abs_diff = std::max(max_abs_diff, diff);
        if (diff != 0) {
            if (mismatches < 10) {
                std::cerr << "mismatch case=" << i
                          << " phase=" << c.phase
                          << " scalar=" << scalar_score
                          << " checked=" << stream12_checked_score
                          << " unchecked=" << stream12_unchecked_score
                          << " idx8=" << stream12_idx8_score
                          << " diff=" << diff << std::endl;
            }
            ++mismatches;
        }
    }

    std::cout << "cases=" << n_cases
              << " seed=" << seed
              << " mismatches=" << mismatches
              << " max_abs_diff=" << max_abs_diff
              << " scalar_checksum=" << scalar_checksum
              << " stream12_checked_checksum=" << stream12_checked_checksum
              << " stream12_unchecked_checksum=" << stream12_unchecked_checksum
              << " stream12_idx8_checksum=" << stream12_idx8_checksum
              << std::endl;
    return mismatches == 0 ? 0 : 2;
#endif
}
