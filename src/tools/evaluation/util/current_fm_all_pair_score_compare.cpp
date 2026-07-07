/*
    Egaroucid Project

    @file current_fm_all_pair_score_compare.cpp
        Direct consistency check for current-model all-pair FM scalar and SIMD scoring
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
#include "../../../engine/evaluate_experiment_current_fm_simdopt_common.hpp"

constexpr int CURRENT_FM_SCORE_COMPARE_N_PATTERN_FEATURES = 64;
constexpr int CURRENT_FM_SCORE_COMPARE_N_ACTIVE =
    CURRENT_FM_SCORE_COMPARE_N_PATTERN_FEATURES + 1;

struct FmCompareCase {
    int phase;
    int ids[CURRENT_FM_SCORE_COMPARE_N_ACTIVE];
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
        for (int i = 0; i < CURRENT_FM_SCORE_COMPARE_N_PATTERN_FEATURES; ++i) {
            const int pattern_idx = i / 4;
            const int local = (int)(rng() % (uint32_t)pattern_span(pattern_idx));
            c.ids[i] = current_fm_pattern_offsets[pattern_idx] + local;
        }
        c.ids[CURRENT_FM_SCORE_COMPARE_N_PATTERN_FEATURES] =
            CURRENT_FM_N_PATTERN_PARAMS_RAW + (int)(rng() % MAX_STONE_NUM);
    }
    return cases;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "usage: current_fm_all_pair_score_compare <eval.egev4> [cases] [seed]" << std::endl;
        return 1;
    }
#if !USE_SIMD
    std::cerr << "this checker must be built with SIMD enabled" << std::endl;
    return 1;
#else
    const std::string eval_file = argv[1];
    const uint64_t n_cases = argc >= 3 ? std::strtoull(argv[2], nullptr, 10) : 1000000ULL;
    const uint32_t seed = argc >= 4 ? (uint32_t)std::strtoul(argv[3], nullptr, 10) : 20260707U;

    if (!current_fm_load_egev4(eval_file.c_str(), true)) {
        return 1;
    }
    if (current_fm_dim != 16) {
        std::cerr << "SIMD direct check currently expects dim=16, found dim=" << current_fm_dim << std::endl;
        return 1;
    }

    const std::vector<FmCompareCase> cases = make_cases(n_cases, seed);
    uint64_t mismatches = 0;
    int max_abs_diff = 0;
    int64_t scalar_checksum = 0;
    int64_t simd_checksum = 0;

    for (uint64_t i = 0; i < n_cases; ++i) {
        const FmCompareCase &c = cases[(size_t)i];
        const int scalar_score = current_fm_score_from_ids_quant(
            c.phase, c.ids, CURRENT_FM_SCORE_COMPARE_N_ACTIVE);
        const int simd_score = current_fm_score_from_ids_simd16(
            c.phase, c.ids, CURRENT_FM_SCORE_COMPARE_N_ACTIVE);
        scalar_checksum += scalar_score;
        simd_checksum += simd_score;

        const int diff = std::abs(scalar_score - simd_score);
        max_abs_diff = std::max(max_abs_diff, diff);
        if (diff != 0) {
            if (mismatches < 10) {
                std::cerr << "mismatch case=" << i
                          << " phase=" << c.phase
                          << " scalar=" << scalar_score
                          << " simd=" << simd_score
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
              << " simd_checksum=" << simd_checksum
              << std::endl;
    return mismatches == 0 ? 0 : 2;
#endif
}
