/*
    Egaroucid Project

    @file edax_official_fm_score_speed_check.cpp
        Isolated score-speed check for official-Edax linear plus FM vectors.
    @date 2026
    @author Takuto Yamana, Codex
    @license GPL-3.0-or-later
*/

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#ifndef EVALUATE_EXPERIMENT_EDAX_OFFICIAL_FM
    #define EVALUATE_EXPERIMENT_EDAX_OFFICIAL_FM
#endif

#include "../../../engine/bit.hpp"
#include "../../../engine/board.hpp"
#include "../../../engine/common.hpp"
#include "../../../engine/evaluate.hpp"
#include "../../../engine/flip.hpp"
#include "../../../engine/last_flip.hpp"
#include "../../../engine/mobility.hpp"
#include "../../../engine/search.hpp"

struct EdaxOfficialFmSpeedCase {
    int phase;
    uint16_t values[N_PATTERN_FEATURES];
    int ids[N_PATTERN_FEATURES];
};

inline uint64_t now_ms() {
    return (uint64_t)std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()
    ).count();
}

std::vector<EdaxOfficialFmSpeedCase> make_cases(const size_t n_cases) {
    std::mt19937 rng(20260710);
    std::vector<EdaxOfficialFmSpeedCase> cases(n_cases);
    for (EdaxOfficialFmSpeedCase &c: cases) {
        c.phase = (int)(rng() % N_PHASES);
        for (int feature = 0; feature < N_PATTERN_FEATURES; ++feature) {
            const int n_digits = pattern_sizes[feature_to_pattern[feature]];
            const int span = pow3[n_digits];
            c.values[feature] = (uint16_t)(rng() % (uint32_t)span);
            c.ids[feature] = edax_official_fm_active_id(feature, c.values[feature]);
        }
    }
    return cases;
}

inline int score_linear_only_from_case(const EdaxOfficialFmSpeedCase &c) {
#if USE_SIMD
    const int raw = edax_official_calc_raw(c.phase, 0, c.values);
#else
    const int raw = edax_official_calc_raw(c.phase, c.values, false);
#endif
    return edax_official_fm_round_score(raw, 0.0);
}

inline int score_fm_from_case(const EdaxOfficialFmSpeedCase &c) {
#if USE_SIMD
    const int raw = edax_official_calc_raw(c.phase, 0, c.values);
#else
    const int raw = edax_official_calc_raw(c.phase, c.values, false);
#endif
    return edax_official_fm_round_score(
        raw,
        edax_official_fm_interaction_from_ids(c.phase, c.ids, N_PATTERN_FEATURES)
    );
}

inline int score_fm_scalar_from_case(const EdaxOfficialFmSpeedCase &c) {
#if USE_SIMD
    const int raw = edax_official_calc_raw(c.phase, 0, c.values);
#else
    const int raw = edax_official_calc_raw(c.phase, c.values, false);
#endif
    return edax_official_fm_round_score(
        raw,
        edax_official_fm_interaction_from_ids_quant(c.phase, c.ids, N_PATTERN_FEATURES)
    );
}

template<typename ScoreFunc>
void run_benchmark(
    const char *name,
    const std::vector<EdaxOfficialFmSpeedCase> &cases,
    const uint64_t iterations,
    ScoreFunc score_func
) {
    int64_t checksum = 0;
    size_t case_idx = 0;
    const uint64_t start = now_ms();
    for (uint64_t i = 0; i < iterations; ++i) {
        checksum += score_func(cases[case_idx]);
        ++case_idx;
        if (case_idx == cases.size()) {
            case_idx = 0;
        }
    }
    const uint64_t elapsed = std::max<uint64_t>(1, now_ms() - start);
    std::cout << name
              << " iterations=" << iterations
              << " elapsed_ms=" << elapsed
              << " nps=" << (iterations * 1000ULL / elapsed)
              << " checksum=" << checksum
              << std::endl;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "usage: edax_official_fm_score_speed_check <eval.dat[@fm.egev4]> [iterations] [case_count]" << std::endl;
        return 1;
    }
    const std::string eval_spec = argv[1];
    const uint64_t iterations = argc >= 3 ? std::strtoull(argv[2], nullptr, 10) : 5000000ULL;
    const size_t case_count = argc >= 4 ? (size_t)std::strtoull(argv[3], nullptr, 10) : 65536ULL;

    bit_init();
    mobility_init();
    flip_init();
    last_flip_init();
    if (!evaluate_init(eval_spec, "", true)) {
        return 1;
    }
    const std::vector<EdaxOfficialFmSpeedCase> cases = make_cases(case_count);
    run_benchmark("official_linear_only", cases, iterations, score_linear_only_from_case);
    run_benchmark("official_fm_scalar", cases, iterations, score_fm_scalar_from_case);
    run_benchmark("official_fm_selected", cases, iterations, score_fm_from_case);
    return 0;
}
