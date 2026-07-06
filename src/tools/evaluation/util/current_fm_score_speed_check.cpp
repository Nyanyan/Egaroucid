/*
    Egaroucid Project

    @file current_fm_score_speed_check.cpp
        Isolated speed check for current-model dual-type FM scoring
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

#include "../../../engine/common.hpp"
#include "../../../engine/evaluate_common.hpp"

#if defined(FM_SPEED_USE_SIMDOPT)
    #include "../../../engine/evaluate_experiment_current_fm_dual_type_simdopt_common.hpp"
#else
    #include "../../../engine/evaluate_experiment_current_fm_dual_type_common.hpp"
#endif

struct FmSpeedCase {
    int phase;
    int ids[CURRENT_FM_N_PATTERN_FEATURES + 1];
};

inline uint64_t now_ms() {
    return (uint64_t)std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()
    ).count();
}

inline int pattern_span(const int pattern_idx) {
    if (pattern_idx + 1 < N_PATTERNS) {
        return current_fm_pattern_offsets[pattern_idx + 1] - current_fm_pattern_offsets[pattern_idx];
    }
    return CURRENT_FM_N_PATTERN_PARAMS_RAW - current_fm_pattern_offsets[pattern_idx];
}

std::vector<FmSpeedCase> make_cases(const size_t n_cases) {
    std::mt19937 rng(20260706);
    std::vector<FmSpeedCase> cases(n_cases);
    for (FmSpeedCase &c : cases) {
        c.phase = (int)(rng() % N_PHASES);
        for (int i = 0; i < CURRENT_FM_N_PATTERN_FEATURES; ++i) {
            const int pattern_idx = i / 4;
            const int local = (int)(rng() % (uint32_t)pattern_span(pattern_idx));
            c.ids[i] = current_fm_pattern_offsets[pattern_idx] + local;
        }
        c.ids[CURRENT_FM_N_PATTERN_FEATURES] = CURRENT_FM_N_PATTERN_PARAMS_RAW + (int)(rng() % MAX_STONE_NUM);
    }
    return cases;
}

inline int score_linear_only_from_ids(const int phase_idx, const int active_ids[], const int n_active) {
    if (!current_fm_loaded || phase_idx < 0 || N_PHASES <= phase_idx) {
        return 0;
    }
    const size_t phase_base = (size_t)phase_idx * CURRENT_FM_N_PARAMS_PER_PHASE;
    const double linear_scale = current_fm_cross_file.linear_scale[phase_idx];
    int64_t linear_quant_sum = 0;
    for (int i = 0; i < n_active; ++i) {
        const int id = active_ids[i];
        if (id < 0 || CURRENT_FM_N_PARAMS_PER_PHASE <= id) {
            continue;
        }
        linear_quant_sum += current_fm_cross_file.linear_quant[phase_base + (size_t)id];
    }
    return std::clamp((int)std::llround((double)linear_quant_sum * linear_scale), -SCORE_MAX, SCORE_MAX);
}

template<typename ScoreFunc>
void run_benchmark(const char *name, const std::vector<FmSpeedCase> &cases, const uint64_t iterations, ScoreFunc score_func) {
    int64_t checksum = 0;
    size_t case_idx = 0;
    const uint64_t start = now_ms();
    for (uint64_t i = 0; i < iterations; ++i) {
        const FmSpeedCase &c = cases[case_idx];
        checksum += score_func(c.phase, c.ids, CURRENT_FM_N_PATTERN_FEATURES + 1);
        ++case_idx;
        if (case_idx == cases.size()) {
            case_idx = 0;
        }
    }
    const uint64_t elapsed = std::max<uint64_t>(1, now_ms() - start);
    const uint64_t nps = iterations * 1000ULL / elapsed;
    std::cout << name
              << " iterations=" << iterations
              << " elapsed_ms=" << elapsed
              << " nps=" << nps
              << " checksum=" << checksum
              << std::endl;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "usage: current_fm_score_speed_check <cross.egev4@same.egev4@cross_weight@same_weight> [iterations] [case_count]" << std::endl;
        return 1;
    }
    const std::string eval_spec = argv[1];
    const uint64_t iterations = argc >= 3 ? std::strtoull(argv[2], nullptr, 10) : 5000000ULL;
    const size_t case_count = argc >= 4 ? (size_t)std::strtoull(argv[3], nullptr, 10) : 65536ULL;
    if (!current_fm_load_egev4(eval_spec.c_str(), true)) {
        return 1;
    }
    std::vector<FmSpeedCase> cases = make_cases(case_count);
    run_benchmark("linear_only", cases, iterations, score_linear_only_from_ids);
    run_benchmark("dual_type_fm", cases, iterations, current_fm_score_from_ids);
    return 0;
}
