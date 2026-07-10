/*
    Egaroucid Project

    @file eval77_fm_simdopt_score_speed_check.cpp
        Isolated score-speed check for the 7.7 beta + FM SIMDOPT evaluator.
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

#ifndef EVALUATE_EXPERIMENT_7_7_FM_SIMDOPT
    #define EVALUATE_EXPERIMENT_7_7_FM_SIMDOPT
#endif

#include "../../../engine/bit.hpp"
#include "../../../engine/board.hpp"
#include "../../../engine/common.hpp"
#include "../../../engine/evaluate.hpp"
#include "../../../engine/flip.hpp"
#include "../../../engine/last_flip.hpp"
#include "../../../engine/mobility.hpp"
#include "../../../engine/search.hpp"

struct Eval77FmSpeedCase {
    int phase;
    int ids[N_PATTERN_FEATURES + 1];
};

inline uint64_t eval77_now_ms() {
    return (uint64_t)std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()
    ).count();
}

std::vector<Eval77FmSpeedCase> eval77_make_cases(const size_t n_cases, const int phase_min, const int phase_max) {
    std::mt19937 rng(20260710);
    std::vector<Eval77FmSpeedCase> cases(n_cases);
    const int lo = std::clamp(phase_min, 0, N_PHASES - 1);
    const int hi = std::clamp(phase_max, lo, N_PHASES - 1);
    for (Eval77FmSpeedCase &c: cases) {
        c.phase = lo + (int)(rng() % (uint32_t)(hi - lo + 1));
        for (int feature = 0; feature < N_PATTERN_FEATURES; ++feature) {
            const int pattern_idx = feature / 4;
            const int local = (int)(rng() % (uint32_t)pow3[10]);
            c.ids[feature] = eval77_fm_pattern_offsets[pattern_idx] + local;
        }
        c.ids[N_PATTERN_FEATURES] = EVAL77_FM_N_PATTERN_PARAMS_RAW + (int)(rng() % MAX_STONE_NUM);
    }
    return cases;
}

inline int eval77_score_linear_only_from_case(const Eval77FmSpeedCase &c) {
    return std::clamp(
        (int)std::llround(eval77_fm_linear_from_ids(c.phase, c.ids, N_PATTERN_FEATURES + 1)),
        -SCORE_MAX,
        SCORE_MAX
    );
}

inline int eval77_score_fm_from_case(const Eval77FmSpeedCase &c) {
    return eval77_fm_score_from_ids(c.phase, c.ids, N_PATTERN_FEATURES + 1);
}

inline int eval77_score_fm_scalar_from_case(const Eval77FmSpeedCase &c) {
    const double score = eval77_fm_linear_from_ids(c.phase, c.ids, N_PATTERN_FEATURES + 1) +
        eval77_fm_interaction_from_ids_quant(c.phase, c.ids, N_PATTERN_FEATURES + 1);
    return std::clamp((int)std::llround(score), -SCORE_MAX, SCORE_MAX);
}

template<typename ScoreFunc>
void eval77_run_benchmark(
    const char *name,
    const std::vector<Eval77FmSpeedCase> &cases,
    const uint64_t iterations,
    ScoreFunc score_func
) {
    int64_t checksum = 0;
    size_t case_idx = 0;
    const uint64_t start = eval77_now_ms();
    for (uint64_t i = 0; i < iterations; ++i) {
        checksum += score_func(cases[case_idx]);
        ++case_idx;
        if (case_idx == cases.size()) {
            case_idx = 0;
        }
    }
    const uint64_t elapsed = std::max<uint64_t>(1, eval77_now_ms() - start);
    std::cout << name
              << " iterations=" << iterations
              << " elapsed_ms=" << elapsed
              << " nps=" << (iterations * 1000ULL / elapsed)
              << " checksum=" << checksum
              << std::endl;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "usage: eval77_fm_simdopt_score_speed_check <eval.egev4> [iterations] [case_count] [phase_min] [phase_max]" << std::endl;
        return 1;
    }
    const std::string eval_file = argv[1];
    const uint64_t iterations = argc >= 3 ? std::strtoull(argv[2], nullptr, 10) : 5000000ULL;
    const size_t case_count = argc >= 4 ? (size_t)std::strtoull(argv[3], nullptr, 10) : 65536ULL;
    const int phase_min = argc >= 5 ? std::atoi(argv[4]) : 0;
    const int phase_max = argc >= 6 ? std::atoi(argv[5]) : N_PHASES - 1;

    bit_init();
    mobility_init();
    flip_init();
    last_flip_init();
    if (!evaluate_init(eval_file, "bin/resources/eval_move_ordering_end.egev", true)) {
        return 1;
    }

    const std::vector<Eval77FmSpeedCase> cases = eval77_make_cases(case_count, phase_min, phase_max);
    eval77_run_benchmark("eval77_linear_only", cases, iterations, eval77_score_linear_only_from_case);
    eval77_run_benchmark("eval77_fm_scalar", cases, iterations, eval77_score_fm_scalar_from_case);
    eval77_run_benchmark("eval77_fm_selected", cases, iterations, eval77_score_fm_from_case);
    return 0;
}
