/*
    Egaroucid Project

    @file edax_official_score_sample.cpp
        Score sampler for SIMD/Generic comparison of the isolated Edax official evaluator.
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

#ifndef EVALUATE_EXPERIMENT_EDAX_OFFICIAL
    #define EVALUATE_EXPERIMENT_EDAX_OFFICIAL
#endif

#include "../../../engine/bit.hpp"
#include "../../../engine/board.hpp"
#include "../../../engine/common.hpp"
#include "../../../engine/evaluate.hpp"
#include "../../../engine/flip.hpp"
#include "../../../engine/last_flip.hpp"
#include "../../../engine/mobility.hpp"
#include "../../../engine/search.hpp"

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "usage: edax_official_score_sample <eval.dat> [positions] [seed]" << std::endl;
        return 1;
    }
    const std::string eval_file = argv[1];
    const uint64_t n_positions = argc >= 3 ? std::strtoull(argv[2], nullptr, 10) : 1000000ULL;
    const uint32_t seed = argc >= 4 ? (uint32_t)std::strtoul(argv[3], nullptr, 10) : 20260710U;

    bit_init();
    mobility_init();
    flip_init();
    last_flip_init();
    if (!evaluate_init(eval_file, "", false)) {
        return 1;
    }

    std::mt19937_64 rng(seed);
    uint64_t generated = 0;
    int min_score = SCORE_INF;
    int max_score = -SCORE_INF;
    uint64_t min_clamped = 0;
    uint64_t max_clamped = 0;
    int64_t checksum = 0;
    while (generated < n_positions) {
        const uint64_t player = rng();
        const uint64_t opponent = rng() & ~player;
        Board board(player, opponent);
        const int n_discs = board.n_discs();
        if (n_discs < 4 || n_discs >= HW2) {
            continue;
        }
        const int score = mid_evaluate(&board);
        min_score = std::min(min_score, score);
        max_score = std::max(max_score, score);
        min_clamped += score == -SCORE_MAX;
        max_clamped += score == SCORE_MAX;
        checksum += score;
        ++generated;
    }

    std::cout << "positions=" << n_positions
              << " seed=" << seed
              << " min_score=" << min_score
              << " max_score=" << max_score
              << " min_clamped=" << min_clamped
              << " max_clamped=" << max_clamped
              << " checksum=" << checksum
              << std::endl;
    return 0;
}
