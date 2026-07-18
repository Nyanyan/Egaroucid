/*
    Egaroucid Project

    @file eval77_fm_egev2_feature_compatibility_check.cpp
        Check whether the EGEV2 linear evaluator can use the FM evaluator's
        feature state after adding the EGEV2 global pattern offsets.
    @date 2026
    @author Takuto Yamana, Codex
    @license GPL-3.0-or-later
*/

#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <string>

#ifndef EVALUATE_EXPERIMENT_7_7_FM_HYBRID_78_RECOMPUTE
#define EVALUATE_EXPERIMENT_7_7_FM_HYBRID_78_RECOMPUTE
#endif

#include "../../../engine/bit.hpp"
#include "../../../engine/board.hpp"
#include "../../../engine/common.hpp"
#include "../../../engine/evaluate.hpp"
#include "../../../engine/flip.hpp"
#include "../../../engine/last_flip.hpp"
#include "../../../engine/mobility.hpp"

inline int pick_nth_compatibility_bit(uint64_t bits, int nth) {
    for (int pos = first_bit(&bits); bits; pos = next_bit(&bits)) {
        if (nth-- == 0) {
            return pos;
        }
    }
    return -1;
}

bool check_feature_compatibility(
    Board *board,
    const int game,
    const uint64_t position
) {
    Eval_features fm_features;
    Eval_features linear_features;
    Eval_features transformed_features;
    eval77_hybrid_recalculate_fm_features(board, &fm_features);
    eval77_hybrid_recalculate_linear_features(board, &linear_features);
    transformed_features = fm_features;
    transformed_features.f256[0] = _mm256_add_epi16(
        transformed_features.f256[0],
        eval78_recompute::eval_simd_offsets_simple[0]
    );
    transformed_features.f256[1] = _mm256_add_epi16(
        transformed_features.f256[1],
        eval78_recompute::eval_simd_offsets_simple[1]
    );

    if (std::memcmp(
            &transformed_features,
            &linear_features,
            sizeof(Eval_features)) == 0) {
        return true;
    }

    alignas(32) int16_t fm_values[N_PATTERN_FEATURES];
    alignas(32) int16_t transformed_values[N_PATTERN_FEATURES];
    alignas(32) int16_t linear_values[N_PATTERN_FEATURES];
    std::memcpy(fm_values, &fm_features, sizeof(Eval_features));
    std::memcpy(
        transformed_values,
        &transformed_features,
        sizeof(Eval_features)
    );
    std::memcpy(linear_values, &linear_features, sizeof(Eval_features));
    std::cerr
        << "feature_mismatch game=" << game
        << " position=" << position
        << " discs=" << board->n_discs()
        << std::endl;
    for (int lane = 0; lane < N_PATTERN_FEATURES; ++lane) {
        if (transformed_values[lane] != linear_values[lane]) {
            std::cerr
                << "lane=" << lane
                << " fm=" << fm_values[lane]
                << " transformed=" << transformed_values[lane]
                << " egev2=" << linear_values[lane]
                << std::endl;
        }
    }
    return false;
}

int main(int argc, char **argv) {
    const int games = argc >= 2 ? std::stoi(argv[1]) : 1000;
    const uint32_t seed = argc >= 3 ?
        static_cast<uint32_t>(std::stoul(argv[2])) : 20260737U;

    bit_init();
    mobility_init();
    flip_init();
    last_flip_init();
    const std::string spec =
        "bin/resources/eval.egev4@bin/resources/eval.egev2@"
        "fffffffffffffff";
    if (!evaluate_init(
            spec,
            "bin/resources/eval_move_ordering_end.egev",
            false)) {
        return 1;
    }

    std::mt19937 rng(seed);
    uint64_t checked = 0;
    for (int game = 0; game < games; ++game) {
        Board board;
        board.reset();
        bool previous_pass = false;
        while (true) {
            if (!check_feature_compatibility(&board, game, checked)) {
                return 2;
            }
            ++checked;
            const uint64_t legal = board.get_legal();
            if (legal == 0) {
                if (previous_pass) {
                    break;
                }
                board.pass();
                previous_pass = true;
                continue;
            }
            previous_pass = false;
            const int move = pick_nth_compatibility_bit(
                legal,
                static_cast<int>(
                    rng() % static_cast<uint32_t>(pop_count_ull(legal))
                )
            );
            Flip flip;
            calc_flip(&flip, &board, move);
            board.move_board(&flip);
        }
    }

    std::cout
        << "ok games=" << games
        << " positions=" << checked
        << " seed=" << seed
        << std::endl;
    return 0;
}
