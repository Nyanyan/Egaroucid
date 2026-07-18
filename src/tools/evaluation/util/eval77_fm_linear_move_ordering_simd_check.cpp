/*
    Egaroucid Project

    @file eval77_fm_linear_move_ordering_simd_check.cpp
        Compare the SIMD and scalar implementations of the FM model's linear
        move-ordering score on random game positions.
    @date 2026
    @author Takuto Yamana, Codex
    @license GPL-3.0-or-later
*/

#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <string>

#ifndef EVALUATE_EXPERIMENT_7_7_FM_GROUPED
#define EVALUATE_EXPERIMENT_7_7_FM_GROUPED
#endif
#ifndef EVALUATE_EXPERIMENT_7_7_FM_LINEAR_MOVE_ORDERING
#define EVALUATE_EXPERIMENT_7_7_FM_LINEAR_MOVE_ORDERING
#endif
#ifndef EVALUATE_EXPERIMENT_7_7_FM_LINEAR_MOVE_ORDERING_SIMD
#define EVALUATE_EXPERIMENT_7_7_FM_LINEAR_MOVE_ORDERING_SIMD
#endif

#include "../../../engine/bit.hpp"
#include "../../../engine/board.hpp"
#include "../../../engine/common.hpp"
#include "../../../engine/evaluate.hpp"
#include "../../../engine/flip.hpp"
#include "../../../engine/last_flip.hpp"
#include "../../../engine/mobility.hpp"
#include "../../../engine/search.hpp"

inline int pick_nth_linear_check_bit(uint64_t bits, int nth) {
    for (int pos = first_bit(&bits); bits; pos = next_bit(&bits)) {
        if (nth-- == 0) {
            return pos;
        }
    }
    return -1;
}

int scalar_linear_move_ordering_score(Search *search) {
    const int phase_idx = search->phase();
    const int num0 = pop_count_ull(search->board.player);
    int active_ids[N_PATTERN_FEATURES + 1];
    int n_active = 0;
    collect_eval77_fm_fast_simd_active_ids(
        &search->eval.features[search->eval.feature_idx],
        num0,
        active_ids,
        n_active
    );
    const Eval77FmFastPhasePtrs phase_ptrs =
        eval77_fm_fast_phase_ptrs(phase_idx);
    int32_t linear_quant = 0;
    for (int i = 0; i < n_active; ++i) {
        int16_t value;
        std::memcpy(
            &value,
            eval77_fm_fast_linear_ptr(phase_ptrs, active_ids[i]),
            sizeof(value)
        );
        linear_quant += value;
    }
    return eval77_fm_fast_finish_linear_quant(phase_idx, linear_quant);
}

int main(int argc, char **argv) {
    const int games = argc >= 2 ? std::stoi(argv[1]) : 1000;
    const uint32_t seed = argc >= 3 ?
        static_cast<uint32_t>(std::stoul(argv[2])) : 20260739U;
    const std::string eval_file = argc >= 4 ?
        argv[3] :
        "model/eval77_fm_losslessbase54_pair54first_init/"
        "eval_losslessbase54_pair54first.egev10";
    const bool verbose = argc >= 5;

    bit_init();
    mobility_init();
    flip_init();
    last_flip_init();
    if (!evaluate_init(
            eval_file,
            "bin/resources/eval_move_ordering_end.egev",
            false)) {
        return 1;
    }

    std::mt19937 rng(seed);
    uint64_t checked = 0;
    for (int game = 0; game < games; ++game) {
        Board board;
        board.reset();
        Search search(&board);
        bool previous_pass = false;
        int iterations = 0;
        while (true) {
            if (++iterations > 200) {
                std::cerr
                    << "iteration_limit game=" << game
                    << " discs=" << search.board.n_discs()
                    << std::endl;
                return 3;
            }
            if (search.phase() >= N_PHASES) {
                break;
            }
            if (verbose) {
                std::cerr
                    << "checking game=" << game
                    << " iteration=" << iterations
                    << " discs=" << search.board.n_discs()
                    << std::endl;
            }
            const int simd_score =
                mid_evaluate_linear_move_ordering(&search);
            const int scalar_score =
                scalar_linear_move_ordering_score(&search);
            if (simd_score != scalar_score) {
                std::cerr
                    << "score_mismatch game=" << game
                    << " position=" << checked
                    << " discs=" << search.board.n_discs()
                    << " simd=" << simd_score
                    << " scalar=" << scalar_score
                    << std::endl;
                return 2;
            }
            ++checked;
            if (search.board.n_discs() >= HW2 - 1) {
                break;
            }
            const uint64_t legal = search.board.get_legal();
            if (legal == 0) {
                if (previous_pass) {
                    break;
                }
                search.pass();
                previous_pass = true;
                continue;
            }
            previous_pass = false;
            const int move = pick_nth_linear_check_bit(
                legal,
                static_cast<int>(
                    rng() % static_cast<uint32_t>(pop_count_ull(legal))
                )
            );
            Flip flip;
            calc_flip(&flip, &search.board, move);
            search.move(&flip);
        }
    }

    std::cout
        << "ok games=" << games
        << " positions=" << checked
        << " seed=" << seed
        << std::endl;
    return 0;
}
