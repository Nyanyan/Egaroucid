/*
    Egaroucid Project

    @file eval77_hybrid78_state_consistency_check.cpp
        Check the incremental EGEV2 feature state used by the 7.7-FM/EGEV2
        hybrid evaluator against feature recalculation from the board.
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
#include "../../../engine/search.hpp"

inline int pick_nth_bit(uint64_t bits, int nth) {
    for (int pos = first_bit(&bits); bits; pos = next_bit(&bits)) {
        if (nth-- == 0) {
            return pos;
        }
    }
    return -1;
}

bool hybrid78_state_matches(
    Search *search,
    const char *operation,
    int game
) {
    Eval_features recalculated_feature;
    const int phase_idx = search->phase();
#if defined(EVALUATE_EXPERIMENT_7_7_FM_EGEV2_MOVE_ORDERING)
    const bool uses_fm = true;
#else
    const bool uses_fm = eval77_hybrid_uses_fm(phase_idx);
#endif
    if (uses_fm) {
        Eval_search recalculated;
        eval77_hybrid_fm_calc_eval_features(
            &search->board,
            &recalculated
        );
        recalculated_feature = recalculated.features[0];
    } else {
        Eval78HybridSearchState recalculated{};
        eval78_recompute::calc_eval_features(
            &search->board,
            &recalculated
        );
        recalculated_feature = recalculated.features[0];
    }
    if (std::memcmp(
            &search->eval.features[search->eval.feature_idx],
            &recalculated_feature,
            sizeof(Eval_features)) != 0) {
        alignas(32) int16_t incremental_values[N_PATTERN_FEATURES];
        alignas(32) int16_t recalculated_values[N_PATTERN_FEATURES];
        std::memcpy(
            incremental_values,
            &search->eval.features[search->eval.feature_idx],
            sizeof(Eval_features)
        );
        std::memcpy(
            recalculated_values,
            &recalculated_feature,
            sizeof(Eval_features)
        );
        std::cerr
            << "feature_mismatch game=" << game
            << " discs=" << search->board.n_discs()
            << " operation=" << operation
            << " incremental_index="
            << static_cast<int>(search->eval.feature_idx)
            << " evaluator=" << (uses_fm ? "fm" : "egev2")
            << std::endl;
        for (int i = 0; i < N_PATTERN_FEATURES; ++i) {
            if (incremental_values[i] != recalculated_values[i]) {
                std::cerr
                    << "lane=" << i
                    << " incremental=" << incremental_values[i]
                    << " recalculated=" << recalculated_values[i]
                    << std::endl;
            }
        }
        return false;
    }
#if defined(EVALUATE_EXPERIMENT_7_7_FM_EGEV2_MOVE_ORDERING)
    const Eval_features &incremental_linear_features =
        search->eval.eval77_egev2_move_ordering_features
            [search->eval.feature_idx];
    Eval_features recalculated_linear_features;
    eval77_hybrid_recalculate_linear_features(
        &search->board,
        &recalculated_linear_features
    );
    if (std::memcmp(
            &incremental_linear_features,
            &recalculated_linear_features,
            sizeof(Eval_features)) != 0) {
        alignas(32) int16_t incremental_values[N_PATTERN_FEATURES];
        alignas(32) int16_t recalculated_values[N_PATTERN_FEATURES];
        std::memcpy(
            incremental_values,
            &incremental_linear_features,
            sizeof(Eval_features)
        );
        std::memcpy(
            recalculated_values,
            &recalculated_linear_features,
            sizeof(Eval_features)
        );
        std::cerr
            << "move_ordering_feature_mismatch game=" << game
            << " discs=" << search->board.n_discs()
            << " operation=" << operation
            << " incremental_index="
            << static_cast<int>(search->eval.feature_idx)
            << std::endl;
        for (int i = 0; i < N_PATTERN_FEATURES; ++i) {
            if (incremental_values[i] != recalculated_values[i]) {
                std::cerr
                    << "lane=" << i
                    << " incremental=" << incremental_values[i]
                    << " recalculated=" << recalculated_values[i]
                    << std::endl;
            }
        }
        return false;
    }
    const int incremental_linear_score =
        mid_evaluate_linear_move_ordering(search);
    const int recalculated_linear_score =
        eval78_recompute::mid_evaluate(&search->board);
    if (incremental_linear_score != recalculated_linear_score) {
        std::cerr
            << "move_ordering_score_mismatch game=" << game
            << " discs=" << search->board.n_discs()
            << " operation=" << operation
            << " incremental=" << incremental_linear_score
            << " recalculated=" << recalculated_linear_score
            << std::endl;
        return false;
    }
#endif
    if (phase_idx >= N_PHASES) {
        return true;
    }
    const int num0 = pop_count_ull(search->board.player);
    const int incremental_score = uses_fm ?
        eval77_hybrid_fm_calc_pattern(
            phase_idx,
            &search->eval.features[search->eval.feature_idx],
            num0
        ) :
        eval77_hybrid_linear_score(
            phase_idx,
            num0,
            &search->eval.features[search->eval.feature_idx]
        );
    int recalculated_score;
    if (uses_fm) {
        recalculated_score = eval77_hybrid_fm_calc_pattern(
            phase_idx,
            &recalculated_feature,
            num0
        );
    } else {
        recalculated_score =
            eval78_recompute::mid_evaluate(&search->board);
    }
    if (incremental_score != recalculated_score) {
        std::cerr
            << "score_mismatch game=" << game
            << " discs=" << search->board.n_discs()
            << " operation=" << operation
            << " incremental=" << incremental_score
            << " recalculated=" << recalculated_score
            << std::endl;
        return false;
    }
    return true;
}

int main(int argc, char **argv) {
    const int games = argc >= 2 ? std::stoi(argv[1]) : 1000;
    const uint32_t seed = argc >= 3 ?
        static_cast<uint32_t>(std::stoul(argv[2])) : 20260731U;
    const std::string phase_mask = argc >= 4 ? argv[3] : "0";

    bit_init();
    mobility_init();
    flip_init();
    last_flip_init();
    const std::string spec =
        "bin/resources/eval.egev4@bin/resources/eval.egev2@" +
        phase_mask;
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
        Search search(&board);
        if (!hybrid78_state_matches(&search, "initial", game)) {
            return 2;
        }
        bool previous_pass = false;
        int iterations = 0;
        while (search.board.n_discs() < HW2 - 1) {
            if (++iterations > 200) {
                std::cerr
                    << "iteration_limit game=" << game
                    << " discs=" << search.board.n_discs()
                    << std::endl;
                return 3;
            }
            const uint64_t legal = search.board.get_legal();
            if (legal == 0) {
                if (previous_pass) {
                    break;
                }
                search.pass();
                if (!hybrid78_state_matches(&search, "pass", game)) {
                    return 2;
                }
                previous_pass = true;
                continue;
            }
            previous_pass = false;
            const int move = pick_nth_bit(
                legal,
                static_cast<int>(
                    rng() % static_cast<uint32_t>(pop_count_ull(legal))
                )
            );
            Flip flip;
            calc_flip(&flip, &search.board, move);
            const Board board_before = search.board.copy();
            const Eval_features features_before =
                search.eval.features[search.eval.feature_idx];

            search.move(&flip);
            if (!hybrid78_state_matches(&search, "move", game)) {
                return 2;
            }
            search.undo(&flip);
            if (search.board != board_before ||
                std::memcmp(
                    &search.eval.features[search.eval.feature_idx],
                    &features_before,
                    sizeof(Eval_features)) != 0 ||
                !hybrid78_state_matches(&search, "undo", game)) {
                std::cerr
                    << "undo_mismatch game=" << game
                    << " discs=" << search.board.n_discs()
                    << std::endl;
                return 2;
            }
            search.move(&flip);
            ++checked;
        }
    }
    std::cout
        << "games=" << games
        << " checked_moves=" << checked
        << " phase_mask=0x" << phase_mask
        << " mismatches=0"
        << std::endl;
    return 0;
}
