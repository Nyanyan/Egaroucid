/*
    Egaroucid Project

    @file edax_official_fm_consistency_check.cpp
        Consistency check for the isolated official-Edax + FM evaluation experiment.
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

inline int pick_nth_bit(uint64_t bits, int nth) {
    for (int pos = first_bit(&bits); bits; pos = next_bit(&bits)) {
        if (nth == 0) {
            return pos;
        }
        --nth;
    }
    return -1;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "usage: edax_official_fm_consistency_check <eval.dat[@fm.egev4]> [games] [seed]" << std::endl;
        return 1;
    }
    const std::string eval_file = argv[1];
    const int n_games = argc >= 3 ? std::atoi(argv[2]) : 1000;
    const uint32_t seed = argc >= 4 ? (uint32_t)std::strtoul(argv[3], nullptr, 10) : 20260710U;

    bit_init();
    mobility_init();
    flip_init();
    last_flip_init();
    if (!evaluate_init(eval_file, "", true)) {
        return 1;
    }

    std::mt19937 rng(seed);
    uint64_t checked = 0;
    uint64_t mismatches = 0;
    int max_abs_diff = 0;
    int min_score = SCORE_INF;
    int max_score = -SCORE_INF;
    int64_t checksum = 0;

    for (int game = 0; game < n_games; ++game) {
        Board board;
        board.reset();
        Search search(&board);
        bool previous_pass = false;
        while (search.board.n_discs() < HW2) {
            const int diff_score = mid_evaluate_diff(&search);
            const int fresh_score = mid_evaluate(&search.board);
            const int diff = std::abs(diff_score - fresh_score);
            if (diff != 0) {
                if (mismatches < 10) {
                    std::cerr << "mismatch game=" << game
                              << " discs=" << search.board.n_discs()
                              << " diff_score=" << diff_score
                              << " fresh_score=" << fresh_score
                              << " board=" << search.board.to_str()
                              << std::endl;
#if !USE_SIMD
                    Eval_search fresh_eval;
                    calc_eval_features(&search.board, &fresh_eval);
                    int feature_mismatches = 0;
                    for (int feature = 0; feature < N_PATTERN_FEATURES; ++feature) {
                        int incremental = search.eval.features[search.eval.feature_idx][feature];
                        if (search.eval.reversed[search.eval.feature_idx]) {
                            incremental = edax_official_swap_player_idx(incremental, pattern_sizes[feature_to_pattern[feature]]);
                        }
                        const int fresh = fresh_eval.features[fresh_eval.feature_idx][feature];
                        if (incremental != fresh) {
                            if (feature_mismatches < 6) {
                                std::cerr << "  feature_mismatch feature=" << feature
                                          << " incremental_as_fresh=" << incremental
                                          << " fresh=" << fresh
                                          << std::endl;
                            }
                            ++feature_mismatches;
                        }
                    }
                    std::cerr << "  feature_mismatches=" << feature_mismatches
                              << " reversed=" << (int)search.eval.reversed[search.eval.feature_idx]
                              << std::endl;
#endif
                }
                ++mismatches;
                max_abs_diff = std::max(max_abs_diff, diff);
            }
            min_score = std::min(min_score, diff_score);
            max_score = std::max(max_score, diff_score);
            checksum += diff_score;
            ++checked;

            uint64_t legal = search.board.get_legal();
            if (legal == 0ULL) {
                if (previous_pass) {
                    break;
                }
                search.pass();
                previous_pass = true;
                continue;
            }
            previous_pass = false;
            const int n_legal = pop_count_ull(legal);
            const int move = pick_nth_bit(legal, (int)(rng() % (uint32_t)n_legal));
            Flip flip;
            calc_flip(&flip, &search.board, move);
            search.move(&flip);
        }
    }

    std::cout << "games=" << n_games
              << " seed=" << seed
              << " checked_positions=" << checked
              << " mismatches=" << mismatches
              << " max_abs_diff=" << max_abs_diff
              << " min_score=" << min_score
              << " max_score=" << max_score
              << " checksum=" << checksum
              << std::endl;
    return mismatches == 0 ? 0 : 2;
}
