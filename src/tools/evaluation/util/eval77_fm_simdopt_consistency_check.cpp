/*
    Egaroucid Project

    @file eval77_fm_simdopt_consistency_check.cpp
        Incremental-vs-fresh consistency check for the 7.7 beta + FM SIMDOPT evaluator.
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

#if !defined(EVALUATE_EXPERIMENT_7_7_FM_SIMDOPT) && !defined(EVALUATE_EXPERIMENT_7_7_FM_SIMDOPT_MMAP) && !defined(EVALUATE_EXPERIMENT_7_7_FM_SUBSET_SIMDOPT)
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

inline int eval77_pick_nth_bit(uint64_t bits, int nth) {
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
        std::cerr << "usage: eval77_fm_simdopt_consistency_check <eval.egev4> [games] [seed]" << std::endl;
        return 1;
    }
    const std::string eval_file = argv[1];
    const int n_games = argc >= 3 ? std::atoi(argv[2]) : 1000;
    const uint32_t seed = argc >= 4 ? (uint32_t)std::strtoul(argv[3], nullptr, 10) : 20260710U;

    bit_init();
    mobility_init();
    flip_init();
    last_flip_init();
    if (!evaluate_init(eval_file, "bin/resources/eval_move_ordering_end.egev", true)) {
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
                }
                ++mismatches;
                max_abs_diff = std::max(max_abs_diff, diff);
            }
            min_score = std::min(min_score, diff_score);
            max_score = std::max(max_score, diff_score);
            checksum += diff_score;
            ++checked;

            if (search.board.n_discs() >= HW2 - 1) {
                break;
            }

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
            const int move = eval77_pick_nth_bit(legal, (int)(rng() % (uint32_t)n_legal));
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
