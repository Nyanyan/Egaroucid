/*
    Egaroucid Project

    @file eval_patterns_check.cpp
        Check the NNUE pattern layout of the pattern evaluation (EGAROUCID_EVAL_NNUE_PATTERNS): incremental
        feature lanes (moves, passes, sibling bases, endgame-only updates) against lanes built from the board,
        and the SIMD pattern sums against a scalar sum over the tables
    @date 2026
    @author Takuto Yamana
    @license GPL-3.0-or-later

    Build: clang++ -O2 -march=native -std=c++20 -DEGAROUCID_EVAL_NNUE_PATTERNS src/tools/nnue/eval_patterns_check.cpp -o eval_patterns_check.exe
    Usage: eval_patterns_check <eval.egev2> <eval_move_ordering_end.egev> [games]
*/

#include <cstring>
#include <iostream>
#include <random>

#include "./../../engine/evaluate.hpp"
#include "./../../engine/search.hpp"

#ifndef EGAROUCID_EVAL_NNUE_PATTERNS
#error "build with -DEGAROUCID_EVAL_NNUE_PATTERNS"
#endif

static int n_bad = 0, n_checked = 0;

static void check_lanes(const Eval_features *f, const Board &b, const char *what, int lane_lo = 0, int lane_hi = 64) {
    uint16_t ref[64], got[64];
    eval_lanes_from_board(b.player, b.opponent, ref);
    std::memcpy(got, f, sizeof(got));
    ++n_checked;
    for (int lane = lane_lo; lane < lane_hi; ++lane) {
        if (ref[lane] != got[lane]) {
            if (n_bad < 10) std::cerr << what << ": lane " << lane << " got " << got[lane] << " want " << ref[lane] << std::endl;
            ++n_bad;
            return;
        }
    }
}

static int scalar_pattern(int phase, const Board &b) {
    uint16_t lanes[64];
    eval_lanes_from_board(b.player, b.opponent, lanes);
    alignas(32) int32_t off[64];
    for (int k = 0; k < 8; ++k) _mm256_storeu_si256((__m256i*)(off + 8 * k), eval_lane_offset_simd[k]);
    int s = 0;
    for (int lane = 0; lane < 64; ++lane) s += pattern_arr[phase][off[lane] + lanes[lane]] - SIMD_EVAL_MAX_VALUE;
    return s;
}

static int scalar_mo_end(const Board &b) {
    uint16_t lanes[64];
    eval_lanes_from_board(b.player, b.opponent, lanes);
    alignas(32) int32_t off[16];
    _mm256_storeu_si256((__m256i*)off, eval_lane_offset_mo_end_simd[0]);
    _mm256_storeu_si256((__m256i*)(off + 8), eval_lane_offset_mo_end_simd[1]);
    int s = 0;
    for (int k = 0; k < 16; ++k) s += pattern_move_ordering_end_arr[off[k] + lanes[32 + k]] - SIMD_EVAL_MAX_VALUE_MO_END;
    return s;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        std::cerr << "usage: eval_patterns_check <eval.egev2> <eval_move_ordering_end.egev> [games]" << std::endl;
        return 1;
    }
    bit_init();
    mobility_init();
    flip_init();
    eval_move_ordering_auto_requested = false;  // the ordering models are plain tables of the same layout
    if (!evaluate_init(std::string(argv[1]), std::string(argv[2]), false)) return 1;
    const int n_games = argc >= 4 ? std::atoi(argv[3]) : 2000;
    std::mt19937_64 rng(20261021);
    int bad_pattern = 0, bad_mo_end = 0, n_values = 0;
    for (int g = 0; g < n_games; ++g) {
        Board b;
        b.reset();
        Eval_search ev;
        calc_eval_features(&b, &ev);
        Eval_search ev_end = ev;  // endgame-only updates (lanes 32-47)
        while (true) {
            if (ev.feature_idx >= HW2 - 4 - 2) {  // a whole game is longer than the feature stack: rebuild
                calc_eval_features(&b, &ev);
                ev_end = ev;
            }
            check_lanes(&ev.features[ev.feature_idx], b, "move");
            check_lanes(&ev_end.features[ev_end.feature_idx], b, "endsearch", 32, 48);
            const int phase = b.n_discs() - 4;
            if (phase < N_PHASES) {
                ++n_values;
                if (calc_pattern(phase, &ev.features[ev.feature_idx]) != scalar_pattern(phase, b)) ++bad_pattern;
                if (calc_pattern_move_ordering_end(&ev.features[ev.feature_idx]) != scalar_mo_end(b)) ++bad_mo_end;
            }
            uint64_t legal = b.get_legal();
            if (!legal) {
                Board t = b;
                t.pass();
                if (!t.get_legal()) break;
                eval_pass(&ev, &b);
                eval_pass_endsearch(&ev_end, &b);
                b.pass();
                continue;
            }
            // every child through eval_move, the sibling base and the endgame-only paths
            Eval_features sibling_base;
            eval_prepare_move_sibling_base(&ev, &b, &sibling_base);
            __m256i end_base;
            eval_prepare_move_endsearch_sibling_base(&ev_end, &b, &end_base);
            std::vector<Flip> moves;
            for (uint64_t l = legal; l; l &= l - 1) {
                Flip flip;
                flip.calc_flip(b.player, b.opponent, (uint_fast8_t)__builtin_ctzll(l));
                moves.push_back(flip);
                Board c = b;
                c.move_board(&flip);
                Eval_search e1 = ev;
                eval_move_from_sibling_base(&e1, &flip, &sibling_base);
                check_lanes(&e1.features[e1.feature_idx], c, "sibling");
                Eval_search e2 = ev_end;
                eval_move_endsearch_from_sibling_base(&e2, &flip, &end_base);
                check_lanes(&e2.features[e2.feature_idx], c, "endsearch sibling", 32, 48);
            }
            const Flip &flip = moves[rng() % moves.size()];
            eval_move(&ev, &flip, &b);
            eval_move_endsearch(&ev_end, &flip, &b);
            b.move_board(&flip);
        }
    }
    std::cout << "lane checks " << n_checked << " mismatches " << n_bad << " | values " << n_values
              << " pattern mismatches " << bad_pattern << " move ordering end mismatches " << bad_mo_end << std::endl;
    return n_bad || bad_pattern || bad_mo_end ? 1 : 0;
}
