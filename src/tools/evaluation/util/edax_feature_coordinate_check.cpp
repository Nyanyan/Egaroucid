/*
    Egaroucid Project

    @file edax_feature_coordinate_check.cpp
        Diagnostic for Edax experiment feature-coordinate conventions
    @date 2026
    @author Takuto Yamana, Codex
    @license GPL-3.0-or-later
*/

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>

#include "../evaluation_definition_experiment_edax_linear.hpp"

inline int edax_pick_pattern_engine_convention(const uint_fast8_t b_arr[], const int pattern_idx) {
    int res = 0;
    for (int i = 0; i < adj_feature_to_coord[pattern_idx].n_cells; ++i) {
        res *= 3;
        res += b_arr[HW2 - 1 - adj_feature_to_coord[pattern_idx].cells[i]];
    }
    return res;
}

inline int edax_pick_pattern_converter_convention(const uint_fast8_t b_arr[], const int pattern_idx) {
    int res = 0;
    for (int i = 0; i < adj_feature_to_coord[pattern_idx].n_cells; ++i) {
        res *= 3;
        res += b_arr[adj_feature_to_coord[pattern_idx].cells[i]];
    }
    return res;
}

int main(int argc, char **argv) {
    const int n_boards = argc >= 2 ? std::atoi(argv[1]) : 10000;
    std::mt19937_64 rng(20260706);
    uint64_t mismatch = 0;
    uint64_t total = 0;
    int first_board = -1;
    int first_feature = -1;
    int first_converter = -1;
    int first_engine = -1;

    for (int board_idx = 0; board_idx < n_boards; ++board_idx) {
        const uint64_t a = rng();
        const uint64_t b = rng();
        Board board(a & ~b, b & ~a);
        uint_fast8_t b_arr[HW2];
        board.translate_to_arr_player(b_arr);
        for (int feature = 0; feature < ADJ_N_FEATURES; ++feature) {
            const int converter_idx = edax_pick_pattern_converter_convention(b_arr, feature);
            const int engine_idx = edax_pick_pattern_engine_convention(b_arr, feature);
            if (converter_idx != engine_idx) {
                ++mismatch;
                if (first_board < 0) {
                    first_board = board_idx;
                    first_feature = feature;
                    first_converter = converter_idx;
                    first_engine = engine_idx;
                }
            }
            ++total;
        }
    }

    std::cout << "boards=" << n_boards
              << " features_checked=" << total
              << " mismatches=" << mismatch
              << " mismatch_rate=" << (total == 0 ? 0.0 : (double)mismatch / (double)total)
              << std::endl;
    if (first_board >= 0) {
        std::cout << "first_mismatch board=" << first_board
                  << " feature=" << first_feature
                  << " converter_idx=" << first_converter
                  << " engine_idx=" << first_engine
                  << std::endl;
    }
    return mismatch == 0 ? 0 : 2;
}
