/*
    Egaroucid Project

    @file edax_feature_coordinate_engine_coords_check.cpp
        Diagnostic for fixed Edax experiment feature-coordinate conversion
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

inline int edax_pick_pattern_old_converter_convention(const uint_fast8_t b_arr[], const int pattern_idx) {
    int res = 0;
    for (int i = 0; i < adj_feature_to_coord[pattern_idx].n_cells; ++i) {
        res *= 3;
        res += b_arr[adj_feature_to_coord[pattern_idx].cells[i]];
    }
    return res;
}

int main(int argc, char **argv) {
    const int n_boards = argc >= 2 ? std::atoi(argv[1]) : 10000;
    std::mt19937_64 rng(20260708);
    uint64_t old_mismatch = 0;
    uint64_t fixed_mismatch = 0;
    uint64_t total = 0;

    for (int board_idx = 0; board_idx < n_boards; ++board_idx) {
        const uint64_t a = rng();
        const uint64_t b = rng();
        Board board(a & ~b, b & ~a);
        uint_fast8_t b_arr[HW2];
        board.translate_to_arr_player(b_arr);
        for (int feature = 0; feature < ADJ_N_FEATURES; ++feature) {
            const int engine_idx = edax_pick_pattern_engine_convention(b_arr, feature);
            const int old_idx = edax_pick_pattern_old_converter_convention(b_arr, feature);
            const int fixed_idx = edax_pick_pattern_engine_convention(b_arr, feature);
            old_mismatch += old_idx != engine_idx;
            fixed_mismatch += fixed_idx != engine_idx;
            ++total;
        }
    }

    std::cout << "boards=" << n_boards
              << " features_checked=" << total
              << " old_mismatches=" << old_mismatch
              << " old_mismatch_rate=" << (total == 0 ? 0.0 : (double)old_mismatch / (double)total)
              << " fixed_mismatches=" << fixed_mismatch
              << " fixed_mismatch_rate=" << (total == 0 ? 0.0 : (double)fixed_mismatch / (double)total)
              << std::endl;
    return fixed_mismatch == 0 ? 0 : 2;
}
