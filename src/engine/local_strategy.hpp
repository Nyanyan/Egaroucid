/*
    Egaroucid Project

    @file local_strategy.hpp
        Calculate Local Strategy by flipping some discs
    @date 2021-2024
    @author Takuto Yamana
    @license GPL-3.0 license
*/

#pragma once
#include "ai.hpp"

void print_local_strategy(const double arr[]) {
    for (int y = 0; y < HW; ++y) {
        for (int x = 0; x < HW; ++x) {
            int cell = HW2_M1 - (y * HW + x);
            std::cout << std::fixed << std::setprecision(3) << arr[cell] << " ";
        }
        std::cout << std::endl;
    }
}

void calc_local_strategy(Board board, int level, double res[], bool *searching, bool show_log) {
    double value_diffs[HW2];
    for (int cell = 0; cell < HW2; ++cell) {
        value_diffs[cell] = 0;
    }
    Search_result complete_result = ai_searching(board, level, true, 0, true, false, searching);
    if (show_log) {
        board.print();
        std::cerr << "complete result " << complete_result.value << std::endl;
    }
    uint64_t legal = board.get_legal();
    uint64_t stability = calc_stability_bits(&board);
    for (int cell = 0; cell < HW2; ++cell) {
        uint64_t bit = 1ULL << cell;
        if (board.player & bit) { // player
            if ((stability & bit) == 0) {
                // player -> opponent
                board.player ^= bit;
                board.opponent ^= bit;
                    Search_result result = ai_searching(board, level, true, 0, true, false, searching);
                    value_diffs[cell] = std::max(0, complete_result.value - result.value);
                board.player ^= bit;
                board.opponent ^= bit;
            }
        } else if (board.opponent & bit) {
            if ((stability & bit) == 0) {
                // opponent -> player
                board.player ^= bit;
                board.opponent ^= bit;
                    Search_result result = ai_searching(board, level, true, 0, true, false, searching);
                    value_diffs[cell] = std::max(0, complete_result.value - result.value);
                board.player ^= bit;
                board.opponent ^= bit;
            }
        } else { // empty
            if ((board.player | board.opponent) & bit_around[cell]) { // next to disc
                if ((legal & bit) == 0) { // can't put there
                    board.player ^= bit; // put there (no flip)
                    board.pass(); // opponent's move
                        Search_result result = ai_searching(board, level, true, 0, true, false, searching);
                        value_diffs[cell] = std::max(0, complete_result.value - result.value);
                    board.pass();
                    board.player ^= bit;
                }
            }
        }
    }
    if (show_log) {
        std::cerr << "value_diffs" << std::endl;
        print_local_strategy(value_diffs);
        std::cerr << std::endl;
    }
    double max_value_diff = 0;
    double sum_value_diff = 0;
    for (int cell = 0; cell < HW2; ++cell) {
        max_value_diff = std::max(max_value_diff, value_diffs[cell]);
        sum_value_diff += value_diffs[cell];
    }
    double denominator = 0.0;
    for (int cell = 0; cell < HW2; ++cell) {
        res[cell] = (double)value_diffs[cell] / max_value_diff;
    }
    print_local_strategy(res);
}
