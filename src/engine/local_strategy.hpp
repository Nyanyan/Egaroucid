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

void calc_local_strategy(Board board, int level, double res[], bool show_log) {
    int value_diff[HW2];
    for (int cell = 0; cell < HW2; ++cell) {
        value_diff[cell] = 0;
    }
    Search_result complete_result = ai(board, level, true, 0, true, false);
    int n = 0;
    int diff_sum = 0;
    for (int cell = 0; cell < HW2; ++cell) {
        uint64_t bit = 1ULL << cell;
        if ((board.player | board.opponent) & bit) { // flip disc
            board.player ^= bit;
            board.opponent ^= bit;
                Search_result result = ai(board, level, true, 0, true, false);
                value_diff[cell] = std::abs(result.value - complete_result.value);
            board.player ^= bit;
            board.opponent ^= bit;
            ++n;
            diff_sum += value_diff[cell];
        } else {

        }
    }
    for (int cell = 0; cell < HW2; ++cell) {
        res[cell] = (double)value_diff[cell] / diff_sum;
    }
    print_local_strategy(res);
}

void print_local_strategy(const double arr[]) {
    for (int y = 0; y < HW; ++y) {
        for (int x = 0; x < HW; ++x) {
            int cell = HW2_M1 - (y * HW + x);
            std::cout << std::fixed << std::setprecision(2) << arr[cell] << " ";
        }
        std::cout << std::endl;
    }
}
