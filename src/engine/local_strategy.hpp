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

void calc_local_strategy(Board board, int level, double res[], bool show_log) {
    double value_diff_player[HW2]; // something -> player
    double value_diff_opponent[HW2]; // something -> opponent
    double value_diff_empty[HW2]; // something -> empty
    double max_diffs[HW2];
    double min_diffs[HW2];
    for (int cell = 0; cell < HW2; ++cell) {
        value_diff_player[cell] = 0;
        value_diff_opponent[cell] = 0;
        value_diff_empty[cell] = 0;
        max_diffs[cell] = 0;
        min_diffs[cell] = 0;
    }
    Search_result complete_result = ai(board, level, true, 0, true, false);
    if (show_log) {
        std::cerr << "complete result " << complete_result.value << std::endl;
    }
    for (int cell = 0; cell < HW2; ++cell) {
        uint64_t bit = 1ULL << cell;
        if (board.player & bit) { // player
            // player -> opponent
            board.player ^= bit;
            board.opponent ^= bit;
                Search_result result1 = ai(board, level, true, 0, true, false);
                value_diff_opponent[cell] = std::max(0, result1.value - complete_result.value);
                //value_diff_opponent[cell] = std::abs(result1.value - complete_result.value);
                //value_diff_opponent[cell] *= (complete_result.policy != result1.policy);
            board.player ^= bit;
            board.opponent ^= bit;
            // player -> empty
            board.player ^= bit;
                Search_result result2 = ai(board, level, true, 0, true, false);
                value_diff_empty[cell] = std::max(0, result2.value - complete_result.value);
                //value_diff_empty[cell] = std::abs(result2.value - complete_result.value);
                //value_diff_empty[cell] *= (complete_result.policy != result2.policy);
            board.player ^= bit;
            max_diffs[cell] = std::max(value_diff_opponent[cell], value_diff_empty[cell]);
            min_diffs[cell] = std::min(value_diff_opponent[cell], value_diff_empty[cell]);
        } else if (board.opponent & bit) {
            // opponent -> player
            board.player ^= bit;
            board.opponent ^= bit;
                Search_result result1 = ai(board, level, true, 0, true, false);
                value_diff_player[cell] = std::max(0, result1.value - complete_result.value);
                //value_diff_player[cell] = std::abs(result1.value - complete_result.value);
                //value_diff_player[cell] *= (complete_result.policy != result1.policy);
            board.player ^= bit;
            board.opponent ^= bit;
            // opponent -> empty
            board.opponent ^= bit;
                Search_result result2 = ai(board, level, true, 0, true, false);
                value_diff_empty[cell] = std::max(0, result2.value - complete_result.value);
                //value_diff_empty[cell] = std::abs(result2.value - complete_result.value);
                //value_diff_empty[cell] *= (complete_result.policy != result2.policy);
            board.opponent ^= bit;
            max_diffs[cell] = std::max(value_diff_player[cell], value_diff_empty[cell]);
            min_diffs[cell] = std::min(value_diff_player[cell], value_diff_empty[cell]);
        } else { // empty
            if ((board.player | board.opponent) & bit_around[cell]) { // next to disc
                // empty -> player
                board.player ^= bit;
                    Search_result result1 = ai(board, level, true, 0, true, false);
                    value_diff_player[cell] = std::max(0, result1.value - complete_result.value);
                    //value_diff_player[cell] = std::abs(result1.value - complete_result.value);
                    //value_diff_player[cell] *= (complete_result.policy != result1.policy);
                board.player ^= bit;
                // empty -> opponent
                board.opponent ^= bit;
                    Search_result result2 = ai(board, level, true, 0, true, false);
                    value_diff_opponent[cell] = std::max(0, result2.value - complete_result.value);
                    //value_diff_opponent[cell] = std::abs(result2.value - complete_result.value);
                    //value_diff_opponent[cell] *= (complete_result.policy != result1.policy);
                board.opponent ^= bit;
                max_diffs[cell] = std::max(value_diff_player[cell], value_diff_opponent[cell]);
                min_diffs[cell] = std::min(value_diff_player[cell], value_diff_opponent[cell]);
            }
        }
    }
    if (show_log) {
        std::cerr << "?->player" << std::endl;
        print_local_strategy(value_diff_player);
        std::cerr << "?->opponent" << std::endl;
        print_local_strategy(value_diff_opponent);
        std::cerr << "?->empty" << std::endl;
        print_local_strategy(value_diff_empty);
        std::cerr << "max_diffs" << std::endl;
        print_local_strategy(max_diffs);
        std::cerr << "min_diffs" << std::endl;
        print_local_strategy(min_diffs);
        std::cerr << std::endl;
    }
    double max_max_diff = 0;
    double max_min_diff = 0;
    double sum_max_diff = 0;
    double sum_min_diff = 0;
    for (int cell = 0; cell < HW2; ++cell) {
        max_max_diff = std::max(max_max_diff, max_diffs[cell]);
        max_min_diff = std::max(max_min_diff, min_diffs[cell]);
        sum_max_diff += max_diffs[cell];
        sum_min_diff += min_diffs[cell];
    }
    double denominator = 0.0;
    for (int cell = 0; cell < HW2; ++cell) {
        res[cell] = (double)max_diffs[cell] / sum_max_diff;
        //res[cell] = (double)min_diffs[cell] / sum_min_diff;
        //res[cell] = std::exp(max_diffs[cell] - max_max_diff); // softmax
        //res[cell] = std::exp(min_diffs[cell] - max_min_diff); // softmax
        //denominator += res[cell];
    }
    /*
    for (int cell = 0; cell < HW2; ++cell) {
        res[cell] /= denominator;
    }
    */
    print_local_strategy(res);
}
