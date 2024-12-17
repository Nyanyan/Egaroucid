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
    Search_result complete_result = ai_searching(board, level, true, 0, true, false, searching);
    if (show_log) {
        board.print();
        std::cerr << "complete result " << complete_result.value << std::endl;
    }
    uint64_t legal = board.get_legal();
    //uint64_t stability = calc_edge_stability_bits(&board); // edge stability
    //uint64_t stability = calc_stability_bits(&board); // stability
    uint64_t stability = 0;
    double value_diffs[HW2];
    for (int cell = 0; cell < HW2; ++cell) {
        value_diffs[cell] = 0;
    }
    for (int cell = 0; cell < HW2; ++cell) {
        uint64_t bit = 1ULL << cell;
        if (board.player & bit) { // player
            if ((stability & bit) == 0) {
                // player -> opponent
                board.player ^= bit;
                board.opponent ^= bit;
                    Search_result result = ai_searching(board, level, true, 0, true, false, searching);
                    value_diffs[cell] = complete_result.value - result.value;
                board.player ^= bit;
                board.opponent ^= bit;
            }
        } else if (board.opponent & bit) {
            if ((stability & bit) == 0) {
                // opponent -> player
                board.player ^= bit;
                board.opponent ^= bit;
                    Search_result result = ai_searching(board, level, true, 0, true, false, searching);
                    value_diffs[cell] = complete_result.value - result.value;
                    /*
                    uint64_t legal_diff = ~board.get_legal() & legal;
                    for (uint_fast8_t nolegal_cell = first_bit(&legal_diff); legal_diff; nolegal_cell = next_bit(&legal_diff)) {
                        value_diffs[nolegal_cell] = value_diffs[cell]; // nolegal_cell belongs to actual legal
                    }
                    */
                board.player ^= bit;
                board.opponent ^= bit;
            }
        } else { // empty
            if ((board.player | board.opponent) & bit_around[cell]) { // next to disc
                if ((legal & bit) == 0) { // can't put there
                    board.player ^= bit; // put there (no flip)
                    board.pass(); // opponent's move
                        int alpha = -SCORE_MAX; 
                        int beta = -complete_result.value; // just want to know better move than complete_result
                        Search_result result = ai_window_searching(board, alpha, beta, level, true, 0, true, false, searching);
                        int result_value = -result.value; // need -1 because it's opponent move
                        value_diffs[cell] = -std::max(0, result_value - complete_result.value); // need max because if the move is bad, just don't put it
                    board.pass();
                    board.player ^= bit;
                }
            }
        }
    }
    double values[HW2];
    for (int cell = 0; cell < HW2; ++cell) {
        values[cell] = std::tanh(0.2 * value_diffs[cell]);
    }
    if (show_log) {
        std::cerr << "value_diffs" << std::endl;
        print_local_strategy(value_diffs);
        std::cerr << "values" << std::endl;
        print_local_strategy(values);
        std::cerr << std::endl;
    }
    double max_abs_value = 0.0;
    for (int cell = 0; cell < HW2; ++cell) {
        max_abs_value = std::max(max_abs_value, std::abs(values[cell]));
    }
    double denominator = 0.0;
    for (int cell = 0; cell < HW2; ++cell) {
        if (max_abs_value > 0.0001) {
            res[cell] = values[cell] / max_abs_value;
        } else {
            res[cell] = 0.0;
        }
    }
    if (show_log) {
        print_local_strategy(res);
    }
}


void calc_local_strategy_player(Board board, int level, double res[], int player, bool *searching, bool show_log) {
    calc_local_strategy(board, level, res, searching, show_log);
    if (player == WHITE) {
        for (int cell = 0; cell < HW2; ++cell) {
            res[cell] *= -1;
        }
    }
}