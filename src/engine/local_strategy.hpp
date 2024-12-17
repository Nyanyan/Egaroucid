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
    constexpr double cell_weight[HW2][N_CELL_TYPE] = {
        2714,  147,   69,  -18,  -18,   69,  147, 2714, 
         147, -577, -186, -153, -153, -186, -577,  147, 
          69, -186, -379, -122, -122, -379, -186,   69, 
         -18, -153, -122, -169, -169, -122, -153,  -18, 
         -18, -153, -122, -169, -169, -122, -153,  -18, 
          69, -186, -379, -122, -122, -379, -186,   69, 
         147, -577, -186, -153, -153, -186, -577,  147, 
        2714,  147,   69,  -18,  -18,   69,  147, 2714
    };
    Search_result complete_result = ai_searching(board, level, true, 0, true, false, searching);
    if (show_log) {
        board.print();
        std::cerr << "complete result " << complete_result.value << std::endl;
    }
    uint64_t legal = board.get_legal();
    double value_diffs[HW2];
    for (int cell = 0; cell < HW2; ++cell) {
        value_diffs[cell] = 0;
    }
    for (int cell = 0; cell < HW2; ++cell) {
        uint64_t bit = 1ULL << cell;
        if (board.player & bit) { // player
            // player -> opponent
            board.player ^= bit;
            board.opponent ^= bit;
                Search_result result = ai_searching(board, level, true, 0, true, false, searching);
                value_diffs[cell] = (complete_result.value - result.value);
                //value_diffs[cell] = (complete_result.value - result.value) - 2.0 * cell_weight[cell] / 256.0;
                //std::cerr << idx_to_coord(cell) << " " << complete_result.value << " " << result.value << " " << 2.0 * cell_weight[cell] / 256.0 << " " << value_diffs[cell] << std::endl;
            board.player ^= bit;
            board.opponent ^= bit;
        } else if (board.opponent & bit) {
            // opponent -> player
            board.player ^= bit;
            board.opponent ^= bit;
                Search_result result = ai_searching(board, level, true, 0, true, false, searching);
                value_diffs[cell] = (complete_result.value - result.value);
                //value_diffs[cell] = (complete_result.value - result.value) + 2.0 * cell_weight[cell] / 256.0;
                //std::cerr << idx_to_coord(cell) << " " << complete_result.value << " " << result.value << " " << 2.0 * cell_weight[cell] / 256.0 << " " << value_diffs[cell] << std::endl;
                /*
                uint64_t legal_diff = ~board.get_legal() & legal;
                for (uint_fast8_t nolegal_cell = first_bit(&legal_diff); legal_diff; nolegal_cell = next_bit(&legal_diff)) {
                    value_diffs[nolegal_cell] = value_diffs[cell]; // nolegal_cell belongs to actual legal
                }
                */
            board.player ^= bit;
            board.opponent ^= bit;
        } else { // empty
            /*
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
            */
        }
    }
    /*
    double values[HW2];
    for (int cell = 0; cell < HW2; ++cell) {
        values[cell] = std::tanh(0.2 * value_diffs[cell]);
    }
    */
    if (show_log) {
        std::cerr << "value_diffs" << std::endl;
        print_local_strategy(value_diffs);
        //std::cerr << "values" << std::endl;
        //print_local_strategy(values);
        std::cerr << std::endl;
    }
    /*
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
    */
    for (int cell = 0; cell < HW2; ++cell) {
        res[cell] = std::tanh(0.2 * value_diffs[cell]); // 10 discs ~ 1.0
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


#if TUNE_LOCAL_STRATEGY

void tune_local_strategy() {
    int n = 100; // n per n_discs per cell_type
    int level = 10;

    double res[HW2][N_CELL_TYPE];
    int count[HW2][N_CELL_TYPE];

    for (int i = 0; i < HW2; ++i) {
        for (int j = 0; j < N_CELL_TYPES; ++j) {
            res[i][j] = 0.0;
            count[i][j] = 0;
        }
    }

    Board board;
    Flip flip;
    for (int n_discs = 4; n_discs <= HW2; ++n_discs) {
        std::cerr << '\r' << "                                                                   ";
        std::cerr << '\r' << "n_discs " << n_discs << " type ";
        for (int cell_type = 0; cell_type < N_CELL_TYPE; ++cell_type) {
            std::cerr << cell_type << " ";
            for (int i = 0; i < n; ++i) {
                board.reset();
                while (board.n_discs() < n_discs && board.check_pass()) {
                    uint64_t legal = board.get_legal();
                    int random_idx = myrandrange(0, pop_count_ull(legal));
                    int t = 0;
                    for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)) {
                        if (t == random_idx) {
                            calc_flip(&flip, &board, cell);
                            break;
                        }
                        ++t;
                    }
                    board.move_board(&flip);
                }
                if (board.check_pass()) {
                    if ((board.player | board.opponent) & cell_type_mask[cell_type]) {
                        uint64_t can_be_masked = (board.player | board.opponent) & cell_type_mask[cell_type];
                        int random_idx = myrandrange(0, pop_count_ull(can_be_masked));
                        int t = 0;
                        uint64_t flipped = 0;
                        for (uint_fast8_t cell = first_bit(&can_be_masked); can_be_masked; cell = next_bit(&can_be_masked)) {
                            if (t == random_idx) {
                                flipped = 1ULL << cell;
                                break;
                            }
                            ++t;
                        }
                        Search_result complete_result = ai(board, level, true, 0, true, false);
                        int sgn = -1; // opponent -> player
                        if (board.player & flipped) { // player -> opponent
                            sgn = 1;
                        }
                        board.player ^= flipped;
                        board.opponent ^= flipped;
                            Search_result flipped_result = ai(board, level, true, 0, true, false);
                        board.player ^= flipped;
                        board.opponent ^= flipped;
                        double diff = sgn * (complete_result.value - flipped_result.value);
                        res[n_discs][cell_type] += diff;
                        ++count[n_discs][cell_type];
                    }
                }
            }
            if (count[n_discs][cell_type]) {
                res[n_discs][cell_type] /= count[n_discs][cell_type];
            }
        }
        std::cerr << '\r';
        std::cout << "{";
        for (int j = 0; j < N_CELL_TYPE; ++j) {
            std::cout << res[n_discs][j];
            if (j < N_CELL_TYPE - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "}," << std::endl;

    }
    std::cerr << std::endl;

    for (int i = 0; i < HW2; ++i) {
        for (int j = 0; j < N_CELL_TYPE; ++j) {
            if (count[i][j]) {
                res[i][j] /= count[i][j];
            }
        }
    }

    std::cerr << "count" << std::endl;
    for (int i = 0; i < HW2; ++i) {
        std::cout << "{";
        for (int j = 0; j < N_CELL_TYPE; ++j) {
            std::cerr << count[i][j];
            if (j < N_CELL_TYPE - 1) {
                std::cerr << ", ";
            }
        }
        std::cout << "}," << std::endl;
    }

    for (int i = 0; i < HW2; ++i) {
        std::cout << "{";
        for (int j = 0; j < N_CELL_TYPE; ++j) {
            std::cout << res[i][j];
            if (j < N_CELL_TYPE - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "}," << std::endl;
    }
}

#endif
