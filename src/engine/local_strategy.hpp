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

constexpr double local_strategy_cell_weight[HW2_P1][N_CELL_TYPE] = {
    {0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 35.3860},
    {0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.9020, 21.0220},
    {0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 64.0000, 20.5070, 16.9640},
    {0.0000, 0.0000, 0.0000, 0.0000, 0.0000, -13.5921, 0.0000, -0.9307, -0.2610, 2.2170},
    {0.0000, 0.0000, 0.0000, 0.0000, 36.2143, -9.2581, 11.7791, 11.7674, 1.8760, 1.7650},
    {0.0000, -20.0000, -10.4444, 0.8421, -48.0364, -8.8548, -8.7290, -2.6907, -2.4760, -0.5190},
    {43.7500, -10.5714, 3.0556, 6.6000, -39.8737, -8.7437, -4.9110, -0.0612, -2.1250, -1.5570},
    {41.6000, -19.5000, 1.8605, -1.5000, -46.9281, -9.9832, -8.0231, -3.7193, -3.0490, -1.6620},
    {45.7778, -18.9000, 0.2586, -0.1143, -43.2944, -9.3027, -7.3422, -2.9195, -2.3860, -2.0350},
    {45.3478, -19.6857, -1.5748, -2.2500, -43.5533, -9.3612, -7.9646, -3.9869, -3.0940, -1.7127},
    {45.7632, -17.3070, -2.5464, -1.9220, -40.6646, -9.2375, -7.7400, -3.4582, -3.4330, -2.0280},
    {43.3902, -19.5875, -0.6425, -2.9129, -39.6268, -8.9183, -8.0612, -3.9213, -3.1772, -2.1792},
    {41.9412, -18.3054, -0.0806, -3.1751, -39.2948, -9.1364, -7.6299, -3.6446, -3.4975, -2.4220},
    {41.2022, -18.0415, -0.8289, -2.9828, -37.5992, -8.8076, -7.4736, -3.2621, -2.9270, -2.3560},
    {41.8542, -16.3105, -1.4244, -3.0622, -36.2691, -8.4120, -7.0310, -3.5887, -3.0120, -1.9190},
    {38.1089, -16.1379, -0.7133, -2.9390, -35.5654, -8.6481, -6.7105, -3.2885, -2.9550, -1.6430},
    {36.4578, -15.4610, -1.9824, -3.6298, -34.1426, -8.1130, -6.6434, -3.2209, -3.1490, -1.8190},
    {38.6576, -15.0203, -1.5627, -2.7743, -32.6191, -8.2371, -6.7245, -3.1306, -3.1411, -1.6690},
    {37.6412, -14.7873, -1.6799, -3.1248, -31.2655, -8.3163, -6.6018, -3.2101, -3.0860, -1.8468},
    {35.2248, -13.6768, -1.2429, -3.1980, -30.3311, -7.7462, -6.3970, -2.7452, -2.9099, -1.7350},
    {34.7974, -12.2182, -1.5704, -2.9760, -28.0975, -7.7503, -6.1825, -2.9709, -2.7050, -1.5120},
    {35.1975, -12.4918, -0.8090, -1.8640, -27.3923, -7.4855, -5.9620, -2.6419, -2.6030, -1.4920},
    {34.5735, -13.1752, -0.7908, -2.9545, -26.5946, -7.3407, -6.0582, -2.4885, -2.5245, -1.3133},
    {32.6114, -11.2426, -1.8237, -2.4955, -24.5064, -6.9690, -5.6182, -2.3962, -2.3984, -1.2242},
    {32.3363, -11.1334, -2.1149, -2.4895, -24.5064, -7.0390, -5.4970, -2.2725, -2.4610, -1.2040},
    {32.0000, -10.6630, -0.7202, -1.8502, -20.7834, -6.3280, -5.1932, -2.4266, -2.4454, -1.0270},
    {30.5926, -10.1055, -1.1270, -2.0456, -22.2437, -6.7440, -4.9258, -2.1520, -2.1321, -1.0020},
    {29.3009, -9.3584, -1.2132, -2.0609, -19.6927, -6.2533, -4.7387, -1.8358, -2.2750, -0.6496},
    {29.4569, -8.8725, -0.6927, -1.2495, -19.9066, -6.0370, -4.7610, -1.9180, -2.0310, -1.1320},
    {29.0768, -8.3167, -1.4532, -1.1828, -18.5728, -5.5876, -4.1860, -1.8468, -1.9890, -0.7508},
    {27.6703, -8.1128, -1.1343, -1.2223, -17.1459, -5.3363, -4.3830, -1.9097, -1.8248, -0.7020},
    {27.1541, -6.5143, -1.3032, -1.1256, -16.9855, -5.4565, -4.1540, -1.6930, -1.7487, -0.9279},
    {25.8451, -6.5659, -0.9113, -0.8063, -15.9699, -5.2250, -3.7538, -1.5310, -1.3800, -0.5860},
    {26.2974, -6.8002, -0.5298, -0.6616, -14.8595, -4.8590, -3.7948, -1.5105, -1.3433, -0.5475},
    {25.2647, -6.2925, -1.0773, -0.5375, -14.4410, -4.6363, -3.9210, -1.6540, -1.2763, -0.4250},
    {23.5926, -4.7128, -1.0654, -0.8417, -13.1194, -4.4690, -3.7020, -1.5796, -1.1722, -0.3303},
    {23.5879, -4.9716, -0.5432, -0.1934, -12.6755, -4.9090, -3.4850, -1.6980, -1.3834, -0.1353},
    {22.0628, -4.1643, -0.1600, -0.5938, -12.2838, -4.0160, -3.2823, -1.3387, -1.1800, -0.2340},
    {21.4495, -4.8313, -0.4990, -0.4775, -11.7871, -3.7460, -3.1742, -1.3357, -1.1530, -0.2270},
    {22.2895, -4.0755, -0.3009, -0.1802, -10.3045, -3.6950, -2.6270, -0.8880, -1.0501, -0.3133},
    {20.6312, -3.7137, -0.2140, 0.3872, -10.0121, -3.8240, -2.8680, -1.2820, -0.8440, 0.0762},
    {20.4072, -2.8048, 0.1922, 0.4790, -9.8995, -3.6920, -2.7548, -0.9570, -0.9189, 0.1040},
    {19.5103, -2.9049, 0.3226, 0.1280, -8.6166, -3.4068, -2.3980, -1.1171, -0.8120, 0.2020},
    {18.4127, -2.1441, 0.3660, 0.4560, -7.7960, -3.1860, -2.1840, -0.7788, -0.5920, 0.4660},
    {18.7218, -1.2112, 0.4280, 0.6727, -7.8637, -3.1371, -1.9680, -1.1892, -0.5926, -0.0240},
    {17.6415, -1.2020, 0.0882, 0.8809, -7.2618, -2.8500, -2.0220, -0.7540, -0.4064, 0.2700},
    {17.3812, -1.2960, 0.5920, 0.4220, -6.4224, -2.4404, -1.7760, -0.8860, -0.4340, 0.1443},
    {16.5105, -1.1371, 0.5385, 1.2866, -6.4950, -2.3223, -1.7137, -0.6607, -0.4460, 0.1542},
    {15.6963, -0.5285, 0.5640, 1.7618, -5.4955, -2.0220, -1.2465, -0.3664, -0.1440, 0.2280},
    {14.7206, -0.0740, 0.8898, 1.1000, -4.9469, -2.0260, -1.2580, -0.1620, -0.1900, 0.1760},
    {14.0204, 0.2182, 0.9940, 0.9710, -4.4660, -1.5960, -0.8620, -0.5180, 0.0581, 0.4004},
    {13.4248, 0.9449, 1.4014, 1.8176, -3.9800, -0.9240, -0.4660, -0.4340, 0.0840, 0.3707},
    {12.6298, 1.2212, 1.4880, 1.0500, -3.3660, -0.9960, -0.8328, 0.1500, 0.3123, 0.6847},
    {10.6888, 1.0030, 1.9360, 1.3173, -2.5760, -0.3203, -0.3440, -0.0720, 0.4480, 0.7508},
    {9.7146, 1.3680, 1.5856, 1.5740, -1.8058, -0.3507, 0.0220, 0.1181, 0.7460, 0.9349},
    {8.8689, 1.7337, 1.5291, 2.0320, -1.6096, 0.6760, 0.3924, 0.2823, 0.7788, 0.7920},
    {8.0341, 1.7660, 1.5776, 1.9600, -0.6980, 0.8096, 0.5271, 1.1214, 0.9690, 0.9280},
    {6.3520, 2.3200, 2.0801, 1.7820, 0.3864, 0.7920, 0.9540, 0.7880, 1.1400, 1.4810},
    {4.6060, 1.9980, 1.8176, 1.8979, 2.0120, 1.4810, 1.9118, 1.3206, 1.5912, 1.2800},
    {2.9212, 2.1996, 2.1636, 1.9758, 5.8869, 2.1623, 2.2402, 1.9496, 1.5911, 1.9310},
    {2.0000, 2.0000, 2.0000, 2.0000, 2.0000, 2.0000, 2.0000, 2.0000, 2.0000, 2.0000}
};

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
                value_diffs[cell] = -(result.value - complete_result.value + local_strategy_cell_weight[board.n_discs()][cell_type[cell]]);
                std::cerr << idx_to_coord(cell) << " " << complete_result.value << " " << result.value << " " << local_strategy_cell_weight[board.n_discs()][cell_type[cell]] << " " << value_diffs[cell] << std::endl;
            board.player ^= bit;
            board.opponent ^= bit;
        } else if (board.opponent & bit) {
            // opponent -> player
            board.player ^= bit;
            board.opponent ^= bit;
                Search_result result = ai_searching(board, level, true, 0, true, false, searching);
                value_diffs[cell] = -(result.value - complete_result.value - local_strategy_cell_weight[board.n_discs()][cell_type[cell]]);
                std::cerr << idx_to_coord(cell) << " " << complete_result.value << " " << result.value << " " << local_strategy_cell_weight[board.n_discs()][cell_type[cell]] << " " << value_diffs[cell] << std::endl;
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
    int n_max = 10000000; // n per n_discs per cell_type
    int n_min = 10000;
    int level = 10;

    double res[HW2_P1][N_CELL_TYPE];
    int count[HW2_P1][N_CELL_TYPE];

    for (int i = 0; i < HW2; ++i) {
        for (int j = 0; j < N_CELL_TYPE; ++j) {
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
            for (int i = 0; i < n_max && count[n_discs][cell_type] < n_min; ++i) {
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
                //if (board.check_pass()) {
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
                //}
            }
            if (count[n_discs][cell_type]) {
                res[n_discs][cell_type] /= count[n_discs][cell_type];
            }
        }
        std::cerr << '\r' << n_discs << " ";
        for (int j = 0; j < N_CELL_TYPE; ++j) {
            std::cerr << count[n_discs][j] << " ";
        }
        std::cerr << std::endl;
        std::cout << "{";
        for (int j = 0; j < N_CELL_TYPE; ++j) {
            std::cout << std::fixed << std::setprecision(4) << res[n_discs][j];
            if (j < N_CELL_TYPE - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "}," << std::endl;
    }
    std::cerr << std::endl;

    std::cerr << "count" << std::endl;
    for (int i = 0; i <= HW2; ++i) {
        std::cout << "{";
        for (int j = 0; j < N_CELL_TYPE; ++j) {
            std::cerr << count[i][j];
            if (j < N_CELL_TYPE - 1) {
                std::cerr << ", ";
            }
        }
        std::cout << "}," << std::endl;
    }
    std::cerr << "res" << std::endl;
    for (int i = 0; i <= HW2; ++i) {
        std::cout << "{";
        for (int j = 0; j < N_CELL_TYPE; ++j) {
            std::cout << std::fixed << std::setprecision(4) << res[i][j];
            if (j < N_CELL_TYPE - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "}," << std::endl;
    }
}

#endif
