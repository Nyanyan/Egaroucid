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

constexpr double local_strategy_cell_weight[HW2][N_CELL_TYPE] = {
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0.3721},
    {0, 0, 0, 0, 0, 0, 0, 0, 0.005, 0.2128},
    {0, 0, 0, 0, 0, 0, 0, 2.06452, 0.1893, 0.1788},
    {0, 0, 0, 0, 0, -0.553819, 0, 0.00367309, -0.0223, 0.0181},
    {0, 0, 0, 0, 18.5, -0.353469, 0.308594, 0.229867, 0.0385, 0.0154},
    {0, 0, -3.22222, 5, -11.16, -0.213563, -0.272222, -0.0128, -0.0207, -0.0029},
    {45, 0, -1, 4.92, -3.21939, -0.179209, -0.138683, 0.00442907, -0.0128, -0.0165},
    {0, -6.4375, 0.333333, -0.333333, -2.90306, -0.1341, -0.124734, -0.044765, -0.0217, -0.0168},
    {33, -12, 0.413223, -0.15625, -1.32507, -0.143424, -0.11585, -0.0476996, -0.0229, -0.019},
    {0, -1.90123, 0.609375, -0.191358, -1.89876, -0.109021, -0.0960555, -0.0491358, -0.0305, -0.026},
    {11.8125, -1.47449, 0.244444, -0.0237812, -1.39126, -0.123583, -0.0882566, -0.0520778, -0.0343, -0.0257},
    {6.97222, -0.935185, -0.00619835, -0.0505051, -0.860677, -0.100054, -0.090931, -0.038155, -0.0384, -0.0236},
    {6.52778, -0.66482, -0.075, -0.0995918, -0.941701, -0.0946481, -0.0887964, -0.0335734, -0.0376, -0.0223},
    {4.61728, -0.575556, 0.104938, -0.0574921, -0.753649, -0.0889757, -0.0742382, -0.0384738, -0.0306, -0.0228},
    {4.57, -0.444598, -0.0623583, -0.0272972, -0.718954, -0.0955024, -0.0712992, -0.0377968, -0.0333, -0.0267},
    {4.20661, -0.429388, -0.0387397, -0.0636, -0.576458, -0.0845832, -0.076735, -0.0272803, -0.0318, -0.0288},
    {2.36, -0.412809, -0.0641975, -0.0544218, -0.475104, -0.0837, -0.0752811, -0.0297793, -0.0308, -0.0254},
    {2.73373, -0.283396, 0.00924745, -0.0421029, -0.58695, -0.0787, -0.077033, -0.0359225, -0.0249, -0.0177},
    {2.12889, -0.22963, -0.0362663, -0.0524584, -0.407769, -0.0858076, -0.069, -0.026464, -0.0274, -0.0141},
    {1.38134, -0.25079, -0.0343994, -0.0293221, -0.410584, -0.0751964, -0.0632, -0.0272421, -0.0308, -0.0157},
    {1.6525, -0.23365, -0.0528047, -0.0351285, -0.337059, -0.0733, -0.0633, -0.0171, -0.0323, -0.0258},
    {1.59722, -0.163111, -0.0270312, -0.0281969, -0.348781, -0.0686, -0.0544, -0.0234, -0.0212, -0.0204061},
    {1.05762, -0.214111, -0.0379747, -0.0412346, -0.2975, -0.0652, -0.0566269, -0.0291807, -0.028, -0.0111},
    {0.883379, -0.162132, 0.00865186, -0.0180785, -0.286139, -0.0699, -0.0511, -0.0245, -0.0157, -0.0144},
    {0.663043, -0.193373, -0.0161184, -0.0459538, -0.294673, -0.0742, -0.0515, -0.0232, -0.0227, -0.009},
    {0.873702, -0.10845, -0.0251524, -0.00963989, -0.256457, -0.06, -0.0554, -0.0219365, -0.026, -0.0121},
    {0.816609, -0.112216, 0.00308642, -0.0331598, -0.241646, -0.0748, -0.0503, -0.0264259, -0.0184, -0.0089},
    {0.825485, -0.112223, -0.00941828, -0.0304709, -0.220166, -0.0633609, -0.0627, -0.0232, -0.0225, -0.0062},
    {0.608988, -0.106814, -0.0129663, -0.0159422, -0.219834, -0.0529, -0.0463, -0.0158, -0.0146, -0.0127},
    {0.620158, -0.0751852, -0.0141059, -0.00249566, -0.230165, -0.0502, -0.0414, -0.0186, -0.0245, -0.0041},
    {0.507233, -0.0874238, -0.00874636, -0.00142843, -0.194825, -0.0532, -0.0389, -0.0172, -0.0209, -0.0065},
    {0.469136, -0.069719, -0.0176, -0.0119, -0.164654, -0.0602, -0.0433, -0.0102, -0.0153, -0.00928477},
    {0.450059, -0.0527, -0.00374844, 0.0003, -0.180758, -0.048, -0.0449, -0.0147, -0.0197, -0.0101},
    {0.398753, -0.0832248, -0.00770512, -0.00367309, -0.124674, -0.0544842, -0.0376, -0.0116, -0.0143, -0.005},
    {0.37759, -0.058561, -0.0062474, 0.0034, -0.1484, -0.0557, -0.0389, -0.0123, -0.0155, -0.0076},
    {0.33102, -0.0579427, 0.0117, 0.0016, -0.1103, -0.0437, -0.0352, -0.0182, -0.0159, -0.0028},
    {0.301118, -0.0487297, 0.00265279, -0.00612182, -0.114784, -0.0327518, -0.0367, -0.0061, -0.0031, -0.0042},
    {0.291786, -0.030099, -0.0015, -0.0008, -0.122345, -0.0351, -0.0314, -0.0179, -0.0115, 0.0056},
    {0.263374, -0.0304, 0.0012, 0.000816243, -0.0903, -0.0441, -0.031, -0.0158, -0.0071, 0.0106},
    {0.253401, -0.0345883, 0.0008, -0.0004, -0.0933, -0.0387, -0.0321, -0.0148, -0.0144, -0.0034},
    {0.231972, -0.0566, 0.0014, 0.0026, -0.0836649, -0.0422, -0.0286, -0.0046, -0.0114, -0.0032},
    {0.244475, -0.0364, 0.0086, 0.0044, -0.083869, -0.0386, -0.0232, -0.0126, -0.0122, 0.0004},
    {0.211389, -0.0412, 0.013, -0.0096, -0.0866306, -0.0328, -0.0268, -0.017, -0.0078, 0.007},
    {0.200317, -0.0192, -0.003, 0.0052, -0.0799918, -0.0308, -0.0184, -0.0164, -0.0076, 0.0004},
    {0.20768, -0.0052, 0.0018, 0.0106, -0.0714, -0.0276, -0.017, -0.0076, -0.0032, 0},
    {0.185432, -0.0304, 0.0156, -0.00530558, -0.066, -0.018, -0.0196, -0.0074, -0.001, 0.0028},
    {0.169575, -0.0238, -0.0214, -0.0078, -0.0808, -0.0318, -0.0166, -0.0054, -0.002, 0.0012},
    {0.171441, -0.003, 0.00489746, 0.016, -0.0602, -0.0224, -0.0165289, -0.015, -0.0112, -0.001},
    {0.16976, -0.0126, 0.0182, 0.0132, -0.0532, -0.0264, -0.0154, -0.0102, -0.0028, 0.0062},
    {0.155165, 0.0072, 0.0162, 0.0194, -0.065, -0.0206, -0.0098, -0.0036, 0.0032, 0.0026},
    {0.1544, 0.0094, 0.01, 0.0108, -0.0546, -0.0176, -0.0048, -0.00734619, -0.0016, 0.00224467},
    {0.137537, -0.0084, 0.0114, -0.0008, -0.0276, -0.0104, -0.0114, -0.001, 0.0054, 0.00714213},
    {0.131823, 0.0048, 0.0186, 0.014, -0.0288, -0.008, -0.0098, -0.0044, 0.00285685, 0.0028},
    {0.1108, 0.0216, 0.0154, 0.0174, -0.0276, 0.0108, -0.0022, 0.0036, 0.0104, 0.0072},
    {0.1006, 0.014, 0.0088, 0.0152, 0, 0.001, 0.0112, -0.0004, 0.0022, 0.0140802},
    {0.1004, 0.0197939, 0.0244, 0.02, -0.0248, -0.0028, 0.0022, 0.0066, 0.0094, 0.00979492},
    {0.0876, 0.0112, 0.023, 0.0116, -0.0026, 0.0094, 0.0088, 0.0026, 0.0082, 0.0166},
    {0.0492, 0.0222, 0.0226, 0.0186, 0.0074, 0.0256, 0.0098, 0.014, 0.0098, 0.0128},
    {0.0354, 0.0362, 0.0204, 0.0184, 0.0334, 0.0102, 0.0258, 0.0155086, 0.0156, 0.0146},
    {0.012703, 0.039, 0.0163249, 0.0204, 0.0408122, 0.0284, 0.0328, 0.0177533, 0.0166, 0.0173452}
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
                value_diffs[cell] = -(result.value - complete_result.value + 2.0 * local_strategy_cell_weight[board.n_discs()][cell_type[cell]]);
                //value_diffs[cell] = (complete_result.value - result.value) - 2.0 * cell_weight[cell] / 256.0;
                std::cerr << idx_to_coord(cell) << " " << complete_result.value << " " << result.value << " " << 2.0 * local_strategy_cell_weight[board.n_discs()][cell_type[cell]] << " " << value_diffs[cell] << std::endl;
            board.player ^= bit;
            board.opponent ^= bit;
        } else if (board.opponent & bit) {
            // opponent -> player
            board.player ^= bit;
            board.opponent ^= bit;
                Search_result result = ai_searching(board, level, true, 0, true, false, searching);
                value_diffs[cell] = -(result.value - complete_result.value - 2.0 * local_strategy_cell_weight[board.n_discs()][cell_type[cell]]);
                //value_diffs[cell] = (complete_result.value - result.value) + 2.0 * cell_weight[cell] / 256.0;
                std::cerr << idx_to_coord(cell) << " " << complete_result.value << " " << result.value << " " << 2.0 * local_strategy_cell_weight[board.n_discs()][cell_type[cell]] << " " << value_diffs[cell] << std::endl;
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
    int n = 1000; // n per n_discs per cell_type
    int level = 10;

    double res[HW2][N_CELL_TYPE];
    int count[HW2][N_CELL_TYPE];

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
