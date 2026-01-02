/*
    Egaroucid Project

    @file local_strategy.hpp
        Calculate Local Strategy by flipping some discs
    @date 2021-2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once
#include "ai.hpp"

#define LOCAL_STRATEGY_POLICY_NOT_CHANGED 0
//#define LOCAL_STRATEGY_POLICY_CHANGED_GOOD_MOVE_DISC 16 // that is a good move because this disc is this color
//#define LOCAL_STRATEGY_POLICY_CHANGED_BAD_MOVE_DISC 32 // that move is NOT good because this disc is this color
#define LOCAL_STRATEGY_POLICY_CHANGED_GOOD_MOVE_FLIPPED 1
#define LOCAL_STRATEGY_POLICY_CHANGED_GOOD_MOVE_UNFLIPPED 2
#define LOCAL_STRATEGY_POLICY_CHANGED_BAD_MOVE_FLIPPED 4
#define LOCAL_STRATEGY_POLICY_CHANGED_BAD_MOVE_UNFLIPPED 8
#define LOCAL_STRATEGY_POLICY_CHANGED_PLAYER_CANPUT 16 // when unflipped
#define LOCAL_STRATEGY_POLICY_CHANGED_PLAYER_CANNOTPUT 32 // when unflipped
#define LOCAL_STRATEGY_POLICY_CHANGED_OPPONENT_CANPUT 64 // when flipped
#define LOCAL_STRATEGY_POLICY_CHANGED_OPPONENT_CANNOTPUT 128 // when flipped
// #define LOCAL_STRATEGY_POLICY_CHANGE_N_THRESHOLD 3 // use best 3 moves to see policy changed
#define LOCAL_STRATEGY_POLICY_CHANGE_BAD_MOVE_LOSS_THRESHOLD 2 // use best moves until value loss is 2 or less than 2
#define LOCAL_STRATEGY_POLICY_CHANGE_GOOD_MOVE_LOSS_THRESHOLD 6 // use best moves until value loss is 6 or less than 6

constexpr int MAX_LOCAL_STRATEGY_LEVEL = 25;

constexpr double local_strategy_cell_weight[HW2_P1][N_CELL_TYPE] = {
    {0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 35.9446},
    {0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.8169, 20.3119},
    {0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 64.0000, 20.8794, 17.2430},
    {0.0000, 0.0000, 0.0000, 0.0000, 0.0000, -13.9275, 0.0000, -0.8433, -0.7167, 2.1834},
    {0.0000, 0.0000, 0.0000, 0.0000, 36.0008, -9.9566, 11.0838, 12.2988, 2.7371, 1.4415},
    {0.0000, -18.1895, -9.0195, 1.4466, -49.5676, -8.7377, -9.4421, -1.8396, -2.5018, -0.4200},
    {62.1743, -16.7449, 5.5544, 5.0160, -41.6905, -8.5904, -5.9412, -0.7858, -1.4242, -1.5837},
    {46.5245, -19.1576, 0.6120, 0.2820, -45.8374, -10.0821, -8.1021, -3.6418, -2.8625, -1.5600},
    {45.1350, -17.4974, 0.7622, -1.4379, -42.7405, -9.0215, -7.6316, -3.1431, -2.6992, -1.9420},
    {44.9177, -18.3463, -0.0504, -2.1638, -42.5875, -9.5728, -8.0137, -3.7735, -3.2138, -1.9275},
    {43.7127, -17.8035, -0.4463, -2.4987, -41.0401, -9.2611, -7.7909, -3.6852, -2.9848, -2.2829},
    {42.9512, -17.6589, -0.8615, -2.8725, -39.8712, -9.3284, -7.6042, -3.7108, -3.2181, -2.1245},
    {42.0100, -17.1074, -1.2992, -2.8872, -38.3728, -9.1961, -7.4694, -3.6001, -3.1493, -2.1718},
    {40.7323, -16.7183, -1.3629, -3.0117, -37.5244, -8.9338, -7.4134, -3.5138, -3.2001, -2.0746},
    {40.0607, -16.3237, -1.5769, -3.3093, -36.2262, -8.6924, -7.1242, -3.4378, -3.2001, -2.0596},
    {39.1865, -15.6711, -1.5825, -3.1300, -34.9004, -8.5086, -6.9490, -3.3355, -3.1654, -2.0326},
    {38.5542, -15.1432, -1.6038, -3.1474, -33.6294, -8.3487, -6.8920, -3.2454, -3.1092, -2.1573},
    {37.3155, -14.7787, -1.7092, -2.8592, -32.0315, -8.1581, -6.6162, -3.0240, -3.0348, -1.7330},
    {36.7710, -14.1155, -1.8509, -2.9666, -31.2291, -8.0166, -6.4025, -3.0232, -2.9446, -1.7699},
    {35.7192, -13.5841, -1.6730, -2.7950, -29.7329, -7.7212, -6.2553, -2.8064, -2.7118, -1.6369},
    {35.1697, -13.2008, -1.6428, -2.8529, -28.3166, -7.5234, -6.0568, -2.8223, -2.7098, -1.6151},
    {34.0325, -12.5179, -1.7374, -2.6701, -27.2623, -7.3511, -5.8705, -2.6096, -2.5994, -1.5347},
    {33.3272, -11.9877, -1.7716, -2.5502, -26.2366, -7.1923, -5.7395, -2.5535, -2.5442, -1.3363},
    {32.2909, -11.3421, -1.7772, -2.2925, -24.4818, -6.9314, -5.5031, -2.3790, -2.3640, -1.2098},
    {32.0634, -11.0835, -1.8913, -2.1174, -23.6842, -6.6992, -5.4155, -2.3789, -2.3459, -1.2215},
    {31.0442, -10.2439, -1.4812, -1.9173, -22.4231, -6.4460, -5.1011, -2.2346, -2.2707, -1.0441},
    {30.5160, -9.8175, -1.5251, -1.8144, -21.5843, -6.3501, -5.0990, -2.1064, -2.2721, -0.9242},
    {29.3691, -9.3899, -1.3252, -1.8733, -20.2964, -6.0612, -4.8104, -2.0378, -2.0178, -0.8728},
    {29.0703, -8.8353, -1.3830, -1.6139, -19.2314, -5.8829, -4.6959, -1.9525, -2.0162, -0.7884},
    {28.0705, -8.3286, -1.0337, -1.3260, -18.2903, -5.6623, -4.5172, -1.7964, -1.8427, -0.6929},
    {27.6494, -7.8301, -1.3695, -1.2985, -17.4495, -5.6024, -4.3762, -1.8209, -1.7366, -0.6917},
    {26.8963, -7.3670, -1.2737, -1.0379, -16.2258, -5.2222, -4.0763, -1.6847, -1.6581, -0.6373},
    {26.4801, -6.9864, -1.0709, -1.0081, -15.7679, -5.2048, -3.9779, -1.6550, -1.5738, -0.4424},
    {25.2689, -6.3167, -0.9900, -0.7440, -14.8929, -4.8388, -3.8532, -1.6587, -1.4679, -0.4744},
    {25.2522, -6.2574, -0.8425, -0.8805, -14.0071, -4.7250, -3.6572, -1.5583, -1.4360, -0.3408},
    {23.8502, -5.4857, -0.7664, -0.6897, -13.3102, -4.4711, -3.5523, -1.4488, -1.2720, -0.3942},
    {23.6233, -5.2749, -0.8128, -0.4643, -12.8026, -4.3704, -3.4111, -1.4482, -1.2409, -0.2346},
    {22.7648, -4.5089, -0.5584, -0.2763, -11.5602, -4.1107, -3.2100, -1.3291, -1.1733, -0.1977},
    {22.3242, -4.3111, -0.3170, 0.0012, -10.8503, -3.9270, -3.0174, -1.2792, -1.0730, -0.1479},
    {21.4076, -3.7998, -0.3753, -0.1576, -10.3660, -3.8177, -2.9003, -1.2135, -0.9481, -0.1471},
    {20.9582, -3.4324, 0.0446, 0.1502, -10.4060, -3.9096, -2.8884, -1.2502, -1.0162, -0.0750},
    {20.1086, -3.0138, -0.0236, 0.2738, -9.5010, -3.5025, -2.5952, -1.1278, -0.7740, -0.0042},
    {19.5744, -2.5730, 0.1534, 0.2918, -9.0620, -3.4494, -2.5402, -1.0436, -0.7788, 0.0720},
    {18.7738, -2.4116, 0.2340, 0.4364, -8.3068, -3.1902, -2.2212, -0.9684, -0.6186, 0.0572},
    {18.2082, -1.9922, 0.3182, 0.6426, -7.8138, -2.9436, -2.1834, -0.9604, -0.5476, 0.0516},
    {17.3450, -1.3742, 0.3772, 0.7106, -6.9870, -2.7112, -1.9148, -0.8046, -0.4786, 0.1226},
    {16.8198, -1.1282, 0.7694, 0.8940, -6.6340, -2.6640, -1.7026, -0.7358, -0.3107, 0.1200},
    {15.7418, -0.7934, 0.8556, 0.9290, -5.9022, -2.2322, -1.4278, -0.6124, -0.3724, 0.2206},
    {15.3268, -0.3386, 0.8984, 1.0956, -5.4134, -2.2084, -1.4526, -0.6556, -0.1786, 0.3306},
    {14.4972, -0.0308, 0.8974, 1.1072, -4.6116, -1.8592, -1.0333, -0.4014, -0.0218, 0.3482},
    {13.5282, 0.1494, 1.1921, 1.1264, -4.3238, -1.5056, -0.9340, -0.3048, 0.0336, 0.4498},
    {12.7834, 0.5172, 1.2282, 1.4554, -3.6248, -1.2640, -0.5988, -0.1998, 0.1786, 0.4418},
    {11.8684, 0.6774, 1.2466, 1.3520, -3.0590, -0.9594, -0.5746, 0.0300, 0.2978, 0.6152},
    {10.7788, 1.1992, 1.4726, 1.5150, -2.5382, -0.5088, -0.1092, 0.0940, 0.5534, 0.7082},
    {9.8364, 1.4818, 1.4300, 1.5604, -1.6806, -0.0258, 0.3310, 0.4557, 0.6984, 0.8032},
    {8.5496, 1.7428, 1.5438, 1.6206, -0.8102, 0.3442, 0.5180, 0.4762, 0.7564, 0.9188},
    {6.8470, 1.7904, 1.8358, 1.6008, 0.4054, 1.0384, 1.1456, 0.8252, 1.2544, 1.0964},
    {5.1758, 2.1170, 1.8724, 1.4832, 2.1156, 1.9396, 1.5902, 1.2552, 1.1616, 1.0056},
    {2.2740, 2.1902, 1.0258, 1.3252, 4.2662, 2.7912, 2.1468, 1.4662, 1.3404, 0.9242},
    {-2.1394, 2.2954, 0.2222, -0.6540, 8.4924, 3.8908, 2.6462, 1.2518, 1.3726, 0.3190},
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

void print_local_strategy(const int arr[]) {
    for (int y = 0; y < HW; ++y) {
        for (int x = 0; x < HW; ++x) {
            int cell = HW2_M1 - (y * HW + x);
            std::cout << std::fixed << std::setprecision(3) << arr[cell] << " ";
        }
        std::cout << std::endl;
    }
}


void calc_local_strategy_player(Board board, int max_level, double res[], int player, bool *searching, int *done_level, bool show_log) {
    for (int cell = 0; cell < HW2; ++cell) {
        res[cell] = 0.0;
    }
    double value_diffs[HW2];
    constexpr uint64_t edge_bits[4] = {0x7E00000000000000ULL, 0x0001010101010100ULL, 0x000000000000007EULL, 0x0080808080808000ULL};
    constexpr uint64_t corner_bits_next_to_edge[4] = {0x8100000000000000ULL, 0x0100000000000001ULL, 0x0000000000000081ULL, 0x8000000000000080ULL};
    for (int level = 1; level < max_level && *searching && global_searching; ++level) {
        Search_result actual_result = ai_searching(board, level, true, 0, true, false, searching);
        if (show_log) {
            std::cerr << "level " << level << " result " << actual_result.value << std::endl;
        }
        for (int cell = 0; cell < HW2; ++cell) {
            value_diffs[cell] = 0;
        }
        uint64_t done_cells = 0;
        for (int cell = 0; cell < HW2 && *searching && global_searching; ++cell) {
            uint64_t bit = 1ULL << cell;
            if (done_cells & bit) {
                continue;
            }
            if ((board.player | board.opponent) & bit) { // discs
                uint64_t bits = bit;
                for (int i = 0; i < 4; ++i) {
                    if ((edge_bits[i] & bit) && (corner_bits_next_to_edge[i] & (board.player | board.opponent)) == 0) {
                        bits = edge_bits[i] & (board.player | board.opponent);
                        break;
                    }
                }
                // player -> opponent, opponent -> player
                board.player ^= bits;
                board.opponent ^= bits;
                    Search_result result = ai_searching(board, level, true, 0, true, false, searching);
                    uint64_t bits_cpy = bits;
                    int n_bits = pop_count_ull(bits);
                    for (uint_fast8_t c = first_bit(&bits_cpy); bits_cpy; c = next_bit(&bits_cpy)) {
                        value_diffs[c] = -(double)(result.value - actual_result.value) / (double)n_bits;
                    }
                board.player ^= bits;
                board.opponent ^= bits;
                done_cells |= bits;
            } else { // empty
                done_cells |= bit;
            }
        }
        if (*searching && global_searching) {
            // if (show_log) {
            //     std::cerr << "value_diffs" << std::endl;
            //     print_local_strategy(value_diffs);
            //     std::cerr << std::endl;
            // }
            for (int cell = 0; cell < HW2; ++cell) {
                // std::cerr << cell << " " << value_diffs[cell] << std::endl;
                int sgn = player == BLACK ? 1 : -1;
                res[cell] = sgn * std::tanh(0.2 * value_diffs[cell]); // 10 discs ~ 1.0
            }
            // if (show_log) {
            //     print_local_strategy(res);
            // }
            *done_level = level;
            if (show_log) {
                std::cerr << "local strategy level " << level << std::endl;
            }
        }
    }
}






/*
// disc change
void calc_local_strategy_policy(Board board, int max_level, int policy_res[HW2][HW2], bool *searching, int *done_level, bool show_log) {
    for (int policy = 0; policy < HW2; ++policy) {
        for (int cell = 0; cell < HW2; ++cell) {
            policy_res[policy][cell] = LOCAL_STRATEGY_POLICY_NOT_CHANGED;
        }
    }
    int policy_changed[HW2][HW2]; // [policy][cell]
    constexpr uint64_t edge_bits[4] = {0x7E00000000000000ULL, 0x0001010101010100ULL, 0x000000000000007EULL, 0x0080808080808000ULL};
    constexpr uint64_t corner_bits_next_to_edge[4] = {0x8100000000000000ULL, 0x0100000000000001ULL, 0x0000000000000081ULL, 0x8000000000000080ULL};
    for (int level = 1; level < max_level && *searching && global_searching; ++level) {
        std::vector<Search_result> actual_results = ai_best_moves_loss_searching(board, level, true, 0, true, false, 1, searching); // best moves or best-1 moves
        std::vector<int> actual_best_moves;
        for (const Search_result &result: actual_results) {
            actual_best_moves.push_back(result.policy);
        }
        uint64_t legal = board.get_legal();
        for (int policy = first_bit(&legal); legal && *searching; policy = next_bit(&legal)) {
            bool policy_is_good_move = std::find(actual_best_moves.begin(), actual_best_moves.end(), policy) != actual_best_moves.end();
            for (int cell = 0; cell < HW2; ++cell) {
                policy_changed[policy][cell] = LOCAL_STRATEGY_POLICY_NOT_CHANGED;
            }
            uint64_t done_cells = 0;
            for (int cell = 0; cell < HW2 && *searching && global_searching; ++cell) {
                uint64_t bit = 1ULL << cell;
                if (done_cells & bit) {
                    continue;
                }
                if ((board.player | board.opponent) & bit) { // discs
                    uint64_t bits = bit;
                    for (int i = 0; i < 4; ++i) {
                        if ((edge_bits[i] & bit) && (corner_bits_next_to_edge[i] & (board.player | board.opponent)) == 0) {
                            bits = edge_bits[i] & (board.player | board.opponent);
                            break;
                        }
                    }
                    // player -> opponent, opponent -> player
                    board.player ^= bits;
                    board.opponent ^= bits;
                        int accept_loss = policy_is_good_move ? LOCAL_STRATEGY_POLICY_CHANGE_GOOD_MOVE_LOSS_THRESHOLD : LOCAL_STRATEGY_POLICY_CHANGE_BAD_MOVE_LOSS_THRESHOLD;
                        std::vector<Search_result> results = ai_best_moves_loss_searching(board, level, true, 0, true, false, accept_loss, searching);
                        //std::vector<Search_result> results = ai_best_n_moves_searching(board, level, true, 0, true, false, LOCAL_STRATEGY_POLICY_CHANGE_N_THRESHOLD, searching);
                        std::vector<int> best_moves;
                        for (const Search_result &result: results) {
                            best_moves.push_back(result.policy);
                        }
                        // std::cerr << idx_to_coord(cell) << " normal " << idx_to_coord(policy) << " changed: ";
                        // for (const Search_result &result: results) {
                        //     std::cerr << idx_to_coord(result.policy) << "@" << result.value << " ";
                        // }
                        // std::cerr << std::endl;
                        if (policy_is_good_move) { // policy is good move in the actual board
                            if (std::find(best_moves.begin(), best_moves.end(), policy) == best_moves.end()) { // good move is NOT selected now
                                uint64_t bits_cpy = bits;
                                for (uint_fast8_t c = first_bit(&bits_cpy); bits_cpy; c = next_bit(&bits_cpy)) {
                                    policy_changed[policy][c] = LOCAL_STRATEGY_POLICY_CHANGED_GOOD_MOVE_DISC;
                                }
                            }
                        } else { // policy is bad move in the actual board
                            if (std::find(best_moves.begin(), best_moves.end(), policy) != best_moves.end()) { // bad move is selected now
                                uint64_t bits_cpy = bits;
                                for (uint_fast8_t c = first_bit(&bits_cpy); bits_cpy; c = next_bit(&bits_cpy)) {
                                    policy_changed[policy][c] = LOCAL_STRATEGY_POLICY_CHANGED_BAD_MOVE_DISC;
                                }
                            }
                        }
                    board.player ^= bits;
                    board.opponent ^= bits;
                    done_cells |= bits;
                } else { // empty
                    done_cells |= bit;
                }
            }
            if (*searching && global_searching) {
                for (int cell = 0; cell < HW2; ++cell) {
                    policy_res[policy][cell] = policy_changed[policy][cell];
                }
            }
        }
        if (*searching && global_searching) {
            *done_level = level;
            if (show_log) {
                std::cerr << "local strategy policy level " << level << std::endl;
            }
        }
    }
}
*/

uint64_t get_connected_bits_dir4(uint64_t candidate, uint64_t bits, uint64_t cell_bit) {
    uint64_t neighbours = (cell_bit & 0x7F7F7F7F7F7F7F7FULL) << 1;
    neighbours |= (cell_bit & 0xFEFEFEFEFEFEFEFEULL) >> 1;
    neighbours |= cell_bit << 8;
    neighbours |= cell_bit >> 8;
    neighbours &= candidate;
    neighbours &= ~bits;
    if (neighbours) {
        bits |= neighbours;
        for (uint_fast8_t cell = first_bit(&neighbours); neighbours; cell = next_bit(&neighbours)) {
            bits |= get_connected_bits_dir4(candidate, bits, 1ULL << cell);
        }
    }
    return bits;
}

uint64_t get_connected_bits_dir8(uint64_t candidate, uint64_t bits, uint64_t cell_bit) {
    uint64_t neighbours = (cell_bit & 0x7F7F7F7F7F7F7F7FULL) << 1;
    neighbours |= (cell_bit & 0xFEFEFEFEFEFEFEFEULL) >> 1;
    neighbours |= cell_bit << 8;
    neighbours |= cell_bit >> 8;
    neighbours |= (cell_bit & 0x007F7F7F7F7F7F7FULL) << 9;
    neighbours |= (cell_bit & 0xFEFEFEFEFEFEFE00ULL) >> 9;
    neighbours |= (cell_bit & 0x00FEFEFEFEFEFEFEULL) << 7;
    neighbours |= (cell_bit & 0x7F7F7F7F7F7F7F00ULL) >> 7;
    neighbours &= candidate;
    neighbours &= ~bits;
    if (neighbours) {
        bits |= neighbours;
        for (uint_fast8_t cell = first_bit(&neighbours); neighbours; cell = next_bit(&neighbours)) {
            bits |= get_connected_bits_dir8(candidate, bits, 1ULL << cell);
        }
    }
    return bits;
}

uint64_t get_connected_bits_horizontal(uint64_t candidate, uint64_t bits, uint64_t cell_bit) {
    uint64_t neighbours = (cell_bit & 0x7F7F7F7F7F7F7F7FULL) << 1;
    neighbours |= (cell_bit & 0xFEFEFEFEFEFEFEFEULL) >> 1;
    neighbours &= candidate;
    neighbours &= ~bits;
    if (neighbours) {
        bits |= neighbours;
        for (uint_fast8_t cell = first_bit(&neighbours); neighbours; cell = next_bit(&neighbours)) {
            bits |= get_connected_bits_horizontal(candidate, bits, 1ULL << cell);
        }
    }
    return bits;
}

uint64_t get_connected_bits_vertical(uint64_t candidate, uint64_t bits, uint64_t cell_bit) {
    uint64_t neighbours = cell_bit << 8;
    neighbours |= cell_bit >> 8;
    neighbours &= candidate;
    neighbours &= ~bits;
    if (neighbours) {
        bits |= neighbours;
        for (uint_fast8_t cell = first_bit(&neighbours); neighbours; cell = next_bit(&neighbours)) {
            bits |= get_connected_bits_vertical(candidate, bits, 1ULL << cell);
        }
    }
    return bits;
}

uint64_t get_connected_bits_diagonal9(uint64_t candidate, uint64_t bits, uint64_t cell_bit) {
    uint64_t neighbours = (cell_bit & 0x007F7F7F7F7F7F7FULL) << 9;
    neighbours |= (cell_bit & 0xFEFEFEFEFEFEFE00ULL) >> 9;
    neighbours &= candidate;
    neighbours &= ~bits;
    if (neighbours) {
        bits |= neighbours;
        for (uint_fast8_t cell = first_bit(&neighbours); neighbours; cell = next_bit(&neighbours)) {
            bits |= get_connected_bits_diagonal9(candidate, bits, 1ULL << cell);
        }
    }
    return bits;
}

uint64_t get_connected_bits_diagonal7(uint64_t candidate, uint64_t bits, uint64_t cell_bit) {
    uint64_t neighbours = (cell_bit & 0x00FEFEFEFEFEFEFEULL) << 7;
    neighbours |= (cell_bit & 0x7F7F7F7F7F7F7F00ULL) >> 7;
    neighbours &= candidate;
    neighbours &= ~bits;
    if (neighbours) {
        bits |= neighbours;
        for (uint_fast8_t cell = first_bit(&neighbours); neighbours; cell = next_bit(&neighbours)) {
            bits |= get_connected_bits_diagonal7(candidate, bits, 1ULL << cell);
        }
    }
    return bits;
}

uint64_t get_left_bit(uint64_t cell_bit) {
    return (cell_bit & 0x7F7F7F7F7F7F7F7FULL) << 1;
}

uint64_t get_right_bit(uint64_t cell_bit) {
    return (cell_bit & 0xFEFEFEFEFEFEFEFEULL) >> 1;
}

uint64_t get_up_bit(uint64_t cell_bit) {
    return cell_bit << 8;
}

uint64_t get_down_bit(uint64_t cell_bit) {
    return cell_bit >> 8;
}

uint64_t get_d7up_bit(uint64_t cell_bit) {
    return (cell_bit & 0x00FEFEFEFEFEFEFEULL) << 7;
}

uint64_t get_d7down_bit(uint64_t cell_bit) {
    return (cell_bit & 0x7F7F7F7F7F7F7F00ULL) >> 7;
}

uint64_t get_d9up_bit(uint64_t cell_bit) {
    return (cell_bit & 0x007F7F7F7F7F7F7FULL) << 9;
}

uint64_t get_d9down_bit(uint64_t cell_bit) {
    return (cell_bit & 0xFEFEFEFEFEFEFE00ULL) >> 9;
}

uint64_t (*get_neighbour_bit[])(uint64_t) = {get_left_bit, get_right_bit, get_up_bit, get_down_bit, get_d7up_bit, get_d7down_bit, get_d9up_bit, get_d9down_bit};

uint64_t get_connected_bits(uint64_t candidate, uint64_t bits, uint64_t cell_bit, uint64_t (*neighbour_bit_func)(uint64_t)) {
    uint64_t neighbours = neighbour_bit_func(cell_bit);
    neighbours &= candidate;
    neighbours &= ~bits;
    if (neighbours) {
        bits |= neighbours;
        for (uint_fast8_t cell = first_bit(&neighbours); neighbours; cell = next_bit(&neighbours)) {
            bits |= get_connected_bits(candidate, bits, 1ULL << cell, neighbour_bit_func);
        }
    }
    return bits;
}

// flip change
void calc_local_strategy_policy(Board board, int max_level, int policy_res[HW2][HW2], bool *searching, int *done_level, bool show_log) {
    for (int policy = 0; policy < HW2; ++policy) {
        for (int cell = 0; cell < HW2; ++cell) {
            policy_res[policy][cell] = LOCAL_STRATEGY_POLICY_NOT_CHANGED;
        }
    }
    int policy_changed[HW2][HW2]; // [policy][cell]
    const uint64_t actual_legal_player = board.get_legal();
    board.pass();
        const uint64_t actual_legal_opponent = board.get_legal();
    board.pass();
    for (int level = 1; level < max_level && *searching && global_searching; ++level) {
        std::vector<Search_result> actual_results = ai_best_moves_loss_searching(board, level, true, 0, true, false, 1, searching); // best moves or best-1 moves
        std::vector<int> actual_best_moves;
        for (const Search_result &result: actual_results) {
            actual_best_moves.push_back(result.policy);
        }
        uint64_t legal = board.get_legal();
        Flip flip;
        for (uint_fast8_t policy = first_bit(&legal); legal && *searching; policy = next_bit(&legal)) {
            bool policy_is_good_move = std::find(actual_best_moves.begin(), actual_best_moves.end(), policy) != actual_best_moves.end();
            for (int cell = 0; cell < HW2; ++cell) {
                policy_changed[policy][cell] = LOCAL_STRATEGY_POLICY_NOT_CHANGED;
            }
            calc_flip(&flip, &board, policy);
            uint64_t policy_bit = 1ULL << policy;
            for (int dir = 0; dir < 8; ++dir) {
                //if ((board.player | board.opponent) & get_neighbour_bit[dir](get_neighbour_bit[dir](policy_bit))) { // there are at least 2 discs for this direction
                uint64_t can_be_flipped_1dir = get_connected_bits(board.opponent, 0, policy_bit, get_neighbour_bit[dir]);
                if (can_be_flipped_1dir) {
                    flip.flip ^= can_be_flipped_1dir;
                    board.move_board(&flip);
                        int g = -ai_searching(board, level, true, 0, true, false, searching).value;
                    board.undo_board(&flip);
                    flip.flip ^= can_be_flipped_1dir;
                    int n_flipped = pop_count_ull(can_be_flipped_1dir);
                    if (g >= actual_results[0].value - 1) { // now policy becomes a GOOD move
                        if (!policy_is_good_move) { // policy is BAD move in the actual board (bad -> good)
                            for (uint_fast8_t c = first_bit(&can_be_flipped_1dir); can_be_flipped_1dir; c = next_bit(&can_be_flipped_1dir)) {
                                if (flip.flip & can_be_flipped_1dir) {
                                    policy_changed[policy][c] |= LOCAL_STRATEGY_POLICY_CHANGED_BAD_MOVE_FLIPPED; // the policy is a good move if the disc is flipped
                                } else if (n_flipped > 1) {
                                    policy_changed[policy][c] |= LOCAL_STRATEGY_POLICY_CHANGED_BAD_MOVE_UNFLIPPED; // the policy is a good move if the disc is NOT flipped
                                }
                            }
                        }
                    } else { // now policy becomes a BAD move
                        if (policy_is_good_move) { // policy is GOOD move in the actual board (good -> bad)
                            for (uint_fast8_t c = first_bit(&can_be_flipped_1dir); can_be_flipped_1dir; c = next_bit(&can_be_flipped_1dir)) {
                                if (flip.flip & can_be_flipped_1dir) {
                                    policy_changed[policy][c] |= LOCAL_STRATEGY_POLICY_CHANGED_GOOD_MOVE_FLIPPED; // the flipped cell is important for the policy to be a good move
                                } else if (n_flipped > 1) {
                                    policy_changed[policy][c] |= LOCAL_STRATEGY_POLICY_CHANGED_GOOD_MOVE_UNFLIPPED; // the UNFLIPPED cell is important for the policy to be a good move
                                }
                            }
                        }
                    }
                }
                //}
            }
            board.move_board(&flip);
                uint64_t modified_legal_opponent = board.get_legal();
                board.pass();
                    uint64_t modified_legal_player = board.get_legal();
                board.pass();
            board.undo_board(&flip);
            uint64_t non_legal_to_legal_player = ~actual_legal_player & modified_legal_player; // player's non-legal -> legal
            uint64_t legal_to_non_legal_player = (actual_legal_player & ~modified_legal_player) & ~(1ULL << policy); // player's legal -> non-legal
            uint64_t non_legal_to_legal_opponent = ~actual_legal_opponent & modified_legal_opponent; // opponent's non-legal -> legal
            uint64_t legal_to_non_legal_opponent = actual_legal_opponent & ~modified_legal_opponent; // opponent's legal -> non-legal
            if (policy_is_good_move) {
                for (uint_fast8_t c = first_bit(&non_legal_to_legal_player); non_legal_to_legal_player; c = next_bit(&non_legal_to_legal_player)) {
                    policy_changed[policy][c] |= LOCAL_STRATEGY_POLICY_CHANGED_PLAYER_CANPUT; // player can put there because the cell's flip is changed
                }
                for (uint_fast8_t c = first_bit(&legal_to_non_legal_opponent); legal_to_non_legal_opponent; c = next_bit(&legal_to_non_legal_opponent)) {
                    policy_changed[policy][c] |= LOCAL_STRATEGY_POLICY_CHANGED_OPPONENT_CANNOTPUT; // opponent can NOT put there because the cell's flip is changed
                }
            } else { // policy is BAD move
                for (uint_fast8_t c = first_bit(&legal_to_non_legal_player); legal_to_non_legal_player; c = next_bit(&legal_to_non_legal_player)) {
                    policy_changed[policy][c] |= LOCAL_STRATEGY_POLICY_CHANGED_PLAYER_CANNOTPUT; // player can NOT put there because the cell's flip is changed
                }
                for (uint_fast8_t c = first_bit(&non_legal_to_legal_opponent); non_legal_to_legal_opponent; c = next_bit(&non_legal_to_legal_opponent)) {
                    policy_changed[policy][c] |= LOCAL_STRATEGY_POLICY_CHANGED_OPPONENT_CANPUT; // opponent can put there because the cell's flip is changed
                }
            }
            /*
            for (uint_fast8_t cell = first_bit(&can_be_flipped); can_be_flipped && (*searching); cell = next_bit(&can_be_flipped)) {
                uint64_t bit = 1ULL << cell;
                flip.flip ^= bit;
                board.move_board(&flip);
                    int g = -ai_searching(board, level, true, 0, true, false, searching).value;
                board.undo_board(&flip);
                flip.flip ^= bit;
                if (g >= actual_results[0].value - 1) { // now policy becomes a good move
                    if (!policy_is_good_move) { // policy is bad move in the actual board (bad -> good)
                        policy_changed[policy][cell] |= LOCAL_STRATEGY_POLICY_CHANGED_BAD_MOVE_FLIP; // the policy is a good move if the cell was not flipped
                    }
                } else { // now policy becomes a bad move
                    if (policy_is_good_move) { // policy is good move in the actual board (good -> bad)
                        policy_changed[policy][cell] |= LOCAL_STRATEGY_POLICY_CHANGED_GOOD_MOVE_FLIP; // the flipped cell is important for the policy to be a good move
                    }
                }
            }
            */
            if (*searching && global_searching) {
                for (int cell = 0; cell < HW2; ++cell) {
                    policy_res[policy][cell] = policy_changed[policy][cell];
                }
            }
        }
        if (*searching && global_searching) {
            *done_level = level;
            if (show_log) {
                std::cerr << "local strategy policy level " << level << std::endl;
            }
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
                    Search_result actual_result = ai(board, level, true, 0, true, false);
                    int sgn = -1; // opponent -> player
                    if (board.player & flipped) { // player -> opponent
                        sgn = 1;
                    }
                    board.player ^= flipped;
                    board.opponent ^= flipped;
                        Search_result flipped_result = ai(board, level, true, 0, true, false);
                    board.player ^= flipped;
                    board.opponent ^= flipped;
                    double diff = sgn * (actual_result.value - flipped_result.value);
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