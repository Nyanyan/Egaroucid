/*
    Egaroucid Project

    @file policy_value_network_sample.cpp
        Sample C++ policy-value distribution viewer.

    Related issue: #613
*/

#include "policy_value_network.hpp"

#include <iomanip>
#include <iostream>
#include <string>

namespace {

void print_usage(const char *exe) {
    std::cerr
        << "usage:\n"
        << "  " << exe << " weights.bin --board BOARD [--side black|white] [--top N]\n"
        << "  " << exe << " weights.bin --transcript MOVES [--top N]\n";
}

void print_board(const egaroucid_policy_value::BoardState &board) {
    for (int y = 0; y < egaroucid_policy_value::HW; ++y) {
        for (int x = 0; x < egaroucid_policy_value::HW; ++x) {
            const int policy = egaroucid_policy_value::HW2_M1 - (y * egaroucid_policy_value::HW + x);
            const uint64_t bit = egaroucid_policy_value::bit_from_policy(policy);
            if (board.black & bit) {
                std::cout << 'X';
            } else if (board.white & bit) {
                std::cout << 'O';
            } else {
                std::cout << '-';
            }
        }
        std::cout << '\n';
    }
    std::cout << "side " << (board.side_to_move == egaroucid_policy_value::BLACK ? "black" : "white") << '\n';
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 4) {
        print_usage(argv[0]);
        return 1;
    }
    const std::string weights_path = argv[1];
    const std::string mode = argv[2];
    const std::string input = argv[3];
    int top_n = 10;
    int side_override = -1;
    for (int i = 4; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--top" && i + 1 < argc) {
            top_n = std::stoi(argv[++i]);
        } else if (arg == "--side" && i + 1 < argc) {
            const std::string side = argv[++i];
            if (side == "black" || side == "b" || side == "B" || side == "X" || side == "x") {
                side_override = egaroucid_policy_value::BLACK;
            } else if (side == "white" || side == "w" || side == "W" || side == "O" || side == "o") {
                side_override = egaroucid_policy_value::WHITE;
            } else {
                std::cerr << "invalid side: " << side << '\n';
                return 1;
            }
        } else {
            print_usage(argv[0]);
            return 1;
        }
    }

    egaroucid_policy_value::PolicyValueNetwork network;
    std::string error;
    if (!network.load(weights_path, &error)) {
        std::cerr << "failed to load weights: " << error << '\n';
        return 1;
    }

    egaroucid_policy_value::BoardState board;
    if (mode == "--board") {
        if (!egaroucid_policy_value::parse_board_65(input, &board)) {
            std::cerr << "invalid board string\n";
            return 1;
        }
        if (side_override >= 0) {
            board.side_to_move = side_override;
        }
    } else if (mode == "--transcript") {
        if (!egaroucid_policy_value::play_transcript(input, &board, &error)) {
            std::cerr << "invalid transcript: " << error << '\n';
            return 1;
        }
    } else {
        print_usage(argv[0]);
        return 1;
    }

    const uint64_t player = board.side_to_move == egaroucid_policy_value::BLACK ? board.black : board.white;
    const uint64_t opponent = board.side_to_move == egaroucid_policy_value::BLACK ? board.white : board.black;
    const uint64_t legal = egaroucid_policy_value::legal_moves(player, opponent);
    const auto prediction = network.predict(player, opponent);

    std::cout << "board:\n";
    print_board(board);
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "\nvalue(player perspective): " << prediction.value
              << "  disc-diff estimate: " << prediction.value * 64.0F << '\n';

    std::cout << "\ntop " << top_n << ":\n";
    for (const auto &entry : egaroucid_policy_value::top_k(prediction.policy, top_n)) {
        const int policy = entry.first;
        std::cout << std::setw(2) << policy << " "
                  << egaroucid_policy_value::coord_from_policy(policy) << " "
                  << std::setw(10) << entry.second;
        if (legal & egaroucid_policy_value::bit_from_policy(policy)) {
            std::cout << " legal";
        }
        std::cout << '\n';
    }
    return 0;
}
