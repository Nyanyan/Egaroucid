/*
    Egaroucid Project

    @file policy_network_sample.cpp
        Sample C++ policy distribution viewer.

    Related issue: #613
*/

#include "policy_network.hpp"

#include <iomanip>
#include <iostream>
#include <string>

namespace {

void print_usage(const char *exe) {
    std::cerr
        << "usage:\n"
        << "  " << exe << " weights.bin --board BOARD [--side black|white] [--top N]\n"
        << "  " << exe << " weights.bin --transcript MOVES [--top N]\n"
        << "\n"
        << "BOARD is 64 board chars, optionally followed by a side-to-move char. X/0/* = black, O/1 = white, -/. = empty.\n"
        << "Output index uses Egaroucid policy coordinates: a1 -> 63, h8 -> 0.\n";
}

void print_board(const egaroucid_policy::BoardState &board) {
    for (int y = 0; y < egaroucid_policy::HW; ++y) {
        for (int x = 0; x < egaroucid_policy::HW; ++x) {
            const int policy = egaroucid_policy::HW2_M1 - (y * egaroucid_policy::HW + x);
            const uint64_t bit = egaroucid_policy::bit_from_policy(policy);
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
    std::cout << "side " << (board.side_to_move == egaroucid_policy::BLACK ? "black" : "white") << '\n';
}

void print_distribution(const std::array<float, 64> &prob) {
    std::cout << "\npolicy distribution board order:\n";
    std::cout << std::fixed << std::setprecision(4);
    for (int y = 0; y < egaroucid_policy::HW; ++y) {
        for (int x = 0; x < egaroucid_policy::HW; ++x) {
            const int policy = egaroucid_policy::HW2_M1 - (y * egaroucid_policy::HW + x);
            std::cout << std::setw(7) << prob[policy];
        }
        std::cout << '\n';
    }
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
                side_override = egaroucid_policy::BLACK;
            } else if (side == "white" || side == "w" || side == "W" || side == "O" || side == "o") {
                side_override = egaroucid_policy::WHITE;
            } else {
                std::cerr << "invalid side: " << side << '\n';
                return 1;
            }
        } else {
            print_usage(argv[0]);
            return 1;
        }
    }

    egaroucid_policy::PolicyNetwork network;
    std::string error;
    if (!network.load(weights_path, &error)) {
        std::cerr << "failed to load weights: " << error << '\n';
        return 1;
    }

    egaroucid_policy::BoardState board;
    if (mode == "--board") {
        if (!egaroucid_policy::parse_board_65(input, &board)) {
            std::cerr << "invalid board string\n";
            return 1;
        }
        if (side_override >= 0) {
            board.side_to_move = side_override;
        }
    } else if (mode == "--transcript") {
        if (!egaroucid_policy::play_transcript(input, &board, &error)) {
            std::cerr << "invalid transcript: " << error << '\n';
            return 1;
        }
    } else {
        print_usage(argv[0]);
        return 1;
    }

    std::cout << "board:\n";
    print_board(board);

    const uint64_t player = board.side_to_move == egaroucid_policy::BLACK ? board.black : board.white;
    const uint64_t opponent = board.side_to_move == egaroucid_policy::BLACK ? board.white : board.black;
    const auto prob = network.predict(player, opponent);
    const uint64_t legal = egaroucid_policy::legal_moves(player, opponent);

    std::cout << "\ntop " << top_n << ":\n";
    std::cout << std::fixed << std::setprecision(6);
    for (const auto &entry : egaroucid_policy::top_k(prob, top_n)) {
        const int policy = entry.first;
        std::cout << std::setw(2) << policy << " "
                  << egaroucid_policy::coord_from_policy(policy) << " "
                  << std::setw(10) << entry.second;
        if (legal & egaroucid_policy::bit_from_policy(policy)) {
            std::cout << " legal";
        }
        std::cout << '\n';
    }

    print_distribution(prob);
    return 0;
}
