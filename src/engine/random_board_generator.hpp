/*
    Egaroucid Project

    @file random_board_generator.hpp
        Random Board Generator
    @date 2021-2025
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/
#pragma once
#include "ai.hpp"

std::vector<int> random_board_generator(int score_range_min, int score_range_max, int n_moves, int light_level, int adjustment_level, bool *searching) {
    int light_n_moves = std::max(0, n_moves - 2);
    int adjustment_n_moves = n_moves - light_n_moves;
    bool success = false;
    std::vector<int> res;
    while (!success && *searching) {
        bool failed = false;

        Board board;
        board.reset();
        res.clear();
        Flip flip;

        std::random_device seed_gen;
        std::default_random_engine engine(seed_gen());
        std::normal_distribution<> dist(0.0, 4.0); // acceptable loss avg = 0.0, sd = 4.0 discs
        for (int i = 0; i < light_n_moves; ++i) {
            if (board.is_end()) {
                failed = true;
                break;
            }
            if (board.get_legal() == 0) {
                board.pass();
            }
            int acceptable_loss = std::abs(std::round(dist(engine)));
            Search_result search_result = ai_accept_loss(board, light_level, acceptable_loss);
            int policy = search_result.policy;
            res.emplace_back(policy);
            std::cerr << "light " << acceptable_loss << " " << idx_to_coord(policy) << " " << search_result.value << std::endl;
            calc_flip(&flip, &board, policy);
            board.move_board(&flip);
        }
        if (!failed) {
            for (int i = 0; i < adjustment_n_moves; ++i) {
                if (board.is_end()) {
                    failed = true;
                    break;
                }
                if (board.get_legal() == 0) {
                    board.pass();
                }
                Search_result search_result = ai_range(board, adjustment_level, score_range_min, score_range_max, searching);
                if (search_result.value == SCORE_UNDEFINED) {
                    failed = true;
                    break;
                }
                int policy = search_result.policy;
                res.emplace_back(policy);
                std::cerr << "adjust " << idx_to_coord(policy) << " " << search_result.value << std::endl;
                calc_flip(&flip, &board, policy);
                board.move_board(&flip);
            }
        }
        if (!failed) {
            success = true;
            std::cerr << "random board generated" << std::endl;
        }
    }
    if (res.size() != light_n_moves + adjustment_n_moves) {
        res.clear();
    }
    return res;
}