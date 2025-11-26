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
    int adjustment_n_moves = 1;
    int light_n_moves = std::max(0, n_moves - adjustment_n_moves);
    std::cerr << "light_n_moves " << light_n_moves << std::endl;
    bool success = false;
    constexpr int N_MAX_TRY = 10;
    std::vector<int> best_try_result;
    int best_try_error = INF;
    for (int try_count = 0; try_count < N_MAX_TRY && !success && *searching; ++try_count) {
        std::cerr << "try " << try_count << std::endl;
        bool failed = false;

        Board board;
        board.reset();
        int player = BLACK;
        std::vector<int> res;
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
                player ^= 1;
            }
            int acceptable_loss = std::abs(std::round(dist(engine)));
            Search_result search_result = ai_accept_loss(board, light_level, acceptable_loss);
            int policy = search_result.policy;
            res.emplace_back(policy);
            std::cerr << "light " << acceptable_loss << " " << idx_to_coord(policy) << " " << search_result.value << std::endl;
            calc_flip(&flip, &board, policy);
            board.move_board(&flip);
            player ^= 1;
        }
        if (!failed) {
            if (board.is_end()) {
                failed = true;
                break;
            }
            if (board.get_legal() == 0) {
                board.pass();
                player ^= 1;
            }
            int alpha, beta;
            if (player == WHITE) { // next: black
                alpha = score_range_min;
                beta = score_range_max;
            } else { // next: white
                alpha = -score_range_max;
                beta = -score_range_min;
            }
            Search_result search_result = ai_range(board, adjustment_level, alpha, beta, searching);
            int policy = search_result.policy;
            if (search_result.value == SCORE_UNDEFINED) {
                failed = true;
                Search_result search_result2 = ai_range(board, adjustment_level, -SCORE_MAX, alpha, searching);
                int policy2 = search_result2.policy;
                std::cerr << "can't get exact move, new range: " << -SCORE_MAX << " " << alpha << " get " << search_result.value << " " << idx_to_coord(policy2) << std::endl;
                if (search_result2.value != SCORE_UNDEFINED && is_valid_policy(policy2)) {
                    res.emplace_back(policy2);
                    int error = std::abs(alpha - search_result2.value);
                    if (error < best_try_error) {
                        std::cerr << "update error result " << error << std::endl;
                        best_try_error = error;
                        best_try_result = res;
                    }
                } else {
                    int acceptable_loss = std::abs(std::round(dist(engine))) + 2;
                    Search_result search_result3 = ai_accept_loss(board, adjustment_level, acceptable_loss);
                    int policy3 = search_result3.policy;
                    res.emplace_back(policy3);
                    int error = std::abs(alpha - search_result3.value);
                    if (error < best_try_error) {
                        std::cerr << "update error result " << error << std::endl;
                        best_try_error = error;
                        best_try_result = res;
                    }
                }
            } else {
                res.emplace_back(policy);
                std::cerr << "adjust " << idx_to_coord(policy) << " " << search_result.value << std::endl;
            }
        }
        if (!failed) {
            success = true;
            std::cerr << "random board generated" << std::endl;
            return res;
        }
    }
    return best_try_result;
}