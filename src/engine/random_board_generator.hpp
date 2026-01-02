/*
    Egaroucid Project

    @file random_board_generator.hpp
        Random Board Generator
    @date 2021-2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/
#pragma once
#include "ai.hpp"

std::vector<int> random_board_generator(int score_range_min, int score_range_max, int n_moves, int light_level, int adjustment_level, bool *searching) {
    uint64_t strt = tim();
    int light_n_moves = std::max(0, n_moves - 2);
    int adjustment_n_moves = n_moves - light_n_moves;
    bool success = false;
    std::vector<int> res;
    constexpr int MAX_N_TRY = 40;
    for (int try_count = 0; try_count < MAX_N_TRY && !success && *searching; ++try_count) {
        std::string ms_per_itr_str = "-";
        if (try_count > 0) {
            uint64_t ms_per_itr = (tim() - strt) / try_count;
            ms_per_itr_str = std::to_string(ms_per_itr);
        }
        std::cerr << "try " << try_count << " start at " << tim() - strt << " ms " << ms_per_itr_str << " ms/it" << std::endl;
        bool failed = false;

        Board board;
        board.reset();
        int player = BLACK;
        res.clear();
        Flip flip;

        std::random_device seed_gen;
        std::default_random_engine engine(seed_gen());
        std::normal_distribution<> dist(0.0, 6.0); // acceptable loss avg = 0.0, sd = 6.0 discs
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
            // std::cerr << "light " << acceptable_loss << " " << idx_to_coord(policy) << " " << search_result.value << std::endl;
            calc_flip(&flip, &board, policy);
            board.move_board(&flip);
            player ^= 1;
        }
        if (!failed) {
            for (int i = 0; i < adjustment_n_moves; ++i) {
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
                int adjustment_level_now = (adjustment_level * (i + 1) + light_level * (adjustment_n_moves - 1 - i)) / adjustment_n_moves;
                Search_result search_result = ai_range(board, adjustment_level_now, alpha, beta, searching);
                if (search_result.value == SCORE_UNDEFINED) { // adjust failed
                    if (i == adjustment_n_moves - 1) { // last move
                        failed = true;
                        break;
                    } else {
                        // use accept_loss search
                        int acceptable_loss = std::abs(std::round(dist(engine)));
                        Search_result search_result2 = ai_accept_loss(board, light_level, acceptable_loss);
                        int policy2 = search_result2.policy;
                        res.emplace_back(policy2);
                        calc_flip(&flip, &board, policy2);
                        board.move_board(&flip);
                        player ^= 1;
                    }
                } else { // adjusted
                    int policy = search_result.policy;
                    res.emplace_back(policy);
                    // std::cerr << "adjust " << idx_to_coord(policy) << " " << search_result.value << std::endl;
                    calc_flip(&flip, &board, policy);
                    board.move_board(&flip);
                    player ^= 1;
                }
            }
        }
        if (!failed) {
            success = true;
            std::cerr << "random board generated in " << tim() - strt << " ms" << std::endl;
        }
    }
    if (res.size() != n_moves) {
        res.clear();
    }
    return res;
}