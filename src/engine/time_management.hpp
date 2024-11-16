/*
    Egaroucid Project

    @file time_management.hpp
        Time management system
    @date 2021-2024
    @author Takuto Yamana
    @license GPL-3.0 license
*/

#pragma once
#include "board.hpp"



#define TIME_MANAGEMENT_REMAINING_TIME_OFFSET 200 // ms / move
//#define TIME_MANAGEMENT_REMAINING_MOVES_OFFSET 12 // moves (fast complete search = 24 moves)
#define TIME_MANAGEMENT_REMAINING_MOVES_OFFSET 5

uint64_t calc_time_limit_ply(const Board board, uint64_t remaining_time_msec, bool show_log) {
    int n_empties = HW2 - board.n_discs();

    // try complete search
    // Nodes(depth) = a * exp(b * depth)
    constexpr double complete_const_a = 0.6;
    constexpr double complete_const_b = 0.75;
    constexpr double complete_nps = 120000000.0;
    double complete_use_time = (double)remaining_time_msec * 0.8;
    double complete_search_depth = log(complete_use_time / 1000.0 * complete_nps / complete_const_a) / complete_const_b;

    // try endgame search
    // Nodes(depth) = a * exp(b * depth)
    constexpr double endgame_const_a = 0.3;
    constexpr double endgame_const_b = 0.65;
    constexpr double endgame_nps = 90000000.0;
    double endgame_use_time = (double)remaining_time_msec * 0.25;
    double endgame_search_depth = log(endgame_use_time / 1000.0 * endgame_nps / endgame_const_a) / endgame_const_b;


    if (show_log) {
        std::cerr << "complete search depth " << complete_search_depth << " endgame search depth " << endgame_search_depth << std::endl;
    }
    if (n_empties <= complete_search_depth) {
        if (show_log) {
            std::cerr << "try complete search" << std::endl;
        }
        return complete_use_time;
    }
    if (n_empties <= endgame_search_depth) {
        if (show_log) {
            std::cerr << "try endgame search" << std::endl;
        }
        return endgame_use_time;
    }

    // midgame search
    int remaining_moves = (n_empties + 1) / 2;
    if (remaining_time_msec > TIME_MANAGEMENT_REMAINING_TIME_OFFSET * remaining_moves) {
        uint64_t remaining_time_msec_proc = remaining_time_msec - TIME_MANAGEMENT_REMAINING_TIME_OFFSET * remaining_moves;
        int remaining_moves_proc = std::max(2, remaining_moves - TIME_MANAGEMENT_REMAINING_MOVES_OFFSET);
        return remaining_time_msec_proc / remaining_moves_proc;
    }
    return 1;
}

