/*
    Egaroucid Project

    @file time_management.hpp
        Time management system
    @date 2021-2025
    @author Takuto Yamana
    @license GPL-3.0 license
*/

#pragma once
#include "ai.hpp"

#if IS_GGS_TOURNAMENT
constexpr int TIME_MANAGEMENT_INITIAL_N_EMPTIES = 50;
#endif

#define TIME_MANAGEMENT_REMAINING_TIME_OFFSET 10 // ms / move
#define TIME_MANAGEMENT_REMAINING_MOVES_OFFSET 13 // 13 * 2 = 26 moves
#define TIME_MANAGEMENT_N_MOVES_COE 0.9 // 10% early break

Search_result ai(Board board, int level, bool use_book, int book_acc_level, bool use_multi_thread, bool show_log);

uint64_t calc_time_limit_ply(const Board board, uint64_t remaining_time_msec, bool show_log) {
    int n_empties = HW2 - board.n_discs();
    int remaining_moves = (n_empties + 1) / 2;
    uint64_t remaining_time_msec_margin = remaining_time_msec;
    if (remaining_time_msec > TIME_MANAGEMENT_REMAINING_TIME_OFFSET * remaining_moves) {
        remaining_time_msec_margin -= TIME_MANAGEMENT_REMAINING_TIME_OFFSET * remaining_moves;
    } else {
        remaining_time_msec_margin = 1;
    }

#if IS_GGS_TOURNAMENT
    // first move
    if (n_empties == TIME_MANAGEMENT_INITIAL_N_EMPTIES) {
        if (show_log) {
            std::cerr << "first move time limit" << std::endl;
        }
        return remaining_time_msec_margin * 0.15;
    }
#endif

    // try complete search
    // Nodes(depth) = a * exp(b * depth)
    constexpr double complete_const_a = 0.70; //2.1747;
    constexpr double complete_const_b = 0.76;
    constexpr double complete_nps = 6.0e8;
    double complete_use_time = (double)remaining_time_msec_margin * 0.9;
    double complete_search_depth = log(complete_use_time / 1000.0 * complete_nps / complete_const_a) / complete_const_b;

    // try endgame search
    // Nodes(depth) = a * exp(b * depth)
    constexpr double endgame_const_a = 0.05; //1.8654;
    constexpr double endgame_const_b = 0.62;
    constexpr double endgame_nps = 3.5e8;
    double endgame_use_time = (double)remaining_time_msec_margin * 0.15;
    double endgame_search_depth = log(endgame_use_time / 1000.0 * endgame_nps / endgame_const_a) / endgame_const_b;

    if (show_log) {
        std::cerr << "complete search depth " << complete_search_depth << " endgame search depth " << endgame_search_depth << " n_empties " << n_empties << std::endl;
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
    int remaining_moves_proc = std::max(2, (int)round((remaining_moves - TIME_MANAGEMENT_REMAINING_MOVES_OFFSET) * TIME_MANAGEMENT_N_MOVES_COE)); // at least 2 moves
    return std::max<uint64_t>(1ULL, remaining_time_msec_margin / remaining_moves_proc);
}

uint64_t request_more_time(Board board, uint64_t remaining_time_msec, uint64_t time_limit, bool show_log) {
    int n_empties = HW2 - board.n_discs();
    int remaining_moves = (n_empties + 1) / 2;
    uint64_t remaining_time_msec_margin = remaining_time_msec;
    if (remaining_time_msec > TIME_MANAGEMENT_REMAINING_TIME_OFFSET * remaining_moves) {
        remaining_time_msec_margin -= TIME_MANAGEMENT_REMAINING_TIME_OFFSET * remaining_moves;
    } else {
        remaining_time_msec_margin = 1;
    }
    std::cerr << "requesting more time remaining " << remaining_time_msec << " remaining_margin " << remaining_time_msec_margin << " now tl " << time_limit << std::endl;
    if (remaining_time_msec_margin > time_limit) {
        int remaining_moves_proc = std::max(2, (int)round((remaining_moves - TIME_MANAGEMENT_REMAINING_MOVES_OFFSET) * TIME_MANAGEMENT_N_MOVES_COE)); // at least 2 moves
        uint64_t additional_time = (remaining_time_msec_margin - time_limit) / remaining_moves_proc * 0.4;
        if (show_log) {
            std::cerr << "additional time " << additional_time << std::endl;
        }
        time_limit += additional_time;
    }
    return time_limit;
}


void selfplay(Board board, int level) {
    Search_result result;
    Flip flip;
    while (board.check_pass()) {
        result = ai(board, level, true, 0, false, false);
        calc_flip(&flip, &board, result.policy);
        board.move_board(&flip);
    }
}


void time_management_selfplay(Board board, bool show_log, uint64_t use_legal, uint64_t time_limit) {
    uint64_t start_time = tim();
    if (show_log) {
        std::cerr << "self play time " << time_limit << " ms" << std::endl;
    }
    std::vector<Flip> move_list(pop_count_ull(use_legal));
    int idx = 0;
    for (uint_fast8_t cell = first_bit(&use_legal); use_legal; cell = next_bit(&use_legal)) {
        calc_flip(&move_list[idx], &board, cell);
        ++idx;
    }
    int select_idx = 0;
    uint64_t count = 0;
    while (tim() - start_time < time_limit) {
        board.move_board(&move_list[select_idx]);
            selfplay(board, 11);
        board.undo_board(&move_list[select_idx]);
        ++count;
        ++select_idx;
        select_idx %= move_list.size();
    }
    if (show_log) {
        std::cerr << "self play count " << count << std::endl;
    }
}