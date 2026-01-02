/*
    Egaroucid Project

    @file time_management.hpp
        Time management system
    @date 2021-2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once
#include "ai.hpp"

// #if IS_GGS_TOURNAMENT
// constexpr int TIME_MANAGEMENT_INITIAL_N_EMPTIES = 50; // 64 - 14 (s8r14)
// #endif

#define TIME_MANAGEMENT_REMAINING_TIME_OFFSET 400 // ms / move
#define TIME_MANAGEMENT_REMAINING_TIME_OFFSET_BASE 5000 // ms
// #define TIME_MANAGEMENT_REMAINING_MOVES_OFFSET 15 // 15 * 2 = 30 moves
#define TIME_MANAGEMENT_N_MOVES_COE_30_OR_MORE 1.4 // 1.2
#define TIME_MANAGEMENT_N_MOVES_COE_40_OR_MORE_ADDITIONAL 0.2 // 0.5 // additional search
#define TIME_MANAGEMENT_N_MOVES_COE_30_OR_MORE_NOTIME 1.2
#define TIME_MANAGEMENT_ADDITIONAL_TIME_COE_BASE 1.8
#define TIME_MANAGEMENT_ADDITIONAL_TIME_COE_ADD 1.8
//#define TIME_MANAGEMENT_N_MOVES_COE_ADDITIONAL_TIME 0.97

Search_result ai(Board board, int level, bool use_book, int book_acc_level, bool use_multi_thread, bool show_log);

uint64_t calc_time_limit_ply(const Board board, uint64_t remaining_time_msec, bool show_log) {
    int n_empties = HW2 - board.n_discs();
    double remaining_moves = (double)(n_empties + 1) / 2.0;
    uint64_t remaining_time_msec_margin = remaining_time_msec;
    if (remaining_time_msec > TIME_MANAGEMENT_REMAINING_TIME_OFFSET * remaining_moves + TIME_MANAGEMENT_REMAINING_TIME_OFFSET_BASE) {
        remaining_time_msec_margin -= TIME_MANAGEMENT_REMAINING_TIME_OFFSET * remaining_moves + TIME_MANAGEMENT_REMAINING_TIME_OFFSET_BASE;
    } else {
        if (show_log) {
            std::cerr << "don't have enough time! remaining " << remaining_time_msec_margin << std::endl;
        }
    }

// #if IS_GGS_TOURNAMENT
//     // first move
//     if (n_empties == TIME_MANAGEMENT_INITIAL_N_EMPTIES) {
//         if (show_log) {
//             std::cerr << "first move time limit" << std::endl;
//         }
//         return remaining_time_msec_margin * 0.15;
//     }
// #endif

    // try complete search
    // Nodes(depth) = a * exp(b * depth)
    constexpr double complete_const_a = 0.60; //2.1747;
    constexpr double complete_const_b = 0.75;
    constexpr double complete_nps = 7.0e8;
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

    // midgame search time
    double remaining_moves_proc = 0;
    if (remaining_time_msec_margin < remaining_time_msec) {
        if (remaining_moves >= 30 / 2) { // 30 or more
            remaining_moves_proc += (remaining_moves - (30 / 2)) * TIME_MANAGEMENT_N_MOVES_COE_30_OR_MORE;
        }
        if (remaining_moves >= 40 / 2) { // 40 or more
            remaining_moves_proc += (remaining_moves - (40 / 2)) * TIME_MANAGEMENT_N_MOVES_COE_40_OR_MORE_ADDITIONAL;
        }
    } else {
        remaining_moves_proc += (remaining_moves - (26 / 2)) * TIME_MANAGEMENT_N_MOVES_COE_30_OR_MORE_NOTIME;
    }
    remaining_moves_proc = std::max(2.0, remaining_moves_proc); // at least 2 moves
    uint64_t midgame_use_time = std::max<uint64_t>(1ULL, (uint64_t)(remaining_time_msec_margin / remaining_moves_proc));

    if (n_empties <= complete_search_depth) {
        if (show_log) {
            std::cerr << "try complete search tl max(" << complete_use_time << ", " << midgame_use_time << ")" << std::endl;
        }
        return std::max((uint64_t)complete_use_time, midgame_use_time);
    }
    if (n_empties <= endgame_search_depth) {
        if (show_log) {
            std::cerr << "try endgame search tl max(" << endgame_use_time << ", " << midgame_use_time << ")" << std::endl;
        }
        return std::max((uint64_t)endgame_use_time, midgame_use_time);
    }
    return midgame_use_time;
}

uint64_t calc_time_limit_ply_MCTS(const Board board, uint64_t remaining_time_msec, bool show_log) {
    int n_empties = HW2 - board.n_discs();
    double remaining_moves = (double)(n_empties + 1) / 2.0;
    uint64_t remaining_time_msec_margin = remaining_time_msec;
    if (remaining_time_msec > TIME_MANAGEMENT_REMAINING_TIME_OFFSET * remaining_moves + TIME_MANAGEMENT_REMAINING_TIME_OFFSET_BASE) {
        remaining_time_msec_margin -= TIME_MANAGEMENT_REMAINING_TIME_OFFSET * remaining_moves + TIME_MANAGEMENT_REMAINING_TIME_OFFSET_BASE;
    } else {
        if (show_log) {
            std::cerr << "don't have enough time! remaining " << remaining_time_msec_margin << std::endl;
        }
    }

// #if IS_GGS_TOURNAMENT
//     // first move
//     if (n_empties == TIME_MANAGEMENT_INITIAL_N_EMPTIES || n_empties == TIME_MANAGEMENT_INITIAL_N_EMPTIES - 1) {
//         if (show_log) {
//             std::cerr << "first move time limit" << std::endl;
//         }
//         return remaining_time_msec_margin * 0.4;
//     }
// #endif

    // try complete search
    // Nodes(depth) = a * exp(b * depth)
    constexpr double complete_const_a = 0.60; //2.1747;
    constexpr double complete_const_b = 0.75;
    constexpr double complete_nps = 7.0e8;
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

    // midgame search time
    double remaining_moves_proc = 0;
    if (remaining_moves > 30 / 2) {
        remaining_moves_proc += remaining_moves - 30 / 2;
    }
    remaining_moves_proc = std::max(2.0, remaining_moves_proc); // at least 2 moves
    double coe = 1.0;
    if (n_empties >= 35) {
        coe += 1.5 * (double)(n_empties - 35) / (60.0 - 35.0);
    }
    std::cerr << "n_empties " << n_empties << " coe " << coe << std::endl;
    uint64_t midgame_use_time = std::max<uint64_t>(1ULL, (uint64_t)(coe * remaining_time_msec_margin / remaining_moves_proc));

    if (n_empties <= complete_search_depth) {
        if (show_log) {
            std::cerr << "try complete search tl max(" << complete_use_time << ", " << midgame_use_time << ")" << std::endl;
        }
        return std::max((uint64_t)complete_use_time, midgame_use_time);
    }
    if (n_empties <= endgame_search_depth) {
        if (show_log) {
            std::cerr << "try endgame search tl max(" << endgame_use_time << ", " << midgame_use_time << ")" << std::endl;
        }
        return std::max((uint64_t)endgame_use_time, midgame_use_time);
    }
    return midgame_use_time;
}

uint64_t request_more_time(Board board, uint64_t remaining_time_msec, uint64_t time_limit, bool show_log) {
    int n_empties = HW2 - board.n_discs();
    double remaining_moves = (double)(n_empties + 1) / 2.0;
    uint64_t remaining_time_msec_margin = remaining_time_msec;
    if (remaining_time_msec > TIME_MANAGEMENT_REMAINING_TIME_OFFSET * remaining_moves + TIME_MANAGEMENT_REMAINING_TIME_OFFSET_BASE) {
        remaining_time_msec_margin -= TIME_MANAGEMENT_REMAINING_TIME_OFFSET * remaining_moves + TIME_MANAGEMENT_REMAINING_TIME_OFFSET_BASE;
    } else {
        remaining_time_msec_margin = 1;
    }
    if (show_log) {
        std::cerr << "requesting more time remaining " << remaining_time_msec << " remaining_margin " << remaining_time_msec_margin << " tl before " << time_limit << std::endl;
    }
    if (remaining_time_msec_margin > time_limit && remaining_time_msec_margin > 40000ULL) {
        // int remaining_moves_proc = std::max(2, (int)round((remaining_moves - TIME_MANAGEMENT_REMAINING_MOVES_OFFSET) * TIME_MANAGEMENT_N_MOVES_COE_ADDITIONAL_TIME)); // at least 2 moves
        int remaining_moves_proc = 0;
        if (remaining_moves >= 30 / 2) { // 30 or more
            remaining_moves_proc += std::round((remaining_moves - (30 / 2)) * TIME_MANAGEMENT_N_MOVES_COE_30_OR_MORE);
        }
        if (remaining_moves >= 40 / 2) { // 40 or more
            remaining_moves_proc += std::round((remaining_moves - (40 / 2)) * TIME_MANAGEMENT_N_MOVES_COE_40_OR_MORE_ADDITIONAL);
        }
        remaining_moves_proc = std::max(2, remaining_moves_proc); // at least 2 moves
        double coe = TIME_MANAGEMENT_ADDITIONAL_TIME_COE_BASE;
        if (remaining_moves >= 40 / 2) { // 40 or more
            coe += TIME_MANAGEMENT_ADDITIONAL_TIME_COE_ADD * (double)(remaining_moves - (40.0 / 2.0)) / (60.0 / 2.0 - 40.0 / 2.0);
        }
        if (show_log) {
            std::cerr << "time request coe " << coe << std::endl;
        }
        uint64_t additional_time = (remaining_time_msec_margin - time_limit) / remaining_moves_proc * coe;
        additional_time = std::min(additional_time, remaining_time_msec_margin / 2);
        if (show_log) {
            std::cerr << "additional time " << additional_time << std::endl;
        }
        time_limit += additional_time;
    }
    if (show_log) {
        std::cerr << "more time requested: new time limit " << time_limit << std::endl;
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