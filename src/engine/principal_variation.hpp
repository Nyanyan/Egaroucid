/*
    Egaroucid Project

    @file principal_variation.hpp
        Get Principal Variation
    @date 2021-2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once
#include <iostream>
#include <future>
#include "setting.hpp"
#include "ai.hpp"

constexpr int PV_LENGTH_SETTING_MIN = 2;
constexpr int PV_LENGTH_SETTING_MAX = 40;

void get_principal_variation_str(Board board, int depth, int max_level, std::string *res) {
    Flip flip;
    for (int level = 1; level <= max_level && global_searching; ++level) {
        std::string pv;
        Board board_cpy = board.copy();
        for (int i = 0; i < depth && !board_cpy.is_end(); ++i) {
            if (board_cpy.get_legal() == 0) {
                board_cpy.pass();
            }
            Search_result search_result = ai_specified(board_cpy, level, true, 0, true, false);
            if (global_searching) {
                int best_move = search_result.policy;
                pv += idx_to_coord(best_move);
                calc_flip(&flip, &board_cpy, best_move);
                board_cpy.move_board(&flip);
            }
        }
        //std::cerr << "pv level " << level << " " << pv << std::endl;
        *res = pv;
    }
}

std::string get_principal_variation_str_tt(Board board, int depth) {
    Flip flip;
    std::string pv;
    Board board_cpy = board.copy();
    for (int i = 0; i < depth && !board_cpy.is_end(); ++i) {
        if (board_cpy.get_legal() == 0) {
            board_cpy.pass();
        }

        Book_value book_result = book.get_specified_best_move(&board_cpy, board_cpy.get_legal());
        if (is_valid_policy(book_result.policy)) {
            pv += idx_to_coord(book_result.policy);
            calc_flip(&flip, &board_cpy, book_result.policy);
            board_cpy.move_board(&flip);
        } else {
            int lower = -SCORE_MAX, upper = SCORE_MAX;
            uint_fast8_t moves[2] = {MOVE_UNDEFINED, MOVE_UNDEFINED};
            int depth;
            uint_fast8_t mpc_level;
            transposition_table.get_info(board_cpy, &lower, &upper, moves, &depth, &mpc_level);
            if (moves[0] != MOVE_UNDEFINED) {
                pv += idx_to_coord(moves[0]);
                calc_flip(&flip, &board_cpy, book_result.policy);
                board_cpy.move_board(&flip);
            } else {
                break;
            }
        }
    }
    return pv;
}