/*
    Egaroucid Project

    @file principal_variation.hpp
        Get Principal Variation
    @date 2021-2024
    @author Takuto Yamana
    @license GPL-3.0 license
*/

#pragma once
#include <iostream>
#include <future>
#include "setting.hpp"
#include "ai.hpp"

#define PRINCIPAL_VARIATION_MAX_LEN 7

void get_principal_variation_str(Board board, int max_level, std::string *res){
    Flip flip;
    for (int level = 1; level <= max_level; ++level){
        std::string pv;
        Board board_cpy = board.copy();
        for (int i = 0; i < PRINCIPAL_VARIATION_MAX_LEN && !board_cpy.is_end(); ++i){
            if (board_cpy.get_legal() == 0){
                board_cpy.pass();
            }
            Search_result search_result = ai_specified(board_cpy, level, true, 0, true, false);
            int best_move = search_result.policy;
            pv += idx_to_coord(best_move);
            calc_flip(&flip, &board_cpy, best_move);
            board_cpy.move_board(&flip);
        }
        std::cerr << "pv level " << level << " " << pv << std::endl;
        *res = pv;
    }
}