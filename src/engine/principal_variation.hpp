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
#include "midsearch.hpp"
#include "book.hpp"
#include "util.hpp"
#include "clogsearch.hpp"
#include "lazy_smp.hpp"

#define PRINCIPAL_VARIATION_MAX_LEN 7

std::string get_principal_variation_str(Board board){
    std::string res;
    Flip flip;
    int len = 0;
    bool pv_search_failed = false;
    while (!pv_search_failed && len < PRINCIPAL_VARIATION_MAX_LEN){
        pv_search_failed = true;
        if (board.is_end()){
            break; // game over
        }
        if (board.get_legal() == 0){
            board.pass();
        }
        uint64_t legal = board.get_legal();
        int best_move_book = book.get_specified_best_move(&board).policy;
        if (is_valid_policy(best_move_book) && (legal & (1ULL << best_move_book))){ // search best move by book
            res += idx_to_coord(best_move_book);
            ++len;
            calc_flip(&flip, &board, best_move_book);
            board.move_board(&flip);
            pv_search_failed = false;
        } else{                                                                     // search pv by TT
            //int best_move_tt = transposition_table.get_best_move(&board, board.hash());
            //if (is_valid_policy(best_move_tt) && (legal & (1ULL << best_move_tt))){
            int l, u;
            int best_move_tt_expanded = MOVE_UNDEFINED;
            int max_val = -SCORE_MAX;
            uint64_t legal2 = legal;
            for (uint_fast8_t cell = first_bit(&legal2); legal2; cell = next_bit(&legal2)){
                calc_flip(&flip, &board, cell);
                board.move_board(&flip);
                    l = -SCORE_MAX;
                    u = SCORE_MAX;
                    transposition_table.get_bounds_any_level(&board, board.hash(), &l, &u);
                    if (max_val < -u || (max_val <= -u && u == l)){
                        max_val = -u;
                        best_move_tt_expanded = cell;
                    }
                board.undo_board(&flip);
            }
            if (best_move_tt_expanded != MOVE_UNDEFINED){
                res += idx_to_coord(best_move_tt_expanded);
                ++len;
                calc_flip(&flip, &board, best_move_tt_expanded);
                board.move_board(&flip);
                pv_search_failed = false;
            }
            //}
        }
    }
    return res;
}