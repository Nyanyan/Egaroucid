/*
    Egaroucid Project

    @file board.hpp
        Board class
    @date 2021-2024
    @author Takuto Yamana
    @license GPL-3.0 license
*/

#pragma once
#include "board.hpp"

uint64_t perft(Board *board, int depth, bool passed){
    if (depth == 0){
        return 1ULL;
    }
    uint64_t res = 0;
    uint64_t legal = board->get_legal();
    if (legal == 0){
        if (passed){
            return 1ULL; // game over
        }
        board->pass();
            res = perft(board, depth - 1, true); // pass counted as 1 move
        board->pass();
        return res;
    }
    if (depth == 1){
        return pop_count_ull(legal); // speedup with bulk-counting
    }
    Flip flip;
    for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)){
        calc_flip(&flip, board, cell);
        board->move_board(&flip);
            res += perft(board, depth - 1, false);
        board->undo_board(&flip);
    }
    return res;
}

uint64_t perft_no_pass_count(Board *board, int depth, bool passed){
    if (depth == 0){
        return 1ULL;
    }
    uint64_t res = 0;
    uint64_t legal = board->get_legal();
    if (legal == 0){
        if (passed){
            return 1ULL; // game over
        }
        board->pass();
            res = perft(board, depth, true); // pass not counted as 1 move
        board->pass();
        return res;
    }
    if (depth == 1){
        return pop_count_ull(legal); // speedup with bulk-counting
    }
    Flip flip;
    for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)){
        calc_flip(&flip, board, cell);
        board->move_board(&flip);
            res += perft(board, depth - 1, false);
        board->undo_board(&flip);
    }
    return res;
}