/*
    Egaroucid Project

    @file minimax.hpp
        Minimax Search
    @date 2021-2024
    @author Takuto Yamana
    @license GPL-3.0 license
*/

#pragma once
#include "evaluate.hpp"

int minimax_p(Search *search, int depth, bool passed) {
    if (depth == 0) {
        return mid_evaluate_diff(search);
    }
    int v = -SCORE_INF;
    uint64_t legal = search->board.get_legal();
    if (legal == 0) {
        if (passed) {
            return search->board.score_player(); // game over
        }
        search->pass();
            v = -minimax_p(search, depth, true); // pass NOT counted as 1 move
        search->pass();
        return v;
    }
    Flip flip;
    for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)) {
        calc_flip(&flip, &search->board, cell);
        search->move(&flip);
            v = std::max(v, minimax_p(search, depth - 1, false));
        search->undo(&flip);
    }
    return v;
}

int minimax(Board *board, int depth) {
    Search search(board);
    return minimax_p(&search, depth, false);
}