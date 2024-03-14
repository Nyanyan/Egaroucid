/*
    Egaroucid Project

    @file parallel.hpp
        Structures for parallel search
    @date 2021-2024
    @author Takuto Yamana
    @license GPL-3.0 license
*/

#pragma once
#include <iostream>
#include <functional>
#include "setting.hpp"
#include "board.hpp"

/*
    @brief Structure for Parallel search

    @param value                search result
    @param n_nodes              number of nodes visited
    @param cell                 last move (for transposition table)
*/
struct Parallel_task{
    int value;
    uint64_t n_nodes;
    uint_fast8_t cell;
    int move_idx;
};