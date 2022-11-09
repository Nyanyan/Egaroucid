/*
    Egaroucid Project

    @file parallel.hpp
        Structures for parallel search
    @date 2021-2022
    @author Takuto Yamana (a.k.a. Nyanyan)
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
    @param mpc_used             MPC used? (for transposition table)
*/
struct Parallel_task{
    int value;
    uint64_t n_nodes;
    uint_fast8_t cell;
    bool mpc_used;

    Parallel_task copy(){
        Parallel_task res;
        res.value = value;
        res.n_nodes = n_nodes;
        res.cell = cell;
        res.mpc_used = mpc_used;
        return res;
    }
};
