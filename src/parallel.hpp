/*
    Egaroucid Project

    @date 2021-2022
    @author Takuto Yamana (a.k.a Nyanyan)
    @license GPL-3.0 license
*/

#pragma once
#include <iostream>
#include <functional>
#include "setting.hpp"
#include "board.hpp"

struct Parallel_task{
    int value;
    uint64_t n_nodes;
    uint_fast8_t cell;

    Parallel_task copy(){
        Parallel_task res;
        res.value = value;
        res.n_nodes = n_nodes;
        res.cell = cell;
        return res;
    }
};

struct Parallel_args{
    uint64_t player;
    uint64_t opponent;
    uint_fast8_t n_discs;
    uint_fast8_t parity;
    bool use_mpc;
    double mpct;
    int alpha;
    int beta;
    int depth;
    uint64_t legal;
    bool is_end_search;
    int policy;
};

struct Parallel_node{
    bool is_waiting;
    bool is_helping;
    function<Parallel_task()> help_task;
    int help_res;
    Parallel_node *parent;
};
