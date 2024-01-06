/*
    Egaroucid Project

    @file move_ordering_tune_end.hpp
        Move ordering tuning (endsearch)
    @date 2021-2024
    @author Takuto Yamana
    @license GPL-3.0 license
*/

#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <future>
#include "common.hpp"
#include "board.hpp"
#include "search.hpp"
#include "midsearch.hpp"
#include "stability.hpp"

#define N_MOVE_ORDERING_END_PARAMS 4
int W_END_MOBILITY = 1;
int W_END_PARITY = 1;
int W_END_PLAYER_POTENTIAL_MOBILITY = 1;
int W_END_OPPONENT_POTENTIAL_MOBILITY = 1;
int *move_ordering_end_params[N_MOVE_ORDERING_END_PARAMS] = {&W_END_MOBILITY, &W_END_PARITY, &W_END_PLAYER_POTENTIAL_MOBILITY, &W_END_OPPONENT_POTENTIAL_MOBILITY};
int move_ordering_end_params_shift[N_MOVE_ORDERING_END_PARAMS] = {5, 5, 5, 5};
uint64_t move_ordering_end_best_nodes = 0xFFFFFFFFFFFFFFFFULL;
std::vector<Board> move_ordering_end_test_cases;

int nega_alpha_end(Search *search, int alpha, int beta, bool skipped, uint64_t legal, const bool *searching);

uint64_t do_task_test_move_ordering_end(int id, Board board){
    Search search;
    search.n_nodes = 0ULL;
    search.use_multi_thread = false;
    search.use_mpc = false;
    search.mpct = MPC_100_LEVEL;
    const bool searching = true;
    search.init_board(&board);
    //calc_features(&search);
    nega_alpha_end(&search, -SCORE_MAX, SCORE_MAX, false, LEGAL_UNDEFINED, &searching);
    return search.n_nodes;
}

uint64_t test_move_ordering_end(){
    for (int i = 0; i < N_MOVE_ORDERING_END_PARAMS; ++i){
        *move_ordering_end_params[i] = 1 << move_ordering_end_params_shift[i];
    }
    std::vector<std::future<uint64_t>> tasks;
    for (Board board: move_ordering_end_test_cases)
        tasks.emplace_back(thread_pool.push(&do_task_test_move_ordering_end, board));
    uint64_t n_nodes = 0ULL;
    for (std::future<uint64_t> &task: tasks)
        n_nodes += task.get();
    std::cerr << "\r" << n_nodes << " ";
    for (int i = 0; i < N_MOVE_ORDERING_END_PARAMS; ++i){
        std::cerr << move_ordering_end_params_shift[i] << " ";
    }
    return n_nodes;
}

void tune_move_ordering_end(std::string file){
    std::ifstream ifs(file);
    if (ifs.fail()){
        std::cerr << "[ERROR] [FATAL] no problem file found" << std::endl;
        return;
    }
    std::string line;
    Board board;
    while (std::getline(ifs, line)){
        board.player = 0ULL;
        board.opponent = 0ULL;
        for (int i = 0; i < HW2; ++i){
            if (line[i] == 'p')
                board.player |= 1ULL << i;
            else if (line[i] == 'o')
                board.opponent |= 1ULL << i;
        }
        move_ordering_end_test_cases.emplace_back(board);
    }
    while (true){
        int idx = myrandrange(0, N_MOVE_ORDERING_END_PARAMS);
        int delta = myrandom() >= 0.5 ? 1 : -1;
        move_ordering_end_params_shift[idx] += delta;
        if (move_ordering_end_params_shift[idx] > 15 || move_ordering_end_params_shift[idx] < 0)
            move_ordering_end_params_shift[idx] -= delta;
        else{
            uint64_t n_nodes = test_move_ordering_end();
            if (n_nodes < move_ordering_end_best_nodes){
                move_ordering_end_best_nodes = n_nodes;
                std::cerr << std::endl;
            } else
                move_ordering_end_params_shift[idx] -= delta;
        }
    }
}