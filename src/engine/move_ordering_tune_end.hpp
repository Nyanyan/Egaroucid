/*
    Egaroucid Project

    @file move_ordering_tune_end.hpp
        Move ordering tuning (endsearch)
    @date 2021-2022
    @author Takuto Yamana (a.k.a. Nyanyan)
    @license GPL-3.0 license
*/

#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
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
int move_ordering_end_params_shift[N_MOVE_ORDERING_END_PARAMS] = {0, 0, 0, 0};
uint64_t move_ordering_end_best_nodes = 0xFFFFFFFFFFFFFFFFULL;
std::vector<Board> move_ordering_end_test_cases;

int nega_alpha_end(Search *search, int alpha, int beta, bool skipped, uint64_t legal, const bool *searching);

void test_move_ordering_end(int param_idx){
    if (param_idx >= N_MOVE_ORDERING_END_PARAMS){
        value_transposition_table.init();
        best_move_transposition_table.init();
        Search search;
        search.n_nodes = 0ULL;
        search.use_multi_thread = false;
        search.use_mpc = false;
        search.mpct = NOMPC;
        const bool searching = true;
        for (Board &board: move_ordering_end_test_cases){
            search.init_board(&board);
            //calc_features(&search);
            nega_alpha_end(&search, -SCORE_MAX, SCORE_MAX, false, LEGAL_UNDEFINED, &searching);
        }
        std::cerr << "\r" << search.n_nodes << " ";
        for (int i = 0; i < N_MOVE_ORDERING_END_PARAMS; ++i){
            std::cerr << move_ordering_end_params_shift[i] << " ";
        }
        if (search.n_nodes < move_ordering_end_best_nodes){
            move_ordering_end_best_nodes = search.n_nodes;
            std::cerr << std::endl;
        }
        return;
    }
    for (move_ordering_end_params_shift[param_idx] = 0; move_ordering_end_params_shift[param_idx] <= 20; ++move_ordering_end_params_shift[param_idx]){
        *move_ordering_end_params[param_idx] = 1 << move_ordering_end_params_shift[param_idx];
        test_move_ordering_end(param_idx + 1);
    }
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
    test_move_ordering_end(0);
}