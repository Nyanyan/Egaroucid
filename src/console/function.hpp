/*
    Egaroucid Project

    @file function.hpp
        Functions about engine
    @date 2021-2023
    @author Takuto Yamana
    @license GPL-3.0 license
*/

#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include "./../engine/engine_all.hpp"
#include "command.hpp"

#if USE_THREAD_MONITOR
    #include "./../engine/thread_monitor.hpp"
#endif

void setboard(Board_info *board, std::string board_str);
Search_result go_noprint(Board_info *board, Options *options, State *state);
void print_search_result_head();
void print_search_result_body(Search_result result, int level);
void go(Board_info *board, Options *options, State *state);

void solve_problems(std::string file, Options *options, State *state){
    std::ifstream ifs(file);
    if (ifs.fail()){
        std::cerr << "[ERROR] [FATAL] no problem file found" << std::endl;
        return;
    }
    std::string line;
    Board_info board;
    board.reset();
    print_search_result_head();
    Search_result total;
    total.nodes = 0;
    total.time = 0;
    while (std::getline(ifs, line)){
        transposition_table.init();
        setboard(&board, line);
        #if USE_THREAD_MONITOR
            start_thread_monitor();
        #endif
        Search_result res = go_noprint(&board, options, state);
        print_search_result_body(res, options->level);
        total.nodes += res.nodes;
        total.time += res.time;
    }
    std::cout << "total " << total.nodes << " nodes in " << ((double)total.time / 1000) << "s NPS " << (total.nodes * 1000 / total.time) << std::endl;
}

void execute_special_tasks(){
    // move ordering tuning (endsearch)
    #if TUNE_MOVE_ORDERING_END
        std::cout << "tune move ordering (endsearch)" << std::endl;
        tune_move_ordering_end("problem/13_13.txt");
        std::exit(0);
    #endif
}

bool execute_special_tasks_loop(Board_info *board, State *state, Options *options){
    if (options->mode == MODE_HUMAN_AI && board->player == WHITE && !board->board.is_end()){
        go(board, options, state);
        return true;
    } else if (options->mode == MODE_AI_HUMAN && board->player == BLACK && !board->board.is_end()){
        go(board, options, state);
        return true;
    } else if (options->mode == MODE_AI_AI && !board->board.is_end()){
        go(board, options, state);
        return true;
    }
    return false;
}