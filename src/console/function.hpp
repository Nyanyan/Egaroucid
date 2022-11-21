/*
    Egaroucid Project

    @file function.hpp
        Functions about engine
    @date 2021-2022
    @author Takuto Yamana (a.k.a. Nyanyan)
    @license GPL-3.0 license
*/

#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include "./../engine/engine_all.hpp"
#include "command.hpp"

void setboard(Board_info *board, std::string board_str);
Search_result go_noprint(Board_info *board, Options *options);
void print_search_result_head();
void print_search_result_body(Search_result result, int level);

void solve_problems(std::string file, Options *options){
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
        setboard(&board, line);
        Search_result res = go_noprint(&board, options);
        print_search_result_body(res, options->level);
        total.nodes += res.nodes;
        total.time += res.time;
    }
    std::cout << "total " << total.nodes << " nodes in " << ((double)total.time / 1000) << "s NPS " << (total.nodes * 1000 / total.time) << std::endl;
}