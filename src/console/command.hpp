/*
    Egaroucid Project

    @file command.hpp
        Commands for Egaroucid
    @date 2021-2022
    @author Takuto Yamana (a.k.a. Nyanyan)
    @license GPL-3.0 license
*/

#pragma once
#include <iostream>
#include <string>
#include "./../engine/engine_all.hpp"
#include "board_info.hpp"
#include "option.hpp"
#include "state.hpp"

void check_command(Board_info *board, State *state, Options options){
    std::cout << "> ";
    std::string cmd_line;
    std::getline(std::cin, cmd_line);
}