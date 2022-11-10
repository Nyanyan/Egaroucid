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
#include <sstream>
#include "./../engine/engine_all.hpp"
#include "board_info.hpp"
#include "option.hpp"
#include "state.hpp"
#include "close.hpp"
#include "print.hpp"
#include "command_definition.hpp"

std::string get_command_line(){
    std::cout << "> ";
    std::string cmd_line;
    std::getline(std::cin, cmd_line);
    return cmd_line;
}

void split_cmd_arg(std::string cmd_line, std::string *cmd, std::string *arg){
    std::istringstream iss(cmd_line);
    iss >> *cmd;
    if (cmd->length() + 1 < cmd_line.length())
        *arg = cmd_line.substr(cmd->length() + 1);
}

int get_command_id(std::string cmd){
    for (int i = 0; i < N_COMMANDS; ++i){
        if (std::find(command_data[i].names.begin(), command_data[i].names.end(), cmd) != command_data[i].names.end())
            return command_data[i].id;
    }
    return COMMAND_NOT_FOUND;
}

void init_board(Board_info *board){
    board->reset();
}

void new_board(Board_info *board){
    board->board = board->first_board.copy();
    board->player = board->first_player;
}

void check_command(Board_info *board, State *state, Options *options){
    std::cout << std::endl;
    print_board_info(board);
    std::string cmd_line = get_command_line();
    std::string cmd, arg;
    split_cmd_arg(cmd_line, &cmd, &arg);
    int cmd_id = get_command_id(cmd);
    if (cmd_id == COMMAND_NOT_FOUND)
        std::cout << "[ERROR] command `" << cmd << "` not found" << std::endl;
    else if (cmd_id == CMD_ID_HELP)
        print_commands_list();
    else if (cmd_id == CMD_ID_EXIT)
        close(state, options);
    else if (cmd_id == CMD_ID_VERSION)
        print_version();
    else if (cmd_id == CMD_ID_INIT)
        init_board(board);
    else if (cmd_id == CMD_ID_NEW)
        new_board(board);
}