/*
    Egaroucid Project

    @file gtp_command.hpp
        Commands for GTP (Gp Text Protocol)
    @date 2021-2022
    @author Takuto Yamana (a.k.a. Nyanyan)
    @license GPL-3.0 license
*/

#pragma once
#include <iostream>
#include <string>
#include <sstream>
#include <cctype>
#include <algorithm>
#include "./../engine/engine_all.hpp"
#include "board_info.hpp"
#include "option.hpp"
#include "state.hpp"
#include "close.hpp"
#include "print.hpp"
#include "gtp_command_definition.hpp"
#include "command.hpp"

#define GTP_VERSION 2
#define GTP_ENDL "\n\n"
#define GTP_ID_NOT_FOUND -1000000000

std::string gtp_get_command_line(){
    std::string cmd_line;
    std::getline(std::cin, cmd_line);
    return cmd_line;
}

void gtp_split_cmd_arg(int *id, std::string cmd_line, std::string *cmd, std::string *arg){
    std::istringstream iss(cmd_line);
    std::string first_elem;
    iss >> first_elem;
    try{
        *id = std::stoi(first_elem);
        iss >> *cmd;
    } catch (const std::invalid_argument& e) {
        *id = GTP_ID_NOT_FOUND;
        *cmd = first_elem;
    } catch (const std::out_of_range& e) {
        *id = GTP_ID_NOT_FOUND;
        *cmd = first_elem;
    }
    std::getline(iss, *arg);
}

int gtp_get_command_id(std::string cmd){
    for (int i = 0; i < N_GTP_COMMANDS; ++i){
        if (std::find(gtp_command_data[i].names.begin(), gtp_command_data[i].names.end(), cmd) != gtp_command_data[i].names.end())
            return gtp_command_data[i].id;
    }
    return COMMAND_NOT_FOUND;
}

std::string str_gtp_head(int id){
    if (id == GTP_ID_NOT_FOUND)
        return "=";
    return "=" + std::to_string(id);
}

std::string str_gtp_error_head(int id){
    if (id == GTP_ID_NOT_FOUND)
        return "?";
    return "?" + std::to_string(id);
}

void print_gtp_version(int id){
    std::cout << str_gtp_head(id) << " " << GTP_VERSION << GTP_ENDL;
}

void gtp_check_command(Board_info *board, State *state, Options *options){
    std::string cmd_line = gtp_get_command_line();
    std::string cmd, arg;
    int id;
    gtp_split_cmd_arg(&id, cmd_line, &cmd, &arg);
    int cmd_id = gtp_get_command_id(cmd);
    if (cmd_id == COMMAND_NOT_FOUND)
        std::cout << str_gtp_error_head(id) << " " << "[ERROR] command not found" << GTP_ENDL;
    else if (cmd_id == GTP_CMD_ID_QUIT){
        std::cout << str_gtp_head(id) << GTP_ENDL;
        close(state, options);
    } else if (cmd_id == GTP_CMD_ID_GTP_VERSION)
        print_gtp_version(id);
}