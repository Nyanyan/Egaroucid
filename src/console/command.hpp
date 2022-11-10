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
    board->board = board->boards[0].copy();
    board->player = board->players[0];
    board->boards.clear();
    board->players.clear();
    board->boards.emplace_back(board->board);
    board->players.emplace_back(board->player);
    board->ply_vec = 0;
}

bool outside(int y, int x){
    return y < 0 || HW <= y || x < 0 || HW <= x;
}

void play(Board_info *board, std::string transcript){
    if (transcript.length() % 2){
        std::cerr << "[ERROR] invalid transcript length" << std::endl;
        return;
    }
    Board_info board_bak = board->copy();
    Flip flip;
    for (int i = 0; i < (int)transcript.length(); i += 2){
        int x = HW_M1 - (int)(transcript[i] - 'a');
        if (x < 0)
            x = HW_M1 - (int)(transcript[i] - 'A');
        int y = HW_M1 - (int)(transcript[i + 1] - '1');
        if (outside(y, x)){
            std::cerr << "[ERROR] invalid coordinate " << transcript[i] << transcript[i + 1] << std::endl;
            *board = board_bak;
            return;
        }
        calc_flip(&flip, &board->board, y * HW + x);
        if (flip.flip == 0ULL){
            std::cerr << "[ERROR] invalid move " << transcript[i] << transcript[i + 1] << std::endl;
            *board = board_bak;
            return;
        }
        board->board.move_board(&flip);
        board->player ^= 1;
        if (board->board.is_end() && i < (int)transcript.length() - 2){
            std::cerr << "[ERROR] game over found before checking all transcript. remaining codes ignored." << std::endl;
            return;
        }
        if (board->board.get_legal() == 0ULL){
            board->board.pass();
            board->player ^= 1;
        }
        board->boards.emplace_back(board->board);
        board->players.emplace_back(board->player);
    }
}

void undo(Board_info *board){
    if (board->ply_vec <= 0)
        return;
    --board->ply_vec;
    board->board = board->boards[board->ply_vec].copy();
    board->player = board->players[board->ply_vec];
}

void redo(Board_info *board){
    if (board->ply_vec >= (int)board->boards.size())
        return;
    ++board->ply_vec;
    board->board = board->boards[board->ply_vec].copy();
    board->player = board->players[board->ply_vec];
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
    else if (cmd_id == CMD_ID_PLAY)
        play(board, arg);
    else if (cmd_id == CMD_ID_UNDO)
        undo(board);
    else if (cmd_id == CMD_ID_REDO)
        redo(board);
}