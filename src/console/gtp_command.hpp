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

#define GTP_VERSION "2.0"
#define GTP_ENDL "\n\n"
#define GTP_ID_NOT_FOUND -1000000000
#define GTP_POLICY_UNDEFINED 127
#define GTP_PLAYER_UNDEFINED 127

std::string gtp_get_command_line(){
    std::string cmd_line;
    std::getline(std::cin, cmd_line);
    return cmd_line;
}

void gtp_split_cmd_arg(std::string cmd_line, int *id, std::string *cmd, std::string *arg){
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
    iss.get();
    std::getline(iss, *arg);
}

int gtp_get_command_id(std::string cmd){
    for (int i = 0; i < N_GTP_COMMANDS; ++i){
        if (std::find(gtp_command_data[i].names.begin(), gtp_command_data[i].names.end(), cmd) != gtp_command_data[i].names.end())
            return gtp_command_data[i].id;
    }
    return COMMAND_NOT_FOUND;
}

std::string gtp_head(int id){
    if (id == GTP_ID_NOT_FOUND)
        return "=";
    return "=" + std::to_string(id);
}

std::string gtp_error_head(int id){
    if (id == GTP_ID_NOT_FOUND)
        return "?";
    return "?" + std::to_string(id);
}

int check_color(std::string color){
    int player = GTP_PLAYER_UNDEFINED;
    if (color == "black" || color == "b" || color == "B")
        player = BLACK;
    if (color == "white" || color == "w" || color == "W")
        player = WHITE;
    return player;
}

void gtp_quit(int id, State *state, Options *options){
    std::cout << gtp_head(id) << GTP_ENDL;
    close(state, options);
}

void gtp_print_gtp_version(int id){
    std::cout << gtp_head(id) << " " << GTP_VERSION << GTP_ENDL;
}

void gtp_print_name(int id){
    std::cout << gtp_head(id) << " " << EGAROUCID_NAME << GTP_ENDL;
}

void gtp_print_version(int id){
    std::cout << gtp_head(id) << " " << EGAROUCID_VERSION << GTP_ENDL;
}

void gtp_known_command(int id, std::string arg){
    std::string res = "false";
    for (Command_info cmd: gtp_command_data){
        if (cmd.names[0] == arg)
            res = "true";
    }
    std::cout << gtp_head(id) << " " << res << GTP_ENDL;
}

void gtp_print_list_commands(int id){
    for (Command_info cmd: gtp_command_data){
        std::cout << gtp_head(id) << " " << cmd.names[0] << "\n";
    }
    std::cout << "\n";
}

void gtp_boardsize(int id){
    std::cout << gtp_head(id) << GTP_ENDL;
}

void gtp_clear_board(int id, Board_info *board){
    board->player = BLACK;
    board->board.reset();
    std::cout << gtp_head(id) << GTP_ENDL;
}

void gtp_komi(int id){
    std::cout << gtp_head(id) << GTP_ENDL;
}

void gtp_play(int id, std::string arg, Board_info *board){
    std::string color, coord;
    uint_fast8_t policy;
    uint_fast8_t player = GTP_PLAYER_UNDEFINED;
    try{
        std::istringstream iss(arg);
        iss >> color >> coord;
        player = check_color(color);
        policy = GTP_POLICY_UNDEFINED;
        if (coord.size() == 2){
            int x = coord[0] - 'a';
            if (x < 0 || HW <= x)
                x = coord[0] - 'A';
            int y = coord[1] - '1';
            if (0 <= x && x < HW && 0 < y && y < HW)
                policy = HW2_M1 - (y * HW + x);
        }
    } catch (const std::invalid_argument& e) {
        policy = GTP_POLICY_UNDEFINED;
    } catch (const std::out_of_range& e) {
        policy = GTP_POLICY_UNDEFINED;
    }
    if (player == GTP_PLAYER_UNDEFINED || policy == GTP_POLICY_UNDEFINED){
        std::cout << gtp_error_head(id) << " " << "illegal move" << GTP_ENDL;
        return;
    }
    if (player != board->player){
        board->board.pass();
        board->player ^= 1;
    }
    Flip flip;
    calc_flip(&flip, &board->board, policy);
    //if (flip.flip == 0ULL){
    //    std::cout << gtp_error_head(id) << " " << "illegal move" << GTP_ENDL;
    //    return;
    //}
    board->board.move_board(&flip);
    board->player ^= 1;
    std::cout << gtp_head(id) << GTP_ENDL;
}

void gtp_genmove(int id, std::string arg, Board_info *board, State *state, Options *options){
    uint_fast8_t player = GTP_PLAYER_UNDEFINED;
    try{
        player = check_color(arg);
    } catch (const std::invalid_argument& e) {
        player = GTP_PLAYER_UNDEFINED;
    } catch (const std::out_of_range& e) {
        player = GTP_PLAYER_UNDEFINED;
    }
    if (player != BLACK && player != WHITE){
        std::cout << gtp_error_head(id) << " " << "illegal color" << GTP_ENDL;
        return;
    }
    if (player != board->player){
        board->board.pass();
        board->player ^= 1;
    }
    int policy = ai(board->board, options->level, true, true, true, state->date).policy;
    ++state->date;
    state->date = manage_date(state->date);
    std::cout << gtp_head(id) << " " << idx_to_coord(policy) << GTP_ENDL;
}

void gtp_check_command(Board_info *board, State *state, Options *options){
    std::string cmd_line = gtp_get_command_line();
    std::string cmd, arg;
    int id;
    gtp_split_cmd_arg(cmd_line, &id, &cmd, &arg);
    int cmd_id = gtp_get_command_id(cmd);
    if (cmd_id == COMMAND_NOT_FOUND)
        std::cout << gtp_error_head(id) << " " << "[ERROR] command not found" << GTP_ENDL;
    else if (cmd_id == GTP_CMD_ID_QUIT)
        gtp_quit(id, state, options);
    else if (cmd_id == GTP_CMD_ID_GTP_VERSION)
        gtp_print_gtp_version(id);
    else if (cmd_id == GTP_CMD_ID_NAME)
        gtp_print_name(id);
    else if (cmd_id == GTP_CMD_ID_VERSION)
        gtp_print_version(id);
    else if (cmd_id == GTP_CMD_ID_KNOWN_CMD)
        gtp_known_command(id, arg);
    else if (cmd_id == GTP_CMD_ID_LIST_CMD)
        gtp_print_list_commands(id);
    else if (cmd_id == GTP_CMD_ID_BOARDSIZE)
        gtp_boardsize(id);
    else if (cmd_id == GTP_CMD_ID_CLEAR_BOARD)
        gtp_clear_board(id, board);
    else if (cmd_id == GTP_CMD_ID_KOMI)
        gtp_komi(id);
    else if (cmd_id == GTP_CMD_ID_PLAY)
        gtp_play(id, arg, board);
    else if (cmd_id == GTP_CMD_ID_GENMOVE)
        gtp_genmove(id, arg, board, state, options);
}