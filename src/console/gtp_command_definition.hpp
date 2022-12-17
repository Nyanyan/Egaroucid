/*
    Egaroucid Project

    @file gtp_command_definition.hpp
        Definition of commands for GTP (Gp Text Protocol)
    @date 2021-2022
    @author Takuto Yamana (a.k.a. Nyanyan)
    @license GPL-3.0 license
*/

#pragma once
#include <string>
#include <vector>
#include "command_definition.hpp"

#define N_GTP_COMMANDS 20

#define GTP_CMD_ID_QUIT 0
#define GTP_CMD_ID_GTP_VERSION 1
#define GTP_CMD_ID_NAME 2
#define GTP_CMD_ID_VERSION 3
#define GTP_CMD_ID_KNOWN_CMD 4
#define GTP_CMD_ID_LIST_CMD 5
#define GTP_CMD_ID_BOARDSIZE 6
#define GTP_CMD_ID_CLEAR_BOARD 7
#define GTP_CMD_ID_KOMI 8
#define GTP_CMD_ID_PLAY 9
#define GTP_CMD_ID_GENMOVE 10
#define GTP_CMD_ID_RULES_GAME_ID 11
#define GTP_CMD_ID_RULES_BOARD 12
#define GTP_CMD_ID_RULES_BOARD_SIZE 13
#define GTP_CMD_ID_RULES_LEGAL_MOVES 14
#define GTP_CMD_ID_RULES_SIDE_MOVE 15
#define GTP_CMD_ID_RULES_FINAL_RESULT 16
#define GTP_CMD_ID_SHOWBOARD 17
#define GTP_CMD_ID_UNDO 18
#define GTP_CMD_ID_REG_GENMOVE 19

const Command_info gtp_command_data[N_GTP_COMMANDS] = {
    {GTP_CMD_ID_QUIT,               {"quit"},                                                   "",                 "Quit"},
    {GTP_CMD_ID_GTP_VERSION,        {"protocol_version"},                                       "",                 "See GTP version"},
    {GTP_CMD_ID_NAME,               {"name"},                                                   "",                 "See name"},
    {GTP_CMD_ID_VERSION,            {"version"},                                                "",                 "See Egaroucid version"},
    {GTP_CMD_ID_KNOWN_CMD,          {"known_command"},                                          "",                 ""},
    {GTP_CMD_ID_LIST_CMD,           {"list_commands"},                                          "",                 ""},
    {GTP_CMD_ID_BOARDSIZE,          {"boardsize"},                                              "",                 ""},
    {GTP_CMD_ID_CLEAR_BOARD,        {"clear_board"},                                            "",                 ""},
    {GTP_CMD_ID_KOMI,               {"komi"},                                                   "",                 ""},
    {GTP_CMD_ID_PLAY,               {"play"},                                                   "",                 ""},
    {GTP_CMD_ID_GENMOVE,            {"genmove"},                                                "",                 ""},
    {GTP_CMD_ID_RULES_GAME_ID,      {"gogui-rules_game_id"},                                    "",                 ""},
    {GTP_CMD_ID_RULES_BOARD,        {"gogui-rules_board"},                                      "",                 ""},
    {GTP_CMD_ID_RULES_BOARD_SIZE,   {"gogui-rules_board_size"},                                 "",                 ""},
    {GTP_CMD_ID_RULES_LEGAL_MOVES,  {"gogui-rules_legal_moves"},                                "",                 ""},
    {GTP_CMD_ID_RULES_SIDE_MOVE,    {"gogui-rules_side_to_move"},                               "",                 ""},
    {GTP_CMD_ID_RULES_FINAL_RESULT, {"gogui-rules_final_result"},                               "",                 ""},
    {GTP_CMD_ID_SHOWBOARD,          {"showboard"},                                              "",                 ""},
    {GTP_CMD_ID_UNDO,               {"undo"},                                                   "",                 ""},
    {GTP_CMD_ID_REG_GENMOVE,        {"reg_genmove"},                                            "",                 ""}
};
