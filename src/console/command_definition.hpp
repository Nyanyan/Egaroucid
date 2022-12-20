/*
    Egaroucid Project

    @file command_definition.hpp
        Definition of commands
    @date 2021-2022
    @author Takuto Yamana (a.k.a. Nyanyan)
    @license GPL-3.0 license
*/

#pragma once
#include <string>
#include <vector>
#include "console_common.hpp"

#define N_COMMANDS 15

#define CMD_ID_HELP 0
#define CMD_ID_EXIT 1
#define CMD_ID_VERSION 2
#define CMD_ID_INIT 3
#define CMD_ID_NEW 4
#define CMD_ID_PLAY 5
#define CMD_ID_UNDO 6
#define CMD_ID_REDO 7
#define CMD_ID_GO 8
#define CMD_ID_SETBOARD 9
#define CMD_ID_LEVEL 10
#define CMD_ID_LEVELINFO 11
#define CMD_ID_MODE 12
#define CMD_ID_HINT 13
#define CMD_ID_ANALYZE 14

#define COMMAND_NOT_FOUND -1

struct Command_info{
    int id;
    std::vector<std::string> names;
    std::string arg;
    std::string description;
};

const Command_info command_data[N_COMMANDS] = {
    {CMD_ID_HELP,       {"help", "?"},                                      "",                 "See command help"}, 
    {CMD_ID_EXIT,       {"exit", "quit"},                                   "",                 "Exit"}, 
    {CMD_ID_VERSION,    {"version", "ver"},                                 "",                 "See Egaroucid version"}, 
    {CMD_ID_INIT,       {"init", "reset"},                                  "",                 "Initialize a game"}, 
    {CMD_ID_NEW,        {"new"},                                            "",                 "Reset board to `setboard` position"},
    {CMD_ID_PLAY,       {"play"},                                           "",                 "Play moves with f5D6... notation"},
    {CMD_ID_UNDO,       {"undo"},                                           "<moves>",          "Undo your last <moves> moves. if <moves> is empty, undo last 1 move."},
    {CMD_ID_REDO,       {"redo"},                                           "<moves>",          "Redo your last <moves> moves. if <moves> is empty, redo last 1 move."},
    {CMD_ID_GO,         {"go"},                                             "",                 "Egaroucid puts a disc."},
    {CMD_ID_SETBOARD,   {"setboard"},                                       "<board>",          "Set position to <board>. `B`, `W`, `-` each represents black, white, empty."},
    {CMD_ID_LEVEL,      {"level"},                                          "<level>",          "Set level to <level>"},
    {CMD_ID_LEVELINFO,  {"levelinfo", "linfo"},                             "",                 "See level definition"},
    {CMD_ID_MODE,       {"mode"},                                           "<mode>",           "Set mode to <mode> (0: You vs Egaroucid, 1: Egaroucid vs You, 2: Egaroucid vs Egaroucid, 3: You vs You)"},
    {CMD_ID_HINT,       {"hint"},                                           "<moves>",          "Print best <moves> moves. if <moves> is empty, print best move."},
    {CMD_ID_ANALYZE,    {"analyze", "analyse"},                             "",                 "Analyze last game."}
};