/*
    Egaroucid Project

    @file print.hpp
        Functions about printing on console
    @date 2021-2022
    @author Takuto Yamana (a.k.a. Nyanyan)
    @license GPL-3.0 license
*/

#pragma once
#include <iostream>
#include <unordered_map>
#include "./../engine/engine_all.hpp"
#include "option.hpp"
#include "info.hpp"
#include "command_definition.hpp"
#include "commandline_option_definition.hpp"
#include "function.hpp"

#define COUT_TAB "  "
#define VERSION_TAB_SIZE 10
#define COMMANDLINE_OPTION_HELP_TAB_SIZE 40
#define COMMAND_HELP_TAB_SIZE 40
#define LEVEL_INFO_TAB_SIZE 5
#define LEVEL_MIDGAME_TAB_SIZE 15
#define LEVEL_DEPTH_TAB_SIZE 10
#define SEARCH_RESULT_TAB_SIZE 15

void print_version(){
    std::cout << "Egaroucid " << EGAROUCID_VERSION << std::endl;
    std::cout << COUT_TAB << std::left << std::setw(VERSION_TAB_SIZE) << "@date ";
    std::cout << EGAROUCID_DATE << std::endl;
    std::cout << COUT_TAB << std::left << std::setw(VERSION_TAB_SIZE) << "@author ";
    std::cout << EGAROUCID_AUTHOR << std::endl;
    std::cout << COUT_TAB << std::left << std::setw(VERSION_TAB_SIZE) << "@license ";
    std::cout << EGAROUCID_LICENSE << std::endl;
    std::cout << COUT_TAB << std::left << std::setw(VERSION_TAB_SIZE) << "@website ";
    std::cout << EGAROUCID_URL << std::endl;
    std::cout << std::endl;
}

void print_commandline_options_list(){
    std::cout << "Commandline options:" << std::endl;
    for (int i = 0; i < N_COMMANDLINE_OPTIONS; ++i){
        std::string s;
        for (int j = 0; j < (int)commandline_option_data[i].names.size(); ++j){
            if (j != 0)
                s += "|";
            s += commandline_option_data[i].names[j];
        }
        s += " " + commandline_option_data[i].arg;
        std::cout << COUT_TAB;
        std::cout << std::left << std::setw(COMMANDLINE_OPTION_HELP_TAB_SIZE) << s;
        std::cout << commandline_option_data[i].description;
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void print_commands_list(){
    std::cout << "Commands:" << std::endl;
    for (int i = 0; i < N_COMMANDS; ++i){
        std::string s;
        for (int j = 0; j < (int)command_data[i].names.size(); ++j){
            if (j != 0)
                s += "|";
            s += command_data[i].names[j];
        }
        s += " " + command_data[i].arg;
        std::cout << COUT_TAB;
        std::cout << std::left << std::setw(COMMAND_HELP_TAB_SIZE) << s;
        std::cout << command_data[i].description;
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void print_help(){
    print_version();
    print_commandline_options_list();
    print_commands_list();
}

void print_level_info(){
    std::unordered_map<double, std::string> probability_char;
    probability_char[MPC_81] = "_";
    probability_char[MPC_95] = "-";
    probability_char[MPC_98] = "=";
    probability_char[MPC_99] = "^";
    probability_char[NOMPC] = "#";
    std::cout << "Level definition:" << std::endl;
    std::cout << COUT_TAB;
    std::cout << "Endgame probability" << std::endl;
    std::cout << COUT_TAB << COUT_TAB << " 81%: " << probability_char[MPC_81] << std::endl;
    std::cout << COUT_TAB << COUT_TAB << " 95%: " << probability_char[MPC_95] << std::endl;
    std::cout << COUT_TAB << COUT_TAB << " 98%: " << probability_char[MPC_98] << std::endl;
    std::cout << COUT_TAB << COUT_TAB << " 99%: " << probability_char[MPC_99] << std::endl;
    std::cout << COUT_TAB << COUT_TAB << "100%: " << probability_char[NOMPC] << std::endl;
    std::cout << COUT_TAB;
    std::cout << "|";
    std::cout << std::right << std::setw(LEVEL_INFO_TAB_SIZE) << "Level";
    std::cout << "|";
    std::cout << std::left << std::setw(LEVEL_MIDGAME_TAB_SIZE) << "Midgame";
    std::cout << "|";
    std::string str_endgame_10 = "Endgame " + std::to_string(LEVEL_DEPTH_TAB_SIZE);
    std::cout << std::right << std::setw(LEVEL_DEPTH_TAB_SIZE) << str_endgame_10;
    std::cout << "|";
    for (int depth = LEVEL_DEPTH_TAB_SIZE * 2; depth <= 60; depth += LEVEL_DEPTH_TAB_SIZE){
        std::cout << std::right << std::setw(LEVEL_DEPTH_TAB_SIZE) << depth;
        std::cout << "|";
    }
    std::cout << std::endl;
    for (int level = 0; level < N_LEVEL; ++level){
        std::cout << COUT_TAB;
        std::cout << "|";
        std::cout << std::right << std::setw(LEVEL_INFO_TAB_SIZE) << level;
        std::cout << "|";
        std::string s;
        if (get_level_midsearch(level, 0))
            s = std::to_string(level_definition[level].mid_lookahead) + " moves@" + std::to_string(calc_probability(level_definition[level].mid_mpct)) + "%";
        else
            s = "None";
        std::cout << std::right << std::setw(LEVEL_MIDGAME_TAB_SIZE) << s;
        for (int n_moves = 0; n_moves < HW2 - 4; ++n_moves){
            if (n_moves % LEVEL_DEPTH_TAB_SIZE == 0)
                std::cout << "|";
            bool is_mid_search, use_mpc;
            double mpct;
            int depth;
            get_level(level, n_moves, &is_mid_search, &depth, &use_mpc, &mpct);
            if (is_mid_search)
                std::cout << " ";
            else{
                std::cout << probability_char[mpct];
            }
        }
        std::cout << "|";
        std::cout << std::endl;
    }
}

void print_board_info(Board_info *board){
    uint64_t black = board->board.player;
    uint64_t white = board->board.opponent;
    if (board->player == WHITE)
        std::swap(black, white);
    std::cout << "  ";
    for (int x = 0; x < HW; ++x)
        std::cout << (char)('a' + x) << " ";
    std::cout << std::endl;
    for (int y = 0; y < HW; ++y){
        std::cout << y + 1 << " ";
        for (int x = 0; x < HW; ++x){
            int cell = HW2_M1 - (y * HW + x);
            if (1 & (black >> cell))
                std::cout << "X ";
            else if (1 & (white >> cell))
                std::cout << "O ";
            else
                std::cout << ". ";
        }
        if (y == 2){
            std::cout << COUT_TAB;
            if (board->board.is_end())
                std::cout << "GAME OVER";
            else if (board->player == BLACK)
                std::cout << "BLACK to move";
            else
                std::cout << "WHITE to move";
        } else if (y == 3){
            std::cout << COUT_TAB;
            std::cout << "ply " << board->board.n_discs() - 3 << " " << HW2 - board->board.n_discs() << " empties";
        } else if (y == 4){
            std::cout << COUT_TAB;
            std::cout << "mode " << board->mode << " ";
            if (board->mode == MODE_HUMAN_VS_AI)
                std::cout << "BLACK: Egaroucid  WHITE: You";
            else if (board->mode == MODE_AI_VS_HUMAN)
                std::cout << "BLACK: You  WHITE: Egaroucid";
            else if (board->mode == MODE_AI_VS_AI)
                std::cout << "BLACK: Egaroucid  WHITE: Egaroucid";
            else if (board->mode == MODE_HUMAN_VS_HUMAN)
                std::cout << "BLACK: You  WHITE: You";
        }
        std::cout << std::endl;
    }
}

void print_search_result_body(Search_result result, int level){
    std::string s;
    if (result.depth == SEARCH_BOOK){
        std::cout << "|";
        std::cout << std::right << std::setw(SEARCH_RESULT_TAB_SIZE) << level;
        std::cout << "|";
        std::cout << std::right << std::setw(SEARCH_RESULT_TAB_SIZE) << "Book";
        std::cout << "|";
        std::cout << std::right << std::setw(SEARCH_RESULT_TAB_SIZE) << idx_to_coord(result.policy);
        std::cout << "|";
        if (result.value >= 0)
            s = "+" + std::to_string(result.value);
        else
            s = std::to_string(result.value);
        std::cout << std::right << std::setw(SEARCH_RESULT_TAB_SIZE) << s;
        std::cout << "|";
        std::cout << std::right << std::setw(SEARCH_RESULT_TAB_SIZE) << 0;
        std::cout << "|";
        std::cout << std::right << std::setw(SEARCH_RESULT_TAB_SIZE) << 0;
        std::cout << "|";
        std::cout << std::right << std::setw(SEARCH_RESULT_TAB_SIZE) << 0;
        std::cout << "|";
        std::cout << std::endl;
    } else{
        std::cout << "|";
        std::cout << std::right << std::setw(SEARCH_RESULT_TAB_SIZE) << level;
        std::cout << "|";
        s = std::to_string(result.depth) + "@" + std::to_string(result.probability) + "%";
        std::cout << std::right << std::setw(SEARCH_RESULT_TAB_SIZE) << s;
        std::cout << "|";
        std::cout << std::right << std::setw(SEARCH_RESULT_TAB_SIZE) << idx_to_coord(result.policy);
        std::cout << "|";
        if (result.value >= 0)
            s = "+" + std::to_string(result.value);
        else
            s = std::to_string(result.value);
        std::cout << std::right << std::setw(SEARCH_RESULT_TAB_SIZE) << s;
        std::cout << "|";
        s = std::to_string((double)result.time / 1000);
        s = s.substr(0, s.length() - 3) + "s";
        std::cout << std::right << std::setw(SEARCH_RESULT_TAB_SIZE) << s;
        std::cout << "|";
        std::cout << std::right << std::setw(SEARCH_RESULT_TAB_SIZE) << result.nodes;
        std::cout << "|";
        std::cout << std::right << std::setw(SEARCH_RESULT_TAB_SIZE) << result.nps;
        std::cout << "|";
        std::cout << std::endl;
    }
}

void print_search_result_head(){
    std::cout << "|";
    std::cout << std::right << std::setw(SEARCH_RESULT_TAB_SIZE) << "Level";
    std::cout << "|";
    std::cout << std::right << std::setw(SEARCH_RESULT_TAB_SIZE) << "Depth";
    std::cout << "|";
    std::cout << std::right << std::setw(SEARCH_RESULT_TAB_SIZE) << "Move";
    std::cout << "|";
    std::cout << std::right << std::setw(SEARCH_RESULT_TAB_SIZE) << "Score";
    std::cout << "|";
    std::cout << std::right << std::setw(SEARCH_RESULT_TAB_SIZE) << "Time";
    std::cout << "|";
    std::cout << std::right << std::setw(SEARCH_RESULT_TAB_SIZE) << "Nodes";
    std::cout << "|";
    std::cout << std::right << std::setw(SEARCH_RESULT_TAB_SIZE) << "NPS";
    std::cout << "|";
    std::cout << std::endl;
}

void print_search_result(Search_result result, int level){
    print_search_result_head();
    print_search_result_body(result, level);
}

void print_special_commandline_options(std::vector<Commandline_option> commandline_options, Options *options){
    if (find_commandline_option(commandline_options, ID_VERSION) == OPTION_FOUND){
        print_version();
        std::exit(0);
    }
    if (find_commandline_option(commandline_options, ID_HELP) == OPTION_FOUND){
        print_help();
        std::exit(0);
    }
    if (find_commandline_option(commandline_options, ID_LEVEL_INFO) == OPTION_FOUND){
        print_level_info();
        std::exit(0);
    }
}

void execute_special_commandline_tasks(std::vector<Commandline_option> commandline_options, Options *options){
    if (find_commandline_option(commandline_options, ID_SOLVE) != OPTION_NOT_FOUND){
        solve_problems(find_commandline_option(commandline_options, ID_SOLVE), options);
        std::exit(0);
    }
}