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
#include "option.hpp"
#include "info.hpp"

#define COUT_TAB "  "
#define VERSION_TAB_SIZE 10
#define COMMANDLINE_OPTION_HELP_TAB_SIZE 40
#define LEVEL_INFO_TAB_SIZE 5
#define LEVEL_MIDGAME_TAB_SIZE 15
#define LEVEL_DEPTH_TAB_SIZE 10

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

void print_commandline_options(){
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

void print_help(){
    print_version();
    print_commandline_options();
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

void print_special_commandline_options(std::vector<Commandline_option> commandline_options){
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