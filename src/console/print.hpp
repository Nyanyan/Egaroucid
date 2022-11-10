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
#include "option.hpp"
#include "version.hpp"
#include "url.hpp"

#define COUT_TAB "  "

void print_version(){
    std::cout << "Egaroucid " << EGAROUCID_VERSION << std::endl;
    std::cout << COUT_TAB << "@date " << EGAROUCID_DATE << std::endl;
    std::cout << COUT_TAB << "@author Takuto Yamana (a.k.a. Nyanyan)" << std::endl;
    std::cout << COUT_TAB << "@license GPL-3.0 license" << std::endl;
    std::cout << COUT_TAB << "@website " << WEB_PAGE_URL << std::endl;
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
    print_commandline_options();
}
