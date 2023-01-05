/*
    Egaroucid Project

    @file commandline_option.hpp
        Commandline options
    @date 2021-2023
    @author Takuto Yamana (a.k.a. Nyanyan)
    @license GPL-3.0 license
*/

#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <ios>
#include <iomanip>
#include "commandline_option_definition.hpp"

#define OPTION_FOUND "1"
#define OPTION_NOT_FOUND ""

struct Commandline_option{
    int id;
    std::string value;

    Commandline_option(int id_in, std::string value_in){
        id = id_in;
        value = value_in;
    }
};

std::vector<Commandline_option> get_commandline_options(int argc, char* argv[]){
    std::vector<std::string> argv_string;
    int i;
    for (i = 0; i < argc; ++i){
        std::string tmp = argv[i];
        argv_string.emplace_back(tmp);
    }
    std::vector<Commandline_option> res;
    int idx = 0;
    while (idx < argc){
        for (i = 0; i < N_COMMANDLINE_OPTIONS; ++i){
            if (std::find(commandline_option_data[i].names.begin(), commandline_option_data[i].names.end(), argv_string[idx]) != commandline_option_data[i].names.end()){
                if (commandline_option_data[i].arg != ""){
                    ++idx;
                    if (idx >= argc)
                        break;
                    res.emplace_back(Commandline_option(commandline_option_data[i].id, argv_string[idx]));
                    break;
                } else{
                    res.emplace_back(Commandline_option(commandline_option_data[i].id, OPTION_FOUND));
                    break;
                }
            }
        }
        ++idx;
    }
    return res;
}

std::string find_commandline_option(std::vector<Commandline_option> commandline_options, int id){
    for (Commandline_option option: commandline_options){
        if (option.id == id)
            return option.value;
    }
    return OPTION_NOT_FOUND;
}