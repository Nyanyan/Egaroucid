/*
    Egaroucid Project

    @file commandline_option.hpp
        Commandline options
    @date 2021-2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <ios>
#include <iomanip>
#include "commandline_option_definition.hpp"

struct Commandline_option {
    int id;
    std::vector<std::string> arg;

    Commandline_option(int id_in, std::vector<std::string> arg_in) {
        id = id_in;
        arg = arg_in;
    }
};

std::vector<Commandline_option> get_commandline_options(int argc, char* argv[]) {
    std::vector<std::string> argv_string;
    int i;
    for (i = 0; i < argc; ++i) {
        std::string tmp = argv[i];
        argv_string.emplace_back(tmp);
    }
    std::vector<Commandline_option> res;
    int idx = 0;
    while (idx < argc) {
        for (i = 0; i < N_COMMANDLINE_OPTIONS; ++i) {
            if (std::find(commandline_option_data[i].names.begin(), commandline_option_data[i].names.end(), argv_string[idx]) != commandline_option_data[i].names.end()) {
                std::vector<std::string> args;
                for (int j = 0; j < commandline_option_data[i].n_args; ++j) {
                    ++idx;
                    if (idx >= argc) {
                        break;
                    }
                    args.emplace_back(argv_string[idx]);
                }
                res.emplace_back(Commandline_option(commandline_option_data[i].id, args));
                break;
            }
        }
        ++idx;
    }
    return res;
}

bool find_commandline_option(std::vector<Commandline_option> commandline_options, int id) {
    for (Commandline_option option: commandline_options) {
        if (option.id == id)
            return true;
    }
    return false;
}

std::vector<std::string> get_commandline_option_arg(std::vector<Commandline_option> commandline_options, int id) {
    for (Commandline_option option: commandline_options) {
        if (option.id == id) {
            return option.arg;
        }
    }
    std::vector<std::string> nores;
    return nores;
}