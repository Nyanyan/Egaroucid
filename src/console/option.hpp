/*
    Egaroucid Project

    @file option.hpp
        Commandline options
    @date 2021-2022
    @author Takuto Yamana (a.k.a. Nyanyan)
    @license GPL-3.0 license
*/

#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <ios>
#include <iomanip>

#define N_COMMANDLINE_OPTIONS 9

#define ID_VERSION 0
#define ID_HELP 1
#define ID_LEVEL 2
#define ID_THREAD 3
#define ID_LOG 4
#define ID_HASH 5
#define ID_LEVEL_INFO 6
#define ID_BOOK_FILE 7
#define ID_EVAL_FILE 8

#define OPTION_FOUND "1"

struct Commandline_option_info{
    int id;
    std::vector<std::string> names;
    std::string arg;
    std::string description;
};

const Commandline_option_info commandline_option_data[N_COMMANDLINE_OPTIONS] = {
    {ID_VERSION,    {"-v", "-version", "--version"},                    "",                 "Check Egaroucid for Console version"}, 
    {ID_HELP,       {"-h", "-help", "--help", "-?"},                    "",                 "See help"}, 
    {ID_LEVEL,      {"-l", "-level"},                                   "<level>",          "Set level to <level> (0 to 60)"}, 
    {ID_THREAD,     {"-t", "-thread", "-threads"},                      "<n_threads>",      "Set number of threads (more than 0)"},
    {ID_LOG,        {"-log"},                                           "<log_level>",      "Set log level to <log_level> (0 or 1)"},
    {ID_HASH,       {"-hash"},                                          "<hash_level>",     "Set hash level to <hash_level> (0 to 29)"},
    {ID_LEVEL_INFO, {"-linfo", "-levelinfo"},                           "",                 "See level information"},
    {ID_BOOK_FILE,  {"-b", "-book"},                                    "<book_file>",      "Import <book_file> as Egaroucid's book"},
    {ID_EVAL_FILE,  {"-eval"},                                          "<eval_file>",      "Import <eval_file> as Egaroucid's evaluation function"}
};

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
    bool option_found;
    while (idx < argc){
        option_found = false;
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
    return "";
}
