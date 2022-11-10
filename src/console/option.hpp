/*
    Egaroucid Project

    @file option.hpp
        Options of Egaroucid
    @date 2021-2022
    @author Takuto Yamana (a.k.a. Nyanyan)
    @license GPL-3.0 license
*/

#pragma once
#include <iostream>
#include "./../engine/engine_all.hpp"
#include "commandline_option.hpp"

struct Options{
    int level;
    int n_threads;
    bool show_log;
    int hash_level;
    std::string book_file;
    std::string eval_file;
};

Options get_options(std::vector<Commandline_option> commandline_options){
    Options res;
    std::string str;
    res.level = DEFAULT_LEVEL;
    str = find_commandline_option(commandline_options, ID_LEVEL);
    if (str != OPTION_NOT_FOUND){
        try {
            res.level = std::stoi(str);
            if (res.level < 0 || N_LEVEL <= res.hash_level){
                res.level = DEFAULT_LEVEL;
                std::cerr << "[ERROR] level argument out of range" << std::endl;
            }
        } catch (const std::invalid_argument& e){
            std::cerr << "[ERROR] level argument invalid" << std::endl;
        } catch (const std::out_of_range& e) {
            std::cerr << "[ERROR] level argument out of range" << std::endl;
        }
    }
    res.n_threads = std::min(32, (int)std::thread::hardware_concurrency());
    str = find_commandline_option(commandline_options, ID_THREAD);
    if (str != OPTION_NOT_FOUND){
        try {
            res.n_threads = std::stoi(str);
        } catch (const std::invalid_argument& e){
            std::cerr << "[ERROR] thread argument invalid" << std::endl;
        } catch (const std::out_of_range& e) {
            std::cerr << "[ERROR] thread argument out of range" << std::endl;
        }
    }
    res.show_log = true;
    str = find_commandline_option(commandline_options, ID_LOG);
    if (str != OPTION_NOT_FOUND){
        try {
            res.show_log = std::stoi(str) > 0;
        } catch (const std::invalid_argument& e){
            std::cerr << "[ERROR] log argument invalid" << std::endl;
        } catch (const std::out_of_range& e) {
            std::cerr << "[ERROR] log argument out of range" << std::endl;
        }
    }
    res.hash_level = DEFAULT_HASH_LEVEL;
    str = find_commandline_option(commandline_options, ID_HASH);
    if (str != OPTION_NOT_FOUND){
        try {
            res.hash_level = std::stoi(str);
            if (res.hash_level < 0 || N_HASH_LEVEL <= res.hash_level){
                res.hash_level = DEFAULT_HASH_LEVEL;
                std::cerr << "[ERROR] hash argument out of range" << std::endl;
            }
        } catch (const std::invalid_argument& e){
            std::cerr << "[ERROR] hash argument invalid" << std::endl;
        } catch (const std::out_of_range& e) {
            std::cerr << "[ERROR] hash argument out of range" << std::endl;
        }
    }
}
