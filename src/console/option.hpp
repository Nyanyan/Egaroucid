/*
    Egaroucid Project

    @file option.hpp
        Options of Egaroucid
    @date 2021-2024
    @author Takuto Yamana
    @license GPL-3.0 license
*/

#pragma once
#include <iostream>
#include "./../engine/engine_all.hpp"
#include "commandline_option.hpp"
#include "console_common.hpp"

struct Options{
    std::string binary_path;
    int level;
    int n_threads;
    bool show_log;
    int hash_level;
    std::string book_file;
    std::string eval_file;
    bool nobook;
    int mode;
    bool gtp;
    bool quiet;
};

Options get_options(std::vector<Commandline_option> commandline_options, std::string binary_path){
    Options res;
    res.binary_path = binary_path;
    res.level = DEFAULT_LEVEL;
    if (find_commandline_option(commandline_options, ID_LEVEL)){
        std::vector<std::string> arg = get_commandline_option_arg(commandline_options, ID_LEVEL);
        try {
            res.level = std::stoi(arg[0]);
            if (res.level < 0 || N_LEVEL <= res.level){
                res.level = DEFAULT_LEVEL;
                std::cerr << "[ERROR] level argument out of range" << std::endl;
            }
        } catch (const std::invalid_argument& e){
            std::cerr << "[ERROR] invalid level" << std::endl;
        } catch (const std::out_of_range& e) {
            std::cerr << "[ERROR] level argument out of range" << std::endl;
        }
    }
    res.n_threads = std::min(48, (int)std::thread::hardware_concurrency());
    if (find_commandline_option(commandline_options, ID_THREAD)){
        std::vector<std::string> arg = get_commandline_option_arg(commandline_options, ID_THREAD);
        try {
            res.n_threads = std::stoi(arg[0]);
        } catch (const std::invalid_argument& e){
            std::cerr << "[ERROR] thread argument invalid" << std::endl;
        } catch (const std::out_of_range& e) {
            std::cerr << "[ERROR] thread argument out of range" << std::endl;
        }
    }
    res.show_log = find_commandline_option(commandline_options, ID_LOG);
    res.hash_level = DEFAULT_HASH_LEVEL;
    if (find_commandline_option(commandline_options, ID_HASH)){
        std::vector<std::string> arg = get_commandline_option_arg(commandline_options, ID_HASH);
        try {
            res.hash_level = std::stoi(arg[0]);
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
    res.book_file = binary_path + "resources/book.egbk3";
    if (find_commandline_option(commandline_options, ID_BOOK_FILE)){
        std::vector<std::string> arg = get_commandline_option_arg(commandline_options, ID_BOOK_FILE);
        try {
            res.book_file = arg[0];
        } catch (const std::invalid_argument& e){
            std::cerr << "[ERROR] hash argument invalid" << std::endl;
        } catch (const std::out_of_range& e) {
            std::cerr << "[ERROR] hash argument out of range" << std::endl;
        }
    }
    res.eval_file = binary_path + "resources/eval.egev2";
    if (find_commandline_option(commandline_options, ID_EVAL_FILE)){
        std::vector<std::string> arg = get_commandline_option_arg(commandline_options, ID_EVAL_FILE);
        try {
            res.eval_file = arg[0];
        } catch (const std::invalid_argument& e){
            std::cerr << "[ERROR] hash argument invalid" << std::endl;
        } catch (const std::out_of_range& e) {
            std::cerr << "[ERROR] hash argument out of range" << std::endl;
        }
    }
    res.nobook = find_commandline_option(commandline_options, ID_NOBOOK);
    res.mode = MODE_HUMAN_HUMAN;
    if (find_commandline_option(commandline_options, ID_MODE)){
        std::vector<std::string> arg = get_commandline_option_arg(commandline_options, ID_MODE);
        try {
            res.mode = std::stoi(arg[0]);
            if (res.mode < 0 || 4 <= res.mode){
                res.mode = 0;
                std::cerr << "[ERROR] mode argument out of range" << std::endl;
            }
        } catch (const std::invalid_argument& e){
            std::cerr << "[ERROR] invalid mode" << std::endl;
        } catch (const std::out_of_range& e) {
            std::cerr << "[ERROR] mode argument out of range" << std::endl;
        }
    }
    res.gtp = find_commandline_option(commandline_options, ID_GTP);
    res.quiet = find_commandline_option(commandline_options, ID_QUIET);
    return res;
}