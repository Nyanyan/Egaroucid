/*
    Egaroucid Project

    @file option.hpp
        Options of Egaroucid
    @date 2021-2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once
#include <iostream>
#include "./../engine/engine_all.hpp"
#include "commandline_option.hpp"
#include "console_common.hpp"
#include "util.hpp"

#define TIME_NOT_ALLOCATED -1

struct Options {
    std::string binary_path;
    int level;
    int n_threads;
    bool show_log;
#if USE_CHANGEABLE_HASH_LEVEL
    int hash_level;
#endif
    std::string book_file;
    std::string eval_file;
    bool nobook;
    int mode;
    bool gtp;
    bool quiet;
    int time_allocated_seconds; // -1 (TIME_NOT_ALLOCATED): not allocated
    bool ponder;
    bool noboard;
    bool log_to_file;
    std::string log_file;
    bool noautopass;
    bool show_value;
    bool play_loss;
    double play_loss_ratio;
    int play_loss_max;
#ifdef INCLUDE_GGS
    bool ggs;
    std::string ggs_username;
    std::string ggs_password;
    bool ggs_log_to_file;
    std::string ggs_log_file;
    bool ggs_game_log_to_file;
    std::string ggs_game_log_dir;
    bool ggs_accept_request;
    bool ggs_route_join_tournament;
#endif
};

Options get_options(std::vector<Commandline_option> commandline_options, std::string binary_path) {
    Options res;
    std::string datetime = get_current_datetime_for_file();
    res.binary_path = binary_path;
    res.level = DEFAULT_LEVEL;
    if (find_commandline_option(commandline_options, ID_LEVEL)) {
        std::vector<std::string> arg = get_commandline_option_arg(commandline_options, ID_LEVEL);
        try {
            res.level = std::stoi(arg[0]);
            if (res.level < 0 || N_LEVEL <= res.level) {
                res.level = DEFAULT_LEVEL;
                std::cerr << "[ERROR] level argument out of range" << std::endl;
            }
        } catch (const std::invalid_argument& e) {
            std::cerr << "[ERROR] invalid level" << std::endl;
        } catch (const std::out_of_range& e) {
            std::cerr << "[ERROR] level argument out of range" << std::endl;
        }
    }
    res.n_threads = std::min(48, (int)std::thread::hardware_concurrency());
    if (find_commandline_option(commandline_options, ID_THREAD)) {
        std::vector<std::string> arg = get_commandline_option_arg(commandline_options, ID_THREAD);
        try {
            res.n_threads = std::stoi(arg[0]);
        } catch (const std::invalid_argument& e) {
            std::cerr << "[ERROR] thread argument invalid" << std::endl;
        } catch (const std::out_of_range& e) {
            std::cerr << "[ERROR] thread argument out of range" << std::endl;
        }
    }
    res.show_log = find_commandline_option(commandline_options, ID_LOG);
    #if USE_CHANGEABLE_HASH_LEVEL
        res.hash_level = DEFAULT_HASH_LEVEL;
        if (find_commandline_option(commandline_options, ID_HASH)) {
            std::vector<std::string> arg = get_commandline_option_arg(commandline_options, ID_HASH);
            try {
                res.hash_level = std::stoi(arg[0]);
                if (res.hash_level < 0 || N_HASH_LEVEL <= res.hash_level) {
                    res.hash_level = DEFAULT_HASH_LEVEL;
                    std::cerr << "[ERROR] hash argument out of range" << std::endl;
                }
            } catch (const std::invalid_argument& e) {
                std::cerr << "[ERROR] hash argument invalid" << std::endl;
            } catch (const std::out_of_range& e) {
                std::cerr << "[ERROR] hash argument out of range" << std::endl;
            }
        }
    #endif
    res.book_file = binary_path + "resources/book.egbk3";
    if (find_commandline_option(commandline_options, ID_BOOK_FILE)) {
        std::vector<std::string> arg = get_commandline_option_arg(commandline_options, ID_BOOK_FILE);
        try {
            res.book_file = arg[0];
        } catch (const std::invalid_argument& e) {
            std::cerr << "[ERROR] hash argument invalid" << std::endl;
        } catch (const std::out_of_range& e) {
            std::cerr << "[ERROR] hash argument out of range" << std::endl;
        }
    }
    res.eval_file = binary_path + "resources/eval.egev2";
    if (find_commandline_option(commandline_options, ID_EVAL_FILE)) {
        std::vector<std::string> arg = get_commandline_option_arg(commandline_options, ID_EVAL_FILE);
        try {
            res.eval_file = arg[0];
        } catch (const std::invalid_argument& e) {
            std::cerr << "[ERROR] hash argument invalid" << std::endl;
        } catch (const std::out_of_range& e) {
            std::cerr << "[ERROR] hash argument out of range" << std::endl;
        }
    }
    res.nobook = find_commandline_option(commandline_options, ID_NOBOOK);
    res.mode = MODE_HUMAN_HUMAN;
    if (find_commandline_option(commandline_options, ID_MODE)) {
        std::vector<std::string> arg = get_commandline_option_arg(commandline_options, ID_MODE);
        try {
            res.mode = std::stoi(arg[0]);
            if (res.mode < 0 || 4 <= res.mode) {
                res.mode = 0;
                std::cerr << "[ERROR] mode argument out of range" << std::endl;
            }
        } catch (const std::invalid_argument& e) {
            std::cerr << "[ERROR] invalid mode" << std::endl;
        } catch (const std::out_of_range& e) {
            std::cerr << "[ERROR] mode argument out of range" << std::endl;
        }
    }
    res.gtp = find_commandline_option(commandline_options, ID_GTP);
    res.quiet = find_commandline_option(commandline_options, ID_QUIET);
    res.time_allocated_seconds = TIME_NOT_ALLOCATED;
    if (find_commandline_option(commandline_options, ID_TIME_ALLOCATE)) {
        std::vector<std::string> arg = get_commandline_option_arg(commandline_options, ID_TIME_ALLOCATE);
        try {
            res.time_allocated_seconds = std::stoi(arg[0]);
            if (res.time_allocated_seconds < 1) {
                res.time_allocated_seconds = TIME_NOT_ALLOCATED;
                std::cerr << "[ERROR] time allocation argument out of range" << std::endl;
            }
        } catch (const std::invalid_argument& e) {
            std::cerr << "[ERROR] invalid time allocation" << std::endl;
        } catch (const std::out_of_range& e) {
            std::cerr << "[ERROR] time allocation argument out of range" << std::endl;
        }
    }
    res.ponder = find_commandline_option(commandline_options, ID_PONDER);
    if (find_commandline_option(commandline_options, ID_DISABLE_AUTO_CACHE_CLEAR)) {
        transposition_table_auto_reset_importance = false;
    }
    res.noboard = find_commandline_option(commandline_options, ID_NOBOARD);
    res.log_to_file = false;
    if (find_commandline_option(commandline_options, ID_LOG_TO_FILE)) {
        res.log_to_file = true;
        std::vector<std::string> arg = get_commandline_option_arg(commandline_options, ID_LOG_TO_FILE);
        try {
            res.log_file = arg[0];
        } catch (const std::invalid_argument& e) {
            std::cerr << "[ERROR] invalid log file" << std::endl;
        } catch (const std::out_of_range& e) {
            std::cerr << "[ERROR] log file out of range" << std::endl;
        }
    }
    res.noautopass = find_commandline_option(commandline_options, ID_NOAUTOPASS);
    res.show_value = find_commandline_option(commandline_options, ID_SHOWVALUE);
    res.play_loss = find_commandline_option(commandline_options, ID_PLAY_LOSS);
    if (res.play_loss) {
        std::vector<std::string> arg = get_commandline_option_arg(commandline_options, ID_PLAY_LOSS);
        try {
            res.play_loss_ratio = std::stod(arg[0]);
            res.play_loss_max = std::stoi(arg[1]);
        } catch (const std::invalid_argument& e) {
            std::cerr << "[ERROR] invalid play loss argument" << std::endl;
        } catch (const std::out_of_range& e) {
            std::cerr << "[ERROR] play loss argument out of range" << std::endl;
        }
    }
    if (!res.log_to_file) {
        if (find_commandline_option(commandline_options, ID_LOGDIR)) {
            res.log_to_file = true;
            std::vector<std::string> arg = get_commandline_option_arg(commandline_options, ID_LOGDIR);
            try {
                res.log_file = arg[0] + "/" + datetime + "_search.log";
            } catch (const std::invalid_argument& e) {
                std::cerr << "[ERROR] invalid log dir" << std::endl;
            } catch (const std::out_of_range& e) {
                std::cerr << "[ERROR] log dir out of range" << std::endl;
            }
        }
    }
#ifdef INCLUDE_GGS
    res.ggs = find_commandline_option(commandline_options, ID_GGS);
    if (res.ggs) {
        res.ggs_username = "";
        res.ggs_password = "";
        if (find_commandline_option(commandline_options, ID_GGS)) {
            std::vector<std::string> arg = get_commandline_option_arg(commandline_options, ID_GGS);
            try {
                res.ggs_username = arg[0];
                res.ggs_password = arg[1];
            } catch (const std::invalid_argument& e) {
                std::cerr << "[ERROR] invalid ggs argument" << std::endl;
            } catch (const std::out_of_range& e) {
                std::cerr << "[ERROR] ggs argument out of range" << std::endl;
            }
        }
    }
    res.ggs_log_to_file = find_commandline_option(commandline_options, ID_GGS_LOGFILE);
    if (res.ggs_log_to_file) {
        std::vector<std::string> arg = get_commandline_option_arg(commandline_options, ID_GGS_LOGFILE);
        try {
            res.ggs_log_file = arg[0];
        } catch (const std::invalid_argument& e) {
            std::cerr << "[ERROR] invalid ggs log file" << std::endl;
        } catch (const std::out_of_range& e) {
            std::cerr << "[ERROR] ggs log file out of range" << std::endl;
        }
    }
    if (!res.ggs_log_to_file) {
        if (find_commandline_option(commandline_options, ID_GGS_LOGDIR)) {
            res.ggs_log_to_file = true;
            std::vector<std::string> arg = get_commandline_option_arg(commandline_options, ID_GGS_LOGDIR);
            try {
                res.ggs_log_file = arg[0] + "/" + datetime + "_ggs.log";
            } catch (const std::invalid_argument& e) {
                std::cerr << "[ERROR] invalid ggs log dir" << std::endl;
            } catch (const std::out_of_range& e) {
                std::cerr << "[ERROR] ggs log dir out of range" << std::endl;
            }
        }
    }
    res.ggs_game_log_to_file = find_commandline_option(commandline_options, ID_GGS_GAMELOGDIR);
    if (res.ggs_game_log_to_file) {
        std::vector<std::string> arg = get_commandline_option_arg(commandline_options, ID_GGS_GAMELOGDIR);
        try {
            res.ggs_game_log_dir = arg[0];
        } catch (const std::invalid_argument& e) {
            std::cerr << "[ERROR] invalid ggs game log dir" << std::endl;
        } catch (const std::out_of_range& e) {
            std::cerr << "[ERROR] ggs game log dir out of range" << std::endl;
        }
    }
    res.ggs_accept_request = find_commandline_option(commandline_options, ID_GGS_ACCEPT_REQUEST);
    res.ggs_route_join_tournament = find_commandline_option(commandline_options, ID_GGS_ROUTE_JOIN_TOURNAMENT);
#endif
    return res;
}