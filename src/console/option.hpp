/*
    Egaroucid Project

    @file option.hpp
        Options of Egaroucid
    @date 2021-2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include "./../engine/engine_all.hpp"
#include "commandline_option.hpp"
#include "console_common.hpp"
#include "util.hpp"

#define TIME_NOT_ALLOCATED -1
constexpr int LEVEL_TYPE_CUSTOM = 1001;

struct Depth_prob_range_setting {
    int move_start;
    int move_end;
    int depth;
    uint_fast8_t mpc_level;
    double probability;
};

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
    bool use_depth_prob_ranges;
    std::vector<Depth_prob_range_setting> depth_prob_ranges;
#ifdef INCLUDE_FLIPSIGHT
    bool flipsight, flipsight_fixed_level, flipsight_no_stream, flipsight_no_ponder;
    std::string flipsight_token, flipsight_api;
    std::string flipsight_log_file, flipsight_log_dir;
    int flipsight_threads;
#endif
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

inline bool parse_selectivity_probability(const std::string &probability_str, uint_fast8_t *mpc_level, double *probability) {
    double probability_in;
    try {
        probability_in = std::stod(probability_str);
    } catch (const std::invalid_argument& e) {
        return false;
    } catch (const std::out_of_range& e) {
        return false;
    }
    constexpr double eps = 1e-4;
    for (int level = 0; level < N_SELECTIVITY_LEVEL; ++level) {
        if (std::abs(SELECTIVITY_PERCENTAGE[level] - probability_in) < eps) {
            *mpc_level = level;
            *probability = SELECTIVITY_PERCENTAGE[level];
            return true;
        }
    }
    return false;
}

inline bool get_depth_prob_range_setting(const Options *options, const Board &board, Depth_prob_range_setting *setting) {
    if (!options->use_depth_prob_ranges) {
        return false;
    }
    int move = board.n_discs() - 3;
    for (const Depth_prob_range_setting &elem: options->depth_prob_ranges) {
        if (move < elem.move_start) {
            break;
        }
        if (elem.move_start <= move && move <= elem.move_end) {
            *setting = elem;
            return true;
        }
    }
    return false;
}

inline Search_result ai_with_settings(Board board, const Options *options, bool use_multi_thread, bool show_log) {
    Depth_prob_range_setting setting;
    if (get_depth_prob_range_setting(options, board, &setting)) {
        bool searching = true;
        Search_result res = tree_search_legal(board, -SCORE_MAX, SCORE_MAX, setting.depth, setting.mpc_level, show_log, board.get_legal(), use_multi_thread, TIME_LIMIT_INF, THREAD_ID_NONE, &searching);
        res.level = LEVEL_TYPE_CUSTOM;
        return res;
    }
    return ai(board, options->level, true, 0, use_multi_thread, show_log);
}

inline Search_result ai_legal_with_settings(Board board, const Options *options, bool use_multi_thread, bool show_log, uint64_t legal) {
    Depth_prob_range_setting setting;
    if (get_depth_prob_range_setting(options, board, &setting)) {
        bool searching = true;
        Search_result res = tree_search_legal(board, -SCORE_MAX, SCORE_MAX, setting.depth, setting.mpc_level, show_log, legal, use_multi_thread, TIME_LIMIT_INF, THREAD_ID_NONE, &searching);
        res.level = LEVEL_TYPE_CUSTOM;
        return res;
    }
    return ai_legal(board, options->level, true, 0, use_multi_thread, show_log, legal);
}

inline Search_result ai_legal_window_with_settings(Board board, int alpha, int beta, const Options *options, bool use_multi_thread, bool show_log, uint64_t legal) {
    Depth_prob_range_setting setting;
    if (get_depth_prob_range_setting(options, board, &setting)) {
        bool searching = true;
        Search_result res = tree_search_legal(board, alpha, beta, setting.depth, setting.mpc_level, show_log, legal, use_multi_thread, TIME_LIMIT_INF, THREAD_ID_NONE, &searching);
        res.level = LEVEL_TYPE_CUSTOM;
        return res;
    }
    return ai_legal_window(board, alpha, beta, options->level, true, 0, use_multi_thread, show_log, legal);
}

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
    res.use_depth_prob_ranges = false;
    res.depth_prob_ranges.clear();
    if (find_commandline_option(commandline_options, ID_DEPTH_PROB_RANGE)) {
        std::vector<std::vector<std::string>> range_args_list = get_commandline_option_args(commandline_options, ID_DEPTH_PROB_RANGE);
        for (const std::vector<std::string> &range_args: range_args_list) {
            if (range_args.size() != 4) {
                std::cerr << "[ERROR] -depthprobrange requires 4 arguments: <m> <M> <depth> <prob>" << std::endl;
                std::exit(1);
            }
            int move_start, move_end, depth;
            try {
                move_start = std::stoi(range_args[0]);
                move_end = std::stoi(range_args[1]);
                depth = std::stoi(range_args[2]);
            } catch (const std::invalid_argument& e) {
                std::cerr << "[ERROR] invalid -depthprobrange argument" << std::endl;
                std::exit(1);
            } catch (const std::out_of_range& e) {
                std::cerr << "[ERROR] -depthprobrange argument out of range" << std::endl;
                std::exit(1);
            }
            if (!(1 <= move_start && move_start <= move_end && move_end <= 60)) {
                std::cerr << "[ERROR] -depthprobrange move range must satisfy 1 <= m <= M <= 60" << std::endl;
                std::exit(1);
            }
            if (depth < 1 || 60 < depth) {
                std::cerr << "[ERROR] -depthprobrange depth out of range (1 to 60)" << std::endl;
                std::exit(1);
            }
            uint_fast8_t mpc_level;
            double probability;
            if (!parse_selectivity_probability(range_args[3], &mpc_level, &probability)) {
                std::cerr << "[ERROR] -depthprobrange prob must be one of: 74, 88, 93, 98, 99, 99.9, 100" << std::endl;
                std::exit(1);
            }
            for (const Depth_prob_range_setting &setting: res.depth_prob_ranges) {
                if (!(move_end < setting.move_start || setting.move_end < move_start)) {
                    std::cerr << "[ERROR] overlapping -depthprobrange: [" << move_start << ", " << move_end << "] overlaps [" << setting.move_start << ", " << setting.move_end << "]" << std::endl;
                    std::exit(1);
                }
            }
            res.depth_prob_ranges.emplace_back(Depth_prob_range_setting{move_start, move_end, depth, mpc_level, probability});
        }
        std::sort(res.depth_prob_ranges.begin(), res.depth_prob_ranges.end(), [](const Depth_prob_range_setting &a, const Depth_prob_range_setting &b) {
            return a.move_start < b.move_start;
        });
        res.use_depth_prob_ranges = !res.depth_prob_ranges.empty();
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
    // The options below take a string as it is: assigning a std::string does not throw std::invalid_argument / std::out_of_range,
    // so the try / catch blocks around these assignments (some with the message of the hash option) were removed.
    // The number of arguments is checked in get_commandline_options().
    res.book_file = binary_path + "resources/book.egbk3";
    if (find_commandline_option(commandline_options, ID_BOOK_FILE)) {
        std::vector<std::string> arg = get_commandline_option_arg(commandline_options, ID_BOOK_FILE);
        res.book_file = arg[0];
    }
    res.eval_file = binary_path + "resources/eval.egev2";
    if (find_commandline_option(commandline_options, ID_EVAL_FILE)) {
        std::vector<std::string> arg = get_commandline_option_arg(commandline_options, ID_EVAL_FILE);
        res.eval_file = arg[0];
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
        res.log_file = arg[0];
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
            res.log_file = arg[0] + "/" + datetime + "_search.log";
        }
    }
#ifdef INCLUDE_GGS
    res.ggs = find_commandline_option(commandline_options, ID_GGS);
    if (res.ggs) {
        res.ggs_username = "";
        res.ggs_password = "";
        if (find_commandline_option(commandline_options, ID_GGS)) {
            std::vector<std::string> arg = get_commandline_option_arg(commandline_options, ID_GGS);
            res.ggs_username = arg[0];
            res.ggs_password = arg[1];
        }
    }
    res.ggs_log_to_file = find_commandline_option(commandline_options, ID_GGS_LOGFILE);
    if (res.ggs_log_to_file) {
        std::vector<std::string> arg = get_commandline_option_arg(commandline_options, ID_GGS_LOGFILE);
        res.ggs_log_file = arg[0];
    }
    if (!res.ggs_log_to_file) {
        if (find_commandline_option(commandline_options, ID_GGS_LOGDIR)) {
            res.ggs_log_to_file = true;
            std::vector<std::string> arg = get_commandline_option_arg(commandline_options, ID_GGS_LOGDIR);
            res.ggs_log_file = arg[0] + "/" + datetime + "_ggs.log";
        }
    }
    res.ggs_game_log_to_file = find_commandline_option(commandline_options, ID_GGS_GAMELOGDIR);
    if (res.ggs_game_log_to_file) {
        std::vector<std::string> arg = get_commandline_option_arg(commandline_options, ID_GGS_GAMELOGDIR);
        res.ggs_game_log_dir = arg[0];
    }
    res.ggs_accept_request = find_commandline_option(commandline_options, ID_GGS_ACCEPT_REQUEST);
    res.ggs_route_join_tournament = find_commandline_option(commandline_options, ID_GGS_ROUTE_JOIN_TOURNAMENT);
#endif
#ifdef INCLUDE_FLIPSIGHT
    res.flipsight = find_commandline_option(commandline_options, ID_FLIPSIGHT);
    res.flipsight_fixed_level = find_commandline_option(commandline_options, ID_LEVEL);
    res.flipsight_no_stream = find_commandline_option(commandline_options, ID_FLIPSIGHT_NO_STREAM);
    res.flipsight_no_ponder = find_commandline_option(commandline_options, ID_FLIPSIGHT_NO_PONDER);
    res.flipsight_log_dir = binary_path + "flipsight/log";
    if (find_commandline_option(commandline_options, ID_FLIPSIGHT_LOGDIR))
        res.flipsight_log_dir = get_commandline_option_arg(commandline_options, ID_FLIPSIGHT_LOGDIR)[0];
    if (find_commandline_option(commandline_options, ID_FLIPSIGHT_LOGFILE))
        res.flipsight_log_file = get_commandline_option_arg(commandline_options, ID_FLIPSIGHT_LOGFILE)[0];
    res.flipsight_threads = 4;
    if (find_commandline_option(commandline_options, ID_FLIPSIGHT_THREAD)) {
        const auto text = get_commandline_option_arg(commandline_options, ID_FLIPSIGHT_THREAD)[0];
        try {
            size_t used = 0;
            auto count = std::stoi(text, &used);
            // The service admits at most eight paired matches (sixteen boards).
            if (used != text.size() || count < 1 || count > std::numeric_limits<int>::max() / 16)
                throw std::invalid_argument("threads");
            res.flipsight_threads = count;
        } catch (...) {
            std::cerr << "[ERROR] -fsthread requires a positive integer within the supported thread-count range" << std::endl;
            std::exit(1);
        }
    }
    res.flipsight_api = "https://othelloclub.com/flipsight/bot_match_api.php";
    if (find_commandline_option(commandline_options, ID_FLIPSIGHT_API))
        res.flipsight_api = get_commandline_option_arg(commandline_options, ID_FLIPSIGHT_API)[0];
    if (res.flipsight) {
        res.flipsight_token = get_commandline_option_arg(commandline_options, ID_FLIPSIGHT)[0];
        res.n_threads = res.flipsight_threads * 2;
        if (res.flipsight_fixed_level) {
            const auto text = get_commandline_option_arg(commandline_options, ID_LEVEL)[0];
            try {
                size_t used = 0;
                auto level = std::stoi(text, &used);
                if (used != text.size() || level < 0 || level >= N_LEVEL) throw std::invalid_argument("level");
                res.level = level;
            } catch (...) {
                std::cerr << "[ERROR] FlipSight level must be an integer from 0 to 60" << std::endl;
                std::exit(1);
            }
        }
        bool incompatible = res.gtp;
#ifdef INCLUDE_GGS
        incompatible = incompatible || res.ggs;
#endif
        if (incompatible) {
            std::cerr << "[ERROR] FlipSight must be used as the only network/protocol mode" << std::endl;
            std::exit(1);
        }
    }
#endif
    return res;
}
