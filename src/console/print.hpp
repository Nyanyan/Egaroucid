/*
    Egaroucid Project

    @file print.hpp
        Functions about printing on console
    @date 2021-2023
    @author Takuto Yamana
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
#define ANALYZE_TAB_SIZE 13
#define ANALYZE_SUMMARY_TAB_SIZE 13

struct Analyze_summary{
    int n_ply;
    int n_disagree;
    int sum_disagree;
    int n_mistake;
    int sum_mistake;

    Analyze_summary(){
        n_ply = 0;
        n_disagree = 0;
        sum_disagree = 0;
        n_mistake = 0;
        sum_mistake = 0;
    }
};

void print_version(){
    std::cout << EGAROUCID_NAME << " " << EGAROUCID_VERSION << std::endl;
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
    const std::string probability_char = "_-=+^#";
    std::cout << "Level definition:" << std::endl;
    std::cout << COUT_TAB;
    std::cout << "Endgame probability" << std::endl;
    std::cout << COUT_TAB << COUT_TAB << " 78%: " << probability_char[MPC_78_LEVEL] << std::endl;
    std::cout << COUT_TAB << COUT_TAB << " 81%: " << probability_char[MPC_81_LEVEL] << std::endl;
    std::cout << COUT_TAB << COUT_TAB << " 95%: " << probability_char[MPC_95_LEVEL] << std::endl;
    std::cout << COUT_TAB << COUT_TAB << " 98%: " << probability_char[MPC_98_LEVEL] << std::endl;
    std::cout << COUT_TAB << COUT_TAB << " 99%: " << probability_char[MPC_99_LEVEL] << std::endl;
    std::cout << COUT_TAB << COUT_TAB << "100%: " << probability_char[MPC_100_LEVEL] << std::endl;
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
            s = std::to_string(level_definition[level].mid_lookahead) + " moves@" + std::to_string(SELECTIVITY_PERCENTAGE[level_definition[level].mid_mpc_level]) + "%";
        else
            s = "None";
        std::cout << std::right << std::setw(LEVEL_MIDGAME_TAB_SIZE) << s;
        for (int n_moves = 0; n_moves < HW2 - 4; ++n_moves){
            if (n_moves % LEVEL_DEPTH_TAB_SIZE == 0)
                std::cout << "|";
            bool is_mid_search;
            uint_fast8_t mpc_level;
            int depth;
            get_level(level, n_moves, &is_mid_search, &depth, &mpc_level);
            if (is_mid_search)
                std::cout << " ";
            else{
                std::cout << probability_char[mpc_level];
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
            int black_discs, white_discs;
            black_discs = board->board.count_player();
            white_discs = board->board.count_opponent();
            if (board->player)
                std::swap(black_discs, white_discs);
            std::cout << "BLACK: " << black_discs << " WHITE: " << white_discs;
        }
        std::cout << std::endl;
    }
}

inline void print_search_result_body(Search_result result, int level){
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
        s = ms_to_time(0);
        std::cout << std::right << std::setw(SEARCH_RESULT_TAB_SIZE) << s;
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
        s = ms_to_time(result.time);
        std::cout << std::right << std::setw(SEARCH_RESULT_TAB_SIZE) << s;
        std::cout << "|";
        std::cout << std::right << std::setw(SEARCH_RESULT_TAB_SIZE) << result.nodes;
        std::cout << "|";
        std::cout << std::right << std::setw(SEARCH_RESULT_TAB_SIZE) << result.nps;
        std::cout << "|";
        std::cout << std::endl;
    }
}

inline void print_search_result_head(){
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

inline void print_search_result(Search_result result, int level){
    print_search_result_head();
    print_search_result_body(result, level);
}

void print_search_result_quiet(Search_result result){
    std::cout << idx_to_coord(result.policy) << std::endl;
}

inline void print_analyze_body(Analyze_result result, int ply, int player, std::string judge){
    std::string s;
    std::cout << "|";
    std::cout << std::right << std::setw(ANALYZE_TAB_SIZE) << ply;
    std::cout << "|";
    if (player == BLACK)
        std::cout << std::right << std::setw(ANALYZE_TAB_SIZE) << "Black";
    else
        std::cout << std::right << std::setw(ANALYZE_TAB_SIZE) << "White";
    std::cout << "|";
    std::cout << std::right << std::setw(ANALYZE_TAB_SIZE) << idx_to_coord(result.played_move);
    std::cout << "|";
    if (result.played_depth == SEARCH_BOOK)
        std::cout << std::right << std::setw(ANALYZE_TAB_SIZE) << "Book";
    else{
        s = std::to_string(result.played_depth) + "@" + std::to_string(result.played_probability) + "%";
        std::cout << std::right << std::setw(ANALYZE_TAB_SIZE) << s;
    }
    std::cout << "|";
    if (result.played_score >= 0)
        s = "+" + std::to_string(result.played_score);
    else
        s = std::to_string(result.played_score);
    std::cout << std::right << std::setw(ANALYZE_TAB_SIZE) << s;
    std::cout << "|";
    if (result.alt_move != -1){
        std::cout << std::right << std::setw(ANALYZE_TAB_SIZE) << idx_to_coord(result.alt_move);
        std::cout << "|";
        if (result.alt_depth == SEARCH_BOOK)
            std::cout << std::right << std::setw(ANALYZE_TAB_SIZE) << "Book";
        else{
            s = std::to_string(result.alt_depth) + "@" + std::to_string(result.alt_probability) + "%";
            std::cout << std::right << std::setw(ANALYZE_TAB_SIZE) << s;
        }
        std::cout << "|";
        if (result.alt_score >= 0)
            s = "+" + std::to_string(result.alt_score);
        else
            s = std::to_string(result.alt_score);
        std::cout << std::right << std::setw(ANALYZE_TAB_SIZE) << s;
        std::cout << "|";
    } else{
        std::cout << std::right << std::setw(ANALYZE_TAB_SIZE) << "None";
        std::cout << "|";
        std::cout << std::right << std::setw(ANALYZE_TAB_SIZE) << "";
        std::cout << "|";
        std::cout << std::right << std::setw(ANALYZE_TAB_SIZE) << "";
        std::cout << "|";
    }
    std::cout << std::right << std::setw(ANALYZE_TAB_SIZE) << judge;
    std::cout << "|";
    std::cout << std::endl;
}

inline void print_analyze_head(){
    std::cout << "|";
    std::cout << std::right << std::setw(ANALYZE_TAB_SIZE) << "Ply";
    std::cout << "|";
    std::cout << std::right << std::setw(ANALYZE_TAB_SIZE) << "Player";
    std::cout << "|";
    std::cout << std::right << std::setw(ANALYZE_TAB_SIZE) << "Played";
    std::cout << "|";
    std::cout << std::right << std::setw(ANALYZE_TAB_SIZE) << "Depth";
    std::cout << "|";
    std::cout << std::right << std::setw(ANALYZE_TAB_SIZE) << "Score";
    std::cout << "|";
    std::cout << std::right << std::setw(ANALYZE_TAB_SIZE) << "Alternative";
    std::cout << "|";
    std::cout << std::right << std::setw(ANALYZE_TAB_SIZE) << "Depth";
    std::cout << "|";
    std::cout << std::right << std::setw(ANALYZE_TAB_SIZE) << "Score";
    std::cout << "|";
    std::cout << std::right << std::setw(ANALYZE_TAB_SIZE) << "Judge";
    std::cout << "|";
    std::cout << std::endl;
}

inline void print_analyze_foot(Analyze_summary summary[]){
    std::cout << std::endl;
    std::cout << "|";
    std::cout << std::right << std::setw(ANALYZE_SUMMARY_TAB_SIZE) << "Player";
    std::cout << "|";
    std::cout << std::right << std::setw(ANALYZE_SUMMARY_TAB_SIZE) << "Disagree";
    std::cout << "|";
    std::cout << std::right << std::setw(ANALYZE_SUMMARY_TAB_SIZE) << "Disagree Loss";
    std::cout << "|";
    std::cout << std::right << std::setw(ANALYZE_SUMMARY_TAB_SIZE) << "Disagree Rate";
    std::cout << "|";
    std::cout << std::right << std::setw(ANALYZE_SUMMARY_TAB_SIZE) << "Mistake";
    std::cout << "|";
    std::cout << std::right << std::setw(ANALYZE_SUMMARY_TAB_SIZE) << "Mistake Loss";
    std::cout << "|";
    std::cout << std::right << std::setw(ANALYZE_SUMMARY_TAB_SIZE) << "Mistake Rate";
    std::cout << "|";
    std::cout << std::right << std::setw(ANALYZE_SUMMARY_TAB_SIZE) << "Avg. Error";
    std::cout << "|";
    std::cout << std::endl;
    std::string s;
    for (int i = 0; i < 2; ++i){
        std::cout << "|";
        if (i == BLACK)
            std::cout << std::right << std::setw(ANALYZE_SUMMARY_TAB_SIZE) << "Black";
        else
            std::cout << std::right << std::setw(ANALYZE_SUMMARY_TAB_SIZE) << "White";
        std::cout << "|";
        std::stringstream ss_disagree;
        ss_disagree << std::right << std::setw(2) << summary[i].n_disagree;
        ss_disagree << " / ";
        ss_disagree << std::right << std::setw(2) << summary[i].n_ply;
        std::cout << std::right << std::setw(ANALYZE_SUMMARY_TAB_SIZE) << ss_disagree.str();
        std::cout << "|";
        std::cout << std::right << std::setw(ANALYZE_SUMMARY_TAB_SIZE) << summary[i].sum_disagree;
        std::cout << "|";
        std::cout << std::right << std::setw(ANALYZE_SUMMARY_TAB_SIZE) << std::fixed << std::setprecision(3) << ((double)summary[i].n_disagree / summary[i].n_ply);
        std::cout << "|";
        std::stringstream ss_mistake;
        ss_mistake << std::right << std::setw(2) << summary[i].n_mistake;
        ss_mistake << " / ";
        ss_mistake << std::right << std::setw(2) << summary[i].n_ply;
        std::cout << std::right << std::setw(ANALYZE_SUMMARY_TAB_SIZE) << ss_mistake.str();
        std::cout << "|";
        std::cout << std::right << std::setw(ANALYZE_SUMMARY_TAB_SIZE) << summary[i].sum_mistake;
        std::cout << "|";
        std::cout << std::right << std::setw(ANALYZE_SUMMARY_TAB_SIZE) << std::fixed << std::setprecision(3) << ((double)summary[i].n_mistake / summary[i].n_ply);
        std::cout << "|";
        std::cout << std::right << std::setw(ANALYZE_SUMMARY_TAB_SIZE) << std::fixed << std::setprecision(3) << ((double)(summary[i].sum_disagree + summary[i].sum_mistake) / summary[i].n_ply);
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

void execute_special_commandline_tasks(std::vector<Commandline_option> commandline_options, Options *options, State *state){
    if (find_commandline_option(commandline_options, ID_SOLVE) != OPTION_NOT_FOUND){
        solve_problems(find_commandline_option(commandline_options, ID_SOLVE), options, state);
        std::exit(0);
    } else if (find_commandline_option(commandline_options, ID_SELF_PLAY) != OPTION_NOT_FOUND){
        self_play(find_commandline_option(commandline_options, ID_SELF_PLAY), options, state);
        std::exit(0);
    }
}