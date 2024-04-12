/*
    Egaroucid Project

    @file function.hpp
        Functions about engine
    @date 2021-2024
    @author Takuto Yamana
    @license GPL-3.0 license
*/

#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include "./../engine/engine_all.hpp"
#include "command.hpp"

#if USE_THREAD_MONITOR
    #include "./../engine/thread_monitor.hpp"
#endif

#define SELF_PLAY_N_TRY 1

void setboard(Board_info *board, std::string board_str);
Search_result go_noprint(Board_info *board, Options *options, State *state);
void print_search_result_head();
void print_search_result_body(Search_result result, int level);
void go(Board_info *board, Options *options, State *state);

void solve_problems(std::vector<std::string> arg, Options *options, State *state){
    if (arg.size() < 1){
        std::cerr << "[ERROR] [FATAL] please input problem file" << std::endl;
        return;
    }
    std::string file = arg[0];
    std::ifstream ifs(file);
    if (ifs.fail()){
        std::cerr << "[ERROR] [FATAL] no problem file found" << std::endl;
        return;
    }
    std::string line;
    Board_info board;
    board.reset();
    print_search_result_head();
    Search_result total;
    total.nodes = 0;
    total.time = 0;
    while (std::getline(ifs, line)){
        transposition_table.init();
        setboard(&board, line);
        #if USE_THREAD_MONITOR
            start_thread_monitor();
        #endif
        Search_result res = go_noprint(&board, options, state);
        transposition_table.reset_importance();
        print_search_result_body(res, options->level);
        transposition_table.init();
        total.nodes += res.nodes;
        total.time += res.time;
    }
    std::cout << "total " << total.nodes << " nodes in " << ((double)total.time / 1000) << "s NPS " << (total.nodes * 1000 / total.time) << std::endl;
}

void execute_special_tasks(Options options){
    // move ordering tuning
    #if TUNE_MOVE_ORDERING
        std::cout << "tune move ordering ";
        tune_move_ordering(options.level);
        std::exit(0);
    #endif

    // probcut (midsearch)
    #if TUNE_PROBCUT_MID
        std::cout << "tune probcut (midsearch)" << std::endl;
        get_data_probcut_mid();
        std::exit(0);
    #endif

    // probcut (endsearch)
    #if TUNE_PROBCUT_END
        std::cout << "tune probcut (endsearch)" << std::endl;
        get_data_probcut_end();
        std::exit(0);
    #endif
}

bool execute_special_tasks_loop(Board_info *board, State *state, Options *options){
    if (options->mode == MODE_HUMAN_AI && board->player == WHITE && !board->board.is_end()){
        go(board, options, state);
        return true;
    } else if (options->mode == MODE_AI_HUMAN && board->player == BLACK && !board->board.is_end()){
        go(board, options, state);
        return true;
    } else if (options->mode == MODE_AI_AI && !board->board.is_end()){
        go(board, options, state);
        return true;
    }
    return false;
}


std::string self_play_task(Options *options, bool use_multi_thread, int n_random_moves, int n_try){
    Board board_start;
    Flip flip;
    Search_result result;
    board_start.reset();
    std::string res;
    for (int j = 0; j < n_random_moves && board_start.check_pass(); ++j){
        uint64_t legal = board_start.get_legal();
        int random_idx = myrandrange(0, pop_count_ull(legal));
        int t = 0;
        for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)){
            if (t == random_idx){
                calc_flip(&flip, &board_start, cell);
                break;
            }
            ++t;
        }
        res += idx_to_coord(flip.pos);
        board_start.move_board(&flip);
    }
    std::vector<int> prev_transcript;
    for (int i = 0; i < n_try; ++i){
        Board board = board_start.copy();
        std::vector<int> transcript;
        while (board.check_pass()){
            result = ai(board, options->level, true, 0, false, options->show_log);
            transcript.emplace_back(result.policy);
            calc_flip(&flip, &board, result.policy);
            board.move_board(&flip);
        }
        bool break_flag = true;
        if (prev_transcript.size() != transcript.size()){
            break_flag = false;
        } else{
            for (int i = 0; i < transcript.size(); ++i){
                if (transcript[i] != prev_transcript[i]){
                    break_flag = false;
                    break;
                }
            }
        }
        prev_transcript.clear();
        for (int &elem: transcript){
            prev_transcript.emplace_back(elem);
        }
        if (break_flag){
            break;
        }
    }
    for (int &elem: prev_transcript){
        res += idx_to_coord(elem);
    }
    return res;
}

void self_play(std::vector<std::string> arg, Options *options, State *state){
    int n_games, n_random_moves;
    if (arg.size() < 2){
        std::cerr << "[ERROR] [FATAL] please input arguments" << std::endl;
        std::exit(1);
    }
    std::string str_n_games = arg[0];
    std::string str_n_random_moves = arg[1];
    try{
        n_games = std::stoi(str_n_games);
        n_random_moves = std::stoi(str_n_random_moves);
    } catch (const std::invalid_argument& e) {
        std::cout << str_n_games << " " << str_n_random_moves << " invalid argument" << std::endl;
        std::exit(1);
    } catch (const std::out_of_range& e) {
        std::cout << str_n_games << " " << str_n_random_moves << " out of range" << std::endl;
        std::exit(1);
    }
    std::cerr << n_games << " games with " << n_random_moves << " random moves" << std::endl;
    uint64_t strt = tim();
    if (thread_pool.size() == 0){
        for (int i = 0; i < n_games; ++i){
            std::string transcript = self_play_task(options, false, n_random_moves, SELF_PLAY_N_TRY);
            std::cout << transcript << std::endl;
        }
    } else{
        int n_games_done = 0;
        std::vector<std::future<std::string>> tasks;
        while (n_games_done < n_games){
            if (thread_pool.get_n_idle() && (int)tasks.size() < n_games){
                bool pushed = false;
                tasks.emplace_back(thread_pool.push(&pushed, std::bind(&self_play_task, options, true, n_random_moves, SELF_PLAY_N_TRY)));
                if (!pushed)
                    tasks.pop_back();
            }
            for (std::future<std::string> &task: tasks){
                if (task.valid()){
                    if (task.wait_for(std::chrono::seconds(0)) == std::future_status::ready){
                        std::string transcript = task.get();
                        std::cout << transcript << std::endl;
                        ++n_games_done;
                        break;
                    }
                }
            }
        }
    }
    global_searching = false;
    std::cerr << "done in " << tim() - strt << " ms" << std::endl;
}

/*
void self_play(std::string str_n_games, Options *options, State *state){
    int n_games;
    try{
        n_games = std::stoi(str_n_games);
    } catch (const std::invalid_argument& e) {
        std::cout << str_n_games << " invalid argument" << std::endl;
        std::exit(1);
    } catch (const std::out_of_range& e) {
        std::cout << str_n_games << " out of range" << std::endl;
        std::exit(1);
    }
    uint64_t strt = tim();
    for (int i = 0; i < n_games; ++i){
        int n_random_moves = myrandrange(10, 20);
        Board board;
        Flip flip;
        Search_result result;
        board.reset();
        for (int j = 0; j < n_random_moves && board.check_pass(); ++j){
            uint64_t legal = board.get_legal();
            int random_idx = myrandrange(0, pop_count_ull(legal));
            int t = 0;
            for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)){
                if (t == random_idx){
                    calc_flip(&flip, &board, cell);
                    break;
                }
                ++t;
            }
            std::cout << idx_to_coord(flip.pos);
            board.move_board(&flip);
        }
        while (board.check_pass()){
            result = ai(board, options->level, true, 0, true, options->show_log);
            calc_flip(&flip, &board, result.policy);
            std::cout << idx_to_coord(flip.pos);
            board.move_board(&flip);
        }
        std::cout << std::endl;
    }
    std::cerr << "done in " << tim() - strt << " ms" << std::endl;
}
*/

void self_play_line(std::vector<std::string> arg, Options *options, State *state){
    if (arg.size() < 1){
        std::cerr << "please input opening file" << std::endl;
        std::exit(1);
    }
    std::string opening_file = arg[0];
    std::cerr << "selfplay with opening file " << opening_file << std::endl;
    std::ifstream ifs(opening_file);
    if (!ifs){
        std::cerr << "can't open file " << opening_file << std::endl;
    }
    uint64_t strt = tim();
    std::string line;
    Board board;
    Flip flip;
    Search_result result;
    while (std::getline(ifs, line)){
        board.reset();
        for (int i = 0; i < (int)line.size() - 1; i += 2){
            int x = line[i] - 'a';
            int y = line[i + 1] - '1';
            int coord = HW2_M1 - (y * HW + x);
            calc_flip(&flip, &board, coord);
            std::cout << idx_to_coord(flip.pos);
            board.move_board(&flip);
        }
        while (board.check_pass()){
            result = ai(board, options->level, true, 0, true, options->show_log);
            calc_flip(&flip, &board, result.policy);
            std::cout << idx_to_coord(flip.pos);
            board.move_board(&flip);
        }
        std::cout << std::endl;
    }
    std::cerr << "done in " << tim() - strt << " ms" << std::endl;
}
