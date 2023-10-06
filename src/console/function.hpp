/*
    Egaroucid Project

    @file function.hpp
        Functions about engine
    @date 2021-2023
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

void setboard(Board_info *board, std::string board_str);
Search_result go_noprint(Board_info *board, Options *options, State *state);
void print_search_result_head();
void print_search_result_body(Search_result result, int level);
void go(Board_info *board, Options *options, State *state);

void solve_problems(std::string file, Options *options, State *state){
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
        print_search_result_body(res, options->level);
        transposition_table.init();
        total.nodes += res.nodes;
        total.time += res.time;
    }
    std::cout << "total " << total.nodes << " nodes in " << ((double)total.time / 1000) << "s NPS " << (total.nodes * 1000 / total.time) << std::endl;
}

void execute_special_tasks(){
    // move ordering tuning (endsearch)
    #if TUNE_MOVE_ORDERING_END
        std::cout << "tune move ordering (endsearch)" << std::endl;
        tune_move_ordering_end("problem/13_13.txt");
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

std::string self_play_task(Options *options){
    int n_random_moves = myrandrange(10, 20);
    Board board;
    Flip flip;
    Search_result result;
    board.reset();
    std::string res;
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
        res += idx_to_coord(flip.pos);
        board.move_board(&flip);
    }
    while (board.check_pass()){
        result = ai(board, options->level, true, 0, false, options->show_log); // search in single thread
        calc_flip(&flip, &board, result.policy);
        res += idx_to_coord(flip.pos);
        board.move_board(&flip);
    }
    return res;
}

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
    if (thread_pool.size() == 0){
        for (int i = 0; i < n_games; ++i){
            std::string transcript = self_play_task(options);
            std::cout << transcript << std::endl;
        }
    } else{
        int n_games_done = 0;
        std::vector<std::future<std::string>> tasks;
        while (n_games_done < n_games){
            if (thread_pool.get_n_idle() && (int)tasks.size() < n_games){
                bool pushed = false;
                tasks.emplace_back(thread_pool.push(&pushed, std::bind(&self_play_task, options)));
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
}