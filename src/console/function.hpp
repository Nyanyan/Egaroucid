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

void setboard(Board_info *board, Options *options, State *state, std::string board_str);
Search_result go_noprint(Board_info *board, Options *options, State *state);
void print_search_result_head();
void print_search_result_body(Search_result result, const Options *options, const State *state);
void go(Board_info *board, Options *options, State *state, uint64_t start_time);

void solve_problems(std::vector<std::string> arg, Options *options, State *state) {
    if (arg.size() < 1) {
        std::cerr << "[ERROR] [FATAL] please input problem file" << std::endl;
        return;
    }
    std::string file = arg[0];
    std::ifstream ifs(file);
    if (ifs.fail()) {
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
    while (std::getline(ifs, line)) {
        transposition_table.init();
        setboard(&board, options, state, line);
        #if USE_THREAD_MONITOR
            start_thread_monitor();
        #endif
        Search_result res = go_noprint(&board, options, state);
        transposition_table.reset_importance();
        print_search_result_body(res, options, state);
        transposition_table.init();
        total.nodes += res.nodes;
        total.time += res.time;
    }
    std::cout << "total " << total.nodes << " nodes in " << ((double)total.time / 1000) << "s NPS " << (total.nodes * 1000 / total.time) << std::endl;
}

void execute_special_tasks(Options options) {
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

bool execute_special_tasks_loop(Board_info *board, State *state, Options *options) {
    uint64_t start_time = tim();
    int player_before = board->player;
    if (options->mode == MODE_HUMAN_AI && board->player == WHITE && !board->board.is_end()) {
        go(board, options, state, start_time);
        return true;
    } else if (options->mode == MODE_AI_HUMAN && board->player == BLACK && !board->board.is_end()) {
        go(board, options, state, start_time);
        return true;
    } else if (options->mode == MODE_AI_AI && !board->board.is_end()) {
        go(board, options, state, start_time);
        return true;
    }
    return false;
}


std::string self_play_task(Board board_start, std::string pre_moves_transcript, Options *options, bool use_multi_thread, int n_random_moves_additional, int n_try) {
    Flip flip;
    Search_result result;
    std::string res = pre_moves_transcript;
    for (int j = 0; j < n_random_moves_additional && board_start.check_pass(); ++j) {
        uint64_t legal = board_start.get_legal();
        int random_idx = myrandrange(0, pop_count_ull(legal));
        int t = 0;
        for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)) {
            if (t == random_idx) {
                calc_flip(&flip, &board_start, cell);
                break;
            }
            ++t;
        }
        res += idx_to_coord(flip.pos);
        board_start.move_board(&flip);
    }
    std::vector<int> prev_transcript;
    for (int i = 0; i < n_try; ++i) {
        Board board = board_start.copy();
        std::vector<int> transcript;
        while (board.check_pass()) {
            result = ai(board, options->level, true, 0, false, options->show_log);
            transcript.emplace_back(result.policy);
            calc_flip(&flip, &board, result.policy);
            board.move_board(&flip);
        }
        bool break_flag = true;
        if (i < SELF_PLAY_N_TRY - 1) {
            if (prev_transcript.size() != transcript.size()) {
                break_flag = false;
            } else{
                for (int i = 0; i < transcript.size(); ++i) {
                    if (transcript[i] != prev_transcript[i]) {
                        break_flag = false;
                        break;
                    }
                }
            }
        }
        prev_transcript.clear();
        for (int &elem: transcript) {
            prev_transcript.emplace_back(elem);
        }
        if (break_flag) {
            break;
        }
    }
    for (int &elem: prev_transcript) {
        res += idx_to_coord(elem);
    }
    return res;
}

void self_play(std::vector<std::string> arg, Options *options, State *state) {
    int n_games, n_random_moves;
    if (arg.size() < 2) {
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
    Board board_start;
    board_start.reset();
    if (thread_pool.size() == 0) {
        for (int i = 0; i < n_games; ++i) {
            std::string transcript = self_play_task(board_start, "", options, false, n_random_moves, SELF_PLAY_N_TRY);
            std::cout << transcript << std::endl;
        }
    } else{
        int n_games_done = 0;
        std::vector<std::future<std::string>> tasks;
        while (n_games_done < n_games) {
            if (thread_pool.get_n_idle() && (int)tasks.size() < n_games) {
                bool pushed = false;
                tasks.emplace_back(thread_pool.push(&pushed, std::bind(&self_play_task, board_start, "", options, true, n_random_moves, SELF_PLAY_N_TRY)));
                if (!pushed) {
                    tasks.pop_back();
                }
            }
            for (std::future<std::string> &task: tasks) {
                if (task.valid()) {
                    if (task.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                        std::string transcript = task.get();
                        std::cout << transcript << std::endl;
                        ++n_games_done;
                    }
                }
            }
        }
    }
    global_searching = false;
    std::cerr << "done in " << tim() - strt << " ms" << std::endl;
}

void self_play_line(std::vector<std::string> arg, Options *options, State *state) {
    if (arg.size() < 1) {
        std::cerr << "please input opening file" << std::endl;
        std::exit(1);
    }
    std::string opening_file = arg[0];
    std::cerr << "selfplay with opening file " << opening_file << std::endl;
    std::ifstream ifs(opening_file);
    if (!ifs) {
        std::cerr << "[ERROR] can't open file " << opening_file << std::endl;
        std::exit(1);
    }
    uint64_t strt = tim();
    std::string line;
    Board board_start;
    Flip flip;
    Search_result result;
    std::vector<std::pair<std::string, Board>> board_list;
    while (std::getline(ifs, line)) {
        board_start.reset();
        for (int i = 0; i < (int)line.size() - 1; i += 2) {
            int x = line[i] - 'a';
            int y = line[i + 1] - '1';
            int coord = HW2_M1 - (y * HW + x);
            calc_flip(&flip, &board_start, coord);
            board_start.move_board(&flip);
        }
        board_list.emplace_back(std::make_pair(line, board_start));
    }
    if (thread_pool.size() == 0) {
        for (std::pair<std::string, Board> start_position: board_list) {
            std::string transcript = self_play_task(start_position.second, start_position.first, options, false, 0, SELF_PLAY_N_TRY);
            std::cout << transcript << std::endl;
        }
    } else{
        int print_task_idx = 0;
        std::vector<std::future<std::string>> tasks;
        for (std::pair<std::string, Board> start_position: board_list) {
            bool go_to_next_task = false;
            while (!go_to_next_task) {
                if (thread_pool.get_n_idle() && tasks.size() < board_list.size()) {
                    bool pushed = false;
                    tasks.emplace_back(thread_pool.push(&pushed, std::bind(&self_play_task, start_position.second, start_position.first, options, true, 0, SELF_PLAY_N_TRY)));
                    if (pushed) {
                        go_to_next_task = true;
                    } else {
                        tasks.pop_back();
                    }
                }
                if (tasks.size() > print_task_idx) {
                    if (tasks[print_task_idx].valid()) {
                        if (tasks[print_task_idx].wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                            std::string transcript = tasks[print_task_idx].get();
                            std::cout << transcript << std::endl;
                            ++print_task_idx;
                        }
                    } else {
                        std::cerr << "[ERROR] task not valid" << std::endl;
                        std::exit(1);
                    }
                }
            }
        }
        while (print_task_idx < tasks.size()) {
            if (tasks[print_task_idx].valid()) {
                std::string transcript = tasks[print_task_idx].get();
                std::cout << transcript << std::endl;
                ++print_task_idx;
            } else {
                std::cerr << "[ERROR] task not valid" << std::endl;
                std::exit(1);
            }
        }
    }
    std::cerr << "done in " << tim() - strt << " ms" << std::endl;
}


void self_play_board(std::vector<std::string> arg, Options *options, State *state) {
    if (arg.size() < 1) {
        std::cerr << "please input opening board file" << std::endl;
        std::exit(1);
    }
    std::string opening_board_file = arg[0];
    std::cerr << "selfplay with opening board file " << opening_board_file << std::endl;
    std::ifstream ifs(opening_board_file);
    if (!ifs) {
        std::cerr << "[ERROR] can't open file " << opening_board_file << std::endl;
        std::exit(1);
    }
    uint64_t strt = tim();
    std::string line;
    Flip flip;
    Search_result result;
    std::vector<std::pair<std::string, Board>> board_list;
    while (std::getline(ifs, line)) {
        std::pair<Board, int> board_player = convert_board_from_str(line);
        if (board_player.second != BLACK && board_player.second != WHITE) {
            std::cerr << "[ERROR] can't convert board " << line << std::endl;
            std::exit(1);
        }
        board_list.emplace_back(std::make_pair(line, board_player.first));
    }
    if (thread_pool.size() == 0) {
        for (std::pair<std::string, Board> start_position: board_list) {
            std::string transcript = self_play_task(start_position.second, "", options, false, 0, SELF_PLAY_N_TRY);
            std::cout << start_position.first << " " << transcript << std::endl;
        }
    } else{
        int print_task_idx = 0;
        std::vector<std::future<std::string>> tasks;
        for (std::pair<std::string, Board> start_position: board_list) {
            bool go_to_next_task = false;
            while (!go_to_next_task) {
                if (thread_pool.get_n_idle() && tasks.size() < board_list.size()) {
                    bool pushed = false;
                    tasks.emplace_back(thread_pool.push(&pushed, std::bind(&self_play_task, start_position.second, "", options, true, 0, SELF_PLAY_N_TRY)));
                    if (pushed) {
                        go_to_next_task = true;
                    } else {
                        tasks.pop_back();
                    }
                }
                if (tasks.size() > print_task_idx) {
                    if (tasks[print_task_idx].valid()) {
                        if (tasks[print_task_idx].wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                            std::string transcript = tasks[print_task_idx].get();
                            std::cout << board_list[print_task_idx].first << " " << transcript << std::endl;
                            ++print_task_idx;
                        }
                    } else {
                        std::cerr << "[ERROR] task not valid" << std::endl;
                        std::exit(1);
                    }
                }
            }
        }
        while (print_task_idx < tasks.size()) {
            if (tasks[print_task_idx].valid()) {
                std::string transcript = tasks[print_task_idx].get();
                std::cout << board_list[print_task_idx].first << " " << transcript << std::endl;
                ++print_task_idx;
            } else {
                std::cerr << "[ERROR] task not valid" << std::endl;
                std::exit(1);
            }
        }
    }
    std::cerr << "done in " << tim() - strt << " ms" << std::endl;
}

void self_play_lossless_lines_task(Board board, const std::string starting_board, Options *options, const int to_n_discs, std::vector<int> &transcript) {
    uint64_t legal = board.get_legal();
    if (legal == 0) {
        board.pass();
        legal = board.get_legal();
        if (legal == 0) {
            std::cout << starting_board << " ";
            for (int &cell: transcript) {
                std::cout << idx_to_coord(cell);
            }
            std::cout << " " << board.to_str() << " END" << std::endl;
            return;
        }
    }
    if (board.n_discs() >= to_n_discs) {
        std::cout << starting_board << " ";
        for (int &cell: transcript) {
            std::cout << idx_to_coord(cell);
        }
        std::cout << " " << board.to_str() << std::endl;
        return;
    }
    double hint_values[HW2];
    int hint_types[HW2];
    ai_hint(board, options->level, true, 0, true, false, 35, hint_values, hint_types);
    uint64_t legal_copy = legal;
    int best_score = -SCORE_MAX - 1;
    for (uint_fast8_t cell = first_bit(&legal_copy); legal_copy; cell = next_bit(&legal_copy)) {
        if (hint_values[cell] > best_score) {
            best_score = hint_values[cell];
        }
    }
    legal_copy = legal;
    Flip flip;
    for (uint_fast8_t cell = first_bit(&legal_copy); legal_copy; cell = next_bit(&legal_copy)) {
        if (hint_values[cell] >= best_score - 1) {
            calc_flip(&flip, &board, cell);
            board.move_board(&flip);
            transcript.emplace_back(cell);
                self_play_lossless_lines_task(board, starting_board, options, to_n_discs, transcript);
            transcript.pop_back();
            board.undo_board(&flip);
        }
    }
}

void self_play_board_lossless_lines(std::vector<std::string> arg, Options *options, State *state) {
    if (arg.size() < 2) {
        std::cerr << "please input opening board file and to_n_discs" << std::endl;
        std::exit(1);
    }
    std::string opening_board_file = arg[0];
    int to_n_discs = 0;
    try{
        to_n_discs = std::stoi(arg[1]);
    } catch (const std::invalid_argument& e) {
        std::cout << arg[1] << " invalid argument" << std::endl;
        std::exit(1);
    } catch (const std::out_of_range& e) {
        std::cout << arg[1] << " out of range" << std::endl;
        std::exit(1);
    }
    if (to_n_discs > HW2) {
        to_n_discs = HW2;
    }
    std::cerr << "selfplay with opening board file " << opening_board_file << " to " << to_n_discs << " discs" << std::endl;
    std::ifstream ifs(opening_board_file);
    if (!ifs) {
        std::cerr << "[ERROR] can't open file " << opening_board_file << std::endl;
        std::exit(1);
    }
    uint64_t strt = tim();
    std::string line;
    Flip flip;
    Search_result result;
    std::vector<std::pair<std::string, Board>> board_list;
    while (std::getline(ifs, line)) {
        std::pair<Board, int> board_player = convert_board_from_str(line);
        if (board_player.second != BLACK && board_player.second != WHITE) {
            std::cerr << "[ERROR] can't convert board " << line << std::endl;
            std::exit(1);
        }
        board_list.emplace_back(std::make_pair(line, board_player.first));
    }
    int idx = 0;
    for (std::pair<std::string, Board> start_position: board_list) {
        ++idx;
        double percent = (double)idx / board_list.size() * 100.0;
        std::cerr << idx << "/" << board_list.size() << " " << percent << "%" << std::endl;
        std::vector<int> transcript;
        self_play_lossless_lines_task(start_position.second, start_position.first, options, to_n_discs, transcript);
    }
    std::cerr << "done in " << tim() - strt << " ms" << std::endl;
}


void perft_commandline(std::vector<std::string> arg) {
    if (arg.size() < 2) {
        std::cerr << "please input <depth> <mode>" << std::endl;
        std::exit(1);
    }
    int depth, mode;
    std::string str_depth = arg[0];
    std::string str_mode = arg[1];
    try{
        depth = std::stoi(str_depth);
        mode = std::stoi(str_mode);
    } catch (const std::invalid_argument& e) {
        std::cout << str_depth << " " << str_mode << " invalid argument" << std::endl;
        std::exit(1);
    } catch (const std::out_of_range& e) {
        std::cout << str_depth << " " << str_mode << " out of range" << std::endl;
        std::exit(1);
    }
    if (mode != 1 && mode != 2) {
        std::cout << "mode must be 1 or 2, got " << mode << std::endl;
        std::exit(1);
    }
    if (depth <= 0 || 60 < depth) {
        std::cout << "depth must be in [1, 60], got " << depth << std::endl;
        std::exit(1);
    }
    Board board;
    board.reset();
    uint64_t strt = tim();
    uint64_t res;
    if (mode == 1) {
        res = perft(&board, depth, false);
    } else{
        res = perft_no_pass_count(&board, depth, false);
    }
    std::cout << "perft mode " << mode << " depth " << depth << " " << res << " leaves found in " << tim() - strt << " ms" << std::endl;
}
