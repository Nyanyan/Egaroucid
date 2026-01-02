/*
    Egaroucid Project

    @file function.hpp
        Functions about engine
    @date 2021-2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
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
        print_search_result_body(res, options, state);
        total.nodes += res.nodes;
        total.time += res.time;
    }
    std::cout << "total " << total.nodes << " nodes in " << ((double)total.time / 1000) << "s NPS " << calc_nps(total.nodes, total.time) << std::endl;
}

void solve_problems_transcript_parallel(std::vector<std::string> arg, Options *options, State *state) {
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
    uint64_t strt = tim();
    std::string line;
    std::vector<Board> board_list;
    Flip flip;
    Board board_start;
    while (std::getline(ifs, line)) {
        /*
        std::pair<Board, int> board_player = convert_board_from_str(line);
        if (board_player.second != BLACK && board_player.second != WHITE) {
            std::cerr << "[ERROR] can't convert board " << line << std::endl;
            std::exit(1);
        }
        board_list.emplace_back(board_player.first);
        */
        board_start.reset();
        for (int i = 0; i < (int)line.size() - 1; i += 2) {
            int x = line[i] - 'a';
            int y = line[i + 1] - '1';
            int coord = HW2_M1 - (y * HW + x);
            calc_flip(&flip, &board_start, coord);
            board_start.move_board(&flip);
            if (board_start.get_legal() == 0) {
                board_start.pass();
            }
        }
        board_list.emplace_back(board_start);
    }
    Search_result result;
    if (thread_pool.size() == 0) {
        for (int i = 0; i < (int)board_list.size(); ++i) {
            result = ai(board_list[i], options->level, true, 0, false, options->show_log);
            std::cout << board_list[i].to_str() << " " << result.value << std::endl;
        }
    } else {
        int print_task_idx = 0;
        std::vector<std::future<Search_result>> tasks;
        for (Board &board: board_list) {
            bool go_to_next_task = false;
            while (!go_to_next_task) {
                if (thread_pool.get_n_idle() && tasks.size() < board_list.size()) {
                    bool pushed = false;
                    tasks.emplace_back(thread_pool.push(&pushed, std::bind(&ai, board, options->level, true, 0, false, options->show_log)));
                    if (pushed) {
                        go_to_next_task = true;
                    } else {
                        tasks.pop_back();
                    }
                }
                if (tasks.size() > print_task_idx) {
                    if (tasks[print_task_idx].valid()) {
                        if (tasks[print_task_idx].wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                            result = tasks[print_task_idx].get();
                            std::cout << board_list[print_task_idx].to_str() << " " << result.value << std::endl;
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
                result = tasks[print_task_idx].get();
                std::cout << board_list[print_task_idx].to_str() << " " << result.value << std::endl;
                ++print_task_idx;
            } else {
                std::cerr << "[ERROR] task not valid" << std::endl;
                std::exit(1);
            }
        }
    }
    std::cerr << "done in " << tim() - strt << " ms" << std::endl;
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

    // local strategy
    #if TUNE_LOCAL_STRATEGY
        std::cout << "tune local strategy" << std::endl;
        tune_local_strategy();
        std::exit(0);
    #endif

    #if TEST_ENDGAME_ACCURACY
        endgame_accuracy_test();
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
            result = ai(board, options->level, true, 0, use_multi_thread, options->show_log);
            if (global_searching && is_valid_policy(result.policy)) {
                transcript.emplace_back(result.policy);
                calc_flip(&flip, &board, result.policy);
                board.move_board(&flip);
            } else {
                break;
            }
        }
        if (!global_searching) {
            if (n_try == 1 || i == n_try - 1) {
                prev_transcript.clear();
                for (int &elem: transcript) {
                    prev_transcript.emplace_back(elem);
                }
            }
            break;
        }
        bool break_flag = true;
        if (i < n_try - 1) {
            if (prev_transcript.size() != transcript.size()) {
                break_flag = false;
            } else {
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
    } else {
        int n_games_done = 0;
        std::vector<std::future<std::string>> tasks;
        while (n_games_done < n_games) {
            if (thread_pool.get_n_idle() && (int)tasks.size() < n_games) {
                bool pushed = false;
                tasks.emplace_back(thread_pool.push(&pushed, std::bind(&self_play_task, board_start, "", options, false, n_random_moves, SELF_PLAY_N_TRY)));
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
            if (0 < n_games - n_games_done && n_games - n_games_done < thread_pool.size()) {
                std::vector<std::string> transcripts_mid;
                global_searching = false;
                    for (std::future<std::string> &task: tasks) {
                        if (task.valid()) {
                            std::string transcript_mid = task.get();
                            transcripts_mid.emplace_back(transcript_mid);
                        }
                    }
                global_searching = true;
                for (std::string &transcript_mid: transcripts_mid) {
                    Board board_start_mid = board_start.copy();
                    Flip flip;
                    for (int i = 0; i < transcript_mid.size(); i += 2) {
                        int x = transcript_mid[i] - 'a';
                        int y = transcript_mid[i + 1] - '1';
                        int coord = HW2_M1 - (y * HW + x);
                        calc_flip(&flip, &board_start_mid, coord);
                        board_start_mid.move_board(&flip);
                        if (board_start_mid.get_legal() == 0) {
                            board_start_mid.pass();
                        }
                    }
                    int n_random_moves_additional = std::max(0, n_random_moves - (int)transcript_mid.size() / 2);
                    std::string transcript = self_play_task(board_start_mid, transcript_mid, options, true, n_random_moves_additional, SELF_PLAY_N_TRY);
                    std::cout << transcript << std::endl;
                    ++n_games_done;
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
            if (board_start.get_legal() == 0) {
                board_start.pass();
            }
        }
        board_list.emplace_back(std::make_pair(line, board_start));
    }
    if (thread_pool.size() == 0) {
        for (std::pair<std::string, Board> start_position: board_list) {
            std::string transcript = self_play_task(start_position.second, start_position.first, options, false, 0, SELF_PLAY_N_TRY);
            std::cout << transcript << std::endl;
        }
    } else {
        int task_idx = 0;
        int print_idx = 0;
        int n_running_task = 0;
        std::vector<std::future<std::string>> tasks;
        std::vector<std::string> results;
        int n_games = board_list.size();
        while (print_idx < n_games) {
            // add task
            if (thread_pool.get_n_idle() && tasks.size() < n_games) {
                bool pushed = false;
                tasks.emplace_back(thread_pool.push(&pushed, std::bind(&self_play_task, board_list[task_idx].second, board_list[task_idx].first, options, false, 0, SELF_PLAY_N_TRY)));
                if (pushed) {
                    ++task_idx;
                    ++n_running_task;
                    results.emplace_back("");
                } else {
                    tasks.pop_back();
                }
            }
            // check if task ends
            for (int i = 0; i < tasks.size(); ++i) {
                if (tasks[i].valid()) {
                    if (tasks[i].wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                        --n_running_task;
                        results[i] = tasks[i].get();
                    }
                }
            }
            // print result
            if (tasks.size() > print_idx) {
                for (int i = print_idx; i < results.size(); ++i) {
                    if (results[i] != "") {
                        std::cout << results[i] << std::endl;
                        ++print_idx;
                    } else {
                        break;
                    }
                }
            }
            // special case
            if (task_idx == n_games && 0 < n_running_task && n_running_task < thread_pool.size()) {
                std::vector<std::pair<int, std::string>> mid_tasks;
                global_searching = false;
                    for (int i = 0; i < tasks.size(); ++i) {
                        if (tasks[i].valid()) {
                            std::string transcript_mid = tasks[i].get();
                            //std::cerr << "mid task " << i << " " << transcript_mid << std::endl;
                            mid_tasks.emplace_back(std::make_pair(i, transcript_mid));
                        }
                    }
                global_searching = true;
                for (std::pair<int, std::string> &mid_task: mid_tasks) {
                    Board board_start_mid;
                    board_start_mid.reset();
                    Flip flip;
                    for (int i = 0; i < mid_task.second.size(); i += 2) {
                        int x = mid_task.second[i] - 'a';
                        int y = mid_task.second[i + 1] - '1';
                        int coord = HW2_M1 - (y * HW + x);
                        calc_flip(&flip, &board_start_mid, coord);
                        board_start_mid.move_board(&flip);
                        if (board_start_mid.get_legal() == 0) {
                            board_start_mid.pass();
                        }
                    }
                    //std::cerr << "additional " << mid_task.first << " " << mid_task.second << " " << board_start_mid.to_str() << std::endl;
                    std::string transcript = self_play_task(board_start_mid, mid_task.second, options, true, 0, SELF_PLAY_N_TRY);
                    //std::cerr << "additional got " << mid_task.first << " " << transcript << std::endl;
                    results[mid_task.first] = transcript;
                    --n_running_task;
                    for (int i = print_idx; i < results.size(); ++i) {
                        if (results[i] != "") {
                            std::cout << results[i] << std::endl;
                            ++print_idx;
                        } else {
                            break;
                        }
                    }
                }
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
    } else {
        int print_task_idx = 0;
        std::vector<std::future<std::string>> tasks;
        for (std::pair<std::string, Board> start_position: board_list) {
            bool go_to_next_task = false;
            while (!go_to_next_task) {
                if (thread_pool.get_n_idle() && tasks.size() < board_list.size()) {
                    bool pushed = false;
                    tasks.emplace_back(thread_pool.push(&pushed, std::bind(&self_play_task, start_position.second, "", options, false, 0, SELF_PLAY_N_TRY)));
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
    if (board.n_discs() >= to_n_discs) {
        std::cout << starting_board << " ";
        for (int &cell: transcript) {
            std::cout << idx_to_coord(cell);
        }
        //std::cout << " " << board.to_str() << std::endl;
        //Search_result accurate_search_result = ai(board, 28, true, 0, true, true);
        //int accurate_val = accurate_search_result.value;
        //std::cout << " " << accurate_val << std::endl;
        std::cout << std::endl;
        return;
    }
    if (board.is_end()) {
        std::cout << starting_board << " ";
        for (int &cell: transcript) {
            std::cout << idx_to_coord(cell);
        }
        //std::cout << " " << board.to_str() << " END" << std::endl;
        //std::cout << " " << board.score_player() << std::endl;
        std::cout << std::endl;
        return;
    }
    uint64_t legal = board.get_legal();
    if (legal == 0) {
        board.pass();
        legal = board.get_legal();
    }
    Flip flip;
    Search_result search_result = ai(board, options->level, true, 0, true, false);
    calc_flip(&flip, &board, search_result.policy);
    board.move_board(&flip);
    transcript.emplace_back(search_result.policy);
        self_play_lossless_lines_task(board, starting_board, options, to_n_discs, transcript);
    transcript.pop_back();
    board.undo_board(&flip);
    legal ^= 1ULL << search_result.policy;
    int best_score = search_result.value;
    //int best_move = search_result.policy;
    int alpha = best_score - 2; // accept best - 1
    int beta = best_score;
    while (legal) {
        search_result = ai_legal_window(board, alpha, beta, options->level, true, 0, true, false, legal);
        if (search_result.value <= alpha) {
            break;
        }
        //std::cerr << board.to_str() << " best " << idx_to_coord(best_move) << " " << best_score << " alt " << idx_to_coord(search_result.policy) << " " << search_result.value << " [" << alpha << "," << beta << "]" << std::endl;
        calc_flip(&flip, &board, search_result.policy);
        board.move_board(&flip);
        transcript.emplace_back(search_result.policy);
            self_play_lossless_lines_task(board, starting_board, options, to_n_discs, transcript);
        transcript.pop_back();
        board.undo_board(&flip);
        legal ^= 1ULL << search_result.policy;
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



Board get_random_board(int n_random_moves) {
    Board board;
    Flip flip;
    for (;;) {
        board.reset();
        for (int j = 0; j < n_random_moves && board.check_pass(); ++j) {
            uint64_t legal = board.get_legal();
            int random_idx = myrandrange(0, pop_count_ull(legal));
            int t = 0;
            for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)) {
                if (t == random_idx) {
                    calc_flip(&flip, &board, cell);
                    break;
                }
                ++t;
            }
            board.move_board(&flip);
        }
        if (board.check_pass()) {
            return board;
        }
    }
    return board; // error
}


void solve_random(std::vector<std::string> arg, Options *options, State *state) {
    int n_boards, n_random_moves;
    if (arg.size() < 2) {
        std::cerr << "[ERROR] [FATAL] please input arguments" << std::endl;
        std::exit(1);
    }
    std::string str_n_boards = arg[0];
    std::string str_n_random_moves = arg[1];
    try{
        n_boards = std::stoi(str_n_boards);
        n_random_moves = std::stoi(str_n_random_moves);
    } catch (const std::invalid_argument& e) {
        std::cout << str_n_boards << " " << str_n_random_moves << " invalid argument" << std::endl;
        std::exit(1);
    } catch (const std::out_of_range& e) {
        std::cout << str_n_boards << " " << str_n_random_moves << " out of range" << std::endl;
        std::exit(1);
    }
    std::cerr << n_boards << " boards with " << n_random_moves << " random moves" << std::endl;
    uint64_t strt = tim();
    if (thread_pool.size() == 0) {
        for (int i = 0; i < n_boards; ++i) {
            Board board = get_random_board(n_random_moves);
            Search_result result = ai(board, options->level, true, 0, false, options->show_log);
            std::cout << board.to_str().substr(0, 64) << " " << result.value << std::endl;
        }
    } else {
        int n_boards_done = 0;
        std::vector<std::pair<Board, std::future<Search_result>>> tasks;
        while (n_boards_done < n_boards) {
            if (thread_pool.get_n_idle() && (int)tasks.size() < n_boards) {
                bool pushed = false;
                Board board = get_random_board(n_random_moves);
                tasks.emplace_back(std::make_pair(board, thread_pool.push(&pushed, std::bind(&ai, board, options->level, true, 0, false, options->show_log))));
                if (!pushed) {
                    tasks.pop_back();
                }
            }
            for (std::pair<Board, std::future<Search_result>> &task: tasks) {
                if (task.second.valid()) {
                    if (task.second.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                        Search_result result = task.second.get();
                        std::cout << task.first.to_str().substr(0, 64) << " " << result.value << std::endl;
                        ++n_boards_done;
                    }
                }
            }
            if (0 < n_boards - n_boards_done && n_boards - n_boards_done < thread_pool.size()) {
                std::vector<Board> boards_mid;
                global_searching = false;
                    for (std::pair<Board, std::future<Search_result>> &task: tasks) {
                        if (task.second.valid()) {
                            task.second.get();
                            boards_mid.emplace_back(task.first);
                        }
                    }
                global_searching = true;
                for (Board &board: boards_mid) {
                    Search_result result = ai(board, options->level, true, 0, true, options->show_log);
                    std::cout << board.to_str().substr(0, 64) << " " << result.value << std::endl;
                    ++n_boards_done;
                }
            }
        }
    }
    global_searching = false;
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
    } else {
        res = perft_no_pass_count(&board, depth, false);
    }
    std::cout << "perft mode " << mode << " depth " << depth << " " << res << " leaves found in " << tim() - strt << " ms" << std::endl;
}

void minimax_commandline(std::vector<std::string> arg) {
    if (arg.size() < 1) {
        std::cerr << "please input <depth>" << std::endl;
        std::exit(1);
    }
    int depth;
    std::string str_depth = arg[0];
    try{
        depth = std::stoi(str_depth);
    } catch (const std::invalid_argument& e) {
        std::cout << str_depth << " invalid argument" << std::endl;
        std::exit(1);
    } catch (const std::out_of_range& e) {
        std::cout << str_depth << " out of range" << std::endl;
        std::exit(1);
    }
    if (depth <= 0 || 60 < depth) {
        std::cout << "depth must be in [1, 60], got " << depth << std::endl;
        std::exit(1);
    }
    Board board;
    board.reset();
    uint64_t strt = tim();
    int res = minimax(&board, depth);
    std::cout << "minimax depth " << depth << " value " << res << " in " << tim() - strt << " ms" << std::endl;
}