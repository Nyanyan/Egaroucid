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

#define SELFPLAY_TT_DATE_MARGIN 5

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
        print_search_result_body(res, options->level);
        transposition_table.init();
        total.nodes += res.nodes;
        total.time += res.time;
    }
    std::cout << "total " << total.nodes << " nodes in " << ((double)total.time / 1000) << "s NPS " << (total.nodes * 1000 / total.time) << std::endl;
}

#if TUNE_MOVE_ORDERING
    uint64_t n_nodes_test(Options *options, std::vector<Board> testcase_arr){
        uint64_t n_nodes = 0;
        for (Board &board: testcase_arr){
            int depth;
            bool is_mid_search;
            uint_fast8_t mpc_level;
            get_level(options->level, board.n_discs() - 4, &is_mid_search, &depth, &mpc_level);
            Search search;
            search.init_board(&board);
            calc_features(&search);
            search.n_nodes = 0ULL;
            search.use_multi_thread = true;
            search.mpc_level = mpc_level;
            std::vector<Clog_result> clogs;
            transposition_table.init();
            //board.print();
            std::pair<int, int> result = first_nega_scout(&search, -SCORE_MAX, SCORE_MAX, SCORE_UNDEFINED, depth, !is_mid_search, false, clogs, tim());
            //std::cerr << result.first << " " << result.second << std::endl;
            n_nodes += search.n_nodes;
        }
        return n_nodes;
    }

    void tune_move_ordering(std::string file, Options *options){
        std::ifstream ifs(file);
        if (ifs.fail()){
            std::cerr << "[ERROR] [FATAL] problem file " << file << " not found" << std::endl;
            return;
        }
        std::vector<Board> testcase_arr;
        std::string line;
        Board_info board_info;
        while (std::getline(ifs, line)){
            setboard(&board_info, line);
            testcase_arr.emplace_back(board_info.board);
        }
        uint64_t min_n_nodes = n_nodes_test(options, testcase_arr);
        double min_percentage = 100.0;
        uint64_t first_n_nodes = min_n_nodes;
        std::cerr << "min_n_nodes " << min_n_nodes << std::endl;
        int n_updated = 0;
        int n_try = 0;
        uint64_t tl = 10ULL * 60ULL * 1000ULL; // 10 min
        uint64_t strt = tim();
        while (tim() - strt < tl){
            int idx = myrandrange(0, N_MOVE_ORDERING_PARAM);
            int delta = myrandrange(-2, 3);
            while (delta == 0)
                delta = myrandrange(-2, 3);
            move_ordering_param_array[idx] += delta;
            uint64_t n_nodes = n_nodes_test(options, testcase_arr);
            double percentage = 100.0 * n_nodes / first_n_nodes;

            // simulated annealing
            constexpr double start_temp = 1.0;
            constexpr double end_temp = 0.001;
            double temp = start_temp + (end_temp - start_temp) * (tim() - strt) / tl;
            double prob = exp((min_percentage - percentage) / temp);
            if (prob > myrandom()){
                min_n_nodes = n_nodes;
                min_percentage = percentage;
                ++n_updated;
            } else{
                move_ordering_param_array[idx] -= delta;
            }
            /*
            // hillclimb
            if (n_nodes <= min_n_nodes){
                min_n_nodes = n_nodes;
                min_percentage = percentage;
                ++n_updated;
            } else{
                move_ordering_param_array[idx] -= delta;
            }
            */
            ++n_try;
            std::cerr << "try " << n_try << " updated " << n_updated << " min_n_nodes " << min_n_nodes << " n_nodes " << n_nodes << " " << min_percentage << "% " << tim() - strt << " ms ";
            for (int i = 0; i < N_MOVE_ORDERING_PARAM; ++i){
                std::cerr << " " << move_ordering_param_array[i];
            }
            std::cerr << std::endl;
        }
    }
#endif

#if TUNE_PROBCUT_MID
    void tune_probcut_mid(){
        std::ofstream ofs("probcut_mid.txt");
        Board board;
        Flip flip;
        Search_result short_ans, long_ans;
        for (int i = 0; i < 1000; ++i){
            for (int depth = 2; depth < 14; ++depth){
                for (int n_discs = 4; n_discs < HW2 - depth - 5; ++n_discs){
                    board.reset();
                    for (int j = 4; j < n_discs && board.check_pass(); ++j){ // random move
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
                        board.move_board(&flip);
                    }
                    if (board.check_pass()){
                        int short_depth = myrandrange(1, depth - 1);
                        short_depth &= 0xfffffffe;
                        short_depth |= depth & 1;
                        //int short_depth = mpc_search_depth_arr[0][depth];
                        if (short_depth == 0){
                            short_ans.value = mid_evaluate(&board);
                        } else{
                            short_ans = tree_search(board, short_depth, MPC_100_LEVEL, false, true);
                        }
                        long_ans = tree_search(board, depth, MPC_100_LEVEL, false, true);
                        // n_discs short_depth long_depth error
                        std::cerr << i << " " << n_discs << " " << short_depth << " " << depth << " " << long_ans.value - short_ans.value << std::endl;
                        ofs << n_discs << " " << short_depth << " " << depth << " " << long_ans.value - short_ans.value << std::endl;
                    }
                }
            }
        }
    }
#endif

#if TUNE_PROBCUT_END
    void tune_probcut_end(){
        std::ofstream ofs("probcut_end.txt");
        Board board;
        Flip flip;
        Search_result short_ans, long_ans;
        for (int i = 0; i < 1000; ++i){
            for (int depth = 6; depth < 24; ++depth){
                board.reset();
                for (int j = 0; j < HW2 - 4 - depth && board.check_pass(); ++j){ // random move
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
                    board.move_board(&flip);
                }
                if (board.check_pass()){
                    int short_depth = myrandrange(1, std::min(15, depth - 1));
                    short_depth &= 0xfffffffe;
                    short_depth |= depth & 1;
                    //int short_depth = mpc_search_depth_arr[1][depth];
                    if (short_depth == 0){
                        short_ans.value = mid_evaluate(&board);
                    } else{
                        short_ans = tree_search(board, short_depth, MPC_100_LEVEL, false, true);
                    }
                    long_ans = tree_search(board, depth, MPC_100_LEVEL, false, true);
                    // n_discs short_depth error
                    std::cerr << i << " " << HW2 - depth << " " << short_depth << " " << long_ans.value - short_ans.value << std::endl;
                    ofs << HW2 - depth << " " << short_depth << " " << long_ans.value - short_ans.value << std::endl;
                }
            }
        }
    }
#endif

void execute_special_tasks(Options options){
    // move ordering tuning (endsearch)
    #if TUNE_MOVE_ORDERING
        std::cout << "tune move ordering, please input testcase file" << std::endl;
        std::string file;
        std::cin >> file;
        tune_move_ordering(file, &options);
        std::exit(0);
    #endif

    // probcut (midsearch)
    #if TUNE_PROBCUT_MID
        std::cout << "tune probcut (midsearch)" << std::endl;
        tune_probcut_mid();
        std::exit(0);
    #endif

    // probcut (endsearch)
    #if TUNE_PROBCUT_END
        std::cout << "tune probcut (endsearch)" << std::endl;
        tune_probcut_end();
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


std::string self_play_task(Options *options, bool use_multi_thread, int n_random_moves){
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
        while (use_multi_thread && transposition_table.get_date() >= MAX_DATE - thread_pool.size() - SELFPLAY_TT_DATE_MARGIN);
        result = ai(board, options->level, true, 0, false, options->show_log); // search in single thread
        calc_flip(&flip, &board, result.policy);
        res += idx_to_coord(flip.pos);
        board.move_board(&flip);
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
            std::string transcript = self_play_task(options, false, n_random_moves);
            std::cout << transcript << std::endl;
        }
    } else{
        int n_games_done = 0;
        std::vector<std::future<std::string>> tasks;
        while (n_games_done < n_games){
            if (transposition_table.get_date() >= MAX_DATE - thread_pool.size() - SELFPLAY_TT_DATE_MARGIN){
                //transposition_table.reset_date_new_thread(thread_pool.size());
                transposition_table.reset_date();
            }
            if (thread_pool.get_n_idle() && (int)tasks.size() < n_games){
                bool pushed = false;
                tasks.emplace_back(thread_pool.push(&pushed, std::bind(&self_play_task, options, true, n_random_moves)));
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
