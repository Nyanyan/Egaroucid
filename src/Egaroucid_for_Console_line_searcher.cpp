/*
	Egaroucid Project

	@file Egaroucid_for_Console.cpp
		Main file for Console application
	@date 2021-2026
	@author Takuto Yamana
	@license GPL-3.0-or-later
*/
#include <iostream>
#include <fstream>
#include "engine/engine_all.hpp"
#include "console/console_all.hpp"


void init_console(Options options, std::string binary_path) {
    int thread_size = std::max(0, options.n_threads - 1);
    thread_pool.resize(thread_size);
    if (options.show_log)
        std::cerr << "thread size = " << thread_size + 1 << std::endl;
    bit_init();
    mobility_init();
    flip_init();
    last_flip_init();
    endsearch_init();
    #if USE_MPC_PRE_CALCULATION
        mpc_init();
    #endif
    move_ordering_init();
    #if USE_CHANGEABLE_HASH_LEVEL
        hash_resize(DEFAULT_HASH_LEVEL, options.hash_level, options.binary_path, options.show_log);
    #else
        hash_tt_init(options.binary_path, options.show_log);
    #endif
    stability_init();
    std::string mo_end_file = binary_path + "resources/eval_move_ordering_end.egev"; // filename fixed
    //std::string mo_mid_file = binary_path + "resources/eval_move_ordering_mid.egev"; // filename fixed
    if (!evaluate_init(options.eval_file, mo_end_file, options.show_log))
        std::exit(0);
    if (!options.nobook)
        book_init(options.book_file, options.show_log);
    if (options.show_log)
        std::cerr << "initialized" << std::endl;
}

void search_lines(Board &board, int depth, int player, int black_score_min, int black_score_max, std::vector<int> &line, int last_move_player, std::vector<int> &last_move_cells) {
    // TBD:
    // 0. depth <= 0ならreturn
    // 1. boardから打てる手を列挙し、全部打ってみる
    //   a. 打てる手がなかったらパスし、player ^= 1
    // 2. 各局面での評価値が黒目線で見た時[black_score_min, black_score_max]に収まるかを判定し、収まる手だけを残す
    // 3. 各条件に合致した局面に対して、以下を行う:
    //   a. line.emplace_back(その手)
    //   b. もし、条件に合致した手がlast_move_cellsの中に存在し、player == last_move_playerであれば、lineを棋譜形式にしたものをstd::coutする
    //   c. そうでなければ、search_lines(その手を打ったあとのボード、depth - 1, hogehoge...)
}

int main(int argc, char* argv[]) {
    State state;
    std::string binary_path = get_binary_path();
    std::vector<Commandline_option> commandline_options = get_commandline_options(argc, argv);
    Options options = get_options(commandline_options, binary_path);
    std::ofstream ofs;
    if (options.log_to_file) {
        ofs.open(options.log_file);
        if (!ofs) {
            std::cout << "[ERROR] can't open log file " << options.log_file << std::endl;
            std::exit(1);
        }
        std::cerr.rdbuf(ofs.rdbuf());
    }
    print_special_commandline_options(commandline_options);
    init_console(options, binary_path);
    execute_special_tasks(options); // tuning etc.
    execute_special_commandline_tasks(commandline_options, &options, &state); // solve problems etc.
    Board_info board;
    init_board(&board, &options, &state);

    // while (true) {
    //     Board b;
    //     b.player = myrand_ull();
    //     b.opponent = ~b.player;
    //     uint_fast8_t empty_cell = myrand_uint() & 63;
    //     b.player &= ~(1ULL << empty_cell);
    //     b.opponent &= ~(1ULL << empty_cell);
    //     Flip flip;
    //     calc_flip(&flip, &b, empty_cell);
    //     int n_flipped_correct = pop_count_ull(flip.flip);
    //     int n_flipped_last = count_last_flip(b.player, empty_cell);
    //     std::cerr << n_flipped_correct << " " << n_flipped_last << std::endl;
    //     b.print();
    //     if (n_flipped_correct != n_flipped_last) {
    //         std::cerr << "ERROR" << std::endl;
    //         std::cerr << n_flipped_correct << " " << n_flipped_last << std::endl;
    //         b.print();
    //     }
    // }

    
    std::string initial_line = "f5d6";
    int n_max_moves = 10;
    int black_score_min = -6;
    int black_score_max = 0;
    int last_move_player = BLACK;
    std::vector<int> last_move_cells = {get_coord_from_chars('b', '2'), get_coord_from_chars('b', '7'), get_coord_from_chars('g', '2'), get_coord_from_chars('g', '7')};

    std::vector<int> initial_line_vec;
    for (int i = 0; i < initial_line.size(); i += 2) {
        initial_line_vec.emplace_back(get_coord_from_chars(initial_line[i], initial_line[i + 1]));
    }
    Board board;
    board.reset();
    int player = BLACK;
    // TBD: initial lineを打ったあとの局面を生成する
    // 打つごとに(パスがあればパスでも)player ^= 1する。
    int n_initial_moves = initial_line_vec.size();
    search_lines(board, player, n_max_moves - n_initial_moves, black_score_min, black_score_max, initial_line_vec, last_move_player, last_move_cells);


    while (true) {
        if (options.gtp) {
            if (options.ponder) {
                state.ponder_searching = true;
                state.ponder_future = std::async(std::launch::async, ai_ponder, board.board, options.show_log, THREAD_ID_NONE, &state.ponder_searching);
            }
            gtp_check_command(&board, &state, &options);
        }else {
            if (!options.quiet && !options.noboard) {
                print_board_info(&board, &state, &options);
                std::cout << std::endl;
                //std::cerr << "val " << mid_evaluate(&board.board) << std::endl;
            }
            if (!execute_special_tasks_loop(&board, &state, &options)) {
                if (options.ponder) {
                    if (board.board.n_discs() > 4) {
                        state.ponder_searching = true;
                        state.ponder_future = std::async(std::launch::async, ai_ponder, board.board, options.show_log, THREAD_ID_NONE, &state.ponder_searching);
                    } //else {
                    //    transposition_table.reset_importance();
                    //}
                }
                check_command(&board, &state, &options);
            }
        }
    }
    return 0;
}