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
#include <algorithm>
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

void search_lines(Board &board, int player, int depth, int black_score_min, int black_score_max, std::vector<int> &line, int last_move_player, std::vector<int> &last_move_cells, int level = 21) {
    if (depth <= 0) {
        return;
    }

    Board root = board.copy();
    int root_player = player;
    uint64_t legal = root.get_legal();
    if (legal == 0ULL) {
        root.pass();
        root_player ^= 1;
        legal = root.get_legal();
        if (legal == 0ULL) {
            return;
        }
    }

    for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)) {
        Flip flip;
        calc_flip(&flip, &root, cell);
        Board next_board = root.move_copy(&flip);
        int next_player = root_player ^ 1;

        int value = ai(next_board, level, false, 0, false, false).value;
        int black_value = (next_player == BLACK) ? value : -value;
        if (black_value < black_score_min || black_score_max < black_value) {
            continue;
        }

        line.emplace_back(cell);
        if (root_player == last_move_player && std::find(last_move_cells.begin(), last_move_cells.end(), cell) != last_move_cells.end()) {
            std::string transcript;
            for (const int &coord: line) {
                transcript += idx_to_coord(coord);
            }
            std::cout << transcript << std::endl;
        } else {
            search_lines(next_board, next_player, depth - 1, black_score_min, black_score_max, line, last_move_player, last_move_cells, level);
        }
        line.pop_back();
    }
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
    int search_level = 21;
    int black_score_min = -6;
    int black_score_max = 0;
    int last_move_player = BLACK;
    std::vector<int> last_move_cells = {get_coord_from_chars('b', '2'), get_coord_from_chars('b', '7'), get_coord_from_chars('g', '2'), get_coord_from_chars('g', '7')};

    std::vector<int> initial_line_vec;
    for (int i = 0; i < initial_line.size(); i += 2) {
        initial_line_vec.emplace_back(get_coord_from_chars(initial_line[i], initial_line[i + 1]));
    }
    Board search_board;
    search_board.reset();
    int player = BLACK;
    Flip flip;
    for (const int &coord: initial_line_vec) {
        uint64_t legal = search_board.get_legal();
        if (legal == 0ULL) {
            search_board.pass();
            player ^= 1;
            legal = search_board.get_legal();
            if (legal == 0ULL) {
                std::cerr << "[ERROR] game over before initial line is fully applied" << std::endl;
                std::exit(1);
            }
        }
        if ((legal & (1ULL << coord)) == 0ULL) {
            std::cerr << "[ERROR] illegal move in initial line: " << idx_to_coord(coord) << std::endl;
            std::exit(1);
        }
        calc_flip(&flip, &search_board, coord);
        search_board.move_board(&flip);
        player ^= 1;
    }

    int n_initial_moves = initial_line_vec.size();
    search_lines(search_board, player, n_max_moves - n_initial_moves, black_score_min, black_score_max, initial_line_vec, last_move_player, last_move_cells, search_level);


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