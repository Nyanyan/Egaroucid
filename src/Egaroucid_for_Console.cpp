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
    disable_power_throttling();
    check_thread_stack_reserve(options.show_log);
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
#ifdef EGAROUCID_EVAL_NNUE_PATTERNS
    // the endgame move ordering table belongs to the pattern set: it is read beside the main evaluation file
    const size_t eval_dir_end = options.eval_file.find_last_of("/\\");
    std::string mo_end_file = (eval_dir_end == std::string::npos ? std::string() : options.eval_file.substr(0, eval_dir_end + 1)) + "eval_move_ordering_end.egev";
#else
    std::string mo_end_file = binary_path + "resources/eval_move_ordering_end.egev"; // filename fixed
#endif
#ifdef EGAROUCID_EVAL_NNUE_PATTERNS
    // the NNUE pattern set has no Offset/Decay models: those situations use the main evaluation
    eval_move_ordering_auto_requested = false;
#else
    eval_move_ordering_auto_requested = !options.eval_move_ordering_auto_off && options.eval_move_ordering_mid_file.empty();
#endif
    if (!evaluate_init(options.eval_file, mo_end_file, options.show_log))
        std::exit(1);
    if (!options.eval_move_ordering_mid_file.empty() &&
        !load_eval_move_ordering_mid_file(options.eval_move_ordering_mid_file.c_str(), options.show_log)) {
        std::exit(1);
    }
    eval_move_ordering_mid_shallow_max_depth = options.eval_move_ordering_shallow_max_depth;
    eval_move_ordering_mid_shallow_share_tt = options.eval_move_ordering_shallow_share_tt;
    eval_move_ordering_mid_shallow_end_only = options.eval_move_ordering_shallow_end_only;
    if (!options.eval_move_ordering_aux_file.empty() &&
        options.eval_move_ordering_aux_file != options.eval_move_ordering_mid_file &&
        !load_eval_move_ordering_aux_file(options.eval_move_ordering_aux_file.c_str(), options.show_log)) {
        std::exit(1);
    }
#if !USE_DIM0_ONLY_EVALUATION
    eval_fm_use_dim0_mpc_search = options.fm_mpc_dim0;
#endif
    if (!options.nobook)
        book_init(options.book_file, options.show_log);
    if (options.show_log)
        std::cerr << "initialized" << std::endl;
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
    if (options.random_seed_specified) {
        raw_myrandom.seed(options.random_seed);
    }
    if (options.show_log) {
        if (options.random_seed_specified) {
            std::cerr << "random seed = " << options.random_seed << std::endl;
        } else {
            std::cerr << "random seed = unspecified" << std::endl;
        }
        #if IS_GGS_TOURNAMENT
            std::cerr << "ggs tournament build = true" << std::endl;
        #else
            std::cerr << "ggs tournament build = false" << std::endl;
        #endif
        #if IS_GGS_3MIN
            std::cerr << "ggs 3min build = true" << std::endl;
        #else
            std::cerr << "ggs 3min build = false" << std::endl;
        #endif
        print_build_info(std::cerr);
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
