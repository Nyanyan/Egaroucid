/*
	Egaroucid Project

	@file Egaroucid_console.cpp
		Main file for Console application
	@date 2021-2024
	@author Takuto Yamana
	@license GPL-3.0 license
*/

#include <iostream>
#include "engine/engine_all.hpp"
#include "console/console_all.hpp"

void init_console(Options options, std::string binary_path){
    thread_pool.resize(std::max(0, options.n_threads - 1));
    bit_init();
    mobility_init();
    flip_init();
    last_flip_init();
    endsearch_init();
    #if USE_MPC_PRE_CALCULATION
        mpc_init();
    #endif
    hash_resize(DEFAULT_HASH_LEVEL, options.hash_level, options.binary_path, options.show_log);
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

int main(int argc, char* argv[]){
    State state;
    std::string binary_path = get_binary_path();
    std::vector<Commandline_option> commandline_options = get_commandline_options(argc, argv);
    Options options = get_options(commandline_options, binary_path);
    print_special_commandline_options(commandline_options);
    init_console(options, binary_path);
    execute_special_tasks(options);
    execute_special_commandline_tasks(commandline_options, &options, &state);
    Board_info board;
    board.reset();
    while (true){
        if (options.gtp){
            gtp_check_command(&board, &state, &options);
        }else {
            if (!options.quiet){
                print_board_info(&board);
                std::cout << std::endl;
                //std::cerr << "val " << mid_evaluate(&board.board) << std::endl;
            }
            if (!execute_special_tasks_loop(&board, &state, &options))
                check_command(&board, &state, &options);
        }
    }
    return 0;
}