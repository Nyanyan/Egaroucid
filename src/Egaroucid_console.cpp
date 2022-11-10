/*
	Egaroucid Project

	@file Egaroucid_console.cpp
		Main file for Console application
	@date 2021-2022
	@author Takuto Yamana (a.k.a. Nyanyan)
	@license GPL-3.0 license
*/

#include <iostream>
#include "engine/engine_all.hpp"
#include "console/console_all.hpp"

void init_console(Options options){
    thread_pool.resize(options.n_threads);
    bit_init();
    mobility_init();
    hash_resize(DEFAULT_HASH_LEVEL, options.hash_level, options.show_log);
    stability_init();
    if (!evaluate_init(options.eval_file, options.show_log))
        std::exit(0);
    book_init(options.book_file, options.show_log);
    if (options.show_log)
        std::cerr << "initialized" << std::endl;
}

int main(int argc, char* argv[]) {
    std::vector<Commandline_option> commandline_options = get_commandline_options(argc, argv);
    print_special_commandline_options(commandline_options);
    Options options = get_options(commandline_options);
    init_console(options);

    Board_info board;
    board.reset();
    while (true) {
        check_command(&board, options);
    }

    return 0;
}
