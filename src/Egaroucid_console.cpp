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

// ffotest
int main(int argc, char* argv[]) {
    print_version();
    std::vector<Commandline_option> commandline_options = get_commandline_options(argc, argv);
    if (find_commandline_option(commandline_options, ID_HELP) == OPTION_FOUND){
        print_help();
        return 0;
    }
    int n_threads = 10;
    bool show_log = true;
    int hash_level = 24;
    if (argc >= 2) {
        n_threads = atoi(argv[1]);
    }
    if (argc >= 3) {
        show_log = atoi(argv[2]);
    }
    if (argc >= 4) {
        hash_level = atoi(argv[3]);
    }
    thread_pool.resize(n_threads);
    bit_init();
    mobility_init();
    hash_resize(0, hash_level);
    evaluate_init("resources/eval.egev");
    stability_init();
    Board board;
    Search_result res;
    while (true) {
        board = input_board();
        //board.reset();
        if (show_log)
            board.print();
        res = ai(board, 60, false, true, show_log);
        std::cout << "depth " << res.depth << " value " << res.value << " policy " << idx_to_coord(res.policy) << " nodes " << res.nodes << " time " << res.time << " nps " << res.nps << std::endl;
    }

    return 0;
}

/*
// general purpose
int main(int argc, char* argv[]) {
    int n_threads = 8;
    bool show_log = true;
    int hash_level = 24;
    int level = 11;
    if (argc >= 2) {
        level = atoi(argv[1]);
    }
    thread_pool.resize(n_threads);
    bit_init();
    mobility_init();
    hash_resize(0, hash_level);
    evaluate_init("resources/eval.egev");
    stability_init();
    Board board;
    Search_result res;
    while (true) {
        board = input_board();
        //board.reset();
        if (show_log)
            board.print();
        res = ai(board, level, false, true, show_log);
        std::cout << res.value << " " << idx_to_coord(res.policy) << endl;
        //std::cout << "depth " << res.depth << " value " << res.value << " policy " << idx_to_coord(res.policy) << " nodes " << res.nodes << " time " << res.time << " nps " << res.nps << endl;
    }

    return 0;
}
*/
