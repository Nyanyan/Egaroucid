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

int main(int argc, char* argv[]) {
    std::vector<Commandline_option> commandline_options = get_commandline_options(argc, argv);
    special_commandline_options(commandline_options);
    Options options = get_options(commandline_options);

    
    thread_pool.resize(options.n_threads);
    bit_init();
    mobility_init();
    hash_resize(DEFAULT_HASH_LEVEL, options.hash_level);
    evaluate_init(options.eval_file);
    stability_init();
    Board board;
    Search_result res;
    while (true) {
        board = input_board();
        //board.reset();
        res = ai(board, options.level, false, true, options.show_log);
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
