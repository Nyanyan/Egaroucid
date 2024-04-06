/*
    Egaroucid Project

    @file test.hpp
        Several tests
    @date 2021-2024
    @author Takuto Yamana
    @license GPL-3.0 license
*/

#include "ai.hpp"
#include "fstream"


void endgame_accuracy_test(){
    std::string problem_file, answer_file;
    std::cerr << "endgame accuracy test" << std::endl;
    std::cerr << "input [problem_file] [answer_file]" << std::endl;
    std::cin >> problem_file >> answer_file;
    std::ifstream ifs(problem_file);
    if (!ifs) {
        std::cerr << "input file not found" << std::endl;
        return;
    }
    std::ifstream ans(answer_file);
    if (!ans) {
        std::cerr << "answer file not found" << std::endl;
        return;
    }
    std::string line;
    ifs >> line;
    const uint64_t n_data = stoi(line);
    uint64_t n_error = 0;
    for (uint64_t i = 0; i < n_data; ++i) {
        if ((i & 0xff) == 0xff){
            std::cerr << '\r' << i;
        }
        ifs >> line;
        Board board;
        if (input_board_base81(line, &board)) {
            std::cerr << "input file format error at idx " << i << " " << line << std::endl;
            return;
        }
        int v = ai(board, 60, false, 0, true, false).value;
        int v_ans;
        ans >> v_ans;
        if (v != v_ans){
            ++n_error;
            std::cerr << "[ERROR] endgame value wrong idx " << i << " got " << v << " expected " << v_ans << std::endl;
        }
    }
    std::cerr << n_error << " endgame error found" << std::endl;
}
