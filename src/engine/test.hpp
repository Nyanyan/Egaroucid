/*
    Egaroucid Project

    @file test.hpp
        Several tests
    @date 2021-2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#include "ai.hpp"
#include "fstream"


void endgame_accuracy_test() {
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
    //ifs >> line;
    //const uint64_t n_data = stoi(line);
    uint64_t n_error = 0;
    uint64_t n = 0;
    while (getline(ifs, line)) {
        if ((n & 0b111111) == 0b111111) {
            std::cerr << '\r' << n;
        }
        Board board;
        //if (input_board_base81(line, &board)) {
        //    std::cerr << "input file format error at idx " << i << " " << line << std::endl;
        //    return;
        //}
        if (!board.from_str(line)) {
            std::cerr << "input file format error at idx " << n << " " << line << std::endl;
            return;
        }
        int v = ai(board, 60, false, 0, true, false).value;
        int v_ans;
        ans >> v_ans;
        if (v != v_ans) {
            ++n_error;
            std::cerr << "\r[ERROR] endgame value wrong idx " << n << " " << line << " got " << v << " expected " << v_ans << std::endl;
        }
        ++n;
    }
    std::cerr << std::endl;
    std::cerr << n_error << " endgame error found in " << n << " boards" << std::endl;
}