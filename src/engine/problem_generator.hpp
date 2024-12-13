#include "ai.hpp"
#include <fstream>

void problem_generator(int n_empties, int n_problems, int level) {
    std::string problem_file = std::to_string(n_empties) + "_" + std::to_string(n_problems) + "_problem.txt";
    std::string answer_file = std::to_string(n_empties) + "_" + std::to_string(n_problems) + "_answer.txt";
    std::string move_file = std::to_string(n_empties) + "_" + std::to_string(n_problems) + "_move.txt";
    std::ofstream prob_ofs(problem_file);
    std::ofstream ans_ofs(answer_file);
    std::ofstream move_ofs(move_file);
    Board board;
    Flip flip;
    for (int i = 0; i < n_problems; ++i) {
        if ((i + 1) % 100 == 0)
            std::cerr << '\r' << i;
        board.reset();
        for (int j = 0; j < 60 - n_empties; ++j) {
            uint64_t legal = board.get_legal();
            if (legal == 0) {
                board.pass();
                legal = board.get_legal();
                if (legal == 0) {
                    break;
                }
            }
            std::vector<uint_fast8_t> legals;
            for (uint_fast8_t c = first_bit(&legal); legal; c = next_bit(&legal)) {
                legals.emplace_back(c);
            }
            uint_fast8_t cell = legals[myrandrange(0, (int)legals.size())];
            calc_flip(&flip, &board, cell);
            board.move_board(&flip);
        }
        if (board.is_end()) {
            --i;
        } else{
            if (board.get_legal() == 0) {
                board.pass();
            }
            for (int i = 0; i < HW2; ++i) {
                if (1 & (board.player >> i)) {
                    prob_ofs << "X";
                } else if (1 & (board.opponent >> i)) {
                    prob_ofs << "O";
                } else{
                    prob_ofs << "-";
                }
            }
            prob_ofs << "X" << std::endl;
            Search_result res = ai(board, level, false, 0, true, false);
            ans_ofs << res.value << std::endl;
            uint64_t legal = board.get_legal();
            for (int i = 0; i < HW2; ++i) {
                if (i == res.policy) {
                    move_ofs << "B";
                } else if (1 & (legal >> i)) {
                    move_ofs << "O";
                } else{
                    move_ofs << "-";
                }
            }
            move_ofs << std::endl;
        }
    }
    std::cerr << std::endl;
}