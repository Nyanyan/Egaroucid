#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <unordered_set>
#include <random>
#include <algorithm>
#include <ios>
#include <iomanip>
#include <numeric>
#include <iterator>
#include "./../../engine/ai.hpp"

std::string get_str(Board board, int turn_color) {
    std::string res;
    for (int cell = HW2 - 1; cell >= 0; --cell) {
        uint64_t cell_bit = 1ULL << cell;
        if (turn_color == BLACK) {
            if (board.player & cell_bit) {
                res += "X";
            } else if (board.opponent & cell_bit) {
                res += "O";
            } else {
                res += "-";
            }
        } else {
            if (board.player & cell_bit) {
                res += "O";
            } else if (board.opponent & cell_bit) {
                res += "X";
            } else {
                res += "-";
            }
        }
    }
    res += " ";
    if (turn_color == BLACK) {
        res += "X";
    } else {
        res += "O";
    }
    return res;
}

int main(int argc, char *argv[]){
    if (argc < 3){
        std::cerr << "input [n_discs] [n_problems]" << std::endl;
        return 1;
    }

    bit_init();
    mobility_init();
    flip_init();
    book_hash_init_rand();
    last_flip_init();
    endsearch_init();
    mpc_init();
    move_ordering_init();
    stability_init();
    evaluate_init("./../../../bin/resources/eval.egev2", "./../../../bin/resources/eval_move_ordering_end.egev", false);

    int n_discs = atoi(argv[1]);
    int n_problems = atoi(argv[2]);

    if (n_discs < 4) {
        n_discs = 4;
    }

    int turn_color = n_discs % 2; // even: BLACK odd: WHITE

    Board board;

    std::vector<int> cells;
    for (int i = 0; i < HW2; ++i) {
        cells.emplace_back(i);
    }
    std::random_device seed_gen;
    std::mt19937 engine(seed_gen());
    for (int i = 0; i < n_problems; ++i) {
        board.player = 0;
        board.opponent = 0;
        std::shuffle(cells.begin(), cells.end(), engine);
        for (int j = 0; j < n_discs; ++j) {
            if (myrandom() < 0.5) {
                board.player |= 1ULL << cells[j];
            } else {
                board.opponent |= 1ULL << cells[j];
            }
        }
        bool result_used = false;
        Search search(&board, MPC_100_LEVEL, true, false);
        bool searching = true;
        int clog_result = clog_search(&search, 10, &searching);
        if (clog_result == CLOG_NOT_FOUND) {
            //Search_result search_result = ai(board, 21, false, 0, true, false);
            //if (std::abs(search_result.value) <= 10) {
            std::cout << get_str(board, turn_color) << std::endl;
            result_used = true;
            //}
        }
        if (!result_used) {
            --i;
        }
    }
    return 0;
}