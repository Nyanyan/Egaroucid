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

constexpr int THREAD_SIZE = 0;

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

uint64_t shift(int cell, int n_shift) {
    if (cell + n_shift < 0) {
        return 1ULL << (cell + n_shift + 64);
    } else if (cell + n_shift >= 64) {
        return 1ULL << (cell + n_shift - 64);
    }
    return 1ULL << (cell + n_shift);
}

int main(int argc, char *argv[]){
    if (argc < 3){
        std::cerr << "input [n_discs] [n_problems]" << std::endl;
        return 1;
    }

    thread_pool.resize(THREAD_SIZE);
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

    std::vector<int> cells_always = {27, 28, 35, 36};
    std::vector<int> cells;
    for (int i = 0; i < HW2; ++i) {
        if (std::find(cells_always.begin(), cells_always.end(), i) == cells_always.end()) {
            cells.emplace_back(i);
        }
    }
    std::random_device seed_gen;
    std::mt19937 engine(seed_gen());
    for (int i = 0; i < n_problems; ++i) {
        board.player = 0;
        board.opponent = 0;
        int n_shift = myrandrange(-27, 28);
        std::shuffle(cells.begin(), cells.end(), engine);
        for (int cell: cells_always) {
            if (myrandom() < 0.5) {
                board.player |= shift(cell, n_shift);
            } else {
                board.opponent |= shift(cell, n_shift);
            }
        }
        for (int j = 0; j < n_discs - (int)cells_always.size(); ++j) {
            if (myrandom() < 0.5) {
                board.player |= shift(cells[j], n_shift);
            } else {
                board.opponent |= shift(cells[j], n_shift);
            }
        }
        if (board.n_discs() != n_discs) {
            std::cerr << "ERR" << std::endl;
            std::cerr << n_shift << " " << cells[0] << std::endl;
            board.print();
            return 1;
        }
        bool result_used = false;
        if (board.get_legal()) {
            bool searching = true;
            uint64_t n_nodes = 0;
            int clog_size = first_clog_search(board, &n_nodes, 6, board.get_legal(), &searching).size();
            if (clog_size == 0) {
                //Search_result search_result = ai(board, 21, false, 0, true, false);
                //if (std::abs(search_result.value) <= 10) {
                std::cout << get_str(board, turn_color) << std::endl;
                result_used = true;
                //}
            }
        }
        if (!result_used) {
            --i;
        }
    }
    return 0;
}