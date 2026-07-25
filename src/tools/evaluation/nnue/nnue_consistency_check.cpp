/*
    Egaroucid Project

    @file nnue_consistency_check.cpp
        NNUE incremental update consistency checker
    @date 2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "./../../../engine/search.hpp"

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "usage: nnue_consistency_check [eval_nnue.egevnnue] [games=1000] [seed=20260725]\n";
        return 1;
    }
    const std::string eval_file = argv[1];
    const int games = argc >= 3 ? std::atoi(argv[2]) : 1000;
    const uint64_t seed = argc >= 4 ? std::strtoull(argv[3], nullptr, 10) : 20260725ULL;
    bit_init();
    mobility_init();
    flip_init();
    if (!evaluate_init(eval_file, "", true)) {
        return 1;
    }
    std::mt19937_64 rng(seed);
    uint64_t checked = 0;
    for (int g = 0; g < games; ++g) {
        Board board;
        board.reset();
        Search search(&board);
        std::vector<Flip> history;
        for (int ply = 0; ply < HW2 - 4; ++ply) {
            const int diff_value = mid_evaluate_diff(&search);
            const int rebuild_value = mid_evaluate(&search.board);
            if (diff_value != rebuild_value) {
                std::cerr << "[ERROR] value mismatch game " << g
                          << " ply " << ply
                          << " diff " << diff_value
                          << " rebuild " << rebuild_value << "\n";
                search.board.print();
                return 1;
            }
            ++checked;
            uint64_t legal = search.board.get_legal();
            if (legal == 0) {
                search.pass();
                if (search.board.get_legal() == 0) {
                    search.pass();
                    break;
                }
                continue;
            }
            const int n_legal = pop_count_ull(legal);
            int chosen = (int)(rng() % (uint64_t)n_legal);
            uint_fast8_t pos = first_bit(&legal);
            while (chosen-- > 0) {
                pos = next_bit(&legal);
            }
            Flip flip;
            calc_flip(&flip, &search.board, pos);
            history.push_back(flip);
            search.move(&history.back());
        }
    }
    std::cout << "checked " << checked << " positions\n";
    return 0;
}
