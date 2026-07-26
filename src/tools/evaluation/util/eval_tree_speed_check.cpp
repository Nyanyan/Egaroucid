/*
    Egaroucid Project

    @file eval_tree_speed_check.cpp
        Evaluation update and forward speed checker on legal child positions
    @date 2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>

#include "./../../../engine/evaluate.hpp"
#include "./../../../engine/search.hpp"

uint64_t speed_tim_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()
    ).count();
}

uint_fast8_t choose_random_move(uint64_t legal, std::mt19937_64 *rng) {
    const int n_legal = pop_count_ull(legal);
    int chosen = (int)((*rng)() % (uint64_t)n_legal);
    uint_fast8_t pos = first_bit(&legal);
    while (chosen-- > 0) {
        pos = next_bit(&legal);
    }
    return pos;
}

int main(int argc, char **argv) {
    std::string eval_file = "bin/resources/eval.egev2";
    std::string mo_file = "bin/resources/eval_move_ordering_end.egev";
    int games = 20000;
    uint64_t seed = 20260726ULL;
    if (argc >= 2) {
        eval_file = argv[1];
    }
    if (argc >= 3) {
        mo_file = argv[2];
    }
    if (argc >= 4) {
        games = std::atoi(argv[3]);
    }
    if (argc >= 5) {
        seed = std::strtoull(argv[4], nullptr, 10);
    }
    if (games <= 0) {
        std::cerr << "usage: eval_tree_speed_check [eval_file] [mo_file] [games=20000] [seed=20260726]\n";
        return 1;
    }

    bit_init();
    mobility_init();
    flip_init();
    if (!evaluate_init(eval_file, mo_file, true)) {
        return 1;
    }

    std::mt19937_64 rng(seed);
    uint64_t child_evals = 0;
    uint64_t playout_moves = 0;
    uint64_t passes = 0;
    int64_t checksum = 0;
    const uint64_t start_ms = speed_tim_ms();
    for (int g = 0; g < games; ++g) {
        Board board;
        board.reset();
        Search search(&board);
        for (int ply = 0; ply < HW2 - 4; ++ply) {
            if (search.n_discs >= HW2 - 1) {
                break;
            }
            uint64_t legal = search.board.get_legal();
            if (legal == 0) {
                search.pass();
                ++passes;
                if (search.board.get_legal() == 0) {
                    search.pass();
                    break;
                }
                continue;
            }

            uint64_t moves = legal;
            for (uint_fast8_t pos = first_bit(&moves); moves; pos = next_bit(&moves)) {
                Flip flip;
                calc_flip(&flip, &search.board, pos);
                search.move(&flip);
                checksum += mid_evaluate_diff(&search);
                search.undo(&flip);
                ++child_evals;
            }

            const uint_fast8_t chosen = choose_random_move(legal, &rng);
            Flip flip;
            calc_flip(&flip, &search.board, chosen);
            search.move(&flip);
            ++playout_moves;
        }
    }
    const uint64_t elapsed_ms = speed_tim_ms() - start_ms;
    const uint64_t child_evals_per_sec =
        elapsed_ms == 0 ? child_evals * 1000ULL : child_evals * 1000ULL / elapsed_ms;

    std::cerr << "checksum " << checksum << std::endl;
    std::cout << "eval_file " << eval_file << "\n";
    std::cout << "games " << games << " seed " << seed << "\n";
    std::cout << "child_evals " << child_evals << "\n";
    std::cout << "playout_moves " << playout_moves << "\n";
    std::cout << "passes " << passes << "\n";
    std::cout << "elapsed_ms " << elapsed_ms << "\n";
    std::cout << "child_evals_per_sec " << child_evals_per_sec << "\n";
    return 0;
}
