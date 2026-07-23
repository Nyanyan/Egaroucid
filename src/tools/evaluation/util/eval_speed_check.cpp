/*
    Egaroucid Project

    @file eval_speed_check.cpp
        Evaluation function speed checker
    @date 2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "./../../../engine/evaluate.hpp"

uint64_t speed_tim() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()
    ).count();
}

Board make_random_board(std::mt19937_64 *rng) {
    std::array<int, HW2> cells;
    std::iota(cells.begin(), cells.end(), 0);
    std::shuffle(cells.begin(), cells.end(), *rng);
    std::uniform_int_distribution<int> n_discs_dist(4, 63);
    const int n_discs = n_discs_dist(*rng);
    std::uniform_int_distribution<int> n_player_dist(1, n_discs - 1);
    const int n_player = n_player_dist(*rng);
    Board board;
    board.player = 0;
    board.opponent = 0;
    for (int i = 0; i < n_player; ++i) {
        board.player |= 1ULL << cells[i];
    }
    for (int i = n_player; i < n_discs; ++i) {
        board.opponent |= 1ULL << cells[i];
    }
    return board;
}

int main(int argc, char **argv) {
    std::string eval_file = "bin/resources/eval.egev2";
    std::string mo_file = "bin/resources/eval_move_ordering_end.egev";
    uint64_t n_positions = 200000;
    int repeats = 5;
    uint64_t seed = 20260723ULL;
    if (argc >= 2) {
        eval_file = argv[1];
    }
    if (argc >= 3) {
        mo_file = argv[2];
    }
    if (argc >= 4) {
        n_positions = std::strtoull(argv[3], nullptr, 10);
    }
    if (argc >= 5) {
        repeats = std::atoi(argv[4]);
    }
    if (argc >= 6) {
        seed = std::strtoull(argv[5], nullptr, 10);
    }
    if (n_positions == 0 || repeats <= 0) {
        std::cerr << "usage: eval_speed_check [eval_file] [mo_file] [n_positions=200000] [repeats=5] [seed=20260723]" << std::endl;
        return 1;
    }
    if (!evaluate_init(eval_file, mo_file, true)) {
        return 1;
    }
    std::mt19937_64 rng(seed);
    std::vector<Board> boards;
    boards.reserve((size_t)n_positions);
    for (uint64_t i = 0; i < n_positions; ++i) {
        boards.emplace_back(make_random_board(&rng));
    }

    std::vector<Search> searches;
    searches.reserve((size_t)n_positions);
    for (const Board &board: boards) {
        searches.emplace_back((Board*)&board);
    }

    int64_t checksum = 0;
    for (uint64_t i = 0; i < std::min<uint64_t>(n_positions, 1000); ++i) {
        checksum += mid_evaluate(&boards[(size_t)i]);
        checksum += mid_evaluate_diff(&searches[(size_t)i]);
    }
    const uint64_t start_eval_ms = speed_tim();
    for (int r = 0; r < repeats; ++r) {
        for (const Board &board: boards) {
            checksum += mid_evaluate((Board*)&board);
        }
    }
    const uint64_t elapsed_eval = speed_tim() - start_eval_ms;
    const uint64_t start_diff_ms = speed_tim();
    for (int r = 0; r < repeats; ++r) {
        for (Search &search: searches) {
            checksum += mid_evaluate_diff(&search);
        }
    }
    const uint64_t elapsed_diff = speed_tim() - start_diff_ms;
    const uint64_t n_eval = n_positions * (uint64_t)repeats;
    const uint64_t eval_nps = n_eval * 1000ULL / (elapsed_eval + 1);
    const uint64_t diff_nps = n_eval * 1000ULL / (elapsed_diff + 1);
    std::cerr << "checksum " << checksum << std::endl;
    std::cout << "eval_file " << eval_file << "\n";
    std::cout << "positions " << n_positions << " repeats " << repeats << " evals " << n_eval << "\n";
    std::cout << "mid_evaluate_elapsed_ms " << elapsed_eval << " nps " << eval_nps << "\n";
    std::cout << "mid_evaluate_diff_elapsed_ms " << elapsed_diff << " nps " << diff_nps << "\n";
    return 0;
}
