/*
    Egaroucid Project

    @file generate_eval77_fm_distillation_batches.cpp
        Generate phase-separated 7.7-FM training records whose targets are
        scores from an existing EGEV4 evaluator.
    @date 2026
    @author Takuto Yamana, Codex
    @license GPL-3.0-or-later
*/

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#if !defined(EVALUATE_EXPERIMENT_7_7_FM_FAST)
    #define EVALUATE_EXPERIMENT_7_7_FM_FAST
#endif

#include "../../../engine/bit.hpp"
#include "../../../engine/board.hpp"
#include "../../../engine/common.hpp"
#include "../../../engine/evaluate.hpp"
#include "../../../engine/flip.hpp"
#include "../../../engine/last_flip.hpp"
#include "../../../engine/mobility.hpp"
#include "../../../engine/search.hpp"

namespace fs = std::filesystem;

namespace {

constexpr int N_SPLITS = 4;
constexpr std::array<int, 16> PATTERN_OFFSETS = {
    0, 59049, 118098, 177147,
    236196, 295245, 354294, 413343,
    472392, 531441, 590490, 649539,
    708588, 767637, 826686, 885735
};

struct Record {
    int16_t n_discs;
    int16_t player;
    std::array<uint16_t, N_PATTERN_FEATURES + 1> features;
    int16_t score;
};

static_assert(
    sizeof(Record) == 136,
    "distillation record must match the 7.7-FM optimizer format"
);

int pick_random_move(uint64_t legal, std::mt19937 &rng) {
    int remaining = static_cast<int>(
        rng() % static_cast<uint32_t>(pop_count_ull(legal))
    );
    for (int move = first_bit(&legal); legal; move = next_bit(&legal)) {
        if (remaining == 0) {
            return move;
        }
        --remaining;
    }
    return MOVE_UNDEFINED;
}

bool write_position(
    Search &search,
    std::ofstream &output,
    std::array<uint64_t, N_PHASES> &phase_counts,
    int group_size
) {
    const int phase = search.phase();
    if (phase < 0 || phase >= N_PHASES) {
        return false;
    }

    const int player_count = pop_count_ull(search.board.player);
    int active_ids[N_PATTERN_FEATURES + 1];
    int n_active = 0;
    collect_eval77_fm_fast_simd_active_ids(
        &search.eval.features[search.eval.feature_idx],
        player_count,
        active_ids,
        n_active
    );
    if (n_active != N_PATTERN_FEATURES + 1) {
        return false;
    }

    Record record{};
    record.n_discs = static_cast<int16_t>(search.board.n_discs());
    record.player = static_cast<int16_t>(group_size);
    std::array<int, 16> block_counts{};
    for (int i = 0; i < N_PATTERN_FEATURES; ++i) {
        const int block = active_ids[i] / 59049;
        if (block < 0 || block >= static_cast<int>(PATTERN_OFFSETS.size()) ||
            block_counts[block] >= 4) {
            return false;
        }
        const int feature = active_ids[i] - PATTERN_OFFSETS[block];
        if (feature < 0 || feature >= 59049) {
            return false;
        }
        record.features[block * 4 + block_counts[block]] =
            static_cast<uint16_t>(feature);
        ++block_counts[block];
    }
    for (int count : block_counts) {
        if (count != 4) {
            return false;
        }
    }
    record.features[N_PATTERN_FEATURES] =
        static_cast<uint16_t>(player_count);
    record.score = static_cast<int16_t>(calc_pattern(
        phase,
        &search.eval.features[search.eval.feature_idx],
        player_count
    ));

    output.write(
        reinterpret_cast<const char *>(&record),
        static_cast<std::streamsize>(sizeof(record))
    );
    if (!output) {
        return false;
    }
    ++phase_counts[phase];
    return true;
}

} // namespace

int main(int argc, char **argv) {
    if (argc != 6 && argc != 7) {
        std::cerr
            << "usage: generate_eval77_fm_distillation_batches "
            << "<board-text-root> <output-root> <teacher.egev4> "
            << "<games> <seed> [ranking]"
            << std::endl;
        return 1;
    }

    const fs::path input_root = argv[1];
    const fs::path output_root = argv[2];
    const std::string teacher_file = argv[3];
    const int requested_games = std::stoi(argv[4]);
    const uint32_t seed = static_cast<uint32_t>(std::stoul(argv[5]));
    const bool ranking_mode = argc == 7 && std::string(argv[6]) == "ranking";
    if (argc == 7 && !ranking_mode) {
        std::cerr << "[ERROR] optional mode must be ranking" << std::endl;
        return 1;
    }
    if (requested_games <= 0 || !fs::is_directory(input_root)) {
        std::cerr << "[ERROR] invalid input directory or game count" << std::endl;
        return 1;
    }

    bit_init();
    mobility_init();
    flip_init();
    last_flip_init();
    pre_calculate_eval_constant();
    if (!eval77_fm_fast_load_file(teacher_file.c_str(), false)) {
        return 1;
    }

    std::array<std::array<std::ofstream, N_SPLITS>, N_PHASES> outputs;
    for (int phase = 0; phase < N_PHASES; ++phase) {
        const fs::path phase_dir = output_root / std::to_string(phase);
        fs::create_directories(phase_dir);
        for (int split = 0; split < N_SPLITS; ++split) {
            outputs[phase][split].open(
                phase_dir /
                    ("records" + std::to_string(split) + ".dat"),
                std::ios::binary | std::ios::trunc
            );
            if (!outputs[phase][split]) {
                std::cerr << "[ERROR] cannot open output for phase "
                          << phase << " split " << split << std::endl;
                return 1;
            }
        }
    }

    std::vector<fs::path> input_files;
    for (const fs::directory_entry &entry :
         fs::recursive_directory_iterator(input_root)) {
        if (entry.is_regular_file() && entry.path().extension() == ".txt") {
            input_files.push_back(entry.path());
        }
    }
    std::sort(input_files.begin(), input_files.end());

    std::mt19937 rng(seed);
    std::array<std::array<uint64_t, N_PHASES>, N_SPLITS> counts{};
    int games = 0;
    uint64_t invalid_positions = 0;
    int input_passes = 0;
    while (games < requested_games) {
        const int games_before_pass = games;
        for (const fs::path &path : input_files) {
            if (games >= requested_games) {
                break;
            }
            std::ifstream input(path);
            std::string line;
            while (games < requested_games && std::getline(input, line)) {
                Board board;
                if (!board.from_str(line)) {
                    continue;
                }

                const int split = games % N_SPLITS;
                Search search(&board);
                bool previous_pass = false;
                while (search.board.n_discs() < HW2) {
                    const int phase = search.phase();
                    if (phase < 0 || phase >= N_PHASES ||
                        (!ranking_mode &&
                        !write_position(
                            search,
                            outputs[phase][split],
                            counts[split],
                            0
                        ))) {
                        ++invalid_positions;
                        break;
                    }

                    uint64_t legal = search.board.get_legal();
                    if (legal == 0) {
                        if (previous_pass) {
                            break;
                        }
                        search.pass();
                        previous_pass = true;
                        continue;
                    }
                    previous_pass = false;
                    if (ranking_mode) {
                        const int group_size = pop_count_ull(legal);
                        uint64_t remaining = legal;
                        for (
                            int candidate_move = first_bit(&remaining);
                            remaining;
                            candidate_move = next_bit(&remaining)
                        ) {
                            Flip candidate_flip;
                            calc_flip(
                                &candidate_flip,
                                &search.board,
                                candidate_move
                            );
                            search.move(&candidate_flip);
                            const int child_phase = search.phase();
                            if (child_phase < N_PHASES &&
                                !write_position(
                                    search,
                                    outputs[child_phase][split],
                                    counts[split],
                                    group_size
                                )) {
                                ++invalid_positions;
                                search.undo(&candidate_flip);
                                break;
                            }
                            search.undo(&candidate_flip);
                        }
                    }
                    const int move = pick_random_move(legal, rng);
                    if (move == MOVE_UNDEFINED) {
                        ++invalid_positions;
                        break;
                    }
                    Flip flip;
                    calc_flip(&flip, &search.board, move);
                    search.move(&flip);
                }

                ++games;
                if (games % 1000 == 0) {
                    std::cerr << "games " << games << "/"
                              << requested_games << std::endl;
                }
            }
        }
        ++input_passes;
        if (games == games_before_pass) {
            std::cerr << "[ERROR] no valid boards found in input pass "
                      << input_passes << std::endl;
            break;
        }
    }

    std::ofstream summary(output_root / "summary.tsv");
    summary << "teacher\t" << teacher_file << "\n";
    summary << "input_root\t" << input_root.string() << "\n";
    summary << "games\t" << games << "\n";
    summary << "input_passes\t" << input_passes << "\n";
    summary << "seed\t" << seed << "\n";
    summary << "mode\t"
            << (ranking_mode ? "ranking" : "position") << "\n";
    summary << "invalid_positions\t" << invalid_positions << "\n";
    summary << "phase";
    for (int split = 0; split < N_SPLITS; ++split) {
        summary << "\trecords" << split;
    }
    summary << "\n";
    for (int phase = 0; phase < N_PHASES; ++phase) {
        summary << phase;
        for (int split = 0; split < N_SPLITS; ++split) {
            summary << "\t" << counts[split][phase];
        }
        summary << "\n";
    }

    std::cout << "games=" << games
              << " invalid_positions=" << invalid_positions
              << " output=" << output_root.string()
              << std::endl;
    return games == requested_games && invalid_positions == 0 ? 0 : 2;
}
