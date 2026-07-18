/*
    Egaroucid Project

    @file generate_eval77_fm_distillation_from_transcripts.cpp
        Generate phase-separated 7.7-FM distillation records from every legal
        child position along complete game transcripts.
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
#include <sstream>
#include <string>
#include <unordered_set>
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
constexpr int FIRST_TRAINED_PHASE = 6;
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

std::vector<std::string> split_tab(const std::string &line) {
    std::vector<std::string> fields;
    std::stringstream stream(line);
    std::string field;
    while (std::getline(stream, field, '\t')) {
        fields.push_back(field);
    }
    if (!line.empty() && line.back() == '\t') {
        fields.emplace_back();
    }
    return fields;
}

int find_column(const std::vector<std::string> &header, const std::string &name) {
    for (size_t i = 0; i < header.size(); ++i) {
        if (header[i] == name) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

bool write_position(
    Search &search,
    std::ofstream &output,
    std::array<uint64_t, N_PHASES> &phase_counts,
    int group_size
) {
    const int phase = search.phase();
    if (phase < FIRST_TRAINED_PHASE || phase >= N_PHASES) {
        return true;
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

bool process_transcript(
    const std::string &transcript,
    int split,
    std::array<std::array<std::ofstream, N_SPLITS>, N_PHASES> &outputs,
    std::array<std::array<uint64_t, N_PHASES>, N_SPLITS> &counts
) {
    if (transcript.empty() || transcript.size() % 2 != 0) {
        return false;
    }

    Board board;
    board.reset();
    for (size_t offset = 0; offset < transcript.size(); offset += 2) {
        uint64_t legal = board.get_legal();
        if (legal == 0) {
            board.pass();
            legal = board.get_legal();
            if (legal == 0) {
                return false;
            }
        }

        Search search(&board);
        const int group_size = pop_count_ull(legal);
        for (int move = first_bit(&legal); legal; move = next_bit(&legal)) {
            Flip flip;
            calc_flip(&flip, &search.board, move);
            if (flip.flip == 0) {
                return false;
            }
            search.move(&flip);
            const int phase = search.phase();
            if (phase >= FIRST_TRAINED_PHASE &&
                !write_position(
                    search,
                    outputs[phase][split],
                    counts[split],
                    group_size
                )) {
                return false;
            }
            search.undo(&flip);
        }

        const int played_move = get_coord_from_chars(
            transcript[offset],
            transcript[offset + 1]
        );
        if (played_move < 0 || played_move >= HW2 ||
            (board.get_legal() & (uint64_t{1} << played_move)) == 0) {
            return false;
        }
        Flip played_flip;
        calc_flip(&played_flip, &board, played_move);
        if (played_flip.flip == 0) {
            return false;
        }
        board.move_board(&played_flip);
    }
    return true;
}

} // namespace

int main(int argc, char **argv) {
    if (argc != 5) {
        std::cerr
            << "usage: generate_eval77_fm_distillation_from_transcripts "
            << "<input.tsv> <output-root> <teacher.egev4> "
            << "<max-unique-transcripts-or-0>"
            << std::endl;
        return 1;
    }

    const fs::path input_path = argv[1];
    const fs::path output_root = argv[2];
    const std::string teacher_file = argv[3];
    const uint64_t max_unique_transcripts = std::stoull(argv[4]);

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
                phase_dir / ("records" + std::to_string(split) + ".dat"),
                std::ios::binary | std::ios::trunc
            );
            if (!outputs[phase][split]) {
                std::cerr << "[ERROR] cannot open output for phase "
                          << phase << " split " << split << std::endl;
                return 1;
            }
        }
    }

    std::ifstream input(input_path);
    std::string line;
    if (!input || !std::getline(input, line)) {
        std::cerr << "[ERROR] cannot read input header" << std::endl;
        return 1;
    }
    const std::vector<std::string> header = split_tab(line);
    const int battle_column = find_column(header, "battle");
    const int record_column = find_column(header, "record");
    if (battle_column < 0 || record_column < 0) {
        std::cerr << "[ERROR] input must contain battle and record columns"
                  << std::endl;
        return 1;
    }

    std::array<std::array<uint64_t, N_PHASES>, N_SPLITS> counts{};
    std::unordered_set<std::string> seen_transcripts;
    uint64_t input_rows = 0;
    uint64_t duplicate_transcripts = 0;
    uint64_t invalid_transcripts = 0;
    uint64_t unique_transcripts = 0;
    while (std::getline(input, line)) {
        if (max_unique_transcripts > 0 &&
            unique_transcripts >= max_unique_transcripts) {
            break;
        }
        ++input_rows;
        const std::vector<std::string> fields = split_tab(line);
        const int max_column = std::max(battle_column, record_column);
        if (static_cast<int>(fields.size()) <= max_column) {
            ++invalid_transcripts;
            continue;
        }
        const std::string &transcript = fields[record_column];
        if (!seen_transcripts.insert(transcript).second) {
            ++duplicate_transcripts;
            continue;
        }

        int battle = 0;
        try {
            battle = std::stoi(fields[battle_column]);
        } catch (const std::exception &) {
            ++invalid_transcripts;
            continue;
        }
        const int split = ((battle - 1) % N_SPLITS + N_SPLITS) % N_SPLITS;
        if (!process_transcript(transcript, split, outputs, counts)) {
            ++invalid_transcripts;
            continue;
        }
        ++unique_transcripts;
        if (unique_transcripts % 1000 == 0) {
            std::cerr << "unique_transcripts " << unique_transcripts
                      << std::endl;
        }
    }

    std::ofstream summary(output_root / "summary.tsv");
    summary << "teacher\t" << teacher_file << "\n";
    summary << "input\t" << input_path.string() << "\n";
    summary << "input_rows\t" << input_rows << "\n";
    summary << "unique_transcripts\t" << unique_transcripts << "\n";
    summary << "duplicate_transcripts\t" << duplicate_transcripts << "\n";
    summary << "invalid_transcripts\t" << invalid_transcripts << "\n";
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

    std::cout << "input_rows=" << input_rows
              << " unique_transcripts=" << unique_transcripts
              << " duplicate_transcripts=" << duplicate_transcripts
              << " invalid_transcripts=" << invalid_transcripts
              << " output=" << output_root.string()
              << std::endl;
    return invalid_transcripts == 0 ? 0 : 2;
}
