/*
    Egaroucid Project

    @file sample_board_to_indexed_phase_data.cpp
        Create phase-wise indexed teacher data by reservoir sampling raw board data
    @date 2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "./../evaluation_definition.hpp"

struct IndexedDatum {
    int16_t n_discs;
    int16_t player;
    std::array<uint16_t, ADJ_N_FEATURES> features;
    int16_t score;
};

bool read_raw_record(FILE *fp, Board *board, int8_t *player, int8_t *score) {
    int8_t policy = 0;
    if (fread(&board->player, 8, 1, fp) < 1) {
        return false;
    }
    return
        fread(&board->opponent, 8, 1, fp) == 1 &&
        fread(player, 1, 1, fp) == 1 &&
        fread(&policy, 1, 1, fp) == 1 &&
        fread(score, 1, 1, fp) == 1;
}

void add_sample(
    std::vector<IndexedDatum> *reservoir,
    uint64_t *seen_count,
    const size_t max_per_phase,
    const IndexedDatum &datum,
    std::mt19937_64 *rng
) {
    ++(*seen_count);
    if (reservoir->size() < max_per_phase) {
        reservoir->emplace_back(datum);
        return;
    }
    std::uniform_int_distribution<uint64_t> dist(0, *seen_count - 1);
    const uint64_t replace_idx = dist(*rng);
    if (replace_idx < max_per_phase) {
        (*reservoir)[(size_t)replace_idx] = datum;
    }
}

bool write_phase_file(const std::string &out_dir, const int phase, const std::vector<IndexedDatum> &data) {
    const std::filesystem::path phase_dir = std::filesystem::path(out_dir) / std::to_string(phase);
    std::filesystem::create_directories(phase_dir);
    const std::filesystem::path out_file = phase_dir / "teacher_0.dat";
    std::ofstream out(out_file, std::ios::binary | std::ios::trunc);
    if (!out) {
        std::cerr << "[ERROR] can't open output " << out_file.string() << "\n";
        return false;
    }
    for (const IndexedDatum &datum: data) {
        out.write((const char*)&datum.n_discs, sizeof(int16_t));
        out.write((const char*)&datum.player, sizeof(int16_t));
        out.write((const char*)datum.features.data(), sizeof(uint16_t) * ADJ_N_FEATURES);
        out.write((const char*)&datum.score, sizeof(int16_t));
    }
    return (bool)out;
}

int main(int argc, char **argv) {
    if (argc < 8) {
        std::cerr
            << "usage: sample_board_to_indexed_phase_data [input_dir] [start_file] [n_files] [out_dir]"
            << " [start_phase] [end_phase] [max_per_phase] [seed=20260724]\n";
        return 1;
    }
    const std::string input_dir = argv[1];
    const int start_file = std::atoi(argv[2]);
    const int n_files = std::atoi(argv[3]);
    const std::string out_dir = argv[4];
    const int start_phase = std::atoi(argv[5]);
    const int end_phase = std::atoi(argv[6]);
    const size_t max_per_phase = (size_t)std::strtoull(argv[7], nullptr, 10);
    const uint64_t seed = argc >= 9 ? std::strtoull(argv[8], nullptr, 10) : 20260724ULL;

    if (start_file < 0 || n_files <= 0 || start_phase < 0 || end_phase < start_phase ||
        end_phase >= ADJ_N_PHASES || max_per_phase == 0) {
        std::cerr << "[ERROR] invalid arguments\n";
        return 1;
    }

    evaluation_definition_init();
    std::mt19937_64 rng(seed);
    std::array<std::vector<IndexedDatum>, ADJ_N_PHASES> reservoirs;
    std::array<uint64_t, ADJ_N_PHASES> seen_counts = {};
    for (int phase = start_phase; phase <= end_phase; ++phase) {
        reservoirs[(size_t)phase].reserve(max_per_phase);
    }

    uint64_t total_records = 0;
    uint64_t broken_records = 0;
    for (int file_idx = start_file; file_idx < start_file + n_files; ++file_idx) {
        const std::string file = input_dir + "/" + std::to_string(file_idx) + ".dat";
        FILE *fp = nullptr;
        if (fopen_s(&fp, file.c_str(), "rb") != 0) {
            std::cerr << "[WARN] can't open raw data " << file << "\n";
            continue;
        }
        uint64_t file_records = 0;
        while (true) {
            Board board;
            int8_t player = 0;
            int8_t score = 0;
            if (!read_raw_record(fp, &board, &player, &score)) {
                if (!feof(fp)) {
                    ++broken_records;
                }
                break;
            }
            ++file_records;
            ++total_records;
            const int phase = calc_phase(&board, player);
            if (phase < start_phase || phase > end_phase) {
                continue;
            }
            IndexedDatum datum;
            datum.n_discs = (int16_t)pop_count_ull(board.player | board.opponent);
            datum.player = player;
            datum.score = score;
            uint16_t features[ADJ_N_FEATURES];
            adj_calc_features(&board, features);
            for (int i = 0; i < ADJ_N_FEATURES; ++i) {
                datum.features[(size_t)i] = features[i];
            }
            add_sample(&reservoirs[(size_t)phase], &seen_counts[(size_t)phase], max_per_phase, datum, &rng);
        }
        fclose(fp);
        std::cerr << file << " raw_records " << file_records << " total " << total_records << "\n";
    }

    for (int phase = start_phase; phase <= end_phase; ++phase) {
        if (!write_phase_file(out_dir, phase, reservoirs[(size_t)phase])) {
            return 1;
        }
    }

    std::ofstream summary(std::filesystem::path(out_dir) / "sample_summary.txt", std::ios::trunc);
    if (summary) {
        summary << "input_dir " << input_dir << "\n";
        summary << "start_file " << start_file << "\n";
        summary << "n_files " << n_files << "\n";
        summary << "start_phase " << start_phase << "\n";
        summary << "end_phase " << end_phase << "\n";
        summary << "max_per_phase " << max_per_phase << "\n";
        summary << "seed " << seed << "\n";
        summary << "total_records " << total_records << "\n";
        summary << "broken_records " << broken_records << "\n";
        summary << "phase seen sampled\n";
        for (int phase = start_phase; phase <= end_phase; ++phase) {
            summary << phase << " " << seen_counts[(size_t)phase] << " " << reservoirs[(size_t)phase].size() << "\n";
        }
    }

    std::cout << "wrote " << out_dir << " total_records " << total_records << " broken_records " << broken_records << "\n";
    for (int phase = start_phase; phase <= end_phase; ++phase) {
        std::cout << "phase " << phase
                  << " seen " << seen_counts[(size_t)phase]
                  << " sampled " << reservoirs[(size_t)phase].size() << "\n";
    }
    return 0;
}
