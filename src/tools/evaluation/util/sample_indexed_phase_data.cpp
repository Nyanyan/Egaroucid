/*
    Egaroucid Project

    @file sample_indexed_phase_data.cpp
        Reservoir-sample phase-wise indexed teacher data
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

bool read_indexed_record(FILE *fp, IndexedDatum *datum) {
    if (fread(&datum->n_discs, sizeof(int16_t), 1, fp) < 1) {
        return false;
    }
    return
        fread(&datum->player, sizeof(int16_t), 1, fp) == 1 &&
        fread(datum->features.data(), sizeof(uint16_t), ADJ_N_FEATURES, fp) == ADJ_N_FEATURES &&
        fread(&datum->score, sizeof(int16_t), 1, fp) == 1;
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
    if (argc < 6) {
        std::cerr
            << "usage: sample_indexed_phase_data [input_data_dir] [out_dir]"
            << " [start_phase] [end_phase] [max_per_phase] [seed=20260724]\n";
        return 1;
    }
    const std::string input_dir = argv[1];
    const std::string out_dir = argv[2];
    const int start_phase = std::atoi(argv[3]);
    const int end_phase = std::atoi(argv[4]);
    const size_t max_per_phase = (size_t)std::strtoull(argv[5], nullptr, 10);
    const uint64_t seed = argc >= 7 ? std::strtoull(argv[6], nullptr, 10) : 20260724ULL;

    if (start_phase < 0 || end_phase < start_phase || end_phase >= ADJ_N_PHASES || max_per_phase == 0) {
        std::cerr << "[ERROR] invalid arguments\n";
        return 1;
    }

    evaluation_definition_init();
    std::mt19937_64 rng(seed);
    std::vector<uint64_t> seen_counts((size_t)ADJ_N_PHASES, 0);
    std::vector<uint64_t> broken_counts((size_t)ADJ_N_PHASES, 0);
    std::vector<int64_t> score_sums((size_t)ADJ_N_PHASES, 0);

    for (int phase = start_phase; phase <= end_phase; ++phase) {
        std::vector<IndexedDatum> reservoir;
        reservoir.reserve(max_per_phase);
        const std::string file = input_dir + "/" + std::to_string(phase) + "/teacher_0.dat";
        FILE *fp = nullptr;
        if (fopen_s(&fp, file.c_str(), "rb") != 0) {
            std::cerr << "[WARN] can't open indexed data " << file << "\n";
            if (!write_phase_file(out_dir, phase, reservoir)) {
                return 1;
            }
            continue;
        }
        while (true) {
            IndexedDatum datum;
            if (!read_indexed_record(fp, &datum)) {
                if (!feof(fp)) {
                    ++broken_counts[(size_t)phase];
                }
                break;
            }
            score_sums[(size_t)phase] += datum.score;
            add_sample(&reservoir, &seen_counts[(size_t)phase], max_per_phase, datum, &rng);
        }
        fclose(fp);
        if (!write_phase_file(out_dir, phase, reservoir)) {
            return 1;
        }
        std::cout << "phase " << phase
                  << " seen " << seen_counts[(size_t)phase]
                  << " sampled " << reservoir.size()
                  << " broken " << broken_counts[(size_t)phase] << "\n";
    }

    std::ofstream summary(std::filesystem::path(out_dir) / "indexed_sample_summary.txt", std::ios::trunc);
    if (summary) {
        summary << "input_dir " << input_dir << "\n";
        summary << "start_phase " << start_phase << "\n";
        summary << "end_phase " << end_phase << "\n";
        summary << "max_per_phase " << max_per_phase << "\n";
        summary << "seed " << seed << "\n";
        summary << "phase seen sampled broken score_avg\n";
        for (int phase = start_phase; phase <= end_phase; ++phase) {
            const uint64_t seen = seen_counts[(size_t)phase];
            const double score_avg = seen ? (double)score_sums[(size_t)phase] / (double)seen : 0.0;
            const std::filesystem::path out_file = std::filesystem::path(out_dir) / std::to_string(phase) / "teacher_0.dat";
            const uint64_t sampled = std::filesystem::exists(out_file)
                ? (uint64_t)(std::filesystem::file_size(out_file) / (sizeof(int16_t) * (ADJ_N_FEATURES + 3)))
                : 0ULL;
            summary << phase << " " << seen << " " << sampled << " "
                    << broken_counts[(size_t)phase] << " " << score_avg << "\n";
        }
    }
    return 0;
}
