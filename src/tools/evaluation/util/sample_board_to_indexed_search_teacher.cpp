/*
    Egaroucid Project

    @file sample_board_to_indexed_search_teacher.cpp
        Create phase-wise indexed teacher data by searching sampled raw boards
    @date 2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#include <array>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "./../../../engine/engine_all.hpp"
#include "./../evaluation_definition.hpp"

struct RawDatum {
    Board board;
    int16_t player;
    int16_t raw_score;
};

struct IndexedDatum {
    int16_t n_discs;
    int16_t player;
    std::array<uint16_t, ADJ_N_FEATURES> features;
    int16_t score;
};

struct PhaseStats {
    uint64_t seen = 0;
    uint64_t sampled = 0;
    uint64_t searched = 0;
    uint64_t nodes = 0;
    uint64_t elapsed_ms = 0;
    int64_t raw_score_sum = 0;
    int64_t search_score_sum = 0;
    int64_t blended_score_sum = 0;
};

uint64_t search_teacher_tim() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()
    ).count();
}

bool read_raw_record(FILE *fp, Board *board, int8_t *player, int8_t *raw_score) {
    int8_t policy = 0;
    if (fread(&board->player, 8, 1, fp) < 1) {
        return false;
    }
    return
        fread(&board->opponent, 8, 1, fp) == 1 &&
        fread(player, 1, 1, fp) == 1 &&
        fread(&policy, 1, 1, fp) == 1 &&
        fread(raw_score, 1, 1, fp) == 1;
}

void add_sample(
    std::vector<RawDatum> *reservoir,
    uint64_t *seen_count,
    const size_t max_per_phase,
    const RawDatum &datum,
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

bool initialize_engine(const std::string &eval_file, const std::string &mo_file, const int hash_level) {
    thread_pool.resize(0);
    bit_init();
    mobility_init();
    flip_init();
    last_flip_init();
    endsearch_init();
#if USE_MPC_PRE_CALCULATION
    mpc_init();
#endif
    move_ordering_init();
#if USE_CHANGEABLE_HASH_LEVEL
    hash_resize(DEFAULT_HASH_LEVEL, hash_level, false);
#else
    hash_tt_init(false);
#endif
    stability_init();
    return evaluate_init(eval_file, mo_file, false);
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

int blend_score(const int raw_score, const int search_score, const int blend_num, const int blend_den) {
    const double blended = (double)raw_score + (double)blend_num * (double)(search_score - raw_score) / (double)blend_den;
    return std::clamp((int)std::lround(blended), -SCORE_MAX, SCORE_MAX);
}

int main(int argc, char **argv) {
    if (argc < 11) {
        std::cerr
            << "usage: sample_board_to_indexed_search_teacher [input_dir] [start_file] [n_files] [out_dir]"
            << " [start_phase] [end_phase] [max_per_phase] [level] [eval_file] [mo_file]"
            << " [seed=20260724] [hash_level=18] [blend_num=1] [blend_den=1]\n";
        return 1;
    }
    const std::string input_dir = argv[1];
    const int start_file = std::atoi(argv[2]);
    const int n_files = std::atoi(argv[3]);
    const std::string out_dir = argv[4];
    const int start_phase = std::atoi(argv[5]);
    const int end_phase = std::atoi(argv[6]);
    const size_t max_per_phase = (size_t)std::strtoull(argv[7], nullptr, 10);
    const int level = std::atoi(argv[8]);
    const std::string eval_file = argv[9];
    const std::string mo_file = argv[10];
    const uint64_t seed = argc >= 12 ? std::strtoull(argv[11], nullptr, 10) : 20260724ULL;
    const int hash_level = argc >= 13 ? std::atoi(argv[12]) : 18;
    const int blend_num = argc >= 14 ? std::atoi(argv[13]) : 1;
    const int blend_den = argc >= 15 ? std::atoi(argv[14]) : 1;

    if (start_file < 0 || n_files <= 0 || start_phase < 0 || end_phase < start_phase ||
        end_phase >= ADJ_N_PHASES || max_per_phase == 0 || level <= 0 || level > MAX_LEVEL ||
        hash_level <= 0 || hash_level >= N_HASH_LEVEL ||
        blend_den <= 0 || blend_num < 0 || blend_num > blend_den) {
        std::cerr << "[ERROR] invalid arguments\n";
        return 1;
    }

    if (!initialize_engine(eval_file, mo_file, hash_level)) {
        return 1;
    }

    std::mt19937_64 rng(seed);
    std::array<std::vector<RawDatum>, ADJ_N_PHASES> reservoirs;
    std::array<PhaseStats, ADJ_N_PHASES> stats;
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
            int8_t raw_score = 0;
            if (!read_raw_record(fp, &board, &player, &raw_score)) {
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
            RawDatum datum;
            datum.board = board;
            datum.player = player;
            datum.raw_score = raw_score;
            stats[(size_t)phase].raw_score_sum += raw_score;
            add_sample(&reservoirs[(size_t)phase], &stats[(size_t)phase].seen, max_per_phase, datum, &rng);
        }
        fclose(fp);
        std::cerr << file << " raw_records " << file_records << " total " << total_records << "\n";
    }

    for (int phase = start_phase; phase <= end_phase; ++phase) {
        std::vector<IndexedDatum> indexed;
        indexed.reserve(reservoirs[(size_t)phase].size());
        const uint64_t phase_start = search_teacher_tim();
        for (const RawDatum &raw: reservoirs[(size_t)phase]) {
            bool searching = true;
            Search_result search_result = ai_searching(raw.board, level, false, 0, false, false, &searching);
            if (search_result.value == SCORE_UNDEFINED || !is_valid_score(search_result.value)) {
                std::cerr << "[WARN] invalid search score phase " << phase
                          << " value " << search_result.value << "\n";
                continue;
            }
            IndexedDatum datum;
            datum.n_discs = (int16_t)pop_count_ull(raw.board.player | raw.board.opponent);
            datum.player = raw.player;
            uint16_t features[ADJ_N_FEATURES];
            Board feature_board = raw.board.copy();
            adj_calc_features(&feature_board, features);
            for (int i = 0; i < ADJ_N_FEATURES; ++i) {
                datum.features[(size_t)i] = features[i];
            }
            const int blended_score = blend_score(raw.raw_score, search_result.value, blend_num, blend_den);
            datum.score = (int16_t)blended_score;
            indexed.emplace_back(datum);
            stats[(size_t)phase].searched += 1;
            stats[(size_t)phase].nodes += search_result.nodes;
            stats[(size_t)phase].search_score_sum += search_result.value;
            stats[(size_t)phase].blended_score_sum += blended_score;
        }
        stats[(size_t)phase].sampled = reservoirs[(size_t)phase].size();
        stats[(size_t)phase].elapsed_ms = search_teacher_tim() - phase_start;
        if (!write_phase_file(out_dir, phase, indexed)) {
            return 1;
        }
        std::cerr << "phase " << phase
                  << " searched " << stats[(size_t)phase].searched
                  << " elapsed_ms " << stats[(size_t)phase].elapsed_ms
                  << " nodes " << stats[(size_t)phase].nodes << "\n";
    }

    std::ofstream summary(std::filesystem::path(out_dir) / "search_teacher_summary.txt", std::ios::trunc);
    if (summary) {
        summary << "input_dir " << input_dir << "\n";
        summary << "start_file " << start_file << "\n";
        summary << "n_files " << n_files << "\n";
        summary << "start_phase " << start_phase << "\n";
        summary << "end_phase " << end_phase << "\n";
        summary << "max_per_phase " << max_per_phase << "\n";
        summary << "level " << level << "\n";
        summary << "eval_file " << eval_file << "\n";
        summary << "mo_file " << mo_file << "\n";
        summary << "seed " << seed << "\n";
        summary << "hash_level " << hash_level << "\n";
        summary << "blend_num " << blend_num << "\n";
        summary << "blend_den " << blend_den << "\n";
        summary << "total_records " << total_records << "\n";
        summary << "broken_records " << broken_records << "\n";
        summary << "phase seen sampled searched elapsed_ms nodes raw_score_avg search_score_avg blended_score_avg\n";
        for (int phase = start_phase; phase <= end_phase; ++phase) {
            const PhaseStats &s = stats[(size_t)phase];
            const double raw_avg = s.seen ? (double)s.raw_score_sum / (double)s.seen : 0.0;
            const double search_avg = s.searched ? (double)s.search_score_sum / (double)s.searched : 0.0;
            const double blended_avg = s.searched ? (double)s.blended_score_sum / (double)s.searched : 0.0;
            summary << phase << " " << s.seen << " " << s.sampled << " " << s.searched
                    << " " << s.elapsed_ms << " " << s.nodes
                    << " " << raw_avg << " " << search_avg << " " << blended_avg << "\n";
        }
    }

    std::cout << "wrote " << out_dir << " total_records " << total_records
              << " broken_records " << broken_records << "\n";
    for (int phase = start_phase; phase <= end_phase; ++phase) {
        const PhaseStats &s = stats[(size_t)phase];
        const double nps = s.elapsed_ms ? (double)s.nodes * 1000.0 / (double)s.elapsed_ms : 0.0;
        std::cout << "phase " << phase
                  << " seen " << s.seen
                  << " sampled " << s.sampled
                  << " searched " << s.searched
                  << " elapsed_ms " << s.elapsed_ms
                  << " nodes " << s.nodes
                  << " nps " << nps
                  << "\n";
    }
    return 0;
}
