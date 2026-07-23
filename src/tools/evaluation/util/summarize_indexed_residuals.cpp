/*
    Egaroucid Project

    @file summarize_indexed_residuals.cpp
        Summarize indexed teacher residuals against a base linear evaluation
    @date 2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "./../evaluation_definition.hpp"

constexpr int N_ZEROS_PLUS_LOCAL = 1 << 12;

struct IndexedDatum {
    int16_t n_discs;
    int16_t player;
    std::array<uint16_t, ADJ_N_FEATURES> features;
    int16_t score;
};

struct PhaseStats {
    uint64_t count = 0;
    double score_sum = 0.0;
    double linear_raw_sum = 0.0;
    double residual_sum = 0.0;
    double residual_abs_sum = 0.0;
    double residual_square_sum = 0.0;
    double residual_min = std::numeric_limits<double>::infinity();
    double residual_max = -std::numeric_limits<double>::infinity();
    double residual_max_abs = 0.0;
};

template<typename T>
bool read_scalar(FILE *fp, T *v) {
    return fread(v, sizeof(T), 1, fp) == 1;
}

bool read_indexed_record(FILE *fp, IndexedDatum *datum) {
    if (fread(&datum->n_discs, sizeof(int16_t), 1, fp) < 1) {
        return false;
    }
    return
        fread(&datum->player, sizeof(int16_t), 1, fp) == 1 &&
        fread(datum->features.data(), sizeof(uint16_t), ADJ_N_FEATURES, fp) == ADJ_N_FEATURES &&
        fread(&datum->score, sizeof(int16_t), 1, fp) == 1;
}

std::vector<int16_t> load_unzip_egev2_local(const std::string &file) {
    FILE *fp = nullptr;
    if (fopen_s(&fp, file.c_str(), "rb") != 0) {
        std::cerr << "[ERROR] can't open base eval " << file << "\n";
        return {};
    }
    int32_t n_zipped = 0;
    if (!read_scalar(fp, &n_zipped) || n_zipped <= 0) {
        std::cerr << "[ERROR] base eval header broken " << file << "\n";
        fclose(fp);
        return {};
    }
    std::vector<int16_t> zipped((size_t)n_zipped);
    if (fread(zipped.data(), sizeof(int16_t), zipped.size(), fp) < zipped.size()) {
        std::cerr << "[ERROR] base eval payload broken " << file << "\n";
        fclose(fp);
        return {};
    }
    fclose(fp);

    std::vector<int16_t> res;
    res.reserve((size_t)ADJ_N_PHASES * 700000);
    for (const int16_t elem: zipped) {
        if (elem >= N_ZEROS_PLUS_LOCAL) {
            res.insert(res.end(), elem - N_ZEROS_PLUS_LOCAL, 0);
        } else {
            res.emplace_back(elem);
        }
    }
    return res;
}

std::array<int, ADJ_N_FEATURES> make_linear_starts() {
    std::array<int, ADJ_N_FEATURES> starts = {};
    int start = 0;
    for (int i = 0; i < ADJ_N_FEATURES; ++i) {
        if (i > 0 && adj_feature_to_eval_idx[i] > adj_feature_to_eval_idx[i - 1]) {
            start += adj_eval_sizes[adj_feature_to_eval_idx[i - 1]];
        }
        starts[(size_t)i] = start;
    }
    return starts;
}

int linear_params_per_phase() {
    int res = 0;
    for (int i = 0; i < ADJ_N_EVAL; ++i) {
        res += adj_eval_sizes[i];
    }
    return res;
}

int calc_linear_raw(
    const std::vector<int16_t> &linear,
    const std::array<int, ADJ_N_FEATURES> &starts,
    const IndexedDatum &datum,
    const int phase
) {
    const int phase_offset = phase * linear_params_per_phase();
    int res = 0;
    for (int i = 0; i < ADJ_N_FEATURES; ++i) {
        const int eval_idx = adj_feature_to_eval_idx[i];
        if (datum.features[(size_t)i] >= adj_eval_sizes[eval_idx]) {
            return 0;
        }
        res += linear[(size_t)phase_offset + starts[(size_t)i] + datum.features[(size_t)i]];
    }
    return res;
}

void add_stat(PhaseStats *stats, const IndexedDatum &datum, const int linear_raw) {
    const double target_raw = (double)datum.score * (double)ADJ_STEP;
    const double residual = target_raw - (double)linear_raw;
    ++stats->count;
    stats->score_sum += datum.score;
    stats->linear_raw_sum += linear_raw;
    stats->residual_sum += residual;
    stats->residual_abs_sum += std::fabs(residual);
    stats->residual_square_sum += residual * residual;
    stats->residual_min = std::min(stats->residual_min, residual);
    stats->residual_max = std::max(stats->residual_max, residual);
    stats->residual_max_abs = std::max(stats->residual_max_abs, std::fabs(residual));
}

int main(int argc, char **argv) {
    if (argc < 5) {
        std::cerr
            << "usage: summarize_indexed_residuals [base_eval.egev2] [data_dir]"
            << " [start_phase] [end_phase] [max_records_per_phase=0] [out_txt=-]\n";
        return 1;
    }
    const std::string base_eval = argv[1];
    const std::string data_dir = argv[2];
    const int start_phase = std::atoi(argv[3]);
    const int end_phase = std::atoi(argv[4]);
    const uint64_t max_records_per_phase = argc >= 6 ? std::strtoull(argv[5], nullptr, 10) : 0ULL;
    const std::string out_txt = argc >= 7 ? argv[6] : "-";

    if (start_phase < 0 || end_phase < start_phase || end_phase >= ADJ_N_PHASES) {
        std::cerr << "[ERROR] invalid phase range\n";
        return 1;
    }

    evaluation_definition_init();
    const auto linear = load_unzip_egev2_local(base_eval);
    const int expected_linear = ADJ_N_PHASES * linear_params_per_phase();
    if ((int)linear.size() != expected_linear) {
        std::cerr << "[ERROR] invalid eval element count found " << linear.size()
                  << " expected " << expected_linear << "\n";
        return 1;
    }
    const auto starts = make_linear_starts();
    std::array<PhaseStats, ADJ_N_PHASES> phase_stats;
    PhaseStats total;

    for (int phase = start_phase; phase <= end_phase; ++phase) {
        const std::string file = data_dir + "/" + std::to_string(phase) + "/teacher_0.dat";
        FILE *fp = nullptr;
        if (fopen_s(&fp, file.c_str(), "rb") != 0) {
            std::cerr << "[WARN] can't open indexed data " << file << "\n";
            continue;
        }
        uint64_t read_count = 0;
        while (max_records_per_phase == 0 || read_count < max_records_per_phase) {
            IndexedDatum datum;
            if (!read_indexed_record(fp, &datum)) {
                break;
            }
            const int linear_raw = calc_linear_raw(linear, starts, datum, phase);
            add_stat(&phase_stats[(size_t)phase], datum, linear_raw);
            add_stat(&total, datum, linear_raw);
            ++read_count;
        }
        fclose(fp);
    }

    std::ofstream file_out;
    std::ostream *out = &std::cout;
    if (out_txt != "-") {
        const std::filesystem::path out_path(out_txt);
        if (out_path.has_parent_path()) {
            std::filesystem::create_directories(out_path.parent_path());
        }
        file_out.open(out_txt, std::ios::trunc);
        if (!file_out) {
            std::cerr << "[ERROR] can't open output " << out_txt << "\n";
            return 1;
        }
        out = &file_out;
    }

    *out << "base_eval " << base_eval << "\n";
    *out << "data_dir " << data_dir << "\n";
    *out << "start_phase " << start_phase << "\n";
    *out << "end_phase " << end_phase << "\n";
    *out << "max_records_per_phase " << max_records_per_phase << "\n";
    *out << "phase count score_avg linear_raw_avg_disc residual_avg_disc residual_mae_disc residual_rms_disc residual_min_disc residual_max_disc residual_max_abs_disc\n";
    for (int phase = start_phase; phase <= end_phase; ++phase) {
        const PhaseStats &s = phase_stats[(size_t)phase];
        if (s.count == 0) {
            *out << phase << " 0 0 0 0 0 0 0 0 0\n";
            continue;
        }
        const double denom = (double)s.count;
        *out << phase
             << " " << s.count
             << " " << s.score_sum / denom
             << " " << s.linear_raw_sum / denom / (double)ADJ_STEP
             << " " << s.residual_sum / denom / (double)ADJ_STEP
             << " " << s.residual_abs_sum / denom / (double)ADJ_STEP
             << " " << std::sqrt(s.residual_square_sum / denom) / (double)ADJ_STEP
             << " " << s.residual_min / (double)ADJ_STEP
             << " " << s.residual_max / (double)ADJ_STEP
             << " " << s.residual_max_abs / (double)ADJ_STEP
             << "\n";
    }
    if (total.count > 0) {
        const double denom = (double)total.count;
        *out << "total"
             << " " << total.count
             << " " << total.score_sum / denom
             << " " << total.linear_raw_sum / denom / (double)ADJ_STEP
             << " " << total.residual_sum / denom / (double)ADJ_STEP
             << " " << total.residual_abs_sum / denom / (double)ADJ_STEP
             << " " << std::sqrt(total.residual_square_sum / denom) / (double)ADJ_STEP
             << " " << total.residual_min / (double)ADJ_STEP
             << " " << total.residual_max / (double)ADJ_STEP
             << " " << total.residual_max_abs / (double)ADJ_STEP
             << "\n";
    }
    return 0;
}
