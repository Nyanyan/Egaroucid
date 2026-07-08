/*
    Egaroucid Project

    @file current_fm_tri_type_zero_vector_stats.cpp
        Count zero and non-zero FM vectors in a current-model tri-type FM evaluation.
    @date 2026
    @author Takuto Yamana, Codex
    @license GPL-3.0-or-later
*/

#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

#include "../../../engine/common.hpp"
#include "../../../engine/evaluate_common.hpp"
#include "../../../engine/evaluate_experiment_current_fm_tri_type_simdopt_common.hpp"

struct BankStats {
    uint64_t vectors = 0;
    uint64_t nonzero_vectors = 0;
    uint64_t entries = 0;
    uint64_t nonzero_entries = 0;
};

BankStats calc_bank_stats(const CurrentFmTriFile &file, const int phase, const int param_begin, const int param_end) {
    BankStats stats;
    const std::vector<int8_t> &vectors = file.vector_quant_by_phase[phase];
    stats.vectors = (uint64_t)(param_end - param_begin);
    stats.entries = stats.vectors * (uint64_t)file.dim;
    if (vectors.empty()) {
        return stats;
    }
    for (int param = param_begin; param < param_end; ++param) {
        bool vector_nonzero = false;
        const size_t base = (size_t)param * file.dim;
        for (int d = 0; d < file.dim; ++d) {
            const bool nonzero = vectors[base + (size_t)d] != 0;
            vector_nonzero = vector_nonzero || nonzero;
            stats.nonzero_entries += nonzero ? 1 : 0;
        }
        stats.nonzero_vectors += vector_nonzero ? 1 : 0;
    }
    return stats;
}

void print_stats(const char *bank, const int phase, const char *range_name, const BankStats &stats) {
    const double vector_ratio = stats.vectors == 0
        ? 0.0
        : (double)stats.nonzero_vectors / (double)stats.vectors;
    const double entry_ratio = stats.entries == 0
        ? 0.0
        : (double)stats.nonzero_entries / (double)stats.entries;
    std::cout << bank << '\t'
              << phase << '\t'
              << range_name << '\t'
              << stats.vectors << '\t'
              << stats.nonzero_vectors << '\t'
              << std::fixed << std::setprecision(6) << vector_ratio << '\t'
              << stats.entries << '\t'
              << stats.nonzero_entries << '\t'
              << std::fixed << std::setprecision(6) << entry_ratio
              << '\n';
}

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "usage: current_fm_tri_type_zero_vector_stats <cross.egev4@same.egev4@count.egev4@cross_weight@same_weight@count_weight> [phase_begin] [phase_end_exclusive]" << std::endl;
        return 1;
    }
    const std::string eval_spec = argv[1];
    const int phase_begin = argc >= 3 ? std::atoi(argv[2]) : 0;
    const int phase_end = argc >= 4 ? std::atoi(argv[3]) : N_PHASES;
    if (phase_begin < 0 || phase_end < phase_begin || N_PHASES < phase_end) {
        std::cerr << "invalid phase range" << std::endl;
        return 1;
    }
    if (!current_fm_load_egev4(eval_spec.c_str(), true)) {
        return 1;
    }

    std::cout << "bank\tphase\trange\tvectors\tnonzero_vectors\tnonzero_vector_ratio\tentries\tnonzero_entries\tnonzero_entry_ratio\n";
    for (int phase = phase_begin; phase < phase_end; ++phase) {
        print_stats("cross", phase, "pattern", calc_bank_stats(current_fm_cross_file, phase, 0, CURRENT_FM_N_PATTERN_PARAMS_RAW));
        print_stats("same", phase, "pattern", calc_bank_stats(current_fm_same_file, phase, 0, CURRENT_FM_N_PATTERN_PARAMS_RAW));
        print_stats("count", phase, "pattern", calc_bank_stats(current_fm_count_file, phase, 0, CURRENT_FM_N_PATTERN_PARAMS_RAW));
        print_stats("count", phase, "stone_count", calc_bank_stats(current_fm_count_file, phase, CURRENT_FM_N_PATTERN_PARAMS_RAW, CURRENT_FM_N_PARAMS_PER_PHASE));
    }
    return 0;
}
