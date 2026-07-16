/*
    Egaroucid Project

    @file eval77_fm_grouped_error_check.cpp
        Compare current EGEV4 7.7-FM scores against grouped EGEV10 scores on
        random-play positions, using the same active feature IDs for both models.
    @date 2026
    @author Takuto Yamana, Codex
    @license GPL-3.0-or-later
*/

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
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

namespace {

constexpr int COMPARE_GROUPED_VERSION = 10;
constexpr size_t EGEV4_FIXED_HEADER_SIZE = 34;
constexpr size_t EGEV4_SCALES_SIZE = sizeof(float) * N_PHASES * 2;
constexpr size_t EGEV4_PAYLOAD_OFFSET = EGEV4_FIXED_HEADER_SIZE + EGEV4_SCALES_SIZE;
constexpr size_t GROUPED_ALIGNMENT = 64;

struct CompareModel {
    Eval77FmMappedFile mapped;
    bool grouped = false;
    int dim = 0;
    int group_count = 0;
    std::array<uint8_t, N_PHASES> phase_to_group{};
    std::array<double, N_PHASES> linear_scale{};
    std::array<double, N_PHASES> interaction_scale{};
    const unsigned char *linear_payload = nullptr;
    const unsigned char *vector_payload = nullptr;
    size_t linear_param_stride = 0;
    size_t linear_phase_stride = 0;
    size_t vector_param_stride = 0;
    size_t vector_phase_stride = 0;

    bool load_egev4(const std::string &path) {
        grouped = false;
        group_count = N_PHASES;
        if (!mapped.open_readonly(path)) {
            std::cerr << "[ERROR] cannot memory-map baseline " << path << std::endl;
            return false;
        }
        if (mapped.size < EGEV4_PAYLOAD_OFFSET) {
            std::cerr << "[ERROR] baseline file is too short" << std::endl;
            return false;
        }
        const unsigned char *header = mapped.data;
        if (std::memcmp(header + 14, "EGEV", 4) != 0) {
            std::cerr << "[ERROR] baseline is not an EGEV file" << std::endl;
            return false;
        }
        const int version = eval77_fm_read_i32_le(header + 18);
        const int n_phase = eval77_fm_read_i32_le(header + 22);
        const int n_param = eval77_fm_read_i32_le(header + 26);
        dim = eval77_fm_read_i32_le(header + 30);
        if ((version != 7 && version != 8) || n_phase != N_PHASES ||
            n_param != EVAL77_FM_N_PARAMS_PER_PHASE || dim <= 0 || dim > EVAL77_FM_MAX_DIM) {
            std::cerr << "[ERROR] unsupported baseline EGEV4 shape" << std::endl;
            return false;
        }
        std::array<float, N_PHASES> linear_scales{};
        std::array<float, N_PHASES> vector_scales{};
        std::memcpy(linear_scales.data(), header + EGEV4_FIXED_HEADER_SIZE, sizeof(float) * N_PHASES);
        std::memcpy(vector_scales.data(), header + EGEV4_FIXED_HEADER_SIZE + sizeof(float) * N_PHASES,
                    sizeof(float) * N_PHASES);
        for (int phase = 0; phase < N_PHASES; ++phase) {
            phase_to_group[phase] = (uint8_t)phase;
            linear_scale[phase] = (double)linear_scales[phase];
            interaction_scale[phase] = 0.5 * (double)vector_scales[phase] * (double)vector_scales[phase];
        }
        const size_t record_stride = sizeof(int16_t) + (size_t)dim;
        linear_payload = mapped.data + EGEV4_PAYLOAD_OFFSET;
        vector_payload = linear_payload + sizeof(int16_t);
        linear_param_stride = record_stride;
        vector_param_stride = record_stride;
        linear_phase_stride = (size_t)EVAL77_FM_N_PARAMS_PER_PHASE * record_stride;
        vector_phase_stride = linear_phase_stride;
        const size_t expected_size = EGEV4_PAYLOAD_OFFSET + (size_t)N_PHASES * linear_phase_stride;
        if (mapped.size < expected_size) {
            std::cerr << "[ERROR] baseline payload is truncated" << std::endl;
            return false;
        }
        return true;
    }

    bool load_grouped_egev10(const std::string &path) {
        grouped = true;
        if (!mapped.open_readonly(path)) {
            std::cerr << "[ERROR] cannot memory-map grouped model " << path << std::endl;
            return false;
        }
        constexpr size_t header_base_size = 14 + 4 + sizeof(int32_t) * 5 + N_PHASES;
        if (mapped.size < header_base_size + sizeof(float) * N_PHASES) {
            std::cerr << "[ERROR] grouped file is too short" << std::endl;
            return false;
        }
        const unsigned char *header = mapped.data;
        if (std::memcmp(header + 14, "EGEV", 4) != 0) {
            std::cerr << "[ERROR] grouped model is not an EGEV file" << std::endl;
            return false;
        }
        const int version = eval77_fm_read_i32_le(header + 18);
        const int n_phase = eval77_fm_read_i32_le(header + 22);
        const int n_param = eval77_fm_read_i32_le(header + 26);
        dim = eval77_fm_read_i32_le(header + 30);
        group_count = eval77_fm_read_i32_le(header + 34);
        if (version != COMPARE_GROUPED_VERSION || n_phase != N_PHASES ||
            n_param != EVAL77_FM_N_PARAMS_PER_PHASE || dim <= 0 || dim > EVAL77_FM_MAX_DIM ||
            group_count <= 0 || group_count > N_PHASES) {
            std::cerr << "[ERROR] unsupported grouped EGEV10 shape" << std::endl;
            return false;
        }
        std::memcpy(phase_to_group.data(), header + 38, N_PHASES);
        for (int phase = 0; phase < N_PHASES; ++phase) {
            if (phase_to_group[phase] >= group_count) {
                std::cerr << "[ERROR] grouped phase_to_group is out of range" << std::endl;
                return false;
            }
        }
        const size_t header_size = header_base_size + sizeof(float) * N_PHASES +
            sizeof(float) * (size_t)group_count;
        std::array<float, N_PHASES> linear_scales{};
        std::vector<float> group_vector_scales((size_t)group_count);
        std::memcpy(linear_scales.data(), header + header_base_size, sizeof(float) * N_PHASES);
        std::memcpy(group_vector_scales.data(), header + header_base_size + sizeof(float) * N_PHASES,
                    sizeof(float) * (size_t)group_count);
        for (int phase = 0; phase < N_PHASES; ++phase) {
            const double vector_scale = (double)group_vector_scales[phase_to_group[phase]];
            linear_scale[phase] = (double)linear_scales[phase];
            interaction_scale[phase] = 0.5 * vector_scale * vector_scale;
        }
        const size_t linear_offset = eval77_fm_align_up_size(header_size, GROUPED_ALIGNMENT);
        linear_param_stride = sizeof(int16_t);
        linear_phase_stride = (size_t)EVAL77_FM_N_PARAMS_PER_PHASE * sizeof(int16_t);
        vector_param_stride = eval77_fm_align_up_size((size_t)dim, 16);
        vector_phase_stride = (size_t)EVAL77_FM_N_PARAMS_PER_PHASE * vector_param_stride;
        const size_t vector_offset = eval77_fm_align_up_size(
            linear_offset + (size_t)N_PHASES * linear_phase_stride,
            GROUPED_ALIGNMENT
        );
        const size_t expected_size = vector_offset + (size_t)group_count * vector_phase_stride;
        if (mapped.size < expected_size) {
            std::cerr << "[ERROR] grouped payload is truncated" << std::endl;
            return false;
        }
        linear_payload = mapped.data + linear_offset;
        vector_payload = mapped.data + vector_offset;
        return true;
    }

    int score(const int phase_idx, const int active_ids[], const int n_active) const {
        const int vector_group = grouped ? (int)phase_to_group[phase_idx] : phase_idx;
        const unsigned char *linear_phase = linear_payload + (size_t)phase_idx * linear_phase_stride;
        const unsigned char *vector_phase = vector_payload + (size_t)vector_group * vector_phase_stride;
        int32_t linear_quant = 0;
        std::array<int32_t, EVAL77_FM_MAX_DIM> sum{};
        std::array<int32_t, EVAL77_FM_MAX_DIM> sum_sq{};
        for (int i = 0; i < n_active; ++i) {
            const int id = active_ids[i];
            int16_t q;
            std::memcpy(&q, linear_phase + (size_t)id * linear_param_stride, sizeof(q));
            linear_quant += q;
            const int8_t *vector = (const int8_t*)(vector_phase + (size_t)id * vector_param_stride);
            for (int dim_idx = 0; dim_idx < dim; ++dim_idx) {
                const int32_t v = vector[dim_idx];
                sum[(size_t)dim_idx] += v;
                sum_sq[(size_t)dim_idx] += v * v;
            }
        }
        int64_t interaction_quant = 0;
        for (int dim_idx = 0; dim_idx < dim; ++dim_idx) {
            const int64_t s = sum[(size_t)dim_idx];
            interaction_quant += s * s - sum_sq[(size_t)dim_idx];
        }
        const double value = (double)linear_quant * linear_scale[phase_idx] +
            (double)interaction_quant * interaction_scale[phase_idx];
        const int rounded = value >= 0.0 ? (int)(value + 0.5) : (int)(value - 0.5);
        return std::clamp(rounded, -SCORE_MAX, SCORE_MAX);
    }
};

struct ErrorStats {
    uint64_t n = 0;
    int64_t sum_diff = 0;
    uint64_t sum_abs = 0;
    long double sum_sq = 0.0;
    int max_abs = 0;
    int max_base = 0;
    int max_grouped = 0;

    void add(const int base_score, const int grouped_score) {
        const int diff = grouped_score - base_score;
        const int abs_diff = std::abs(diff);
        ++n;
        sum_diff += diff;
        sum_abs += (uint64_t)abs_diff;
        sum_sq += (long double)diff * (long double)diff;
        if (abs_diff > max_abs) {
            max_abs = abs_diff;
            max_base = base_score;
            max_grouped = grouped_score;
        }
    }

    double bias() const {
        return n == 0 ? 0.0 : (double)sum_diff / (double)n;
    }

    double mae() const {
        return n == 0 ? 0.0 : (double)sum_abs / (double)n;
    }

    double rmse() const {
        return n == 0 ? 0.0 : std::sqrt((double)(sum_sq / (long double)n));
    }
};

int pick_nth_bit(uint64_t bits, int nth) {
    for (int pos = first_bit(&bits); bits; pos = next_bit(&bits)) {
        if (nth == 0) {
            return pos;
        }
        --nth;
    }
    return -1;
}

void print_stats_line(const std::string &label, const ErrorStats &stats) {
    std::cout << label
              << " n=" << stats.n
              << " bias=" << std::fixed << std::setprecision(4) << stats.bias()
              << " mae=" << stats.mae()
              << " rmse=" << stats.rmse()
              << " max_abs=" << stats.max_abs
              << " max_pair=" << stats.max_base << '/' << stats.max_grouped
              << std::endl;
}

} // namespace

int main(int argc, char **argv) {
    if (argc < 3 || argc > 5) {
        std::cerr << "usage: eval77_fm_grouped_error_check <baseline.egev4> <grouped.egev10> [games] [seed]" << std::endl;
        return 1;
    }

    const std::string baseline_file = argv[1];
    const std::string grouped_file = argv[2];
    const int n_games = argc >= 4 ? std::atoi(argv[3]) : 2000;
    const uint32_t seed = argc >= 5 ? (uint32_t)std::strtoul(argv[4], nullptr, 10) : 20260716U;
    if (n_games <= 0) {
        std::cerr << "games must be positive" << std::endl;
        return 1;
    }

    CompareModel baseline;
    CompareModel grouped;
    if (!baseline.load_egev4(baseline_file) || !grouped.load_grouped_egev10(grouped_file)) {
        return 1;
    }
    if (baseline.dim != grouped.dim) {
        std::cerr << "[ERROR] dim mismatch baseline=" << baseline.dim << " grouped=" << grouped.dim << std::endl;
        return 1;
    }

    bit_init();
    mobility_init();
    flip_init();
    last_flip_init();
    pre_calculate_eval_constant();
    if (!eval77_fm_fast_load_file(baseline_file.c_str(), false)) {
        return 1;
    }

    std::mt19937 rng(seed);
    ErrorStats overall;
    std::array<ErrorStats, N_PHASES> by_phase{};
    uint64_t sign_disagree = 0;
    uint64_t exact_match = 0;
    uint64_t engine_mismatches = 0;
    int active_ids[N_PATTERN_FEATURES + 1];

    for (int game = 0; game < n_games; ++game) {
        Board board;
        board.reset();
        Search search(&board);
        bool previous_pass = false;
        while (search.board.n_discs() < HW2) {
            const int phase_idx = search.phase();
            const int num0 = pop_count_ull(search.board.player);
            int n_active = 0;
            collect_eval77_fm_fast_simd_active_ids(
                &search.eval.features[search.eval.feature_idx],
                num0,
                active_ids,
                n_active
            );
            const int base_score = baseline.score(phase_idx, active_ids, n_active);
            const int grouped_score = grouped.score(phase_idx, active_ids, n_active);
            const int engine_score = calc_pattern(
                phase_idx,
                &search.eval.features[search.eval.feature_idx],
                num0
            );
            if (base_score != engine_score) {
                if (engine_mismatches < 10) {
                    std::cerr << "engine_mismatch game=" << game
                              << " discs=" << search.board.n_discs()
                              << " phase=" << phase_idx
                              << " custom=" << base_score
                              << " engine=" << engine_score
                              << std::endl;
                }
                ++engine_mismatches;
            }
            overall.add(base_score, grouped_score);
            by_phase[(size_t)phase_idx].add(base_score, grouped_score);
            exact_match += base_score == grouped_score;
            sign_disagree += (base_score < 0 && grouped_score > 0) ||
                (base_score > 0 && grouped_score < 0);

            if (search.board.n_discs() >= HW2 - 1) {
                break;
            }
            uint64_t legal = search.board.get_legal();
            if (legal == 0ULL) {
                if (previous_pass) {
                    break;
                }
                search.pass();
                previous_pass = true;
                continue;
            }
            previous_pass = false;
            const int move = pick_nth_bit(legal, (int)(rng() % (uint32_t)pop_count_ull(legal)));
            Flip flip;
            calc_flip(&flip, &search.board, move);
            search.move(&flip);
        }
    }

    std::cout << "baseline=" << baseline_file
              << " grouped=" << grouped_file
              << " games=" << n_games
              << " seed=" << seed
              << " positions=" << overall.n
              << " exact_match=" << exact_match
              << " exact_rate=" << std::fixed << std::setprecision(6)
              << (overall.n == 0 ? 0.0 : (double)exact_match / (double)overall.n)
              << " sign_disagree=" << sign_disagree
              << " sign_disagree_rate="
              << (overall.n == 0 ? 0.0 : (double)sign_disagree / (double)overall.n)
              << " engine_mismatches=" << engine_mismatches
              << std::endl;
    print_stats_line("overall", overall);
    std::cout << "phase_stats phase n bias mae rmse max_abs max_pair" << std::endl;
    for (int phase = 0; phase < N_PHASES; ++phase) {
        const ErrorStats &stats = by_phase[(size_t)phase];
        if (stats.n == 0) {
            continue;
        }
        std::cout << "phase " << phase
                  << " n=" << stats.n
                  << " bias=" << std::fixed << std::setprecision(4) << stats.bias()
                  << " mae=" << stats.mae()
                  << " rmse=" << stats.rmse()
                  << " max_abs=" << stats.max_abs
                  << " max_pair=" << stats.max_base << '/' << stats.max_grouped
                  << std::endl;
    }
    return engine_mismatches == 0 ? 0 : 2;
}
