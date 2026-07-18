/*
    Select pattern types for a partial 7.7 FM interaction by greedy removal.
*/

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr int N_PHASES = 60;
constexpr int N_PATTERN_TYPES = 16;
constexpr int N_PATTERN_FEATURES = 64;
constexpr int N_FEATURES = 65;
constexpr int N_PATTERN_VALUES = 59049;
constexpr int N_PATTERN_PARAMS_RAW = N_PATTERN_TYPES * N_PATTERN_VALUES;
constexpr int N_PARAMS_PER_PHASE = N_PATTERN_PARAMS_RAW + 65;
constexpr int RECORD_BYTES = 136;
constexpr int SCORE_MAX = 64;
constexpr int MAX_DIM = 16;

struct Model {
    int dim = 0;
    size_t record_stride = 0;
    size_t phase_stride = 0;
    size_t payload_offset = 0;
    std::array<float, N_PHASES> linear_scale{};
    std::array<float, N_PHASES> vector_scale{};
    std::vector<unsigned char> bytes;
};

struct Sample {
    double target = 0.0;
    double linear = 0.0;
    std::array<std::array<float, MAX_DIM>, N_PATTERN_TYPES + 1> sums{};
    std::array<std::array<float, MAX_DIM>, N_PATTERN_TYPES + 1> sums_sq{};
};

int32_t read_i32(const unsigned char *ptr) {
    int32_t value = 0;
    std::memcpy(&value, ptr, sizeof(value));
    return value;
}

Model load_model(const fs::path &path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("cannot open model: " + path.string());
    }
    const std::streamsize size = input.tellg();
    if (size < 514) {
        throw std::runtime_error("model is too short");
    }
    input.seekg(0);

    Model model;
    model.bytes.resize(static_cast<size_t>(size));
    input.read(
        reinterpret_cast<char *>(model.bytes.data()),
        static_cast<std::streamsize>(model.bytes.size())
    );
    if (!input) {
        throw std::runtime_error("cannot read model");
    }

    const unsigned char *header = model.bytes.data();
    if (std::memcmp(header + 14, "EGEV", 4) != 0) {
        throw std::runtime_error("input is not an EGEV file");
    }
    const int version = read_i32(header + 18);
    const int n_phases = read_i32(header + 22);
    const int n_params = read_i32(header + 26);
    model.dim = read_i32(header + 30);
    if (
        (version != 7 && version != 8) ||
        n_phases != N_PHASES ||
        n_params != N_PARAMS_PER_PHASE ||
        model.dim <= 0 ||
        MAX_DIM < model.dim
    ) {
        throw std::runtime_error("unsupported EGEV shape");
    }

    std::memcpy(
        model.linear_scale.data(),
        header + 34,
        sizeof(float) * N_PHASES
    );
    std::memcpy(
        model.vector_scale.data(),
        header + 34 + sizeof(float) * N_PHASES,
        sizeof(float) * N_PHASES
    );
    model.payload_offset = 34 + sizeof(float) * N_PHASES * 2;
    model.record_stride = sizeof(int16_t) + static_cast<size_t>(model.dim);
    model.phase_stride =
        static_cast<size_t>(N_PARAMS_PER_PHASE) * model.record_stride;
    const size_t expected =
        model.payload_offset + static_cast<size_t>(N_PHASES) * model.phase_stride;
    if (model.bytes.size() < expected) {
        throw std::runtime_error("model payload is truncated");
    }
    return model;
}

fs::path data_path(const fs::path &root, int phase, int file_id) {
    return root / std::to_string(phase) /
        ("records" + std::to_string(file_id) + ".dat");
}

void add_feature(
    Sample &sample,
    const unsigned char *phase_payload,
    const Model &model,
    int phase,
    int param_id,
    int type
) {
    const unsigned char *record =
        phase_payload + static_cast<size_t>(param_id) * model.record_stride;
    int16_t linear_quant = 0;
    std::memcpy(&linear_quant, record, sizeof(linear_quant));
    sample.linear +=
        static_cast<double>(linear_quant) *
        static_cast<double>(model.linear_scale[phase]);

    const int8_t *vector = reinterpret_cast<const int8_t *>(
        record + sizeof(int16_t)
    );
    const float scale = model.vector_scale[phase];
    for (int dim = 0; dim < model.dim; ++dim) {
        const float value = static_cast<float>(vector[dim]) * scale;
        sample.sums[type][dim] += value;
        sample.sums_sq[type][dim] += value * value;
    }
}

std::vector<Sample> load_samples(
    const Model &model,
    const fs::path &root,
    int records_per_phase,
    int phase_min,
    int phase_max,
    int file_id
) {
    std::vector<Sample> samples;
    samples.reserve(
        static_cast<size_t>(phase_max - phase_min + 1) *
        static_cast<size_t>(records_per_phase)
    );
    std::array<unsigned char, RECORD_BYTES> raw{};

    for (int phase = phase_min; phase <= phase_max; ++phase) {
        const fs::path path = data_path(root, phase, file_id);
        std::ifstream input(path, std::ios::binary);
        if (!input) {
            throw std::runtime_error("cannot open data: " + path.string());
        }
        const unsigned char *phase_payload =
            model.bytes.data() + model.payload_offset +
            static_cast<size_t>(phase) * model.phase_stride;
        int loaded = 0;
        while (loaded < records_per_phase) {
            input.read(
                reinterpret_cast<char *>(raw.data()),
                static_cast<std::streamsize>(raw.size())
            );
            if (!input) {
                break;
            }
            Sample sample;
            int16_t target = 0;
            std::memcpy(
                &target,
                raw.data() + 4 + N_FEATURES * sizeof(uint16_t),
                sizeof(target)
            );
            sample.target = target;
            for (int feature_idx = 0;
                 feature_idx < N_PATTERN_FEATURES;
                 ++feature_idx) {
                uint16_t feature = 0;
                std::memcpy(
                    &feature,
                    raw.data() + 4 + feature_idx * sizeof(uint16_t),
                    sizeof(feature)
                );
                const int type = feature_idx / 4;
                const int param_id = type * N_PATTERN_VALUES + feature;
                add_feature(
                    sample,
                    phase_payload,
                    model,
                    phase,
                    param_id,
                    type
                );
            }
            uint16_t stone_count = 0;
            std::memcpy(
                &stone_count,
                raw.data() + 4 +
                    N_PATTERN_FEATURES * sizeof(uint16_t),
                sizeof(stone_count)
            );
            add_feature(
                sample,
                phase_payload,
                model,
                phase,
                N_PATTERN_PARAMS_RAW + stone_count,
                N_PATTERN_TYPES
            );
            samples.push_back(std::move(sample));
            ++loaded;
        }
        std::cout << "phase=" << phase << " samples=" << loaded << '\n';
    }
    return samples;
}

double predict(const Sample &sample, uint32_t mask, int dim) {
    double interaction = 0.0;
    for (int d = 0; d < dim; ++d) {
        double sum = sample.sums[N_PATTERN_TYPES][d];
        double sum_sq = sample.sums_sq[N_PATTERN_TYPES][d];
        for (int type = 0; type < N_PATTERN_TYPES; ++type) {
            if ((mask >> type) & 1u) {
                sum += sample.sums[type][d];
                sum_sq += sample.sums_sq[type][d];
            }
        }
        interaction += sum * sum - sum_sq;
    }
    return std::clamp(
        static_cast<double>(std::llround(sample.linear + 0.5 * interaction)),
        static_cast<double>(-SCORE_MAX),
        static_cast<double>(SCORE_MAX)
    );
}

std::pair<double, double> metrics(
    const std::vector<Sample> &samples,
    uint32_t mask,
    int dim
) {
    double absolute_error = 0.0;
    double squared_error = 0.0;
    for (const Sample &sample : samples) {
        const double error = sample.target - predict(sample, mask, dim);
        absolute_error += std::abs(error);
        squared_error += error * error;
    }
    const double count = static_cast<double>(samples.size());
    return {
        absolute_error / count,
        std::sqrt(squared_error / count)
    };
}

} // namespace

int main(int argc, char **argv) {
    try {
        if (argc < 3 || 8 < argc) {
            std::cerr
                << "usage: select_eval77_fm_subset_mask "
                << "<eval.egev4> <data_root> [records_per_phase=1000] "
                << "[phase_min=6] [phase_max=59] [target_types=8] "
                << "[file_id=3]\n";
            return 1;
        }
        const int records_per_phase = argc >= 4 ? std::stoi(argv[3]) : 1000;
        const int phase_min = argc >= 5 ? std::stoi(argv[4]) : 6;
        const int phase_max = argc >= 6 ? std::stoi(argv[5]) : 59;
        const int target_types = argc >= 7 ? std::stoi(argv[6]) : 8;
        const int file_id = argc >= 8 ? std::stoi(argv[7]) : 3;
        if (
            records_per_phase <= 0 ||
            phase_min < 0 ||
            phase_max < phase_min ||
            N_PHASES <= phase_max ||
            target_types < 0 ||
            N_PATTERN_TYPES < target_types ||
            file_id < 0
        ) {
            throw std::runtime_error("invalid numeric argument");
        }

        const Model model = load_model(argv[1]);
        const std::vector<Sample> samples = load_samples(
            model,
            argv[2],
            records_per_phase,
            phase_min,
            phase_max,
            file_id
        );
        if (samples.empty()) {
            throw std::runtime_error("no samples loaded");
        }

        for (int type = 0; type < N_PATTERN_TYPES; ++type) {
            const uint32_t candidate = 0xFFFFu & ~(1u << type);
            const auto [mae, rmse] = metrics(
                samples,
                candidate,
                model.dim
            );
            std::cout
                << "single_removed_type=" << type
                << " mask=0x" << std::hex << std::setw(4)
                << std::setfill('0') << candidate << std::dec
                << std::setfill(' ')
                << " mae=" << std::setprecision(10) << mae
                << " rmse=" << rmse
                << '\n';
        }

        uint32_t mask = 0xFFFFu;
        for (int count = N_PATTERN_TYPES; count >= target_types; --count) {
            const auto [mae, rmse] = metrics(samples, mask, model.dim);
            std::cout
                << "selected_types=" << count
                << " mask=0x" << std::hex << std::setw(4)
                << std::setfill('0') << mask << std::dec
                << std::setfill(' ')
                << " mae=" << std::setprecision(10) << mae
                << " rmse=" << rmse
                << '\n';
            if (count == target_types) {
                break;
            }

            double best_mae = std::numeric_limits<double>::infinity();
            double best_rmse = std::numeric_limits<double>::infinity();
            int best_type = -1;
            for (int type = 0; type < N_PATTERN_TYPES; ++type) {
                if (((mask >> type) & 1u) == 0) {
                    continue;
                }
                const uint32_t candidate = mask & ~(1u << type);
                const auto [candidate_mae, candidate_rmse] =
                    metrics(samples, candidate, model.dim);
                if (
                    candidate_mae < best_mae ||
                    (
                        candidate_mae == best_mae &&
                        candidate_rmse < best_rmse
                    )
                ) {
                    best_mae = candidate_mae;
                    best_rmse = candidate_rmse;
                    best_type = type;
                }
            }
            mask &= ~(1u << best_type);
            std::cout << "removed_type=" << best_type << '\n';
        }
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
