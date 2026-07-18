/*
    Egaroucid Project

    @file project_egev4_fm_pca_experiment.cpp
        Reduce EGEV4 FM dimensions by an independent uncentered principal
        component projection for each phase.
    @date 2026
    @author Takuto Yamana, Codex
    @license GPL-3.0-or-later
*/

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <immintrin.h>

namespace fs = std::filesystem;

constexpr int TIMESTAMP_SIZE = 14;
constexpr int N_PHASES = 60;
constexpr int VERSION_LINEAR_FM_INT16_INT8 = 8;
constexpr int MAX_FM_DIM = 16;
constexpr size_t COVARIANCE_SAMPLE_LIMIT = 65536;
constexpr size_t QUANTIZATION_SAMPLE_LIMIT = 1048576;

struct Config {
    fs::path input;
    fs::path output;
    fs::path summary;
    std::string timestamp;
    int output_dim = 0;
    int quant_max = 127;
    int threads = 0;
    bool balance_components = false;
    bool identity_basis = false;
};

struct Egev4Info {
    std::vector<uint8_t> header;
    std::string input_timestamp;
    std::string output_timestamp;
    int version = 0;
    int n_phases = 0;
    int params_per_phase = 0;
    int input_dim = 0;
    size_t fixed_header_size = 0;
    size_t payload_offset = 0;
    size_t input_record_stride = 0;
    size_t input_phase_stride = 0;
    uint64_t expected_input_size = 0;
    std::array<float, N_PHASES> input_vector_scales{};
    std::array<float, N_PHASES> output_vector_scales{};
};

struct PcaBasis {
    alignas(32) float by_input[MAX_FM_DIM][MAX_FM_DIM]{};
    std::array<double, MAX_FM_DIM> selected_eigenvalues{};
    double retained_energy_ratio = 0.0;
};

struct PhaseSummary {
    int phase = 0;
    float input_vector_scale = 0.0F;
    float output_vector_scale = 0.0F;
    double quantization_step = 0.0;
    double retained_energy_ratio = 0.0;
    double projected_quantization_mse = 0.0;
    double saturation_fraction = 0.0;
    std::array<double, MAX_FM_DIM> selected_eigenvalues{};
};

[[noreturn]] void usage_error(const std::string &message) {
    throw std::runtime_error(
        message +
        "\nusage: project_egev4_fm_pca_experiment"
        " --input FILE --output FILE --output-dim N"
        " [--summary FILE] [--timestamp now|YYYYMMDDhhmmss] [--threads N]"
        " [--quant-max N] [--balance-components|--identity-basis]"
    );
}

std::string require_value(int &index, int argc, char **argv) {
    if (index + 1 >= argc) {
        usage_error(std::string("missing value for ") + argv[index]);
    }
    return argv[++index];
}

Config parse_args(int argc, char **argv) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--input") {
            cfg.input = require_value(i, argc, argv);
        } else if (arg == "--output") {
            cfg.output = require_value(i, argc, argv);
        } else if (arg == "--output-dim") {
            cfg.output_dim = std::stoi(require_value(i, argc, argv));
        } else if (arg == "--quant-max") {
            cfg.quant_max = std::stoi(require_value(i, argc, argv));
        } else if (arg == "--summary") {
            cfg.summary = require_value(i, argc, argv);
        } else if (arg == "--timestamp") {
            cfg.timestamp = require_value(i, argc, argv);
        } else if (arg == "--threads") {
            cfg.threads = std::stoi(require_value(i, argc, argv));
        } else if (arg == "--balance-components") {
            cfg.balance_components = true;
        } else if (arg == "--identity-basis") {
            cfg.identity_basis = true;
        } else if (arg == "--help" || arg == "-h") {
            usage_error("");
        } else {
            usage_error("unknown argument: " + arg);
        }
    }
    if (cfg.input.empty()) {
        usage_error("--input is required");
    }
    if (cfg.output.empty()) {
        usage_error("--output is required");
    }
    if (cfg.output_dim <= 0 || cfg.output_dim > MAX_FM_DIM) {
        usage_error("--output-dim must be between 1 and 16");
    }
    if (cfg.quant_max <= 0 || cfg.quant_max > 127) {
        usage_error("--quant-max must be between 1 and 127");
    }
    if (cfg.balance_components && cfg.identity_basis) {
        usage_error("--balance-components and --identity-basis are mutually exclusive");
    }
    if (cfg.threads < 0) {
        usage_error("--threads must not be negative");
    }
    if (!cfg.timestamp.empty() && cfg.timestamp != "now") {
        if (cfg.timestamp.size() != TIMESTAMP_SIZE ||
            !std::all_of(cfg.timestamp.begin(), cfg.timestamp.end(), [](unsigned char c) {
                return std::isdigit(c);
            })) {
            usage_error("--timestamp must be 'now' or 14 digits");
        }
    }
    return cfg;
}

template<typename T>
T read_value(const std::vector<uint8_t> &data, size_t offset) {
    if (offset + sizeof(T) > data.size()) {
        throw std::runtime_error("read past end of EGEV4 header");
    }
    T value{};
    std::memcpy(&value, data.data() + offset, sizeof(value));
    return value;
}

std::string make_timestamp_now() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t time = std::chrono::system_clock::to_time_t(now);
    std::tm local_time{};
#ifdef _WIN32
    localtime_s(&local_time, &time);
#else
    localtime_r(&time, &local_time);
#endif
    std::ostringstream out;
    out << std::put_time(&local_time, "%Y%m%d%H%M%S");
    return out.str();
}

Egev4Info read_egev4_info(
    std::ifstream &input,
    const fs::path &path,
    const std::string &timestamp
) {
    Egev4Info info;
    info.fixed_header_size = TIMESTAMP_SIZE + 4 + sizeof(int32_t) * 4;
    info.payload_offset = info.fixed_header_size + sizeof(float) * N_PHASES * 2;
    info.header.resize(info.payload_offset);
    input.read(
        reinterpret_cast<char *>(info.header.data()),
        static_cast<std::streamsize>(info.header.size())
    );
    if (!input) {
        throw std::runtime_error("EGEV4 input is too short for its header");
    }

    info.input_timestamp.assign(
        reinterpret_cast<const char *>(info.header.data()),
        TIMESTAMP_SIZE
    );
    if (std::memcmp(info.header.data() + TIMESTAMP_SIZE, "EGEV", 4) != 0) {
        throw std::runtime_error("unexpected EGEV4 magic");
    }
    const size_t fields_offset = TIMESTAMP_SIZE + 4;
    info.version = read_value<int32_t>(info.header, fields_offset);
    info.n_phases = read_value<int32_t>(info.header, fields_offset + 4);
    info.params_per_phase = read_value<int32_t>(info.header, fields_offset + 8);
    info.input_dim = read_value<int32_t>(info.header, fields_offset + 12);
    if (info.version != VERSION_LINEAR_FM_INT16_INT8) {
        throw std::runtime_error("unsupported EGEV4 version: " + std::to_string(info.version));
    }
    if (info.n_phases != N_PHASES || info.params_per_phase <= 0 ||
        info.input_dim <= 0 || info.input_dim > MAX_FM_DIM) {
        throw std::runtime_error(
            "unsupported EGEV4 shape: phases=" + std::to_string(info.n_phases) +
            " params_per_phase=" + std::to_string(info.params_per_phase) +
            " dim=" + std::to_string(info.input_dim)
        );
    }

    const size_t vector_scales_offset = info.fixed_header_size + sizeof(float) * N_PHASES;
    std::memcpy(
        info.input_vector_scales.data(),
        info.header.data() + vector_scales_offset,
        sizeof(info.input_vector_scales)
    );
    info.output_vector_scales = info.input_vector_scales;
    info.output_timestamp =
        timestamp == "now" ? make_timestamp_now() :
        (!timestamp.empty() ? timestamp : info.input_timestamp);
    std::memcpy(info.header.data(), info.output_timestamp.data(), TIMESTAMP_SIZE);

    info.input_record_stride = sizeof(int16_t) + static_cast<size_t>(info.input_dim);
    info.input_phase_stride =
        static_cast<size_t>(info.params_per_phase) * info.input_record_stride;
    info.expected_input_size =
        static_cast<uint64_t>(info.payload_offset) +
        static_cast<uint64_t>(info.n_phases) * info.input_phase_stride;
    const uint64_t actual_size = fs::file_size(path);
    if (actual_size != info.expected_input_size) {
        throw std::runtime_error(
            "EGEV4 input size mismatch: got " + std::to_string(actual_size) +
            " expected " + std::to_string(info.expected_input_size)
        );
    }
    return info;
}

int choose_thread_count(const Config &cfg, size_t work_items) {
    unsigned int detected = std::thread::hardware_concurrency();
    int threads = cfg.threads > 0 ? cfg.threads :
        static_cast<int>(detected == 0 ? 1 : detected);
    threads = std::clamp(threads, 1, 32);
    return std::min<int>(threads, static_cast<int>(std::max<size_t>(1, work_items)));
}

template<typename Function>
void parallel_ranges(size_t size, int thread_count, Function function) {
    if (thread_count <= 1 || size < static_cast<size_t>(thread_count)) {
        function(0, size, 0);
        return;
    }
    std::vector<std::thread> threads;
    threads.reserve(static_cast<size_t>(thread_count));
    for (int thread = 0; thread < thread_count; ++thread) {
        const size_t begin = size * static_cast<size_t>(thread) /
            static_cast<size_t>(thread_count);
        const size_t end = size * static_cast<size_t>(thread + 1) /
            static_cast<size_t>(thread_count);
        threads.emplace_back([=, &function] {
            function(begin, end, thread);
        });
    }
    for (std::thread &thread : threads) {
        thread.join();
    }
}

using Matrix = std::array<std::array<double, MAX_FM_DIM>, MAX_FM_DIM>;

Matrix estimate_uncentered_gram(
    const std::vector<uint8_t> &phase_data,
    size_t record_stride,
    size_t n_records,
    int input_dim
) {
    Matrix gram{};
    const size_t sample_count = std::min(n_records, COVARIANCE_SAMPLE_LIMIT);
    for (size_t sample = 0; sample < sample_count; ++sample) {
        const size_t record = sample * n_records / sample_count;
        const int8_t *vector = reinterpret_cast<const int8_t *>(
            phase_data.data() + record * record_stride + sizeof(int16_t)
        );
        for (int row = 0; row < input_dim; ++row) {
            const double row_value = vector[row];
            for (int column = row; column < input_dim; ++column) {
                gram[row][column] += row_value * static_cast<double>(vector[column]);
            }
        }
    }
    for (int row = 0; row < input_dim; ++row) {
        for (int column = 0; column < row; ++column) {
            gram[row][column] = gram[column][row];
        }
    }
    return gram;
}

void jacobi_eigendecomposition(
    Matrix matrix,
    int dim,
    std::array<double, MAX_FM_DIM> &eigenvalues,
    Matrix &eigenvectors
) {
    eigenvectors = {};
    for (int i = 0; i < dim; ++i) {
        eigenvectors[i][i] = 1.0;
    }

    constexpr int max_iterations = MAX_FM_DIM * MAX_FM_DIM * 100;
    for (int iteration = 0; iteration < max_iterations; ++iteration) {
        int pivot_row = 0;
        int pivot_column = 1;
        double max_off_diagonal = 0.0;
        for (int row = 0; row < dim; ++row) {
            for (int column = row + 1; column < dim; ++column) {
                const double magnitude = std::abs(matrix[row][column]);
                if (magnitude > max_off_diagonal) {
                    max_off_diagonal = magnitude;
                    pivot_row = row;
                    pivot_column = column;
                }
            }
        }

        double diagonal_scale = 0.0;
        for (int i = 0; i < dim; ++i) {
            diagonal_scale = std::max(diagonal_scale, std::abs(matrix[i][i]));
        }
        if (max_off_diagonal <= std::max(1.0, diagonal_scale) * 1.0e-12) {
            break;
        }

        const int p = pivot_row;
        const int q = pivot_column;
        const double angle = 0.5 * std::atan2(
            2.0 * matrix[p][q],
            matrix[q][q] - matrix[p][p]
        );
        const double cosine = std::cos(angle);
        const double sine = std::sin(angle);

        for (int i = 0; i < dim; ++i) {
            if (i == p || i == q) {
                continue;
            }
            const double old_ip = matrix[i][p];
            const double old_iq = matrix[i][q];
            matrix[i][p] = matrix[p][i] = cosine * old_ip - sine * old_iq;
            matrix[i][q] = matrix[q][i] = sine * old_ip + cosine * old_iq;
        }
        const double old_pp = matrix[p][p];
        const double old_qq = matrix[q][q];
        const double old_pq = matrix[p][q];
        matrix[p][p] =
            cosine * cosine * old_pp -
            2.0 * sine * cosine * old_pq +
            sine * sine * old_qq;
        matrix[q][q] =
            sine * sine * old_pp +
            2.0 * sine * cosine * old_pq +
            cosine * cosine * old_qq;
        matrix[p][q] = matrix[q][p] = 0.0;

        for (int row = 0; row < dim; ++row) {
            const double old_vp = eigenvectors[row][p];
            const double old_vq = eigenvectors[row][q];
            eigenvectors[row][p] = cosine * old_vp - sine * old_vq;
            eigenvectors[row][q] = sine * old_vp + cosine * old_vq;
        }
    }
    eigenvalues.fill(0.0);
    for (int i = 0; i < dim; ++i) {
        eigenvalues[i] = matrix[i][i];
    }
}

PcaBasis make_pca_basis(
    const Matrix &gram,
    int input_dim,
    int output_dim,
    bool balance_components
) {
    std::array<double, MAX_FM_DIM> eigenvalues{};
    Matrix eigenvectors{};
    jacobi_eigendecomposition(gram, input_dim, eigenvalues, eigenvectors);

    std::array<int, MAX_FM_DIM> order{};
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.begin() + input_dim, [&](int lhs, int rhs) {
        return eigenvalues[lhs] > eigenvalues[rhs];
    });

    PcaBasis basis;
    double total_energy = 0.0;
    double retained_energy = 0.0;
    for (int dim = 0; dim < input_dim; ++dim) {
        total_energy += std::max(0.0, eigenvalues[dim]);
    }
    for (int component = 0; component < output_dim; ++component) {
        const int source = order[component];
        basis.selected_eigenvalues[component] = eigenvalues[source];
        retained_energy += std::max(0.0, eigenvalues[source]);
    }
    for (int input = 0; input < input_dim; ++input) {
        for (int output = 0; output < output_dim; ++output) {
            double value = 0.0;
            for (int component = 0; component < output_dim; ++component) {
                const double rotation = balance_components ?
                    (component == 0 ?
                        std::sqrt(1.0 / output_dim) :
                        std::sqrt(2.0 / output_dim) *
                        std::cos(
                            std::acos(-1.0) * component * (output + 0.5) /
                            output_dim
                        )) :
                    (component == output ? 1.0 : 0.0);
                value += eigenvectors[input][order[component]] * rotation;
            }
            basis.by_input[input][output] = static_cast<float>(value);
        }
    }
    basis.retained_energy_ratio =
        total_energy == 0.0 ? 0.0 : retained_energy / total_energy;
    return basis;
}

PcaBasis make_identity_basis(int dim) {
    PcaBasis basis;
    for (int i = 0; i < dim; ++i) {
        basis.by_input[i][i] = 1.0F;
    }
    basis.retained_energy_ratio = 1.0;
    return basis;
}

void project_vectors(
    const std::vector<uint8_t> &phase_data,
    size_t record_stride,
    size_t n_records,
    int input_dim,
    int output_dim,
    const PcaBasis &basis,
    int thread_count,
    std::vector<float> &projected
) {
    projected.assign(n_records * static_cast<size_t>(output_dim), 0.0F);
    parallel_ranges(n_records, thread_count, [&](size_t begin, size_t end, int) {
        for (size_t record = begin; record < end; ++record) {
            const int8_t *vector = reinterpret_cast<const int8_t *>(
                phase_data.data() + record * record_stride + sizeof(int16_t)
            );
            __m256 low = _mm256_setzero_ps();
            __m256 high = _mm256_setzero_ps();
            for (int input = 0; input < input_dim; ++input) {
                const __m256 value = _mm256_set1_ps(static_cast<float>(vector[input]));
                low = _mm256_fmadd_ps(
                    value,
                    _mm256_load_ps(basis.by_input[input]),
                    low
                );
                if (output_dim > 8) {
                    high = _mm256_fmadd_ps(
                        value,
                        _mm256_load_ps(basis.by_input[input] + 8),
                        high
                    );
                }
            }
            float *destination =
                projected.data() + record * static_cast<size_t>(output_dim);
            if (output_dim >= 8) {
                _mm256_storeu_ps(destination, low);
                if (output_dim > 8) {
                    alignas(32) float high_values[8];
                    _mm256_store_ps(high_values, high);
                    std::memcpy(
                        destination + 8,
                        high_values,
                        sizeof(float) * static_cast<size_t>(output_dim - 8)
                    );
                }
            } else {
                alignas(32) float low_values[8];
                _mm256_store_ps(low_values, low);
                std::memcpy(
                    destination,
                    low_values,
                    sizeof(float) * static_cast<size_t>(output_dim)
                );
            }
        }
    });
}

double choose_quantization_step(
    const std::vector<float> &projected,
    const int quant_max
) {
    const size_t sample_count =
        std::min(projected.size(), QUANTIZATION_SAMPLE_LIMIT);
    if (sample_count == 0) {
        return 1.0;
    }
    long double sum_sq = 0.0;
    double max_abs = 0.0;
    for (size_t sample = 0; sample < sample_count; ++sample) {
        const size_t index = sample * projected.size() / sample_count;
        const double value = projected[index];
        sum_sq += value * value;
        max_abs = std::max(max_abs, std::abs(value));
    }
    const double rms = std::sqrt(static_cast<double>(sum_sq / sample_count));
    if (rms <= std::numeric_limits<double>::min()) {
        return 1.0;
    }

    constexpr std::array<double, 12> thresholds_in_rms = {
        2.5, 2.75, 3.0, 3.25, 3.5, 3.75,
        4.0, 4.5, 5.0, 6.0, 8.0, 10.0
    };
    std::array<double, thresholds_in_rms.size() + 1> candidates{};
    for (size_t i = 0; i < thresholds_in_rms.size(); ++i) {
        candidates[i] = thresholds_in_rms[i] * rms / quant_max;
    }
    candidates.back() = max_abs / quant_max;

    double best_step = candidates.front();
    long double best_error = std::numeric_limits<long double>::infinity();
    for (double step : candidates) {
        if (step <= std::numeric_limits<double>::min()) {
            continue;
        }
        long double error_sum = 0.0;
        for (size_t sample = 0; sample < sample_count; ++sample) {
            const size_t index = sample * projected.size() / sample_count;
            const double value = projected[index];
            const double quantized = std::clamp(
                std::round(value / step),
                -static_cast<double>(quant_max),
                static_cast<double>(quant_max)
            );
            const double error = value - quantized * step;
            error_sum += error * error;
        }
        if (error_sum < best_error) {
            best_error = error_sum;
            best_step = step;
        }
    }
    return best_step;
}

void quantize_and_pack(
    const std::vector<uint8_t> &input_phase,
    size_t input_record_stride,
    const std::vector<float> &projected,
    int output_dim,
    int quant_max,
    double quantization_step,
    int thread_count,
    std::vector<uint8_t> &output_phase,
    double &mse,
    double &saturation_fraction
) {
    const size_t n_records = projected.size() / static_cast<size_t>(output_dim);
    const size_t output_record_stride = sizeof(int16_t) + static_cast<size_t>(output_dim);
    output_phase.resize(n_records * output_record_stride);
    std::vector<long double> error_sums(static_cast<size_t>(thread_count), 0.0);
    std::vector<uint64_t> saturation_counts(static_cast<size_t>(thread_count), 0);

    parallel_ranges(n_records, thread_count, [&](size_t begin, size_t end, int thread) {
        long double error_sum = 0.0;
        uint64_t saturation_count = 0;
        for (size_t record = begin; record < end; ++record) {
            const uint8_t *input_record =
                input_phase.data() + record * input_record_stride;
            uint8_t *output_record =
                output_phase.data() + record * output_record_stride;
            std::memcpy(output_record, input_record, sizeof(int16_t));
            const float *source =
                projected.data() + record * static_cast<size_t>(output_dim);
            for (int dim = 0; dim < output_dim; ++dim) {
                const int quantized = std::clamp(
                    static_cast<int>(std::lround(source[dim] / quantization_step)),
                    -quant_max,
                    quant_max
                );
                const int8_t signed_value = static_cast<int8_t>(quantized);
                std::memcpy(output_record + sizeof(int16_t) + dim, &signed_value, 1);
                const double error =
                    static_cast<double>(source[dim]) -
                    static_cast<double>(quantized) * quantization_step;
                error_sum += error * error;
                saturation_count += std::abs(quantized) == quant_max;
            }
        }
        error_sums[static_cast<size_t>(thread)] = error_sum;
        saturation_counts[static_cast<size_t>(thread)] = saturation_count;
    });

    const long double error_sum =
        std::accumulate(error_sums.begin(), error_sums.end(), 0.0L);
    const uint64_t saturation_count =
        std::accumulate(saturation_counts.begin(), saturation_counts.end(), uint64_t{0});
    const size_t value_count = projected.size();
    mse = value_count == 0 ? 0.0 :
        static_cast<double>(error_sum / static_cast<long double>(value_count));
    saturation_fraction = value_count == 0 ? 0.0 :
        static_cast<double>(saturation_count) / static_cast<double>(value_count);
}

void write_summary(
    const Config &cfg,
    const Egev4Info &info,
    const std::vector<PhaseSummary> &summaries
) {
    if (cfg.summary.empty()) {
        return;
    }
    if (!cfg.summary.parent_path().empty()) {
        fs::create_directories(cfg.summary.parent_path());
    }
    std::ofstream output(cfg.summary);
    if (!output) {
        throw std::runtime_error("cannot open summary output: " + cfg.summary.string());
    }
    output << std::setprecision(10);
    output << "EGEV4 FM principal-component projection experiment\n";
    output << "input=" << cfg.input.string() << "\n";
    output << "output=" << cfg.output.string() << "\n";
    output << "input_timestamp=" << info.input_timestamp << "\n";
    output << "output_timestamp=" << info.output_timestamp << "\n";
    output << "version=" << info.version << "\n";
    output << "n_phases=" << info.n_phases << "\n";
    output << "params_per_phase=" << info.params_per_phase << "\n";
    output << "input_dim=" << info.input_dim << "\n";
    output << "output_dim=" << cfg.output_dim << "\n";
    output << "quant_max=" << cfg.quant_max << "\n";
    output << "covariance_sample_limit_per_phase=" << COVARIANCE_SAMPLE_LIMIT << "\n";
    output << "method="
           << (cfg.identity_basis ?
               "identity projection independently requantized for each phase" :
               "uncentered principal component analysis independently for each phase")
           << "\n";
    output << "component_rotation="
           << (cfg.identity_basis ? "identity basis" :
               (cfg.balance_components ? "orthonormal DCT within retained subspace" : "none"))
           << "\n";
    output << "note=Linear weights and linear scales are copied unchanged. Projected FM vectors are requantized to integers in the inclusive range [-quant_max, quant_max] and stored as signed 8-bit integers. Each phase FM vector scale is multiplied by its selected quantization step.\n";
    output << "phase\tinput_vector_scale\toutput_vector_scale\tquantization_step\tretained_energy_ratio\tprojected_quantization_mse\tsaturation_fraction\tselected_eigenvalues\n";
    for (const PhaseSummary &summary : summaries) {
        output
            << summary.phase << '\t'
            << summary.input_vector_scale << '\t'
            << summary.output_vector_scale << '\t'
            << summary.quantization_step << '\t'
            << summary.retained_energy_ratio << '\t'
            << summary.projected_quantization_mse << '\t'
            << summary.saturation_fraction << '\t';
        for (int dim = 0; dim < cfg.output_dim; ++dim) {
            if (dim != 0) {
                output << ',';
            }
            output << summary.selected_eigenvalues[dim];
        }
        output << '\n';
    }
}

void project_model(const Config &cfg) {
    std::ifstream input(cfg.input, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open EGEV4 input: " + cfg.input.string());
    }
    Egev4Info info = read_egev4_info(input, cfg.input, cfg.timestamp);
    if (cfg.output_dim > info.input_dim) {
        throw std::runtime_error("--output-dim must not exceed the input dimension");
    }
    if (cfg.identity_basis && cfg.output_dim != info.input_dim) {
        throw std::runtime_error(
            "--identity-basis requires --output-dim to equal the input dimension"
        );
    }
    const int thread_count = choose_thread_count(
        cfg,
        static_cast<size_t>(info.params_per_phase)
    );
    const size_t output_record_stride =
        sizeof(int16_t) + static_cast<size_t>(cfg.output_dim);
    const uint64_t expected_output_size =
        static_cast<uint64_t>(info.payload_offset) +
        static_cast<uint64_t>(N_PHASES) *
        static_cast<uint64_t>(info.params_per_phase) *
        static_cast<uint64_t>(output_record_stride);

    if (!cfg.output.parent_path().empty()) {
        fs::create_directories(cfg.output.parent_path());
    }
    std::ofstream output(cfg.output, std::ios::binary);
    if (!output) {
        throw std::runtime_error("cannot open EGEV4 output: " + cfg.output.string());
    }

    const int32_t output_dim_i32 = cfg.output_dim;
    std::memcpy(
        info.header.data() + TIMESTAMP_SIZE + 4 + 12,
        &output_dim_i32,
        sizeof(output_dim_i32)
    );
    output.write(
        reinterpret_cast<const char *>(info.header.data()),
        static_cast<std::streamsize>(info.header.size())
    );

    std::vector<PhaseSummary> summaries;
    summaries.reserve(N_PHASES);
    std::vector<uint8_t> input_phase(info.input_phase_stride);
    std::vector<float> projected;
    std::vector<uint8_t> output_phase;
    for (int phase = 0; phase < N_PHASES; ++phase) {
        input.read(
            reinterpret_cast<char *>(input_phase.data()),
            static_cast<std::streamsize>(input_phase.size())
        );
        if (!input) {
            throw std::runtime_error("input ended while reading phase " + std::to_string(phase));
        }

        const PcaBasis basis = cfg.identity_basis ?
            make_identity_basis(info.input_dim) :
            make_pca_basis(
                estimate_uncentered_gram(
                    input_phase,
                    info.input_record_stride,
                    static_cast<size_t>(info.params_per_phase),
                    info.input_dim
                ),
                info.input_dim,
                cfg.output_dim,
                cfg.balance_components
            );
        project_vectors(
            input_phase,
            info.input_record_stride,
            static_cast<size_t>(info.params_per_phase),
            info.input_dim,
            cfg.output_dim,
            basis,
            thread_count,
            projected
        );
        const double quantization_step = choose_quantization_step(
            projected, cfg.quant_max
        );
        double projected_quantization_mse = 0.0;
        double saturation_fraction = 0.0;
        quantize_and_pack(
            input_phase,
            info.input_record_stride,
            projected,
            cfg.output_dim,
            cfg.quant_max,
            quantization_step,
            thread_count,
            output_phase,
            projected_quantization_mse,
            saturation_fraction
        );
        output.write(
            reinterpret_cast<const char *>(output_phase.data()),
            static_cast<std::streamsize>(output_phase.size())
        );
        if (!output) {
            throw std::runtime_error("failed while writing phase " + std::to_string(phase));
        }

        info.output_vector_scales[phase] =
            static_cast<float>(
                static_cast<double>(info.input_vector_scales[phase]) *
                quantization_step
            );
        PhaseSummary summary;
        summary.phase = phase;
        summary.input_vector_scale = info.input_vector_scales[phase];
        summary.output_vector_scale = info.output_vector_scales[phase];
        summary.quantization_step = quantization_step;
        summary.retained_energy_ratio = basis.retained_energy_ratio;
        summary.projected_quantization_mse = projected_quantization_mse;
        summary.saturation_fraction = saturation_fraction;
        summary.selected_eigenvalues = basis.selected_eigenvalues;
        summaries.push_back(summary);
        std::cout
            << "phase=" << phase
            << " retained_energy_ratio=" << std::fixed << std::setprecision(8)
            << summary.retained_energy_ratio
            << " quantization_step=" << summary.quantization_step
            << " saturation_fraction=" << summary.saturation_fraction
            << std::endl;
    }
    output.close();
    if (!output) {
        throw std::runtime_error("failed while closing EGEV4 output");
    }

    {
        std::fstream header_output(cfg.output, std::ios::binary | std::ios::in | std::ios::out);
        if (!header_output) {
            throw std::runtime_error("cannot reopen output to update FM vector scales");
        }
        const size_t vector_scales_offset =
            info.fixed_header_size + sizeof(float) * N_PHASES;
        header_output.seekp(static_cast<std::streamoff>(vector_scales_offset));
        header_output.write(
            reinterpret_cast<const char *>(info.output_vector_scales.data()),
            sizeof(info.output_vector_scales)
        );
        if (!header_output) {
            throw std::runtime_error("failed while updating output FM vector scales");
        }
    }

    const uint64_t actual_output_size = fs::file_size(cfg.output);
    if (actual_output_size != expected_output_size) {
        throw std::runtime_error(
            "EGEV4 output size mismatch: got " + std::to_string(actual_output_size) +
            " expected " + std::to_string(expected_output_size)
        );
    }
    write_summary(cfg, info, summaries);
    const double mean_retained_energy = std::accumulate(
        summaries.begin(),
        summaries.end(),
        0.0,
        [](double sum, const PhaseSummary &summary) {
            return sum + summary.retained_energy_ratio;
        }
    ) / static_cast<double>(summaries.size());
    std::cout
        << "wrote " << cfg.output.string() << "\n"
        << "version=" << info.version
        << " phases=" << info.n_phases
        << " params_per_phase=" << info.params_per_phase
        << " input_dim=" << info.input_dim
        << " output_dim=" << cfg.output_dim
        << " quant_max=" << cfg.quant_max
        << " threads=" << thread_count << "\n"
        << "mean_phase_retained_energy_ratio=" << std::setprecision(10)
        << mean_retained_energy << std::endl;
}

int main(int argc, char **argv) {
    try {
        project_model(parse_args(argc, argv));
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "error: " << error.what() << std::endl;
        return 1;
    }
}
