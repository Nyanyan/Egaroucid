#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <ctime>
#include <sstream>
#include <cstdint>
#include <algorithm>
#include <cmath>

constexpr int DEFAULT_FM_DIM = 6;
constexpr int MAGIC_SIZE = 4;
constexpr int TIMESTAMP_SIZE = 14;
constexpr int FM_EVAL_VERSION_PACKED64 = 7;
constexpr int FM_INT8_MAX_ABS = 127;
constexpr int LINEAR_INT16_MAX_ABS = 32767;

struct FMHeaderCommon {
    char magic[MAGIC_SIZE]; // EGEV
    int version;
    int n_phases;
    int n_params;
    int fm_dim;
};

struct FMHeaderScales {
    float linear_scale;
    float factor_scale;
};

static std::string create_timestamp_yyyymmddhhmmss() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm local_tm{};
#ifdef _MSC_VER
    localtime_s(&local_tm, &now_time);
#else
    local_tm = *std::localtime(&now_time);
#endif
    std::ostringstream oss;
    oss << (local_tm.tm_year + 1900);
    if (local_tm.tm_mon + 1 < 10) oss << '0';
    oss << (local_tm.tm_mon + 1);
    if (local_tm.tm_mday < 10) oss << '0';
    oss << local_tm.tm_mday;
    if (local_tm.tm_hour < 10) oss << '0';
    oss << local_tm.tm_hour;
    if (local_tm.tm_min < 10) oss << '0';
    oss << local_tm.tm_min;
    if (local_tm.tm_sec < 10) oss << '0';
    oss << local_tm.tm_sec;
    return oss.str();
}

static bool read_phase_file(
    const std::string& path,
    int n_params,
    int fm_dim,
    std::vector<float>& linear,
    std::vector<float>& factor) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        return false;
    }
    std::vector<double> raw;
    raw.reserve((size_t)n_params * (size_t)(fm_dim + 1));
    double value;
    while (ifs >> value) {
        raw.emplace_back(value);
    }

    const size_t expected_full = (size_t)n_params * (size_t)(fm_dim + 1);
    const size_t expected_factor_only = (size_t)n_params * (size_t)fm_dim;
    if (raw.size() == expected_full) {
        for (int i = 0; i < n_params; ++i) {
            linear[(size_t)i] = (float)raw[(size_t)i];
        }
        size_t offset = (size_t)n_params;
        for (int i = 0; i < n_params * fm_dim; ++i) {
            factor[(size_t)i] = (float)raw[offset + (size_t)i];
        }
        return true;
    }
    if (raw.size() == expected_factor_only) {
        std::fill(linear.begin(), linear.end(), 0.0f);
        for (int i = 0; i < n_params * fm_dim; ++i) {
            factor[(size_t)i] = (float)raw[(size_t)i];
        }
        return true;
    }

    std::cerr << "[ERROR] invalid element count in " << path
              << " found=" << raw.size()
              << " expected=" << expected_full
              << " (or " << expected_factor_only << " for factor-only)" << std::endl;
    return false;
}

static uint64_t pack_param(int16_t linear_q, const int8_t* factor_q6) {
    uint64_t packed = (uint16_t)linear_q;
    for (int f = 0; f < DEFAULT_FM_DIM; ++f) {
        packed |= (uint64_t)(uint8_t)factor_q6[f] << (16 + f * 8);
    }
    return packed;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "input [n_phases] [n_params] [fm_dim=" << DEFAULT_FM_DIM
                  << "] [out_file=trained/eval_fm.egev4] [model_dir=trained]" << std::endl;
        return 1;
    }

    const int n_phases = atoi(argv[1]);
    const int n_params = atoi(argv[2]);
    const int fm_dim = (argc >= 4) ? atoi(argv[3]) : DEFAULT_FM_DIM;
    const std::string out_file = (argc >= 5) ? argv[4] : "trained/eval_fm.egev4";
    const std::string model_dir = (argc >= 6) ? argv[5] : "trained";

    if (n_phases <= 0 || n_params <= 0 || fm_dim != DEFAULT_FM_DIM) {
        std::cerr << "[ERROR] invalid args. n_phases/n_params must be > 0 and fm_dim must be " << DEFAULT_FM_DIM << std::endl;
        return 1;
    }

    std::cerr << "n_phases " << n_phases
              << " n_params " << n_params
              << " fm_dim " << fm_dim
              << " out_file " << out_file
              << " model_dir " << model_dir
              << std::endl;

    std::ofstream fout(out_file, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!fout.is_open()) {
        std::cerr << "[ERROR] can't open " << out_file << std::endl;
        return 1;
    }

    const std::string created_at = create_timestamp_yyyymmddhhmmss();
    if (created_at.size() != TIMESTAMP_SIZE) {
        std::cerr << "[ERROR] failed to create timestamp" << std::endl;
        return 1;
    }

    std::vector<float> linear((size_t)n_params, 0.0f);
    std::vector<float> factor((size_t)n_params * (size_t)fm_dim, 0.0f);
    std::vector<float> linear_scales((size_t)n_phases, 1.0f);
    std::vector<float> factor_scales((size_t)n_phases, 1.0f);

    int loaded = 0;
    int missing = 0;
    float max_abs_linear_global = 0.0f;
    float max_abs_factor_global = 0.0f;

    for (int phase = 0; phase < n_phases; ++phase) {
        std::fill(linear.begin(), linear.end(), 0.0f);
        std::fill(factor.begin(), factor.end(), 0.0f);

        const std::string fm_path = model_dir + "/" + std::to_string(phase) + "_fm.txt";
        if (!read_phase_file(fm_path, n_params, fm_dim, linear, factor)) {
            ++missing;
            std::cerr << "phase " << phase << " missing -> zero fill" << std::endl;
            continue;
        }
        ++loaded;
        float max_abs_linear = 0.0f;
        float max_abs_factor = 0.0f;
        for (float v : linear) {
            max_abs_linear = std::max(max_abs_linear, std::fabs(v));
        }
        for (float v : factor) {
            max_abs_factor = std::max(max_abs_factor, std::fabs(v));
        }
        linear_scales[(size_t)phase] = (max_abs_linear > 0.0f) ? (max_abs_linear / (float)LINEAR_INT16_MAX_ABS) : 1.0f;
        factor_scales[(size_t)phase] = (max_abs_factor > 0.0f) ? (max_abs_factor / (float)FM_INT8_MAX_ABS) : 1.0f;
        max_abs_linear_global = std::max(max_abs_linear_global, max_abs_linear);
        max_abs_factor_global = std::max(max_abs_factor_global, max_abs_factor);
    }

    FMHeaderCommon header_common = {{'E', 'G', 'E', 'V'}, FM_EVAL_VERSION_PACKED64, n_phases, n_params, fm_dim};
    fout.write(created_at.data(), TIMESTAMP_SIZE);
    fout.write((char*)&header_common, sizeof(FMHeaderCommon));
    fout.write((char*)linear_scales.data(), sizeof(float) * linear_scales.size());
    fout.write((char*)factor_scales.data(), sizeof(float) * factor_scales.size());
    std::cerr << "created_at " << created_at
              << " version " << FM_EVAL_VERSION_PACKED64
              << " global_max_linear " << max_abs_linear_global
              << " global_max_factor " << max_abs_factor_global << std::endl;

    loaded = 0;
    missing = 0;
    std::vector<uint64_t> packed((size_t)n_params, 0ULL);

    for (int phase = 0; phase < n_phases; ++phase) {
        std::fill(linear.begin(), linear.end(), 0.0f);
        std::fill(factor.begin(), factor.end(), 0.0f);
        std::fill(packed.begin(), packed.end(), 0ULL);

        const std::string fm_path = model_dir + "/" + std::to_string(phase) + "_fm.txt";

        if (!read_phase_file(fm_path, n_params, fm_dim, linear, factor)) {
            ++missing;
            std::cerr << "phase " << phase << " missing -> zero fill" << std::endl;
        } else {
            ++loaded;
            std::cerr << "phase " << phase << " loaded linear+FM" << std::endl;
            const float linear_scale = linear_scales[(size_t)phase];
            const float factor_scale = factor_scales[(size_t)phase];
            for (int p = 0; p < n_params; ++p) {
                const int q_linear = (int)std::round(linear[(size_t)p] / linear_scale);
                const int16_t q_linear_clamped = (int16_t)std::clamp(q_linear, -LINEAR_INT16_MAX_ABS, LINEAR_INT16_MAX_ABS);
                int8_t q_factor[DEFAULT_FM_DIM];
                for (int f = 0; f < DEFAULT_FM_DIM; ++f) {
                    const float fv = factor[(size_t)p * (size_t)fm_dim + (size_t)f];
                    const int q = (int)std::round(fv / factor_scale);
                    q_factor[f] = (int8_t)std::clamp(q, -FM_INT8_MAX_ABS, FM_INT8_MAX_ABS);
                }
                packed[(size_t)p] = pack_param(q_linear_clamped, q_factor);
            }
        }

        fout.write((char*)packed.data(), sizeof(uint64_t) * packed.size());
    }

    std::cerr << "loaded " << loaded
              << " missing " << missing << std::endl;
    std::cerr << "done" << std::endl;
    return 0;
}