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

constexpr int DEFAULT_FM_DIM = 4;
constexpr int MAGIC_SIZE = 4;
constexpr int TIMESTAMP_SIZE = 14;

struct FMHeader {
    char magic[MAGIC_SIZE]; // EGEV
    int version;
    int n_phases;
    int n_params;
    int fm_dim;
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
        size_t offset = (size_t)n_params;
        for (int i = 0; i < n_params * fm_dim; ++i) {
            factor[(size_t)i] = (float)raw[offset + (size_t)i];
        }
        return true;
    }
    if (raw.size() == expected_factor_only) {
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

    if (n_phases <= 0 || n_params <= 0 || fm_dim <= 0) {
        std::cerr << "[ERROR] invalid args. n_phases, n_params, fm_dim must be > 0" << std::endl;
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
    std::vector<float> factor((size_t)n_params * (size_t)fm_dim, 0.0f);
    std::vector<int8_t> factor_q((size_t)n_params * (size_t)fm_dim, 0);

    int loaded = 0;
    int missing = 0;
    float max_abs = 0.0f;

    for (int phase = 0; phase < n_phases; ++phase) {
        std::fill(factor.begin(), factor.end(), 0.0f);

        const std::string fm_path = model_dir + "/" + std::to_string(phase) + "_fm.txt";
        if (!read_phase_file(fm_path, n_params, fm_dim, factor)) {
            ++missing;
            std::cerr << "phase " << phase << " missing -> zero fill" << std::endl;
            continue;
        }
        ++loaded;
        for (float v : factor) {
            max_abs = std::max(max_abs, std::fabs(v));
        }
    }

    const float factor_scale = (max_abs > 0.0f) ? (max_abs / 127.0f) : 1.0f;
    FMHeader header = {{'E', 'G', 'E', 'V'}, 4, n_phases, n_params, fm_dim, factor_scale};
    fout.write(created_at.data(), TIMESTAMP_SIZE);
    fout.write((char*)&header, sizeof(FMHeader));
    std::cerr << "created_at " << created_at << " factor_scale " << factor_scale << std::endl;

    loaded = 0;
    missing = 0;

    for (int phase = 0; phase < n_phases; ++phase) {
        std::fill(factor.begin(), factor.end(), 0.0f);
        std::fill(factor_q.begin(), factor_q.end(), 0);

        const std::string fm_path = model_dir + "/" + std::to_string(phase) + "_fm.txt";

        if (!read_phase_file(fm_path, n_params, fm_dim, factor)) {
            ++missing;
            std::cerr << "phase " << phase << " missing -> zero fill" << std::endl;
        } else {
            ++loaded;
            std::cerr << "phase " << phase << " loaded FM" << std::endl;
            for (size_t i = 0; i < factor.size(); ++i) {
                const float scaled = factor[i] / factor_scale;
                const int q = (int)std::round(scaled);
                const int q_clamped = std::clamp(q, -127, 127);
                factor_q[i] = (int8_t)q_clamped;
            }
        }

        fout.write((char*)factor_q.data(), sizeof(int8_t) * factor_q.size());
    }

    std::cerr << "loaded " << loaded
              << " missing " << missing << std::endl;
    std::cerr << "done" << std::endl;
    return 0;
}