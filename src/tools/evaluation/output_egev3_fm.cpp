#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <ctime>
#include <sstream>

constexpr int DEFAULT_FM_DIM = 16;
constexpr int MAGIC_SIZE = 4;
constexpr int TIMESTAMP_SIZE = 14;

struct FMHeader {
    char magic[MAGIC_SIZE]; // EGFM
    int version;
    int n_phases;
    int n_linear_params;
    int fm_dim;
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
    int n_linear_params,
    int fm_dim,
    std::vector<float>& linear,
    std::vector<float>& factor,
    bool allow_linear_only) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        return false;
    }
    std::vector<double> raw;
    raw.reserve((size_t)n_linear_params * (size_t)(fm_dim + 1));
    double value;
    while (ifs >> value) {
        raw.emplace_back(value);
    }

    const size_t expected_full = (size_t)n_linear_params * (size_t)(fm_dim + 1);
    const size_t expected_linear = (size_t)n_linear_params;
    if (raw.size() == expected_full) {
        for (int i = 0; i < n_linear_params; ++i) {
            linear[(size_t)i] = (float)raw[(size_t)i];
        }
        size_t offset = (size_t)n_linear_params;
        for (int i = 0; i < n_linear_params * fm_dim; ++i) {
            factor[(size_t)i] = (float)raw[offset + (size_t)i];
        }
        return true;
    }
    if (allow_linear_only && raw.size() == expected_linear) {
        for (int i = 0; i < n_linear_params; ++i) {
            linear[(size_t)i] = (float)raw[(size_t)i];
        }
        return true;
    }

    std::cerr << "[ERROR] invalid element count in " << path
              << " found=" << raw.size()
              << " expected=" << expected_full
              << " (or " << expected_linear << " for linear-only)" << std::endl;
    return false;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "input [n_phases] [n_linear_params] [fm_dim=" << DEFAULT_FM_DIM
                  << "] [out_file=trained/eval_fm.egev3] [model_dir=trained]" << std::endl;
        return 1;
    }

    const int n_phases = atoi(argv[1]);
    const int n_linear_params = atoi(argv[2]);
    const int fm_dim = (argc >= 4) ? atoi(argv[3]) : DEFAULT_FM_DIM;
    const std::string out_file = (argc >= 5) ? argv[4] : "trained/eval_fm.egev3";
    const std::string model_dir = (argc >= 6) ? argv[5] : "trained";

    if (n_phases <= 0 || n_linear_params <= 0 || fm_dim <= 0) {
        std::cerr << "[ERROR] invalid args. n_phases, n_linear_params, fm_dim must be > 0" << std::endl;
        return 1;
    }

    std::cerr << "n_phases " << n_phases
              << " n_linear_params " << n_linear_params
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
    FMHeader header = {{'E', 'G', 'E', 'V'}, 3, n_phases, n_linear_params, fm_dim};
    fout.write(created_at.data(), TIMESTAMP_SIZE);
    fout.write((char*)&header, sizeof(FMHeader));
    std::cerr << "created_at " << created_at << std::endl;

    std::vector<float> linear((size_t)n_linear_params, 0.0f);
    std::vector<float> factor((size_t)n_linear_params * (size_t)fm_dim, 0.0f);

    int loaded_full = 0;
    int loaded_linear_only = 0;
    int missing = 0;

    for (int phase = 0; phase < n_phases; ++phase) {
        std::fill(linear.begin(), linear.end(), 0.0f);
        std::fill(factor.begin(), factor.end(), 0.0f);

        const std::string fm_path = model_dir + "/" + std::to_string(phase) + "_fm.txt";
        const std::string linear_path = model_dir + "/" + std::to_string(phase) + ".txt";

        bool ok = false;
        bool full = false;
        {
            std::ifstream test(fm_path);
            if (test.is_open()) {
                ok = read_phase_file(fm_path, n_linear_params, fm_dim, linear, factor, true);
                if (ok) {
                    std::ifstream ifs_count(fm_path);
                    double tmp;
                    int c = 0;
                    while (ifs_count >> tmp) {
                        ++c;
                    }
                    full = (c == n_linear_params * (fm_dim + 1));
                }
            }
        }
        if (!ok) {
            std::ifstream test(linear_path);
            if (test.is_open()) {
                ok = read_phase_file(linear_path, n_linear_params, fm_dim, linear, factor, true);
                full = false;
            }
        }

        if (!ok) {
            ++missing;
            std::cerr << "phase " << phase << " missing -> zero fill" << std::endl;
        } else if (full) {
            ++loaded_full;
            std::cerr << "phase " << phase << " loaded full FM" << std::endl;
        } else {
            ++loaded_linear_only;
            std::cerr << "phase " << phase << " loaded linear-only" << std::endl;
        }

        fout.write((char*)linear.data(), sizeof(float) * linear.size());
        fout.write((char*)factor.data(), sizeof(float) * factor.size());
    }

    std::cerr << "loaded_full " << loaded_full
              << " loaded_linear_only " << loaded_linear_only
              << " missing " << missing << std::endl;
    std::cerr << "done" << std::endl;
    return 0;
}