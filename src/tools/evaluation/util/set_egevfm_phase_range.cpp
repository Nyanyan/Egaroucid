/*
    Egaroucid Project

    @file set_egevfm_phase_range.cpp
        Set the FM application phase range flag in an egevfm file
    @date 2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

constexpr char EVAL_FM_FILE_MAGIC[8] = {'E', 'G', 'F', 'M', '0', '0', '1', '\0'};
constexpr size_t EVAL_FM_FLAGS_OFFSET = 8 + 7 * sizeof(uint32_t);
constexpr uint32_t EVAL_FM_FLAG_PHASE_RANGE = 0x80000000U;
constexpr uint32_t EVAL_FM_PHASE_START_SHIFT = 16;
constexpr uint32_t EVAL_FM_PHASE_END_SHIFT = 22;
constexpr uint32_t EVAL_FM_PHASE_FLAG_MASK = 0x3FU;
constexpr int N_PHASES_LOCAL = 60;

template<typename T>
bool read_scalar(const std::vector<char> &data, size_t offset, T *value) {
    if (offset + sizeof(T) > data.size()) {
        return false;
    }
    std::memcpy(value, data.data() + offset, sizeof(T));
    return true;
}

template<typename T>
bool write_scalar(std::vector<char> *data, size_t offset, const T value) {
    if (offset + sizeof(T) > data->size()) {
        return false;
    }
    std::memcpy(data->data() + offset, &value, sizeof(T));
    return true;
}

int main(int argc, char **argv) {
    if (argc != 5) {
        std::cerr << "usage: set_egevfm_phase_range [in.egevfm] [out.egevfm] [start_phase] [end_phase]\n";
        return 1;
    }

    const std::string in_file = argv[1];
    const std::string out_file = argv[2];
    const int start_phase = std::stoi(argv[3]);
    const int end_phase = std::stoi(argv[4]);
    if (start_phase < 0 || end_phase < start_phase || end_phase >= N_PHASES_LOCAL) {
        std::cerr << "[ERROR] invalid phase range " << start_phase << "-" << end_phase << "\n";
        return 1;
    }

    std::ifstream in(in_file, std::ios::binary);
    if (!in) {
        std::cerr << "[ERROR] can't open input " << in_file << "\n";
        return 1;
    }
    std::vector<char> data((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (data.size() < 56 || std::memcmp(data.data(), EVAL_FM_FILE_MAGIC, sizeof(EVAL_FM_FILE_MAGIC)) != 0) {
        std::cerr << "[ERROR] input is not egevfm v1: " << in_file << "\n";
        return 1;
    }

    uint32_t old_flags = 0;
    if (!read_scalar(data, EVAL_FM_FLAGS_OFFSET, &old_flags)) {
        std::cerr << "[ERROR] broken flags in " << in_file << "\n";
        return 1;
    }
    uint32_t new_flags = old_flags & 0x0000FFFFU;
    new_flags |= EVAL_FM_FLAG_PHASE_RANGE;
    new_flags |= ((uint32_t)start_phase & EVAL_FM_PHASE_FLAG_MASK) << EVAL_FM_PHASE_START_SHIFT;
    new_flags |= ((uint32_t)end_phase & EVAL_FM_PHASE_FLAG_MASK) << EVAL_FM_PHASE_END_SHIFT;
    if (!write_scalar(&data, EVAL_FM_FLAGS_OFFSET, new_flags)) {
        std::cerr << "[ERROR] failed to write flags\n";
        return 1;
    }

    std::filesystem::path out_path(out_file);
    if (out_path.has_parent_path()) {
        std::filesystem::create_directories(out_path.parent_path());
    }
    std::ofstream out(out_file, std::ios::binary | std::ios::trunc);
    if (!out) {
        std::cerr << "[ERROR] can't open output " << out_file << "\n";
        return 1;
    }
    out.write(data.data(), (std::streamsize)data.size());
    if (!out) {
        std::cerr << "[ERROR] failed to write output " << out_file << "\n";
        return 1;
    }

    std::ofstream summary(out_file + ".summary.txt", std::ios::trunc);
    if (summary) {
        summary << "source " << in_file << "\n";
        summary << "old_flags " << old_flags << "\n";
        summary << "new_flags " << new_flags << "\n";
        summary << "active_pattern_mask 0x" << std::hex << (new_flags & 0xFFFFU) << std::dec << "\n";
        summary << "apply_phase_start " << start_phase << "\n";
        summary << "apply_phase_end " << end_phase << "\n";
    }
    std::cout << "wrote " << out_file
              << " old_flags " << old_flags
              << " new_flags " << new_flags
              << " apply_phase_range " << start_phase << "-" << end_phase << "\n";
    return 0;
}
