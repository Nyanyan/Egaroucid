/*
    Egaroucid Project

    @file scale_egevfm.cpp
        Change FM scale in an egevfm file to attenuate or amplify the FM term
    @date 2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

constexpr char EVAL_FM_FILE_MAGIC[8] = {'E', 'G', 'F', 'M', '0', '0', '1', '\0'};
constexpr size_t FM_SCALE_OFFSET = 8 + 6 * sizeof(uint32_t);

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
    if (argc != 4) {
        std::cerr << "usage: scale_egevfm [in.egevfm] [out.egevfm] [new_fm_scale]" << std::endl;
        return 1;
    }

    const std::string in_file = argv[1];
    const std::string out_file = argv[2];
    int32_t new_scale = 0;
    try {
        new_scale = std::stoi(argv[3]);
    } catch (...) {
        std::cerr << "[ERROR] invalid new_fm_scale" << std::endl;
        return 1;
    }
    if (new_scale <= 0) {
        std::cerr << "[ERROR] new_fm_scale must be positive" << std::endl;
        return 1;
    }

    std::ifstream in(in_file, std::ios::binary);
    if (!in) {
        std::cerr << "[ERROR] can't open input " << in_file << std::endl;
        return 1;
    }
    std::vector<char> data((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (data.size() < 56 || std::memcmp(data.data(), EVAL_FM_FILE_MAGIC, sizeof(EVAL_FM_FILE_MAGIC)) != 0) {
        std::cerr << "[ERROR] input is not egevfm v1: " << in_file << std::endl;
        return 1;
    }

    int32_t old_scale = 0;
    if (!read_scalar(data, FM_SCALE_OFFSET, &old_scale) || old_scale <= 0) {
        std::cerr << "[ERROR] broken fm_scale in " << in_file << std::endl;
        return 1;
    }
    if (!write_scalar(&data, FM_SCALE_OFFSET, new_scale)) {
        std::cerr << "[ERROR] failed to write fm_scale" << std::endl;
        return 1;
    }

    std::ofstream out(out_file, std::ios::binary);
    if (!out) {
        std::cerr << "[ERROR] can't open output " << out_file << std::endl;
        return 1;
    }
    out.write(data.data(), (std::streamsize)data.size());
    if (!out) {
        std::cerr << "[ERROR] failed to write output " << out_file << std::endl;
        return 1;
    }

    const double multiplier = ((double)old_scale * (double)old_scale) / ((double)new_scale * (double)new_scale);
    std::ofstream summary(out_file + ".summary.txt");
    if (summary) {
        summary << "source " << in_file << "\n";
        summary << "old_fm_scale " << old_scale << "\n";
        summary << "new_fm_scale " << new_scale << "\n";
        summary << "fm_term_multiplier " << multiplier << "\n";
    }
    std::cout << "wrote " << out_file
              << " old_fm_scale " << old_scale
              << " new_fm_scale " << new_scale
              << " fm_term_multiplier " << multiplier << std::endl;
    return 0;
}
